"""Unit tests for app/services/console/service.py, derived only from its spec.

These tests exercise AuthService, PermissionService, StaffAdminService,
AuditService and OverrideService directly, injecting hand-written fakes in
place of the repository/audit/rate-limiter collaborators (per
project_rules.testing's "no live infrastructure" rule -- there is no real
database here, only plain Python doubles standing in for the repository
classes named in the spec's Public surface).

Several repository method names are not spelled out verbatim in the spec
(only a handful are, e.g. StaffUserRepository.get_by_username,
SessionRepository.create/revoke_all_for_user, RolePermissionRepository
.is_allowed, AppointmentRepository/OutreachMessageRepository.update_status,
AuditLogRepository "insert"). Where a name is not given, the fakes below
accept *any* attribute access (via __getattr__) and record what was called
with, and assertions are written against the observable values passed
around rather than a guessed method name, so the tests stay anchored to the
spec's behavioural claims rather than to invented internals.
"""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.common.exceptions.errors import (
    AccountLockedError,
    AppError,
    DuplicateStaffAccountError,
    InvalidCredentialsError,
    InvalidResetTokenError,
    IrreversibleActionError,
    NotFoundError,
    RateLimitedError,
)
from app.services.console.service import (
    AuditService,
    AuthService,
    OverrideService,
    PermissionService,
    StaffAdminService,
)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def now_utc():
    return datetime.now(timezone.utc)


def flatten(args, kwargs):
    """Flatten a call's positional + keyword values into one list, so
    assertions can look for a particular literal without depending on
    whether the real implementation passed it positionally or by name."""
    return list(args) + list(kwargs.values())


def call_values(mock_call):
    return flatten(mock_call.args, mock_call.kwargs)


def exception_mentions(exc, value):
    haystacks = [str(exc)]
    for attr in ("details", "args", "context", "data"):
        if hasattr(exc, attr):
            haystacks.append(str(getattr(exc, attr)))
    return any(str(value) in haystack for haystack in haystacks)


def make_staff_user(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        username="jdoe",
        email="jdoe@example.com",
        role="front_office_staff",
        status="active",
        failed_attempt_count=2,
        last_login_at=None,
        # several plausible names for whatever field holds the password
        # hash, so that regardless of which one the implementation reads,
        # an attribute is present (its value is irrelevant since
        # verify_password itself is monkeypatched in these tests).
        password_hash="stub-hash",
        hashed_password="stub-hash",
        password="stub-hash",
        pwd_hash="stub-hash",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def patch_verify_password(monkeypatch, result):
    """Force core.security.verify_password's outcome deterministically,
    wherever app.services.console.service happens to have bound it (either
    `from app.core.security import verify_password` or a module-qualified
    call), so login tests do not depend on a real bcrypt hash matching an
    unspecified StaffUser attribute name."""
    import app.core.security as core_security

    fake = MagicMock(return_value=result)
    monkeypatch.setattr(core_security, "verify_password", fake, raising=False)

    import app.services.console.service as svc_module

    if hasattr(svc_module, "verify_password"):
        monkeypatch.setattr(svc_module, "verify_password", fake, raising=False)
    return fake


class FakeStaffRepo:
    def __init__(self):
        self.get_by_username = AsyncMock(return_value=None)
        self.get_by_email = AsyncMock(return_value=None)
        self.get_by_id = AsyncMock(return_value=None)
        self.increment_failed_attempts = AsyncMock(return_value=1)
        self.update_password = AsyncMock(return_value=None)
        self.create = AsyncMock()


class FakeSessionRepo:
    def __init__(self):
        self.create = AsyncMock(
            return_value=SimpleNamespace(id=uuid.uuid4(), jti=str(uuid.uuid4()))
        )
        self.revoke_all_for_user = AsyncMock(return_value=0)


class FakeResetRepo:
    def __init__(self):
        self.create = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        self.get_by_token = AsyncMock(return_value=None)
        self.mark_used = AsyncMock(return_value=None)


class FakeRolePermRepo:
    def __init__(self):
        self.is_allowed = AsyncMock(return_value=False)


class FakeAudit:
    """Stands in for the injected AuditService collaborator."""

    def __init__(self):
        self.record = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))


class FakeRateLimiter:
    def __init__(self, raise_error=None):
        if raise_error is not None:
            self.check = AsyncMock(side_effect=raise_error)
        else:
            self.check = AsyncMock(return_value=None)


# =========================================================================== #
# AuthService.login
# =========================================================================== #

class TestAuthServiceLogin:
    async def test_successful_login_resets_attempts_creates_session_audits_then_returns_tokens(
        self, monkeypatch
    ):
        user = make_staff_user(failed_attempt_count=4, last_login_at=None)
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_username = AsyncMock(return_value=user)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()
        order = []

        async def _session_create(*args, **kwargs):
            order.append("session_create")
            return SimpleNamespace(id=uuid.uuid4(), jti=str(uuid.uuid4()))

        async def _audit_record(*args, **kwargs):
            order.append("audit_record")
            return SimpleNamespace(id=uuid.uuid4())

        session_repo.create = AsyncMock(side_effect=_session_create)
        audit.record = AsyncMock(side_effect=_audit_record)
        patch_verify_password(monkeypatch, True)

        service = AuthService(staff_repo, session_repo, FakeResetRepo(), audit, FakeRateLimiter())
        db = MagicMock()

        access, refresh, expires_in, returned_user = await service.login(
            db, "jdoe", "correct-horse-battery"
        )

        assert isinstance(access, str) and access
        assert isinstance(refresh, str) and refresh
        assert access != refresh
        assert expires_in == 15 * 60  # access tokens are 15 minutes per project rules
        assert returned_user is user
        assert user.failed_attempt_count == 0
        assert user.last_login_at is not None
        assert session_repo.create.await_count == 1
        assert audit.record.await_count == 1
        values = call_values(audit.record.call_args)
        assert "staff" in values
        assert "auth.login" in values
        assert order == ["session_create", "audit_record"]

    async def test_wrong_password_raises_invalid_credentials_and_increments_failed_attempts(
        self, monkeypatch
    ):
        user = make_staff_user()
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_username = AsyncMock(return_value=user)
        staff_repo.increment_failed_attempts = AsyncMock(
            return_value=user.failed_attempt_count + 1
        )
        session_repo = FakeSessionRepo()
        audit = FakeAudit()
        patch_verify_password(monkeypatch, False)

        service = AuthService(staff_repo, session_repo, FakeResetRepo(), audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(InvalidCredentialsError):
            await service.login(db, "jdoe", "wrong-password")

        assert staff_repo.increment_failed_attempts.await_count == 1
        assert audit.record.await_count == 1
        assert session_repo.create.await_count == 0

    async def test_unknown_username_raises_invalid_credentials_and_is_audited(self, monkeypatch):
        staff_repo = FakeStaffRepo()  # get_by_username -> None
        session_repo = FakeSessionRepo()
        audit = FakeAudit()
        patch_verify_password(monkeypatch, False)

        service = AuthService(staff_repo, session_repo, FakeResetRepo(), audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(InvalidCredentialsError):
            await service.login(db, "ghost", "whatever")

        assert audit.record.await_count == 1
        assert session_repo.create.await_count == 0

    async def test_login_still_returns_401_on_the_request_that_crosses_lockout_threshold(
        self, monkeypatch
    ):
        user = make_staff_user(failed_attempt_count=0)
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_username = AsyncMock(return_value=user)
        # a huge count comfortably clears any real LOGIN_LOCKOUT_THRESHOLD
        staff_repo.increment_failed_attempts = AsyncMock(return_value=9999)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()
        patch_verify_password(monkeypatch, False)

        service = AuthService(staff_repo, session_repo, FakeResetRepo(), audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(InvalidCredentialsError):
            await service.login(db, "jdoe", "wrong-password")

        assert session_repo.create.await_count == 0

    async def test_login_against_already_locked_account_raises_account_locked_without_checking_password(
        self, monkeypatch
    ):
        user = make_staff_user(status="locked")
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_username = AsyncMock(return_value=user)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()
        # Even a "correct" password must not let this request succeed.
        verify_mock = patch_verify_password(monkeypatch, True)

        service = AuthService(staff_repo, session_repo, FakeResetRepo(), audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(AccountLockedError):
            await service.login(db, "jdoe", "correct-horse-battery")

        assert session_repo.create.await_count == 0
        assert verify_mock.called is False

    async def test_login_propagates_audit_write_failure_and_does_not_return_tokens(
        self, monkeypatch
    ):
        user = make_staff_user()
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_username = AsyncMock(return_value=user)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()
        audit.record = AsyncMock(side_effect=AppError(code="AUDIT_WRITE_FAILED"))
        patch_verify_password(monkeypatch, True)

        service = AuthService(staff_repo, session_repo, FakeResetRepo(), audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(AppError) as exc_info:
            await service.login(db, "jdoe", "correct-horse-battery")

        assert exc_info.value.code == "AUDIT_WRITE_FAILED"


# =========================================================================== #
# AuthService.request_reset
# =========================================================================== #

class TestAuthServiceRequestReset:
    async def test_unknown_email_completes_without_raising_and_does_not_create_a_token(self):
        staff_repo = FakeStaffRepo()  # get_by_email -> None
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        audit = FakeAudit()
        rate_limiter = FakeRateLimiter()

        service = AuthService(staff_repo, session_repo, reset_repo, audit, rate_limiter)
        db = MagicMock()

        result = await service.request_reset(db, "nobody@example.com")

        assert result is None
        assert rate_limiter.check.await_count == 1
        assert reset_repo.create.await_count == 0

    async def test_rate_limit_check_runs_first_and_its_error_propagates(self):
        staff_repo = FakeStaffRepo()
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        audit = FakeAudit()
        rate_limiter = FakeRateLimiter(raise_error=RateLimitedError("too many requests"))

        service = AuthService(staff_repo, session_repo, reset_repo, audit, rate_limiter)
        db = MagicMock()

        with pytest.raises(RateLimitedError):
            await service.request_reset(db, "jdoe@example.com")

        assert reset_repo.create.await_count == 0

    async def test_known_email_creates_a_hashed_single_use_token_with_future_expiry(self):
        user = make_staff_user(email="jdoe@example.com")
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_email = AsyncMock(return_value=user)
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        audit = FakeAudit()
        rate_limiter = FakeRateLimiter()

        service = AuthService(staff_repo, session_repo, reset_repo, audit, rate_limiter)
        db = MagicMock()

        result = await service.request_reset(db, "jdoe@example.com")

        assert result is None
        assert rate_limiter.check.await_count == 1
        assert reset_repo.create.await_count == 1
        values = call_values(reset_repo.create.call_args)

        opaque_tokens = [
            v for v in values if isinstance(v, str) and v != user.email and len(v) >= 16
        ]
        assert opaque_tokens, "expected an opaque token_hash string to be persisted"

        expiries = [v for v in values if isinstance(v, datetime)]
        assert expiries, "expected an expires_at datetime to be persisted"
        assert expiries[0] > now_utc()


# =========================================================================== #
# AuthService.confirm_reset
# =========================================================================== #

class TestAuthServiceConfirmReset:
    async def test_unknown_token_raises_invalid_reset_token(self):
        staff_repo = FakeStaffRepo()
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        reset_repo.get_by_token = AsyncMock(return_value=None)
        audit = FakeAudit()

        service = AuthService(staff_repo, session_repo, reset_repo, audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(InvalidResetTokenError) as exc_info:
            await service.confirm_reset(db, "not-a-real-token", "N3wPassw0rd!")

        assert exc_info.value.code == "INVALID_RESET_TOKEN"
        assert staff_repo.update_password.await_count == 0

    async def test_expired_token_raises_invalid_reset_token(self):
        staff_repo = FakeStaffRepo()
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        expired_row = SimpleNamespace(
            id=uuid.uuid4(),
            staff_id=uuid.uuid4(),
            used=False,
            used_at=None,
            expires_at=now_utc() - timedelta(hours=1),
            token_hash="whatever",
        )
        reset_repo.get_by_token = AsyncMock(return_value=expired_row)
        audit = FakeAudit()

        service = AuthService(staff_repo, session_repo, reset_repo, audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(InvalidResetTokenError):
            await service.confirm_reset(db, "expired-token", "N3wPassw0rd!")

        assert staff_repo.update_password.await_count == 0

    async def test_already_used_token_raises_invalid_reset_token(self):
        staff_repo = FakeStaffRepo()
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        used_row = SimpleNamespace(
            id=uuid.uuid4(),
            staff_id=uuid.uuid4(),
            used=True,
            used_at=now_utc() - timedelta(minutes=5),
            expires_at=now_utc() + timedelta(hours=1),
            token_hash="whatever",
        )
        reset_repo.get_by_token = AsyncMock(return_value=used_row)
        audit = FakeAudit()

        service = AuthService(staff_repo, session_repo, reset_repo, audit, FakeRateLimiter())
        db = MagicMock()

        with pytest.raises(InvalidResetTokenError):
            await service.confirm_reset(db, "used-token", "N3wPassw0rd!")

        assert staff_repo.update_password.await_count == 0

    async def test_valid_token_updates_password_marks_it_used_and_audits(self):
        staff_repo = FakeStaffRepo()
        session_repo = FakeSessionRepo()
        reset_repo = FakeResetRepo()
        valid_row = SimpleNamespace(
            id=uuid.uuid4(),
            staff_id=uuid.uuid4(),
            used=False,
            used_at=None,
            expires_at=now_utc() + timedelta(hours=1),
            token_hash="whatever",
        )
        reset_repo.get_by_token = AsyncMock(return_value=valid_row)
        audit = FakeAudit()

        service = AuthService(staff_repo, session_repo, reset_repo, audit, FakeRateLimiter())
        db = MagicMock()

        await service.confirm_reset(db, "good-token", "N3wPassw0rd!")

        assert staff_repo.update_password.await_count == 1
        assert reset_repo.mark_used.await_count == 1
        assert audit.record.await_count == 1
        values = call_values(audit.record.call_args)
        assert "auth.password_reset" in values


# =========================================================================== #
# PermissionService
# =========================================================================== #

class TestPermissionServiceCheck:
    async def test_check_delegates_to_role_permission_repo_and_returns_true(self):
        role_repo = FakeRolePermRepo()
        role_repo.is_allowed = AsyncMock(return_value=True)
        service = PermissionService(role_repo)
        db = MagicMock()

        result = await service.check(db, "clinic_management", "rules", "approve")

        assert result is True
        assert role_repo.is_allowed.await_count == 1
        values = call_values(role_repo.is_allowed.call_args)
        assert "clinic_management" in values
        assert "rules" in values
        assert "approve" in values

    async def test_check_returns_false_when_repo_denies(self):
        role_repo = FakeRolePermRepo()
        role_repo.is_allowed = AsyncMock(return_value=False)
        service = PermissionService(role_repo)
        db = MagicMock()

        result = await service.check(db, "delivery_team", "patients", "read")

        assert result is False


class TestPermissionServiceVisibleWidgets:
    EXPECTED_MODULES = {
        "schedule",
        "waitlist",
        "outreach",
        "recall",
        "rules",
        "dashboards",
        "console",
        "triage",
        "utilisation",
        "education",
        "engagement",
    }

    async def test_returns_one_entry_per_module_with_visibility_from_the_matrix(self):
        role_repo = FakeRolePermRepo()
        allowed_resources = {"waitlist.dashboard", "console.dashboard", "education.dashboard"}

        async def _is_allowed(*args, **kwargs):
            values = flatten(args, kwargs)
            resource = next(
                (v for v in values if isinstance(v, str) and v.endswith(".dashboard")), None
            )
            action = next((v for v in values if v == "read"), None)
            return resource in allowed_resources and action == "read"

        role_repo.is_allowed = AsyncMock(side_effect=_is_allowed)
        service = PermissionService(role_repo)
        db = MagicMock()

        widgets = await service.get_visible_widgets(db, "front_office_staff")

        keys = {w["key"] for w in widgets}
        assert keys == self.EXPECTED_MODULES
        assert role_repo.is_allowed.await_count == len(self.EXPECTED_MODULES)

        for widget in widgets:
            assert {"key", "label", "visible"} <= set(widget.keys())
            assert isinstance(widget["label"], str) and widget["label"]
            expected_visible = f"{widget['key']}.dashboard" in allowed_resources
            assert widget["visible"] is expected_visible

    async def test_no_hardcoded_widget_list_when_matrix_denies_everything(self):
        role_repo = FakeRolePermRepo()
        role_repo.is_allowed = AsyncMock(return_value=False)
        service = PermissionService(role_repo)
        db = MagicMock()

        widgets = await service.get_visible_widgets(db, "delivery_team")

        assert {w["key"] for w in widgets} == self.EXPECTED_MODULES
        assert all(widget["visible"] is False for widget in widgets)


# =========================================================================== #
# StaffAdminService
# =========================================================================== #

class TestStaffAdminServiceProvision:
    async def test_provision_creates_a_new_staff_user_when_email_is_free(self):
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_email = AsyncMock(return_value=None)
        created = make_staff_user(email="new.hire@example.com", role="delivery_team")
        staff_repo.create = AsyncMock(return_value=created)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()

        service = StaffAdminService(staff_repo, session_repo, audit)
        db = MagicMock()

        result = await service.provision(
            db, "New Hire", "new.hire@example.com", "delivery_team", uuid.uuid4()
        )

        assert result is created
        assert staff_repo.create.await_count == 1

    async def test_provision_raises_duplicate_for_existing_active_email_and_names_its_id(self):
        existing = make_staff_user(email="taken@example.com", status="active")
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_email = AsyncMock(return_value=existing)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()

        service = StaffAdminService(staff_repo, session_repo, audit)
        db = MagicMock()

        with pytest.raises(DuplicateStaffAccountError) as exc_info:
            await service.provision(
                db, "Someone Else", "taken@example.com", "front_office_staff", uuid.uuid4()
            )

        assert exception_mentions(exc_info.value, existing.id)
        assert staff_repo.create.await_count == 0

    async def test_provision_raises_duplicate_for_existing_inactive_email(self):
        existing = make_staff_user(email="taken2@example.com", status="inactive")
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_email = AsyncMock(return_value=existing)
        session_repo = FakeSessionRepo()
        audit = FakeAudit()

        service = StaffAdminService(staff_repo, session_repo, audit)
        db = MagicMock()

        with pytest.raises(DuplicateStaffAccountError):
            await service.provision(
                db, "Someone Else", "taken2@example.com", "front_office_staff", uuid.uuid4()
            )

        assert staff_repo.create.await_count == 0


class TestStaffAdminServiceDeactivate:
    async def test_deactivate_sets_inactive_and_revokes_all_sessions_in_one_call(self):
        user = make_staff_user(status="active")
        staff_repo = FakeStaffRepo()
        staff_repo.get_by_id = AsyncMock(return_value=user)
        session_repo = FakeSessionRepo()
        session_repo.revoke_all_for_user = AsyncMock(return_value=3)
        audit = FakeAudit()

        service = StaffAdminService(staff_repo, session_repo, audit)
        db = MagicMock()

        updated_user, revoked_count = await service.deactivate(db, user.id, uuid.uuid4())

        assert updated_user.status == "inactive"
        assert revoked_count == 3
        assert session_repo.revoke_all_for_user.await_count == 1
        values = call_values(session_repo.revoke_all_for_user.call_args)
        assert user.id in values


# =========================================================================== #
# AuditService.record / query
# =========================================================================== #

class TransientlyFailingAuditRepo:
    """Fails whatever insert-style method AuditService.record calls, a
    configurable number of times, before succeeding -- this lets the retry
    contract be exercised without knowing the exact repository method name.
    """

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.call_count = 0
        self.calls = []

    def __getattr__(self, name):
        async def _method(*args, **kwargs):
            self.call_count += 1
            self.calls.append((args, kwargs))
            if self.call_count <= self.fail_times:
                raise RuntimeError("transient database failure")
            return SimpleNamespace(
                id=uuid.uuid4(),
                action_type=kwargs.get("action_type"),
                entity_type=kwargs.get("entity_type"),
            )

        return _method


class RecordingRepo:
    """Records every call regardless of method name and always returns a
    fixed value -- used to test pass-through/delegation behaviour."""

    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def __getattr__(self, name):
        async def _method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.return_value

        return _method


class TestAuditServiceRecord:
    async def test_record_succeeds_on_first_attempt_and_forwards_all_fields(self):
        repo = TransientlyFailingAuditRepo(fail_times=0)
        service = AuditService(repo)
        db = MagicMock()
        actor_id = uuid.uuid4()
        entity_id = uuid.uuid4()

        result = await service.record(
            db, actor_id, "staff", "auth.login", "staff_user", entity_id
        )

        assert repo.call_count == 1
        args, kwargs = repo.calls[-1]
        values = flatten(args, kwargs)
        assert "staff" in values
        assert "auth.login" in values
        assert "staff_user" in values
        assert entity_id in values
        assert hasattr(result, "id")

    async def test_record_retries_on_transient_failure_then_succeeds(self):
        repo = TransientlyFailingAuditRepo(fail_times=1)
        service = AuditService(repo)
        db = MagicMock()

        result = await service.record(
            db, uuid.uuid4(), "staff", "auth.login", "staff_user", uuid.uuid4()
        )

        assert repo.call_count == 2  # one failure, then a successful retry
        assert hasattr(result, "id")

    async def test_record_raises_app_error_after_retries_are_exhausted(self):
        repo = TransientlyFailingAuditRepo(fail_times=1_000_000)
        service = AuditService(repo)
        db = MagicMock()

        with pytest.raises(AppError) as exc_info:
            await service.record(
                db, uuid.uuid4(), "staff", "auth.login", "staff_user", uuid.uuid4()
            )

        assert exc_info.value.code == "AUDIT_WRITE_FAILED"
        # more than a single attempt was made before giving up (i.e. it did retry)
        assert repo.call_count > 1


class TestAuditServiceQuery:
    async def test_query_forwards_filters_and_pagination_and_returns_repo_result_verbatim(self):
        rows = [SimpleNamespace(id=uuid.uuid4())]
        repo = RecordingRepo(return_value=(rows, 1))
        service = AuditService(repo)
        db = MagicMock()
        actor_id = uuid.uuid4()
        date_from = now_utc() - timedelta(days=7)
        date_to = now_utc()

        result_rows, total = await service.query(
            db, actor_id, "staff_user", date_from, date_to, page=2, page_size=25
        )

        assert result_rows is rows
        assert total == 1
        assert len(repo.calls) == 1
        _, args, kwargs = repo.calls[0]
        values = flatten(args, kwargs)
        assert 2 in values
        assert 25 in values
        assert actor_id in values


# =========================================================================== #
# OverrideService.apply_override
# =========================================================================== #

def make_original_action_row(entity_type, entity_id, original_payload):
    return SimpleNamespace(
        id=uuid.uuid4(),
        actor_type="ai_agent",
        actor_staff_id=None,
        action_type="ai.auto_action",
        entity_type=entity_type,
        entity_id=entity_id,
        original_payload=original_payload,
        override_payload=None,
        overridden_of_id=None,
        created_at=now_utc(),
    )


class FakeAuditRepoForOverride:
    def __init__(self, original_row):
        self.rows_by_id = {original_row.id: original_row}
        self.create_calls = []

    async def get_by_id(self, db, row_id):
        return self.rows_by_id.get(row_id)

    async def create(self, db=None, **kwargs):
        self.create_calls.append(kwargs)
        row = SimpleNamespace(id=uuid.uuid4(), **kwargs)
        self.rows_by_id[row.id] = row
        return row

    def __getattr__(self, name):
        async def _fallback(*args, **kwargs):
            # Best-effort simulation of an idempotency lookup performed
            # under some unspecified method name: look for a previously
            # created row referencing the same original action with the
            # same override payload.
            values = flatten(args, kwargs)
            for row in list(self.rows_by_id.values()):
                overridden_of_id = getattr(row, "overridden_of_id", None)
                override_payload = getattr(row, "override_payload", None)
                if overridden_of_id is not None and overridden_of_id in values:
                    if override_payload is None or override_payload in values:
                        return row
            return None

        return _fallback


class RecordingEntityRepo:
    def __init__(self, get_by_id_return=None):
        self._get_by_id_return = get_by_id_return
        self.update_status = AsyncMock(return_value=None)
        self.calls = []

    async def get_by_id(self, db, entity_id):
        return self._get_by_id_return

    def __getattr__(self, name):
        async def _method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return _method


class TestOverrideServiceApplyOverride:
    async def test_raises_not_found_when_action_id_is_unknown(self):
        audit_repo = FakeAuditRepoForOverride(
            make_original_action_row("appointment", uuid.uuid4(), {"status": "booked"})
        )
        service = OverrideService(
            audit_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
        )
        db = MagicMock()

        with pytest.raises(NotFoundError):
            await service.apply_override(
                db, uuid.uuid4(), uuid.uuid4(), {"status": "confirmed"}, "wrong slot"
            )

    async def test_raises_not_found_when_original_action_was_not_ai_originated(self):
        original = make_original_action_row("appointment", uuid.uuid4(), {"status": "booked"})
        original.actor_type = "staff"
        audit_repo = FakeAuditRepoForOverride(original)
        service = OverrideService(
            audit_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
        )
        db = MagicMock()

        with pytest.raises(NotFoundError):
            await service.apply_override(
                db, original.id, uuid.uuid4(), {"status": "confirmed"}, "manual staff action"
            )

    async def test_appointment_override_updates_status_and_writes_a_second_linked_audit_row(self):
        entity_id = uuid.uuid4()
        original = make_original_action_row("appointment", entity_id, {"status": "booked"})
        audit_repo = FakeAuditRepoForOverride(original)
        appointment_repo = RecordingEntityRepo(
            get_by_id_return=SimpleNamespace(id=entity_id, status="booked")
        )
        service = OverrideService(
            audit_repo,
            appointment_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
        )
        db = MagicMock()
        actor_id = uuid.uuid4()
        corrected = {"status": "cancelled"}

        result = await service.apply_override(db, original.id, actor_id, corrected, "double booked")

        assert appointment_repo.update_status.await_count == 1
        assert len(audit_repo.create_calls) == 1

        kwargs = audit_repo.create_calls[0]
        assert kwargs.get("overridden_of_id") == original.id
        assert kwargs.get("actor_type") == "staff"
        assert kwargs.get("original_payload") == original.original_payload
        assert kwargs.get("override_payload") == corrected

        assert result.overridden_of_id == original.id
        assert result.override_payload == corrected

    async def test_outreach_message_override_dispatches_to_update_status(self):
        entity_id = uuid.uuid4()
        original = make_original_action_row(
            "outreach_message", entity_id, {"template": "confirmation"}
        )
        audit_repo = FakeAuditRepoForOverride(original)
        outreach_repo = RecordingEntityRepo(
            get_by_id_return=SimpleNamespace(id=entity_id, status="queued")
        )
        service = OverrideService(
            audit_repo,
            RecordingEntityRepo(),
            outreach_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
        )
        db = MagicMock()

        await service.apply_override(
            db, original.id, uuid.uuid4(), {"status": "cancelled"}, "wrong channel"
        )

        assert outreach_repo.update_status.await_count == 1

    async def test_waitlist_entry_override_dispatches_to_the_waitlist_repo(self):
        entity_id = uuid.uuid4()
        original = make_original_action_row("waitlist_entry", entity_id, {"priority": "standard"})
        audit_repo = FakeAuditRepoForOverride(original)
        waitlist_repo = RecordingEntityRepo(
            get_by_id_return=SimpleNamespace(id=entity_id, status="offered")
        )
        service = OverrideService(
            audit_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            waitlist_repo,
            RecordingEntityRepo(),
        )
        db = MagicMock()

        await service.apply_override(
            db, original.id, uuid.uuid4(), {"priority": "urgent"}, "clinical priority change"
        )

        assert waitlist_repo.update_status.await_count == 1 or len(waitlist_repo.calls) >= 1

    async def test_triage_session_override_dispatches_to_the_triage_repo(self):
        entity_id = uuid.uuid4()
        original = make_original_action_row("triage_session", entity_id, {"outcome": "self_care"})
        audit_repo = FakeAuditRepoForOverride(original)
        triage_repo = RecordingEntityRepo(
            get_by_id_return=SimpleNamespace(id=entity_id, status="closed")
        )
        service = OverrideService(
            audit_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            triage_repo,
        )
        db = MagicMock()

        await service.apply_override(
            db, original.id, uuid.uuid4(), {"outcome": "escalated"}, "missed red flag"
        )

        assert triage_repo.update_status.await_count == 1 or len(triage_repo.calls) >= 1

    async def test_irreversible_action_still_writes_the_audit_trail_but_returns_409(self):
        entity_id = uuid.uuid4()
        original = make_original_action_row(
            "outreach_message", entity_id, {"template": "recall_reminder"}
        )
        audit_repo = FakeAuditRepoForOverride(original)
        outreach_repo = RecordingEntityRepo(
            get_by_id_return=SimpleNamespace(id=entity_id, status="delivered")
        )
        service = OverrideService(
            audit_repo,
            RecordingEntityRepo(),
            outreach_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
        )
        db = MagicMock()

        with pytest.raises(IrreversibleActionError) as exc_info:
            await service.apply_override(
                db, original.id, uuid.uuid4(), {"status": "cancelled"}, "patient asked to stop"
            )

        assert exc_info.value.code == "IRREVERSIBLE_ACTION"
        assert getattr(exc_info.value, "details", None)
        assert len(audit_repo.create_calls) == 1

    async def test_apply_override_is_idempotent_for_a_repeated_action_id_and_payload(self):
        entity_id = uuid.uuid4()
        original = make_original_action_row("appointment", entity_id, {"status": "booked"})
        audit_repo = FakeAuditRepoForOverride(original)
        appointment_repo = RecordingEntityRepo(
            get_by_id_return=SimpleNamespace(id=entity_id, status="booked")
        )
        service = OverrideService(
            audit_repo,
            appointment_repo,
            RecordingEntityRepo(),
            RecordingEntityRepo(),
            RecordingEntityRepo(),
        )
        db = MagicMock()
        actor_id = uuid.uuid4()
        corrected = {"status": "cancelled"}

        first = await service.apply_override(db, original.id, actor_id, corrected, "double booked")
        second = await service.apply_override(db, original.id, actor_id, corrected, "double booked")

        assert first.id == second.id
        assert len(audit_repo.create_calls) == 1
