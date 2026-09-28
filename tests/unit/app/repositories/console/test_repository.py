"""Unit tests for app/repositories/console/repository.py.

Exercises the five console-table repositories directly against a real
async SQLite (in-memory) database created from the same ORM models the
repository queries against (`app.models.console.models`), per the
ENVIRONMENT=test DB substitution described in project_rules.testing.
Nothing here mocks the `AsyncSession` -- these are real ORM round-trips,
so a repository method that constructs the wrong query is caught the
same way it would be against Postgres.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.common.constants import LOGIN_LOCKOUT_MINUTES, LOGIN_LOCKOUT_THRESHOLD
from app.core.database import Base
from app.models.console.models import PasswordResetToken, RolePermission, StaffUser
from app.repositories.console.repository import (
    AuditLogRepository,
    PasswordResetTokenRepository,
    RolePermissionRepository,
    SessionRepository,
    StaffUserRepository,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def db(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


def _unique():
    return uuid.uuid4().hex[:10]


async def _make_staff(db: AsyncSession, **overrides) -> StaffUser:
    unique = _unique()
    defaults = dict(
        username=f"user_{unique}",
        email=f"user_{unique}@clinic.example",
        password_hash="$2b$12$originalhashvalue",
        role="front_office_staff",
    )
    defaults.update(overrides)
    staff = StaffUser(**defaults)
    db.add(staff)
    await db.commit()
    await db.refresh(staff)
    return staff


@pytest_asyncio.fixture
async def staff(db) -> StaffUser:
    return await _make_staff(db)


# ---------------------------------------------------------------------------
# StaffUserRepository
# ---------------------------------------------------------------------------


class TestStaffUserRepository:
    async def test_get_by_username_finds_existing_user(self, db, staff):
        repo = StaffUserRepository()
        found = await repo.get_by_username(db, staff.username)
        assert found is not None
        assert found.id == staff.id

    async def test_get_by_username_returns_none_when_missing(self, db):
        repo = StaffUserRepository()
        assert await repo.get_by_username(db, "no_such_user") is None

    async def test_get_by_id_finds_existing_user(self, db, staff):
        repo = StaffUserRepository()
        found = await repo.get_by_id(db, staff.id)
        assert found is not None
        assert found.username == staff.username

    async def test_get_by_id_returns_none_when_missing(self, db):
        repo = StaffUserRepository()
        assert await repo.get_by_id(db, uuid.uuid4()) is None

    async def test_get_by_email_finds_existing_user(self, db, staff):
        repo = StaffUserRepository()
        found = await repo.get_by_email(db, staff.email)
        assert found is not None
        assert found.id == staff.id

    async def test_get_by_email_returns_none_when_missing(self, db):
        repo = StaffUserRepository()
        assert await repo.get_by_email(db, "nobody@clinic.example") is None

    async def test_create_persists_a_new_staff_user(self, db):
        repo = StaffUserRepository()
        unique = _unique()
        created = await repo.create(
            db,
            username=f"new_{unique}",
            email=f"new_{unique}@clinic.example",
            password_hash="$2b$12$freshhash",
            role="clinic_management",
        )
        assert created.id is not None
        assert created.username == f"new_{unique}"
        assert created.role == "clinic_management"

        reloaded = await StaffUserRepository().get_by_username(db, f"new_{unique}")
        assert reloaded is not None
        assert reloaded.email == f"new_{unique}@clinic.example"

    async def test_update_status_changes_and_persists_status(self, db, staff):
        repo = StaffUserRepository()
        updated = await repo.update_status(db, staff.id, "inactive")
        assert updated.status == "inactive"

        reloaded = await repo.get_by_id(db, staff.id)
        assert reloaded.status == "inactive"

    async def test_increment_failed_attempts_increments_counter_below_threshold(self, db, staff):
        repo = StaffUserRepository()
        updated = await repo.increment_failed_attempts(db, staff.id)
        assert updated.failed_attempt_count == 1
        assert updated.locked_until is None

    async def test_increment_failed_attempts_locks_account_at_threshold(self, db, staff):
        repo = StaffUserRepository()
        before = datetime.now(timezone.utc)
        for _ in range(LOGIN_LOCKOUT_THRESHOLD - 1):
            await repo.increment_failed_attempts(db, staff.id)
        # one call short of the threshold: still unlocked
        not_yet_locked = await repo.get_by_id(db, staff.id)
        assert not_yet_locked.locked_until is None
        assert not_yet_locked.failed_attempt_count == LOGIN_LOCKOUT_THRESHOLD - 1

        locked = await repo.increment_failed_attempts(db, staff.id)
        after = datetime.now(timezone.utc)

        assert locked.failed_attempt_count == 0
        assert locked.locked_until is not None
        expected_earliest = before + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
        expected_latest = after + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
        locked_until = locked.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        assert expected_earliest - timedelta(seconds=5) <= locked_until <= expected_latest + timedelta(
            seconds=5
        )

    async def test_reset_failed_attempts_zeroes_the_counter(self, db, staff):
        repo = StaffUserRepository()
        await repo.increment_failed_attempts(db, staff.id)
        await repo.increment_failed_attempts(db, staff.id)
        mid = await repo.get_by_id(db, staff.id)
        assert mid.failed_attempt_count == 2

        result = await repo.reset_failed_attempts(db, staff.id)
        assert result is None

        reloaded = await repo.get_by_id(db, staff.id)
        assert reloaded.failed_attempt_count == 0

    async def test_update_password_changes_the_stored_hash(self, db, staff):
        repo = StaffUserRepository()
        result = await repo.update_password(db, staff.id, "$2b$12$brandnewhashvalue")
        assert result is None

        reloaded = await repo.get_by_id(db, staff.id)
        assert reloaded.password_hash == "$2b$12$brandnewhashvalue"


# ---------------------------------------------------------------------------
# SessionRepository
# ---------------------------------------------------------------------------


class TestSessionRepository:
    async def test_create_persists_a_session_row(self, db, staff):
        repo = SessionRepository()
        now = datetime.now(timezone.utc)
        created = await repo.create(
            db,
            user_id=staff.id,
            jti="jti-001",
            issued_at=now,
            expires_at=now + timedelta(minutes=15),
        )
        assert created.id is not None
        assert created.user_id == staff.id
        assert created.jti == "jti-001"
        assert created.revoked_at is None

    async def test_get_by_jti_finds_existing_session(self, db, staff):
        repo = SessionRepository()
        now = datetime.now(timezone.utc)
        await repo.create(
            db, user_id=staff.id, jti="jti-lookup", issued_at=now, expires_at=now + timedelta(minutes=15)
        )
        found = await repo.get_by_jti(db, "jti-lookup")
        assert found is not None
        assert found.user_id == staff.id

    async def test_get_by_jti_returns_none_when_missing(self, db):
        repo = SessionRepository()
        assert await repo.get_by_jti(db, "no-such-jti") is None

    async def test_revoke_all_for_user_revokes_only_active_sessions_for_that_user(self, db, staff):
        other_staff = await _make_staff(db)
        repo = SessionRepository()
        now = datetime.now(timezone.utc)

        active_1 = await repo.create(
            db, user_id=staff.id, jti="active-1", issued_at=now, expires_at=now + timedelta(minutes=15)
        )
        active_2 = await repo.create(
            db, user_id=staff.id, jti="active-2", issued_at=now, expires_at=now + timedelta(minutes=15)
        )
        already_revoked = await repo.create(
            db, user_id=staff.id, jti="already-revoked", issued_at=now, expires_at=now + timedelta(minutes=15)
        )
        already_revoked_at = now - timedelta(hours=1)
        already_revoked.revoked_at = already_revoked_at
        await db.commit()

        other_users_session = await repo.create(
            db, user_id=other_staff.id, jti="other-user", issued_at=now, expires_at=now + timedelta(minutes=15)
        )

        count = await repo.revoke_all_for_user(db, staff.id)
        assert count == 2

        reloaded_1 = await repo.get_by_jti(db, "active-1")
        reloaded_2 = await repo.get_by_jti(db, "active-2")
        reloaded_already = await repo.get_by_jti(db, "already-revoked")
        reloaded_other = await repo.get_by_jti(db, "other-user")

        assert reloaded_1.revoked_at is not None
        assert reloaded_2.revoked_at is not None
        # a session already revoked before the call keeps its original timestamp
        reloaded_already_ts = reloaded_already.revoked_at
        if reloaded_already_ts.tzinfo is None:
            reloaded_already_ts = reloaded_already_ts.replace(tzinfo=timezone.utc)
        assert reloaded_already_ts == already_revoked_at
        # a different user's active session is untouched
        assert reloaded_other.revoked_at is None

    async def test_revoke_all_for_user_returns_zero_when_no_active_sessions(self, db, staff):
        repo = SessionRepository()
        count = await repo.revoke_all_for_user(db, staff.id)
        assert count == 0


# ---------------------------------------------------------------------------
# RolePermissionRepository
# ---------------------------------------------------------------------------


class TestRolePermissionRepository:
    async def test_list_for_role_is_empty_before_any_grant_exists(self, db):
        repo = RolePermissionRepository()
        assert await repo.list_for_role(db, "front_office_staff") == []

    async def test_is_allowed_false_when_no_matching_row_exists(self, db):
        repo = RolePermissionRepository()
        assert await repo.is_allowed(db, "front_office_staff", "patients.records", "write") is False

    async def test_is_allowed_true_for_an_explicit_allowed_grant(self, db):
        db.add(RolePermission(role="front_office_staff", resource="patients.records", action="write", allowed=True))
        await db.commit()
        repo = RolePermissionRepository()
        assert await repo.is_allowed(db, "front_office_staff", "patients.records", "write") is True

    async def test_is_allowed_false_when_matching_row_is_explicitly_denied(self, db):
        db.add(RolePermission(role="delivery_team", resource="patients.records", action="write", allowed=False))
        await db.commit()
        repo = RolePermissionRepository()
        assert await repo.is_allowed(db, "delivery_team", "patients.records", "write") is False

    async def test_list_for_role_only_returns_rows_for_that_role(self, db):
        db.add_all(
            [
                RolePermission(role="front_office_staff", resource="patients.records", action="write", allowed=True),
                RolePermission(role="delivery_team", resource="rules.rule_sets", action="write", allowed=True),
            ]
        )
        await db.commit()
        repo = RolePermissionRepository()
        front_office_perms = await repo.list_for_role(db, "front_office_staff")
        assert len(front_office_perms) == 1
        assert front_office_perms[0].resource == "patients.records"

    async def test_seed_defaults_grants_delivery_team_exactly_the_three_named_resources_and_nothing_operational(
        self, db
    ):
        repo = RolePermissionRepository()
        await repo.seed_defaults(db)

        delivery_perms = await repo.list_for_role(db, "delivery_team")
        granted = {(p.resource, p.action) for p in delivery_perms if p.allowed}
        assert granted == {
            ("rules.rule_sets", "write"),
            ("rules.configuration", "write"),
            ("outreach.channels.whatsapp", "write"),
        }

        # FR-E1.3: delivery_team must never be granted an operational resource
        assert await repo.is_allowed(db, "delivery_team", "patients.records", "write") is False
        assert await repo.is_allowed(db, "delivery_team", "scheduling.appointments", "write") is False
        assert await repo.is_allowed(db, "delivery_team", "waitlist.entries", "write") is False

    async def test_seed_defaults_grants_clinic_management_every_front_office_grant_plus_its_extras(self, db):
        repo = RolePermissionRepository()
        await repo.seed_defaults(db)

        front_office_perms = await repo.list_for_role(db, "front_office_staff")
        clinic_mgmt_perms = await repo.list_for_role(db, "clinic_management")

        front_office_pairs = {(p.resource, p.action) for p in front_office_perms if p.allowed}
        clinic_mgmt_pairs = {(p.resource, p.action) for p in clinic_mgmt_perms if p.allowed}
        clinic_mgmt_resources = {p.resource for p in clinic_mgmt_perms if p.allowed}

        assert front_office_pairs, "front_office_staff must have at least one operational grant"
        assert front_office_pairs.issubset(clinic_mgmt_pairs)

        # clinic_management-only extras named in the spec
        assert "console.staff_admin" in clinic_mgmt_resources
        assert "rules.approve" in clinic_mgmt_resources
        assert await repo.is_allowed(db, "clinic_management", "console.audit", "read") is True
        assert await repo.is_allowed(db, "clinic_management", "analytics.financial", "read") is True
        assert await repo.is_allowed(db, "clinic_management", "analytics.baseline_cost", "write") is True

        # those clinic_management-only extras must not leak into front_office_staff
        front_office_resources = {p.resource for p in front_office_perms if p.allowed}
        assert "console.staff_admin" not in front_office_resources
        assert "rules.approve" not in front_office_resources

    async def test_seed_defaults_is_idempotent(self, db):
        repo = RolePermissionRepository()
        await repo.seed_defaults(db)
        first_count = len(await repo.list_for_role(db, "delivery_team"))

        await repo.seed_defaults(db)
        second_count = len(await repo.list_for_role(db, "delivery_team"))

        assert first_count == second_count


# ---------------------------------------------------------------------------
# PasswordResetTokenRepository
# ---------------------------------------------------------------------------


class TestPasswordResetTokenRepository:
    async def test_create_persists_a_reset_token(self, db, staff):
        repo = PasswordResetTokenRepository()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        created = await repo.create(db, user_id=staff.id, token_hash="hash-abc", expires_at=expires_at)
        assert created.id is not None
        assert created.user_id == staff.id
        assert created.token_hash == "hash-abc"
        assert created.used_at is None

    async def test_get_valid_by_hash_returns_none_when_no_such_token(self, db):
        repo = PasswordResetTokenRepository()
        assert await repo.get_valid_by_hash(db, "no-such-hash") is None

    async def test_get_valid_by_hash_returns_the_token_when_unused_and_unexpired(self, db, staff):
        repo = PasswordResetTokenRepository()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        created = await repo.create(db, user_id=staff.id, token_hash="hash-valid", expires_at=expires_at)

        found = await repo.get_valid_by_hash(db, "hash-valid")
        assert found is not None
        assert found.id == created.id

    async def test_get_valid_by_hash_returns_none_when_already_used(self, db, staff):
        repo = PasswordResetTokenRepository()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        token = PasswordResetToken(user_id=staff.id, token_hash="hash-used", expires_at=expires_at)
        token.used_at = datetime.now(timezone.utc)
        db.add(token)
        await db.commit()

        assert await repo.get_valid_by_hash(db, "hash-used") is None

    async def test_get_valid_by_hash_returns_none_when_expired(self, db, staff):
        repo = PasswordResetTokenRepository()
        expired_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        token = PasswordResetToken(user_id=staff.id, token_hash="hash-expired", expires_at=expired_at)
        db.add(token)
        await db.commit()

        assert await repo.get_valid_by_hash(db, "hash-expired") is None

    async def test_mark_used_sets_used_at_so_the_token_becomes_invalid(self, db, staff):
        repo = PasswordResetTokenRepository()
        expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        created = await repo.create(db, user_id=staff.id, token_hash="hash-to-consume", expires_at=expires_at)

        result = await repo.mark_used(db, created.id)
        assert result is None

        reloaded = await db.get(PasswordResetToken, created.id)
        assert reloaded.used_at is not None
        assert await repo.get_valid_by_hash(db, "hash-to-consume") is None


# ---------------------------------------------------------------------------
# AuditLogRepository
# ---------------------------------------------------------------------------


class TestAuditLogRepository:
    async def test_insert_persists_and_returns_the_new_row(self, db, staff):
        repo = AuditLogRepository()
        entity_id = uuid.uuid4()
        created = await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="appointment.cancel",
            entity_type="appointment",
            entity_id=entity_id,
            timestamp=datetime.now(timezone.utc),
        )
        assert created.id is not None
        assert created.actor_staff_id == staff.id
        assert created.action_type == "appointment.cancel"
        assert created.entity_id == entity_id

    async def test_get_by_id_finds_existing_entry(self, db, staff):
        repo = AuditLogRepository()
        created = await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="override.apply",
            entity_type="triage_session",
            entity_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
        )
        found = await repo.get_by_id(db, created.id)
        assert found is not None
        assert found.action_type == "override.apply"

    async def test_get_by_id_returns_none_when_missing(self, db):
        repo = AuditLogRepository()
        assert await repo.get_by_id(db, uuid.uuid4()) is None

    async def test_query_filters_by_actor_id(self, db, staff):
        other_staff = await _make_staff(db)
        repo = AuditLogRepository()
        now = datetime.now(timezone.utc)
        await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="patient.update",
            entity_type="patient",
            entity_id=uuid.uuid4(),
            timestamp=now,
        )
        await repo.insert(
            db,
            actor_staff_id=other_staff.id,
            actor_type="staff",
            action_type="patient.update",
            entity_type="patient",
            entity_id=uuid.uuid4(),
            timestamp=now,
        )

        items, total = await repo.query(
            db, actor_id=staff.id, entity_type=None, date_from=None, date_to=None, page=1, page_size=50
        )
        assert total == 1
        assert len(items) == 1
        assert items[0].actor_staff_id == staff.id

    async def test_query_filters_by_entity_type(self, db, staff):
        repo = AuditLogRepository()
        now = datetime.now(timezone.utc)
        await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="patient.update",
            entity_type="patient",
            entity_id=uuid.uuid4(),
            timestamp=now,
        )
        await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="appointment.cancel",
            entity_type="appointment",
            entity_id=uuid.uuid4(),
            timestamp=now,
        )

        items, total = await repo.query(
            db, actor_id=None, entity_type="appointment", date_from=None, date_to=None, page=1, page_size=50
        )
        assert total == 1
        assert items[0].entity_type == "appointment"

    async def test_query_filters_by_date_range_on_timestamp_not_created_at(self, db, staff):
        # Watch out: audit_log has no created_at column -- the filter must be
        # against `timestamp`.
        repo = AuditLogRepository()
        old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent_ts = datetime.now(timezone.utc)

        await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="old.action",
            entity_type="patient",
            entity_id=uuid.uuid4(),
            timestamp=old_ts,
        )
        await repo.insert(
            db,
            actor_staff_id=staff.id,
            actor_type="staff",
            action_type="recent.action",
            entity_type="patient",
            entity_id=uuid.uuid4(),
            timestamp=recent_ts,
        )

        items, total = await repo.query(
            db,
            actor_id=None,
            entity_type=None,
            date_from=recent_ts - timedelta(minutes=1),
            date_to=recent_ts + timedelta(minutes=1),
            page=1,
            page_size=50,
        )
        assert total == 1
        assert items[0].action_type == "recent.action"

    async def test_query_returns_total_count_independent_of_page_size(self, db, staff):
        repo = AuditLogRepository()
        now = datetime.now(timezone.utc)
        entity_ids = [uuid.uuid4() for _ in range(5)]
        for entity_id in entity_ids:
            await repo.insert(
                db,
                actor_staff_id=staff.id,
                actor_type="staff",
                action_type="patient.update",
                entity_type="patient",
                entity_id=entity_id,
                timestamp=now,
            )

        page_1_items, total_1 = await repo.query(
            db, actor_id=None, entity_type=None, date_from=None, date_to=None, page=1, page_size=2
        )
        page_2_items, total_2 = await repo.query(
            db, actor_id=None, entity_type=None, date_from=None, date_to=None, page=2, page_size=2
        )
        page_3_items, total_3 = await repo.query(
            db, actor_id=None, entity_type=None, date_from=None, date_to=None, page=3, page_size=2
        )

        assert total_1 == total_2 == total_3 == 5
        assert len(page_1_items) == 2
        assert len(page_2_items) == 2
        assert len(page_3_items) == 1

        all_ids = {item.id for item in page_1_items + page_2_items + page_3_items}
        assert len(all_ids) == 5  # no overlap, no gaps across pages

    async def test_query_returns_empty_items_and_zero_total_when_nothing_matches(self, db, staff):
        repo = AuditLogRepository()
        items, total = await repo.query(
            db, actor_id=uuid.uuid4(), entity_type=None, date_from=None, date_to=None, page=1, page_size=50
        )
        assert items == []
        assert total == 0

    def test_exposes_no_update_or_delete_method(self):
        # FR-E1.6: append-only enforcement lives here at the application layer
        # -- there is simply no way to call an update/delete through this class.
        assert not hasattr(AuditLogRepository, "update")
        assert not hasattr(AuditLogRepository, "delete")
