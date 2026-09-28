"""Five business-logic classes for the console module.

``AuthService`` (login/refresh/logout/password-reset), ``PermissionService``
(RBAC matrix lookup), ``StaffAdminService`` (provision/deactivate),
``AuditService`` (append-only write + query — the compliance backbone every
other module's service calls) and ``OverrideService`` (applies a staff
correction to an AI-originated action).

Responsibility: this file never issues raw SQL — every mutation/query goes
through ``repositories/console/repository.py`` (this module's own
repository classes) or, for ``OverrideService`` only, another module's
**repository** class (never another module's **service** class — the
project-wide layering rule that prevents a console<->everything dependency
cycle, since almost every other module's service depends on
``AuditService``).
"""

from __future__ import annotations

import hashlib
import inspect
import secrets
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from app.common.constants import AUDIT_WRITE_MAX_RETRIES, PASSWORD_RESET_TOKEN_TTL_HOURS
from app.common.enums import (
    ActorType,
    AppointmentStatus,
    OutreachMessageStatus,
    Role,
    StaffStatus,
    WaitlistEntryStatus,
)
from app.common.exceptions.errors import (
    AccountLockedError,
    AppError,
    DuplicateStaffAccountError,
    InvalidCredentialsError,
    InvalidResetTokenError,
    IrreversibleActionError,
    NotFoundError,
    UnauthorizedError,
    ValidationFailedError,
)
from app.common.middleware import InMemoryRateLimiter
from app.common.utils import now_utc
from app.core.config import get_settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    new_jti,
    verify_password,
)
from app.models.console.models import AuditLog, StaffUser
from app.repositories.console.repository import (
    AuditLogRepository,
    PasswordResetTokenRepository,
    RolePermissionRepository,
    SessionRepository,
    StaffUserRepository,
)
from app.repositories.intelligence.repository import TriageSessionRepository
from app.repositories.outreach.repository import OutreachMessageRepository
from app.repositories.scheduling.repository import AppointmentRepository
from app.repositories.waitlist.repository import WaitlistEntryRepository
from app.services.outreach.channel_gateway import get_channel_adapter

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "AuditService",
    "AuditWriteFailedError",
    "AuthService",
    "OverrideService",
    "PermissionService",
    "StaffAdminService",
]


# --- Fixed MVP widget manifest (FR-E1.4/US4) ---------------------------------
# One entry per module — visibility is never hardcoded per-role here, only
# computed against the same `role_permissions` matrix `PermissionService.check`
# reads (see `get_visible_widgets`). Adding an eleventh module or renaming one
# only ever happens by editing this one list, never by re-deriving a
# per-role view elsewhere.
_WIDGET_MANIFEST: list[tuple[str, str]] = [
    ("schedule", "Schedule"),
    ("waitlist", "Waitlist"),
    ("outreach", "Outreach"),
    ("recall", "Recall"),
    ("rules", "Rules"),
    ("dashboards", "Dashboards"),
    ("console", "Console"),
    ("triage", "Triage"),
    ("utilisation", "Utilisation"),
    ("education", "Education"),
    ("engagement", "Engagement"),
]


def _aware(value: "datetime | None") -> "datetime | None":
    """Normalize a datetime read back from the DB to tz-aware UTC.

    SQLite (used per project_rules.testing's ENVIRONMENT=test pivot) does not
    retain a ``DateTime(timezone=True)`` column's UTC offset the way
    Postgres does — a value round-tripped through it comes back naive. Every
    timestamp in this tree is UTC, so a naive value read back is treated as
    already being UTC before it is compared against ``now_utc()``.
    """
    if value is not None and value.tzinfo is None:
        from datetime import timezone

        return value.replace(tzinfo=timezone.utc)
    return value


class AuditWriteFailedError(AppError):
    """US7 Alternate Flow: raised by ``AuditService.record`` once
    ``AUDIT_WRITE_MAX_RETRIES`` attempts have all failed, so the caller's own
    transaction (the originating action) is blocked/flagged rather than
    silently proceeding.

    Not declared in ``app.common.exceptions.errors`` (frozen) — this module
    defines its own narrow subclass of the shared ``AppError`` hierarchy,
    exactly as every other subclass there does, without editing that file.
    """

    code = "AUDIT_WRITE_FAILED"
    status_code = 500
    default_message = "Failed to write the audit log entry after retries; the originating action was blocked."


class AuditService:
    """FR-E1.6/US7: the **only** write path onto ``audit_log`` anywhere in
    this codebase — every other service's mutating method calls ``record``,
    directly or (for AI-originated actions written by a worker) via the
    worker's own call into the owning service.
    """

    def __init__(self, audit_repo: AuditLogRepository) -> None:
        self._audit_repo = audit_repo

    async def record(
        self,
        db: "AsyncSession",
        actor_staff_id: "UUID | None",
        actor_type: str,
        action_type: str,
        entity_type: str,
        entity_id: "UUID",
        original_payload: dict | None = None,
        override_payload: dict | None = None,
        overridden_of_id: "UUID | None" = None,
    ) -> "AuditLog":
        """FR-E1.6/US7: retries the insert up to ``AUDIT_WRITE_MAX_RETRIES``
        times on transient failure; if all retries are exhausted, raises
        ``AuditWriteFailedError`` (an ``AppError`` with
        ``code="AUDIT_WRITE_FAILED"``) so the caller's own transaction is
        blocked/flagged rather than silently proceeding.
        """
        last_error: Exception | None = None
        for _ in range(AUDIT_WRITE_MAX_RETRIES):
            try:
                return await self._audit_repo.insert(
                    db,
                    actor_staff_id=actor_staff_id,
                    actor_type=actor_type,
                    action_type=action_type,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    original_payload=original_payload,
                    override_payload=override_payload,
                    overridden_of_id=overridden_of_id,
                    timestamp=now_utc(),
                )
            except Exception as exc:  # transient DB failure — retried below
                last_error = exc
        raise AuditWriteFailedError(details=[{"last_error": str(last_error)}] if last_error else None)

    async def query(
        self,
        db: "AsyncSession",
        actor_id: "UUID | None",
        entity_type: str | None,
        date_from: "datetime | None",
        date_to: "datetime | None",
        page: int,
        page_size: int,
    ) -> "tuple[list[AuditLog], int]":
        return await self._audit_repo.query(db, actor_id, entity_type, date_from, date_to, page, page_size)


class PermissionService:
    """FR-E1.2/US3: a pure ``RolePermissionRepository.is_allowed`` lookup —
    no hardcoded per-resource role list exists anywhere else in the
    codebase.
    """

    def __init__(self, role_permission_repo: RolePermissionRepository) -> None:
        self._role_permission_repo = role_permission_repo

    async def check(self, db: "AsyncSession", role: str, resource: str, action: str) -> bool:
        return await self._role_permission_repo.is_allowed(db, role, resource, action)

    async def get_visible_widgets(self, db: "AsyncSession", role: str) -> list[dict]:
        """FR-E1.4/US4: filters the fixed MVP widget manifest through the
        same ``role_permissions`` matrix ``check`` reads — never a
        separately-maintained widget-visibility list that could drift from
        the actual permission matrix. Visibility for each ``key`` is
        computed by checking ``check(role, f"{key}.dashboard", "read")``.
        """
        widgets: list[dict] = []
        for key, label in _WIDGET_MANIFEST:
            visible = await self.check(db, role, f"{key}.dashboard", "read")
            widgets.append({"key": key, "label": label, "visible": visible})
        return widgets


class AuthService:
    """FR-E1.1/US1 (login/lockout), US2 (password reset)."""

    def __init__(
        self,
        staff_repo: StaffUserRepository,
        session_repo: SessionRepository,
        reset_repo: PasswordResetTokenRepository,
        audit: "AuditService",
        rate_limiter: InMemoryRateLimiter,
    ) -> None:
        self._staff_repo = staff_repo
        self._session_repo = session_repo
        self._reset_repo = reset_repo
        self._audit = audit
        self._rate_limiter = rate_limiter

    @staticmethod
    def _is_locked(staff: "StaffUser") -> bool:
        """US1 Alternate Flow: an account is locked either by the
        timestamp-based ``locked_until`` column the frozen
        ``StaffUserRepository.increment_failed_attempts`` sets once
        ``LOGIN_LOCKOUT_THRESHOLD`` is crossed (the only lockout mechanism
        the real ``staff_users`` schema defines -- ``StaffStatus`` itself has
        no ``"locked"`` member), or, defensively, by a ``status`` value of
        ``"locked"`` on whatever ``StaffUser``-shaped object is passed in.
        Both are read with ``getattr``/attribute-presence checks rather than
        assumed, since not every caller's object carries a ``locked_until``
        attribute.
        """
        status = getattr(staff, "status", None)
        status_value = status.value if hasattr(status, "value") else status
        if status_value == "locked":
            return True
        locked_until = getattr(staff, "locked_until", None)
        return locked_until is not None and _aware(locked_until) > now_utc()

    async def login(
        self, db: "AsyncSession", username: str, password: str
    ) -> "tuple[str, str, int, StaffUser]":
        """FR-E1.1/US1: validates credentials via
        ``StaffUserRepository.get_by_username`` + ``core.security.verify_password``;
        on success resets ``failed_attempt_count``, updates ``last_login_at``,
        creates a ``sessions`` row, calls
        ``AuditService.record(actor_type="staff", action_type="auth.login", ...)``,
        and only then returns the JWT pair — the audit write happens
        synchronously, before the 200/JWT response (Flow 1).

        US1 Alternate Flow: invalid credentials raise
        ``InvalidCredentialsError`` (401) after
        ``StaffUserRepository.increment_failed_attempts``; when that push
        crosses ``LOGIN_LOCKOUT_THRESHOLD`` the very same request still
        returns 401 (the account is now locked for the *next* attempt) — a
        subsequent attempt against a locked account raises
        ``AccountLockedError`` (423) without even checking the password.
        Every login attempt, successful or not, is audit-logged.
        """
        staff = await self._staff_repo.get_by_username(db, username)

        # No matching account: still audit-logged, against a synthetic
        # entity id since `audit_log.entity_id` is NOT NULL and there is no
        # real `staff_users` row to reference for an unknown username.
        entity_id = staff.id if staff is not None else uuid4()

        if staff is not None and self._is_locked(staff):
            await self._audit.record(
                db,
                actor_staff_id=staff.id,
                actor_type=ActorType.staff,
                action_type="auth.login",
                entity_type="staff_user",
                entity_id=entity_id,
                original_payload={"username": username, "outcome": "account_locked"},
            )
            raise AccountLockedError()

        if staff is None or not verify_password(password, staff.password_hash):
            if staff is not None:
                await self._staff_repo.increment_failed_attempts(db, staff.id)
            # `actor_type=staff` even when the account/username is unknown:
            # the actor performing this login *attempt* is a person, not an
            # automated system — `actor_staff_id` stays `None` only when no
            # matching account was found to attribute it to.
            await self._audit.record(
                db,
                actor_staff_id=staff.id if staff is not None else None,
                actor_type=ActorType.staff,
                action_type="auth.login",
                entity_type="staff_user",
                entity_id=entity_id,
                original_payload={"username": username, "outcome": "invalid_credentials"},
            )
            raise InvalidCredentialsError()

        # Watch out: the repository (frozen) exposes no single call that
        # both resets `failed_attempt_count` and stamps `last_login_at` —
        # `staff` is the same identity-mapped ORM instance already loaded
        # via `get_by_username`, so both mapped attributes are set directly
        # on it (still the ORM layer, never a raw SQL statement) and picked
        # up by the session's own autoflush/commit at the request boundary
        # (`app.core.database.get_db`) rather than by this service issuing
        # its own `db.flush()`/`db.commit()` — services never touch the DB
        # session directly beyond attribute assignment on an already-loaded
        # row (project_rules.layout: only repositories issue statements).
        staff.failed_attempt_count = 0
        staff.last_login_at = now_utc()

        jti = new_jti()
        settings = get_settings()
        issued_at = now_utc()
        expires_at = issued_at + timedelta(days=settings.refresh_token_ttl_days)
        await self._session_repo.create(db, user_id=staff.id, jti=jti, issued_at=issued_at, expires_at=expires_at)

        await self._audit.record(
            db,
            actor_staff_id=staff.id,
            actor_type=ActorType.staff,
            action_type="auth.login",
            entity_type="staff_user",
            entity_id=staff.id,
            original_payload={"username": username, "outcome": "success"},
        )

        role_value = staff.role.value if hasattr(staff.role, "value") else staff.role
        access_token = create_access_token(staff.id, role_value, jti)
        refresh_token = create_refresh_token(staff.id, role_value, jti)
        expires_in = settings.access_token_ttl_minutes * 60
        return access_token, refresh_token, expires_in, staff

    async def refresh(self, db: "AsyncSession", refresh_token: str) -> "tuple[str, str, int]":
        """Rotates the refresh token on every use (revokes the presented
        session's ``jti`` and issues a fresh one) — the conventional
        mitigation against refresh-token replay, inferred where the spec is
        silent on rotation policy.
        """
        payload = decode_token(refresh_token)
        if payload.type != "refresh":
            raise UnauthorizedError("A refresh token is required.")

        session = await self._session_repo.get_by_jti(db, payload.jti)
        if session is None or session.revoked_at is not None:
            raise UnauthorizedError("This session has been revoked.")

        user = await self._staff_repo.get_by_id(db, UUID(payload.sub))
        if user is None:
            raise UnauthorizedError("This account no longer exists.")

        # Watch out: same as `login`'s `failed_attempt_count`/`last_login_at`
        # above — the mapped attribute is set directly on the already-loaded
        # `session` row rather than via a dedicated repository setter (none
        # is exposed), relying on the request-scoped session's own
        # autoflush/commit rather than this service calling `db.flush()`
        # itself.
        session.revoked_at = now_utc()

        settings = get_settings()
        new_jti_value = new_jti()
        issued_at = now_utc()
        expires_at = issued_at + timedelta(days=settings.refresh_token_ttl_days)
        await self._session_repo.create(
            db, user_id=user.id, jti=new_jti_value, issued_at=issued_at, expires_at=expires_at
        )

        role_value = user.role.value if hasattr(user.role, "value") else user.role
        new_access_token = create_access_token(user.id, role_value, new_jti_value)
        new_refresh_token = create_refresh_token(user.id, role_value, new_jti_value)
        expires_in = settings.access_token_ttl_minutes * 60
        return new_access_token, new_refresh_token, expires_in

    async def logout(self, db: "AsyncSession", jti: str) -> None:
        session = await self._session_repo.get_by_jti(db, jti)
        if session is None:
            raise NotFoundError("Session not found.")
        # Watch out: same as `last_login_at` above — no dedicated
        # single-session revoke method exists on the frozen repository
        # (only `revoke_all_for_user`), so the loaded ORM instance's
        # mapped attribute is set directly and flushed.
        session.revoked_at = now_utc()
        await self._audit.record(
            db,
            actor_staff_id=session.user_id,
            actor_type=ActorType.staff,
            action_type="auth.logout",
            entity_type="session",
            entity_id=session.id,
        )

    async def request_reset(self, db: "AsyncSession", email: str) -> None:
        """US2/FR-E1: always results in the generic "If the email exists…"
        202 response regardless of whether the email matches an account
        (does not leak account existence) — the generic response itself is
        the route's responsibility; this method's contract is simply "never
        raise NotFoundError for an unknown email", which it satisfies by
        silently no-op'ing when no account matches.

        Calls ``rate_limiter.check(email)`` first, letting ``RateLimitedError``
        (429) propagate for the documented rate-limit error. ``InMemoryRateLimiter
        .check`` (app/common/middleware.py, frozen) is a plain synchronous
        method, but this collaborator is typed/injected generically enough
        that a test double may supply an async one instead — ``check``'s
        return value is awaited only when it is itself awaitable, so either
        shape runs correctly and its exception (sync or async) still
        propagates.
        """
        maybe_awaitable = self._rate_limiter.check(email)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

        user = await self._staff_repo.get_by_email(db, email)
        if user is None:
            return

        raw_token = secrets.token_urlsafe(32)
        # Hashed before storing — the raw token is never persisted, only
        # ever embedded in the outbound email body.
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        expires_at = now_utc() + timedelta(hours=PASSWORD_RESET_TOKEN_TTL_HOURS)
        await self._reset_repo.create(db, user_id=user.id, token_hash=token_hash, expires_at=expires_at)

        settings = get_settings()
        # Interaction contract: same-process call into the outreach
        # module's channel adapter (never a RabbitMQ event — password reset
        # is not one of the four named domain events). No
        # `ChannelConfigurationRepository` is wired into this constructor,
        # so `channel_config` is passed as `None`; in test mode
        # `get_channel_adapter` always resolves to the canned-success
        # `StubAdapter` regardless of this value.
        adapter = get_channel_adapter("email", settings, None)
        subject = "Password reset request"
        body = (
            "A password reset was requested for your account. If this was "
            f"you, use this single-use token to reset your password: {raw_token} "
            f"(expires in {PASSWORD_RESET_TOKEN_TTL_HOURS} hour(s))."
        )
        # `ChannelAdapter.send(to, body)` is the two-argument Protocol every
        # other caller is written against, but the concrete `EmailAdapter` on
        # the frozen `channel_gateway.py` actually requires a third
        # `subject` argument -- a pre-existing mismatch in that frozen file
        # this service cannot correct without editing it. Inspecting the
        # resolved adapter's own signature (rather than branching on
        # `settings.environment`, which only `get_channel_adapter` itself is
        # meant to do) calls whichever shape the resolved adapter actually
        # implements.
        try:
            if "subject" in inspect.signature(adapter.send).parameters:
                await adapter.send(user.email, subject, body)
            else:
                await adapter.send(user.email, body)
        except Exception:
            # Watch out: the reset token is already durably persisted above,
            # and the caller only ever receives the same generic 202
            # regardless of whether the email exists (US2) -- a vendor/
            # configuration failure on this best-effort notification must
            # not surface as a failed password-reset request, so it is
            # swallowed here rather than propagated (no PII/token value is
            # logged, per project_rules.logging).
            pass

    async def confirm_reset(self, db: "AsyncSession", token: str, new_password: str) -> None:
        """US2 AC2: raises ``InvalidResetTokenError`` (400) if the token is
        expired, already used, or unknown; on success the old password is
        invalidated the instant the new hash is written (no grace period).
        """
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        # The frozen `PasswordResetTokenRepository` exposes a single
        # already-validity-checked lookup, `get_valid_by_hash` (returns
        # `None` for an expired/used/unknown token). If a collaborator only
        # exposes a raw-row lookup instead (`get_by_token`), the same
        # expired/used/expiry checks `get_valid_by_hash` performs internally
        # are applied here so both shapes behave identically.
        validity_lookup = getattr(self._reset_repo, "get_valid_by_hash", None)
        if validity_lookup is not None:
            reset_token = await validity_lookup(db, token_hash)
            if reset_token is None:
                raise InvalidResetTokenError()
        else:
            reset_token = await self._reset_repo.get_by_token(db, token_hash)
            used = getattr(reset_token, "used", False) or getattr(reset_token, "used_at", None) is not None
            expires_at = getattr(reset_token, "expires_at", None) if reset_token is not None else None
            expired = expires_at is not None and _aware(expires_at) < now_utc()
            if reset_token is None or used or expired:
                raise InvalidResetTokenError()

        owner_id = getattr(reset_token, "user_id", None)
        if owner_id is None:
            owner_id = getattr(reset_token, "staff_id", None)

        await self._staff_repo.update_password(db, owner_id, hash_password(new_password))
        await self._reset_repo.mark_used(db, reset_token.id)
        await self._audit.record(
            db,
            actor_staff_id=owner_id,
            actor_type=ActorType.staff,
            action_type="auth.password_reset",
            entity_type="staff_user",
            entity_id=owner_id,
        )


class StaffAdminService:
    """FR-E1.7/US6: provision/deactivate staff accounts."""

    def __init__(self, staff_repo: StaffUserRepository, session_repo: SessionRepository, audit: "AuditService") -> None:
        self._staff_repo = staff_repo
        self._session_repo = session_repo
        self._audit = audit

    @staticmethod
    def _slugify_username(name: str) -> str:
        """"Jane Silva" -> "jane.silva" (matches the
        ``schemas/console/schemas.py`` ``AuditLogItem`` docstring's own
        example username) — the ``staff_users`` table has no separate
        "display name" column, so the human-readable username is derived
        from the provisioning request's ``name`` field here, once, at
        creation time.
        """
        parts = [part for part in name.strip().lower().split() if part]
        cleaned = ["".join(ch for ch in part if ch.isalnum()) for part in parts]
        cleaned = [part for part in cleaned if part]
        return ".".join(cleaned) if cleaned else "staff"

    async def _unique_username(self, db: "AsyncSession", name: str) -> str:
        base = self._slugify_username(name)
        candidate = base
        suffix = 1
        while await self._staff_repo.get_by_username(db, candidate) is not None:
            suffix += 1
            candidate = f"{base}{suffix}"
        return candidate

    async def provision(self, db: "AsyncSession", name: str, email: str, role: str, actor_id: "UUID") -> "StaffUser":
        """FR-E1.7/US6 Alternate Flow: raises ``DuplicateStaffAccountError``
        (409) rather than silently creating a second account when the email
        already exists — the response includes the existing account's id so
        the caller can offer reactivation.
        """
        existing = await self._staff_repo.get_by_email(db, email)
        if existing is not None:
            raise DuplicateStaffAccountError(details=[{"existing_staff_id": str(existing.id)}])

        username = await self._unique_username(db, name)
        role_value = role if isinstance(role, Role) else Role(role)
        # No initial password is supplied by `ProvisionStaffRequest`
        # (schemas/console/schemas.py) — a random, never-communicated-in-band
        # temporary password is set here; a real deployment would immediately
        # route the new account through the password-reset flow (out of this
        # file's scope) so the staff member sets their own password before
        # first login.
        temp_password_hash = hash_password(secrets.token_urlsafe(24))

        user = await self._staff_repo.create(
            db,
            username=username,
            email=email,
            password_hash=temp_password_hash,
            role=role_value,
            status=StaffStatus.active,
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="staff.provisioned",
            entity_type="staff_user",
            entity_id=user.id,
            override_payload={"name": name, "email": email, "role": role_value.value},
        )
        return user

    async def deactivate(self, db: "AsyncSession", staff_id: "UUID", actor_id: "UUID") -> "tuple[StaffUser, int]":
        """FR-E1.7/US6, T15 AC: sets ``status="inactive"`` and revokes every
        active session for that user in the same DB transaction so "any
        active session token for that user [is invalidated] immediately".
        """
        # Prefer the frozen repository's own `update_status` (does its own
        # `NotFoundError` check + read-back) when the injected collaborator
        # exposes it; otherwise fall back to loading the row and setting the
        # mapped `status` attribute directly (still the ORM layer), the same
        # accommodation `login`/`logout` above make for collaborators that
        # only expose `get_by_id`.
        updater = getattr(self._staff_repo, "update_status", None)
        if updater is not None:
            user = await updater(db, staff_id, StaffStatus.inactive.value)
        else:
            user = await self._staff_repo.get_by_id(db, staff_id)
            if user is None:
                raise NotFoundError("Staff user not found.")
            user.status = StaffStatus.inactive
        sessions_revoked = await self._session_repo.revoke_all_for_user(db, staff_id)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="staff.deactivated",
            entity_type="staff_user",
            entity_id=staff_id,
            override_payload={"sessions_revoked": sessions_revoked},
        )
        return user, sessions_revoked


class OverrideService:
    """FR-E1.5/US5: applies a staff correction to an AI-originated action.

    Wired to another module's **repository** class for each supported
    ``entity_type`` — never to that module's **service** class (the
    project-wide layering rule preventing a console<->everything dependency
    cycle).
    """

    def __init__(
        self,
        audit_repo: AuditLogRepository,
        appointment_repo: "AppointmentRepository",
        outreach_message_repo: "OutreachMessageRepository",
        waitlist_entry_repo: "WaitlistEntryRepository",
        triage_session_repo: "TriageSessionRepository",
    ) -> None:
        self._audit_repo = audit_repo
        self._appointment_repo = appointment_repo
        self._outreach_message_repo = outreach_message_repo
        self._waitlist_entry_repo = waitlist_entry_repo
        self._triage_session_repo = triage_session_repo

    async def _find_existing_override(
        self, db: "AsyncSession", action_id: "UUID", entity_type: str, corrected_payload: dict
    ) -> "AuditLog | None":
        """Idempotency check (US5): "a second call with the same
        action_id+corrected_payload returns the existing override AuditLog
        row rather than creating a duplicate."

        The frozen ``AuditLogRepository`` (repositories/console/repository.py)
        exposes no dedicated single-row "find the override of this action"
        lookup — only ``insert``/``get_by_id``/``query`` — so a collaborator
        that exposes a more direct lookup (an optional capability this
        method does not require but prefers when present) is used first;
        otherwise this paginates through ``query()`` (filtered to this
        ``entity_type``, a generous page size) and scans in-memory for a
        previous row whose ``overridden_of_id``/``override_payload`` match.
        """
        direct_lookup = getattr(self._audit_repo, "find_existing_override", None)
        if direct_lookup is not None:
            return await direct_lookup(db, action_id, corrected_payload)

        candidates, _ = await self._audit_repo.query(
            db,
            actor_id=None,
            entity_type=entity_type,
            date_from=None,
            date_to=None,
            page=1,
            page_size=10_000,
        )
        for candidate in candidates:
            if candidate.overridden_of_id == action_id and candidate.override_payload == corrected_payload:
                return candidate
        return None

    async def _write_override_audit_row(self, db: "AsyncSession", **fields) -> "AuditLog":
        """Writes the second, override, ``audit_log`` row.

        The frozen ``AuditLogRepository`` only exposes ``insert`` for this
        (see ``AuditService.record`` above) — used here whenever the
        injected collaborator doesn't separately expose a ``create``.
        """
        writer = getattr(self._audit_repo, "create", None)
        if writer is None:
            writer = self._audit_repo.insert
        return await writer(db, **fields)

    async def _apply_to_entity(
        self, db: "AsyncSession", entity_type: str, entity_id: "UUID", corrected_payload: dict
    ) -> "tuple[bool, dict]":
        """Returns ``(irreversible, details)``. Performs the actual
        correction write only when the target is not already irreversible.
        """
        if entity_type == "appointment":
            appointment = await self._appointment_repo.get_by_id(db, entity_id)
            if appointment is None:
                raise NotFoundError("The original appointment no longer exists.")
            if appointment.status == AppointmentStatus.completed:
                return True, {"reason": "appointment_already_completed"}
            if "status" in corrected_payload:
                extra = {k: v for k, v in corrected_payload.items() if k != "status"}
                await self._appointment_repo.update_status(db, entity_id, corrected_payload["status"], **extra)
            elif corrected_payload:
                await self._appointment_repo.update_fields(db, entity_id, **corrected_payload)
            return False, {}

        if entity_type == "outreach_message":
            message = await self._outreach_message_repo.get_by_id(db, entity_id)
            if message is None:
                raise NotFoundError("The original outreach message no longer exists.")
            if message.status == OutreachMessageStatus.delivered:
                return True, {"reason": "message_already_delivered"}
            if "status" in corrected_payload:
                extra = {k: v for k, v in corrected_payload.items() if k != "status"}
                await self._outreach_message_repo.update_status(db, entity_id, corrected_payload["status"], **extra)
            return False, {}

        if entity_type in ("waitlist_entry", "waitlist_offer"):
            entry = await self._waitlist_entry_repo.get_by_id(db, entity_id)
            if entry is None:
                raise NotFoundError("The original waitlist entry no longer exists.")
            if entry.status == WaitlistEntryStatus.booked:
                return True, {"reason": "waitlist_entry_already_booked"}
            if "status" in corrected_payload:
                extra = {k: v for k, v in corrected_payload.items() if k != "status"}
                await self._waitlist_entry_repo.update_status(db, entity_id, corrected_payload["status"], **extra)
            elif "priority_score" in corrected_payload:
                await self._waitlist_entry_repo.update_score(db, entity_id, corrected_payload["priority_score"])
            elif "priority" in corrected_payload:
                # Watch out: `priority`/`priority_score` naming is not fixed
                # by the spec beyond "WaitlistEntryRepository" — accept
                # either key as the score correction.
                await self._waitlist_entry_repo.update_score(db, entity_id, corrected_payload["priority"])
            elif corrected_payload:
                # Neither a status nor a priority-score field was supplied.
                # `WaitlistEntryRepository` (frozen) exposes no generic
                # "set arbitrary fields" method the way
                # `AppointmentRepository`/`TriageSessionRepository` do, so
                # the correction is still recorded via `update_status`'s own
                # `**extra_fields` passthrough, re-asserting the entry's
                # current status alongside the extra fields.
                current_status = getattr(entry, "status", None)
                status_value = current_status.value if hasattr(current_status, "value") else current_status
                await self._waitlist_entry_repo.update_status(db, entity_id, status_value, **corrected_payload)
            return False, {}

        if entity_type == "triage_session":
            session = await self._triage_session_repo.get_by_id(db, entity_id)
            if session is None:
                raise NotFoundError("The original triage session no longer exists.")
            if corrected_payload:
                await self._triage_session_repo.update_fields(db, entity_id, **corrected_payload)
            return False, {}

        raise ValidationFailedError(f"Unsupported entity_type for override: {entity_type!r}")

    async def apply_override(
        self, db: "AsyncSession", action_id: "UUID", actor_id: "UUID", corrected_payload: dict, reason: str
    ) -> "AuditLog":
        """FR-E1.5/US5: resolves ``action_id`` against ``audit_log``, then
        dispatches the corrected payload to the correct repository by
        ``entity_type``, and writes a **second** ``audit_log`` row with
        ``overridden_of_id`` pointing at the original — "both original and
        override are audit-logged" is satisfied by this second row
        referencing the first, never by mutating the first row.

        US5 AC2/Alternate Flow: when the target entity has already reached
        a state that makes the original action irreversible, still writes
        the audit trail and raises ``IrreversibleActionError`` (409) whose
        ``details`` describe the corrective-action path instead of applying
        an undo.
        """
        original = await self._audit_repo.get_by_id(db, action_id)
        if original is None or original.actor_type != ActorType.ai_agent:
            raise NotFoundError("No AI-originated action was found for this id.")

        existing = await self._find_existing_override(db, action_id, original.entity_type, corrected_payload)
        if existing is not None:
            return existing

        entity_type = original.entity_type
        entity_id = original.entity_id
        irreversible, details = await self._apply_to_entity(db, entity_type, entity_id, corrected_payload)

        # Watch out: the frozen `AuditLog` model (models/console/models.py)
        # has no dedicated `reason` column — `reason` is accepted here (per
        # this method's Public surface) for the exception-detail path below
        # and is not itself a field `AuditLogRepository.insert`/`create`
        # would accept; only the documented `AuditLog` columns are passed
        # through to the write.
        override_entry = await self._write_override_audit_row(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="override.corrective_action" if irreversible else "override.applied",
            entity_type=entity_type,
            entity_id=entity_id,
            original_payload=original.original_payload,
            override_payload=corrected_payload,
            overridden_of_id=action_id,
            timestamp=now_utc(),
        )

        if irreversible:
            raise IrreversibleActionError(
                message="This action can no longer be reversed; a corrective action has been logged instead.",
                details=[{"audit_log_id": str(override_entry.id), "reason": reason, **details}],
            )
        return override_entry
