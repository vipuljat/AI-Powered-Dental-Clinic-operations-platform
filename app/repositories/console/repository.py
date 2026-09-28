"""Data-access classes over the five console tables.

Every SQL/ORM statement the console module issues lives here —
``services/console/service.py`` never constructs a query itself. Each method
takes an ``AsyncSession`` explicitly; no repository owns/creates its own
session (that is the composition root's / ``get_db``'s job, per
app/core/database.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.constants import LOGIN_LOCKOUT_MINUTES, LOGIN_LOCKOUT_THRESHOLD
from app.common.enums import Role
from app.common.exceptions.errors import NotFoundError
from app.common.utils import now_utc
from app.models.console.models import (
    AuditLog,
    PasswordResetToken,
    RolePermission,
    Session,
    StaffUser,
)

def _as_aware_utc(value: datetime) -> datetime:
    """Normalize a datetime read back from the DB to tz-aware UTC.

    SQLite (used per project_rules.testing's ENVIRONMENT=test pivot) does not
    retain a ``DateTime(timezone=True)`` column's UTC offset the way Postgres
    does — a value round-tripped through it comes back naive. Every timestamp
    in this tree is UTC (project_rules "all timestamps are timezone-aware
    UTC"), so a naive value read back is treated as already being UTC before
    it is compared against ``now_utc()``.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


__all__ = [
    "AuditLogRepository",
    "PasswordResetTokenRepository",
    "RolePermissionRepository",
    "SessionRepository",
    "StaffUserRepository",
]

# --- Interaction contract: default role_permissions seed matrix -------------
# The exact split named in project_rules.auth. `front_office_staff` gets every
# operational resource; `clinic_management` gets that same set plus staff
# admin/audit/rule-approval/financial grants; `delivery_team` gets only rule
# authoring + eligibility/cadence configuration + WhatsApp channel admin
# (FR-E1.3: delivery_team must never receive an operational grant).
_FRONT_OFFICE_GRANTS: list[tuple[str, str]] = [
    ("patients", "read"),
    ("patients", "write"),
    ("scheduling", "read"),
    ("scheduling", "write"),
    ("waitlist", "read"),
    ("waitlist", "write"),
    ("outreach.exceptions", "read"),
    ("outreach.exceptions", "write"),
    ("recall", "read"),
    ("recall", "write"),
    ("intelligence.triage_review", "read"),
    ("intelligence.triage_review", "write"),
    ("intelligence.utilisation", "read"),
    ("analytics.dashboards", "read"),
]

_CLINIC_MANAGEMENT_EXTRA_GRANTS: list[tuple[str, str]] = [
    ("console.staff_admin", "write"),
    ("rules.approve", "write"),
    ("console.audit", "read"),
    ("analytics.financial", "read"),
    ("analytics.baseline_cost", "write"),
]

_DELIVERY_TEAM_GRANTS: list[tuple[str, str]] = [
    ("rules.rule_sets", "write"),
    ("rules.configuration", "write"),
    ("outreach.channels.whatsapp", "write"),
]

_DEFAULT_ROLE_PERMISSIONS: list[tuple[Role, str, str, bool]] = (
    [(Role.front_office_staff, resource, action, True) for resource, action in _FRONT_OFFICE_GRANTS]
    + [(Role.clinic_management, resource, action, True) for resource, action in _FRONT_OFFICE_GRANTS]
    + [(Role.clinic_management, resource, action, True) for resource, action in _CLINIC_MANAGEMENT_EXTRA_GRANTS]
    + [(Role.delivery_team, resource, action, True) for resource, action in _DELIVERY_TEAM_GRANTS]
)


class StaffUserRepository:
    """Data access for ``staff_users``."""

    async def get_by_username(self, db: AsyncSession, username: str) -> StaffUser | None:
        result = await db.execute(select(StaffUser).where(StaffUser.username == username))
        return result.scalar_one_or_none()

    async def get_by_id(self, db: AsyncSession, id: UUID) -> StaffUser | None:
        result = await db.execute(select(StaffUser).where(StaffUser.id == id))
        return result.scalar_one_or_none()

    async def get_by_email(self, db: AsyncSession, email: str) -> StaffUser | None:
        result = await db.execute(select(StaffUser).where(StaffUser.email == email))
        return result.scalar_one_or_none()

    async def create(self, db: AsyncSession, **fields) -> StaffUser:
        user = StaffUser(**fields)
        db.add(user)
        await db.flush()
        await db.refresh(user)
        return user

    async def update_status(self, db: AsyncSession, id: UUID, status: str) -> StaffUser:
        user = await self.get_by_id(db, id)
        if user is None:
            raise NotFoundError("Staff user not found.")
        user.status = status
        await db.flush()
        await db.refresh(user)
        return user

    async def increment_failed_attempts(self, db: AsyncSession, id: UUID) -> StaffUser:
        """US1 Alternate Flow: locks the account after a configured threshold.

        This is a single atomic read-modify-write against ``staff_users`` that
        must not race with a concurrent login attempt on the same account, so
        the lockout side-effect lives here rather than in ``AuthService``.
        """
        user = await self.get_by_id(db, id)
        if user is None:
            raise NotFoundError("Staff user not found.")
        user.failed_attempt_count += 1
        if user.failed_attempt_count >= LOGIN_LOCKOUT_THRESHOLD:
            user.locked_until = now_utc() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
            user.failed_attempt_count = 0
        await db.flush()
        await db.refresh(user)
        return user

    async def reset_failed_attempts(self, db: AsyncSession, id: UUID) -> None:
        user = await self.get_by_id(db, id)
        if user is None:
            raise NotFoundError("Staff user not found.")
        user.failed_attempt_count = 0
        await db.flush()

    async def update_password(self, db: AsyncSession, id: UUID, password_hash: str) -> None:
        user = await self.get_by_id(db, id)
        if user is None:
            raise NotFoundError("Staff user not found.")
        user.password_hash = password_hash
        await db.flush()


class SessionRepository:
    """Data access for ``sessions`` (JWT session/refresh records)."""

    async def create(
        self,
        db: AsyncSession,
        user_id: UUID,
        jti: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> Session:
        session = Session(
            user_id=user_id,
            jti=jti,
            issued_at=issued_at,
            expires_at=expires_at,
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        return session

    async def get_by_jti(self, db: AsyncSession, jti: str) -> Session | None:
        result = await db.execute(select(Session).where(Session.jti == jti))
        return result.scalar_one_or_none()

    async def revoke_all_for_user(self, db: AsyncSession, user_id: UUID) -> int:
        """Sets ``revoked_at`` on every un-revoked session for ``user_id``
        (force-logout, checked by ``get_current_user``); returns the count revoked.
        """
        result = await db.execute(
            select(Session).where(Session.user_id == user_id, Session.revoked_at.is_(None))
        )
        sessions = result.scalars().all()
        revoked_at = now_utc()
        for session in sessions:
            session.revoked_at = revoked_at
        await db.flush()
        return len(sessions)


class RolePermissionRepository:
    """Data access for the data-driven ``role_permissions`` matrix."""

    async def list_for_role(self, db: AsyncSession, role: str) -> list[RolePermission]:
        result = await db.execute(select(RolePermission).where(RolePermission.role == role))
        return list(result.scalars().all())

    async def is_allowed(self, db: AsyncSession, role: str, resource: str, action: str) -> bool:
        result = await db.execute(
            select(RolePermission.allowed).where(
                RolePermission.role == role,
                RolePermission.resource == resource,
                RolePermission.action == action,
            )
        )
        allowed = result.scalar_one_or_none()
        return bool(allowed)

    async def seed_defaults(self, db: AsyncSession) -> None:
        """Idempotent: inserts the default ``role_permissions`` matrix (see
        this file's module-level Interaction Contract constants) only for
        (role, resource, action) triples not already present.
        """
        result = await db.execute(select(RolePermission.role, RolePermission.resource, RolePermission.action))
        existing = {(row.role, row.resource, row.action) for row in result.all()}
        for role, resource, action, allowed in _DEFAULT_ROLE_PERMISSIONS:
            key = (role, resource, action)
            if key in existing:
                continue
            db.add(RolePermission(role=role, resource=resource, action=action, allowed=allowed))
            existing.add(key)
        await db.flush()


class PasswordResetTokenRepository:
    """Data access for ``password_reset_tokens``."""

    async def create(
        self, db: AsyncSession, user_id: UUID, token_hash: str, expires_at: datetime
    ) -> PasswordResetToken:
        token = PasswordResetToken(user_id=user_id, token_hash=token_hash, expires_at=expires_at)
        db.add(token)
        await db.flush()
        await db.refresh(token)
        return token

    async def get_valid_by_hash(self, db: AsyncSession, token_hash: str) -> PasswordResetToken | None:
        """Returns ``None`` if not found, already used, or expired."""
        result = await db.execute(
            select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
        )
        token = result.scalar_one_or_none()
        if token is None:
            return None
        if token.used_at is not None:
            return None
        if _as_aware_utc(token.expires_at) < now_utc():
            return None
        return token

    async def mark_used(self, db: AsyncSession, id: UUID) -> None:
        result = await db.execute(select(PasswordResetToken).where(PasswordResetToken.id == id))
        token = result.scalar_one_or_none()
        if token is None:
            raise NotFoundError("Password reset token not found.")
        token.used_at = now_utc()
        await db.flush()


class AuditLogRepository:
    """FR-E1.6: audit_log is append-only.

    Only ``insert``/``get_by_id``/``query`` are exposed here — there is no
    ``update``/``delete`` method anywhere on this class, which is the
    enforcement mechanism (application-layer, not a DB trigger, per
    architecture.md §4.2 "immutable at the application layer").
    """

    async def insert(self, db: AsyncSession, **fields) -> AuditLog:
        entry = AuditLog(**fields)
        db.add(entry)
        await db.flush()
        await db.refresh(entry)
        return entry

    async def get_by_id(self, db: AsyncSession, id: UUID) -> AuditLog | None:
        result = await db.execute(select(AuditLog).where(AuditLog.id == id))
        return result.scalar_one_or_none()

    async def query(
        self,
        db: AsyncSession,
        actor_id: UUID | None,
        entity_type: str | None,
        date_from: datetime | None,
        date_to: datetime | None,
        page: int,
        page_size: int,
    ) -> tuple[list[AuditLog], int]:
        """Filters on ``audit_log.timestamp`` (the table has no
        ``created_at``/``updated_at`` pair — only ``timestamp``).
        """
        stmt = select(AuditLog)
        count_stmt = select(func.count()).select_from(AuditLog)
        conditions = []
        if actor_id is not None:
            conditions.append(AuditLog.actor_staff_id == actor_id)
        if entity_type is not None:
            conditions.append(AuditLog.entity_type == entity_type)
        if date_from is not None:
            conditions.append(AuditLog.timestamp >= date_from)
        if date_to is not None:
            conditions.append(AuditLog.timestamp <= date_to)
        for condition in conditions:
            stmt = stmt.where(condition)
            count_stmt = count_stmt.where(condition)

        total = (await db.execute(count_stmt)).scalar_one()

        stmt = stmt.order_by(AuditLog.timestamp.desc()).offset((page - 1) * page_size).limit(page_size)
        items = list((await db.execute(stmt)).scalars().all())
        return items, total
