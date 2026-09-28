"""SQLAlchemy ORM declarations for the console module's five tables.

Pure schema — no query methods, no business logic. Query/mutation behaviour
(including the append-only enforcement on ``AuditLog``) lives in
``repositories/console/repository.py``; this file only declares columns and
constraints.

Interaction contract: every other module's ``models.py`` references
``staff_users.id`` via a plain ``UUID`` foreign-key column only
(``created_by_staff_id``, ``configured_by_staff_id``, etc.) — no cross-module
``relationship()`` is declared here or anywhere else, so no other module's
``models.py`` imports these classes. Only this module's own
``repository.py`` (and, for the two named cross-module repository
dependencies, ``core/dependencies.py`` and ``services/console/service.py``'s
``OverrideService``) import these classes directly.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import ActorType, Role, StaffStatus
from app.core.database import Base
from app.core.db_types import JSONBType


class StaffUser(Base):
    """A console user account.

    FR-E1.7 / US6: ``status`` defaults to ``active``; ``failed_attempt_count``
    and ``locked_until`` back the login-lockout rule (US1 Alternate Flow,
    open_questions[Q-d84091fb]: lock after ``LOGIN_LOCKOUT_THRESHOLD`` failed
    attempts for ``LOGIN_LOCKOUT_MINUTES``).
    """

    __tablename__ = "staff_users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    # bcrypt hash; never logged/displayed.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(SAEnum(Role, native_enum=False), nullable=False)
    status: Mapped[StaffStatus] = mapped_column(
        SAEnum(StaffStatus, native_enum=False),
        nullable=False,
        default=StaffStatus.active,
        server_default=StaffStatus.active.value,
    )
    # [INFERRED] lockout support.
    failed_attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class RolePermission(Base):
    """One (role, resource, action) -> allowed row in the data-driven permission matrix.

    Read by ``PermissionService.check`` (via ``require_role``/``require_permission``
    in ``app/core/dependencies.py``) — never a hardcoded per-route role list.
    """

    __tablename__ = "role_permissions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    role: Mapped[Role] = mapped_column(SAEnum(Role, native_enum=False), nullable=False)
    resource: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PasswordResetToken(Base):
    """A single-use password-reset token issued to a staff user."""

    __tablename__ = "password_reset_tokens"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Session(Base):
    """A JWT session/refresh record, keyed by the token's ``jti``.

    ``get_current_user`` (app/core/dependencies.py) checks ``revoked_at`` on
    every request to support force-logout.
    """

    __tablename__ = "sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
    jti: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """FR-E1.6: an append-only audit trail entry recording an actor + timestamp
    for every patient-affecting action.

    No ORM-level update/delete method is exposed anywhere in this tree — the
    append-only guarantee is enforced at the repository layer (``AuditRepository``
    in ``repositories/console/repository.py`` exposes only inserts/reads), not
    by a DB trigger, per ``[INFERRED]`` in architecture.md §4.2.

    ``entity_type``/``entity_id`` are a polymorphic pair with no DB-level FK:
    every writer across every module passes the target table name as a plain
    string (e.g. ``"appointment"``, ``"patient"``) and the target row's UUID.
    There is no referential-integrity guarantee across this pair, by design,
    so an audited entity may later be deleted while its audit trail is
    retained (FR-E2.5 archival).
    """

    __tablename__ = "audit_log"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Null for system/AI-originated actions.
    actor_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    # [INFERRED]
    actor_type: Mapped[ActorType] = mapped_column(SAEnum(ActorType, native_enum=False), nullable=False)
    action_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[UUID] = mapped_column(nullable=False)
    original_payload: Mapped[dict | None] = mapped_column(JSONBType, nullable=True)
    # [INFERRED] US5
    override_payload: Mapped[dict | None] = mapped_column(JSONBType, nullable=True)
    overridden_of_id: Mapped[UUID | None] = mapped_column(ForeignKey("audit_log.id"), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
