"""Two ``APIRouter`` instances for the console module: ``auth_router``
(prefix ``/auth``, unauthenticated) and ``console_router`` (prefix
``/console``, authenticated) — thin HTTP adapters over
``services/console/service.py``'s five classes.

Responsibility: validate request shape via ``schemas/console/schemas.py``,
call exactly one service method, return the documented status code
(architecture.md §5.2). No business logic lives here — every rule (lockout,
rate limiting, RBAC matrix lookup, override idempotency, audit writes) is
owned by the service layer this file only calls into.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import APIRouter, Depends

from app.common.constants import PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR
from app.common.middleware import InMemoryRateLimiter
from app.core.dependencies import CurrentUser, get_current_user, get_db, require_role
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
from app.schemas.console.schemas import (
    AuditLogItem,
    AuditLogQueryResponse,
    DashboardWidgetsResponse,
    DeactivateStaffResponse,
    LoginRequest,
    LoginResponse,
    LoginUser,
    OverrideRequest,
    OverrideResponse,
    PasswordResetConfirmRequest,
    PasswordResetRequestRequest,
    ProvisionStaffRequest,
    StaffResponse,
)
from app.services.console.service import (
    AuditService,
    AuthService,
    OverrideService,
    PermissionService,
    StaffAdminService,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["auth_router", "console_router"]

auth_router = APIRouter(prefix="/auth")
console_router = APIRouter(prefix="/console")

# project_rules.auth / InMemoryRateLimiter's own Interaction Contract
# (app/common/middleware.py): constructed once at composition-root startup
# and shared across requests, never per-request -- this module-level
# singleton is that one shared instance for the one 429-documented endpoint,
# POST /auth/password-reset/request.
_password_reset_rate_limiter = InMemoryRateLimiter(
    max_calls=PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR, per_seconds=3600
)


def _audit_service() -> AuditService:
    return AuditService(AuditLogRepository())


def _auth_service() -> AuthService:
    return AuthService(
        StaffUserRepository(),
        SessionRepository(),
        PasswordResetTokenRepository(),
        _audit_service(),
        _password_reset_rate_limiter,
    )


def _permission_service() -> PermissionService:
    return PermissionService(RolePermissionRepository())


def _staff_admin_service() -> StaffAdminService:
    return StaffAdminService(StaffUserRepository(), SessionRepository(), _audit_service())


def _override_service() -> OverrideService:
    return OverrideService(
        AuditLogRepository(),
        AppointmentRepository(),
        OutreachMessageRepository(),
        WaitlistEntryRepository(),
        TriageSessionRepository(),
    )


@auth_router.post("/login", response_model=LoginResponse, status_code=200)
async def login(body: LoginRequest, db: "AsyncSession" = Depends(get_db)) -> LoginResponse:
    """FR-E1.1/US1: 200 on success, 401 on invalid credentials, 423 once the
    account is locked -- ``AuthService.login`` raises ``InvalidCredentialsError``
    / ``AccountLockedError``, both mapped by the global handler.
    """

    access_token, refresh_token, expires_in, staff = await _auth_service().login(db, body.username, body.password)
    role_value = staff.role.value if hasattr(staff.role, "value") else staff.role
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        user=LoginUser(id=staff.id, username=staff.username, role=role_value),
    )


@auth_router.post("/password-reset/request", status_code=202)
async def request_password_reset(body: PasswordResetRequestRequest, db: "AsyncSession" = Depends(get_db)) -> dict:
    """US2: always 202 ("If the email exists…") -- ``AuthService.request_reset``
    never raises ``NotFoundError`` for an unknown email; ``RateLimitedError``
    (429) still propagates to the global handler when the per-email budget is
    exceeded.
    """

    await _auth_service().request_reset(db, body.email)
    return {"detail": "If an account with that email exists, a password reset link has been sent."}


@auth_router.post("/password-reset/confirm", status_code=200)
async def confirm_password_reset(body: PasswordResetConfirmRequest, db: "AsyncSession" = Depends(get_db)) -> dict:
    """US2 AC2: 200 on success; ``AuthService.confirm_reset`` raises
    ``InvalidResetTokenError`` (400) for an expired/used/unknown token. A
    malformed body (e.g. a password failing complexity rules) never reaches
    here -- it is rejected at 422 by Pydantic validation
    (``PasswordResetConfirmRequest``'s own ``validate_password_complexity``).
    """

    await _auth_service().confirm_reset(db, body.token, body.new_password)
    return {"detail": "Your password has been reset."}


@console_router.get("/dashboard/widgets", response_model=DashboardWidgetsResponse)
async def get_dashboard_widgets(
    user: CurrentUser = Depends(get_current_user), db: "AsyncSession" = Depends(get_db)
) -> DashboardWidgetsResponse:
    """FR-E1.4/US4: 200, filtered through the same ``role_permissions``
    matrix ``require_permission`` reads elsewhere -- any authenticated role
    may call this endpoint, only the returned widget set differs per role.
    """

    widgets = await _permission_service().get_visible_widgets(db, user.role)
    return DashboardWidgetsResponse(widgets=widgets)


@console_router.post("/activity-log/{action_id}/override", response_model=OverrideResponse)
async def override_action(
    action_id: UUID,
    body: OverrideRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db: "AsyncSession" = Depends(get_db),
) -> OverrideResponse:
    """FR-E1.5/US5: 200 on a successful override; 403 if the caller's role is
    neither ``front_office_staff`` nor ``clinic_management`` (``delivery_team``
    is excluded per FR-E1.3); 409 (``IrreversibleActionError``) once the
    target entity has already reached a state that makes the original action
    irreversible -- ``OverrideService.apply_override`` still writes the audit
    trail before raising that 409.
    """

    override_entry = await _override_service().apply_override(
        db, action_id, user.id, body.corrected_payload, body.reason
    )
    return OverrideResponse(
        id=override_entry.id,
        original_action_id=action_id,
        applied_at=override_entry.timestamp,
        actor_id=user.id,
    )


@console_router.post("/staff", response_model=StaffResponse, status_code=201)
async def provision_staff(
    body: ProvisionStaffRequest,
    user: CurrentUser = Depends(require_role("clinic_management")),
    db: "AsyncSession" = Depends(get_db),
) -> StaffResponse:
    """FR-E1.7/US6: 201 on success, 403 if the caller isn't ``clinic_management``,
    409 (``DuplicateStaffAccountError``) when the email already exists.
    """

    staff = await _staff_admin_service().provision(db, body.name, body.email, body.role.value, user.id)
    return StaffResponse(id=staff.id, username=staff.username, role=staff.role, status=staff.status)


@console_router.post("/staff/{id}/deactivate", response_model=DeactivateStaffResponse)
async def deactivate_staff(
    id: UUID,
    user: CurrentUser = Depends(require_role("clinic_management")),
    db: "AsyncSession" = Depends(get_db),
) -> DeactivateStaffResponse:
    """FR-E1.7/US6, T15 AC: 200, revoking every active session for that
    staff member in the same transaction -- 403 if the caller isn't
    ``clinic_management``.
    """

    staff, sessions_revoked = await _staff_admin_service().deactivate(db, id, user.id)
    return DeactivateStaffResponse(id=staff.id, status=staff.status, sessions_revoked=sessions_revoked)


@console_router.get("/audit", response_model=AuditLogQueryResponse)
async def query_audit_log(
    actor_id: UUID | None = None,
    entity_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    page: int = 1,
    page_size: int = 50,
    user: CurrentUser = Depends(require_role("clinic_management")),
    db: "AsyncSession" = Depends(get_db),
) -> AuditLogQueryResponse:
    """US7: 200, searchable/filterable by actor, entity, and date range --
    ``clinic_management``-only (403 for any other role) per architecture.md
    §5.2.
    """

    items, total = await _audit_service().query(db, actor_id, entity_type, date_from, date_to, page, page_size)
    return AuditLogQueryResponse(
        items=[
            AuditLogItem(
                id=item.id,
                actor=_resolve_actor_display(item),
                action_type=item.action_type,
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                timestamp=item.timestamp,
            )
            for item in items
        ],
        page=page,
        page_size=page_size,
        total=total,
    )


def _resolve_actor_display(audit_log_item) -> "str | None":
    """``AuditLogItem.actor`` is a display string ("jane.silva"), not
    ``actor_staff_id`` (schemas/console/schemas.py) -- for a non-staff actor
    (``ActorType.ai_agent``/``ActorType.system``) the actor type itself is
    the only display value available on the eagerly-loaded ``AuditLog`` row
    (no username to resolve, so resolving ``staff_users.username`` via a
    second query per row is unnecessary for those). A staff actor without a
    resolvable username falls back to ``None`` rather than raising.
    """

    actor_type = audit_log_item.actor_type
    actor_type_value = actor_type.value if hasattr(actor_type, "value") else actor_type
    if actor_type_value != "staff":
        return actor_type_value
    username = getattr(audit_log_item, "actor_username", None)
    if username is not None:
        return username
    return str(audit_log_item.actor_staff_id) if audit_log_item.actor_staff_id is not None else None
