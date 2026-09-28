"""The one place FastAPI ``Depends(...)`` auth/permission wiring is defined.

``get_current_user``, ``require_role``, ``require_permission``, and a
re-exported ``get_db`` — every ``routes/*/routes.py`` file across all 11
business modules imports from here for auth, never re-implements token
decoding or a role check locally. This file does not itself define what a
role is allowed to do (that is ``PermissionService.check`` /
the data-driven ``role_permissions`` table, ``services/console/service.py``)
— it only wires the dependency and turns a denial into the right exception.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.common.enums import ActorType
from app.common.exceptions.errors import InsufficientPermissionsError, UnauthorizedError
from app.core.database import get_db
from app.core.security import decode_token
from app.repositories.console.repository import (
    AuditLogRepository,
    RolePermissionRepository,
    SessionRepository,
)
from app.services.console.service import AuditService, PermissionService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["CurrentUser", "get_current_user", "get_db", "require_permission", "require_role"]

# Extracts the Bearer token from the ``Authorization`` header. ``tokenUrl`` is
# purely documentation metadata for the OpenAPI/Swagger UI "Authorize" flow —
# the actual login endpoint (POST /auth/login) is owned by the console
# module's routes.py, not this file.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")


class CurrentUser(BaseModel):
    """The minimal identity/authorization context resolved from a validated
    JWT — everything a route or downstream dependency needs to know about
    "who is making this request", without a second DB round trip.
    """

    id: UUID
    role: str


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: "AsyncSession" = Depends(get_db),
) -> CurrentUser:
    """FR-E1.1/US1 AC2: the resolved role drives every downstream permission
    check.

    Decodes ``token`` via ``app.core.security.decode_token`` (raises
    ``UnauthorizedError`` itself on an expired/malformed/invalid-signature
    token, propagated unchanged here), then looks up the session by ``jti``
    via ``SessionRepository.get_by_jti``. US6 AC / ``sessions.revoked_at``:
    checking revocation on *every* request (not just at login) is what makes
    a force-logout take effect on the very next request a deactivated staff
    member makes, even with an unexpired JWT still in hand — so a missing
    session row or one with ``revoked_at`` set both raise ``UnauthorizedError``
    (401) here.

    Watch out: deliberately trusts the token's own ``sub``/``role`` claims
    rather than re-querying ``staff_users.role`` on every request (a
    performance choice for the documented p95<500ms budget) — a mid-session
    role change does not take effect until the affected user's token is
    refreshed/re-issued. Acceptable because role changes are not a feature
    this BRD exposes (roles are fixed at account provisioning, US6); this is
    flagged so a per-request DB round trip isn't added later without
    realizing the budget trade-off being made.
    """

    payload = decode_token(token)

    session_repo = SessionRepository()
    session = await session_repo.get_by_jti(db, payload.jti)
    if session is None or session.revoked_at is not None:
        raise UnauthorizedError("This session has expired or been revoked.")

    return CurrentUser(id=UUID(payload.sub), role=payload.role)


def require_role(*allowed_roles: str):
    """FR-E1.2/US3: a simpler/cheaper variant of ``require_permission`` for
    endpoints whose access rule is a flat role allow-list rather than a
    ``(resource, action)`` matrix lookup (e.g. ``clinic_management``-only
    endpoints like staff provisioning).

    Returns a FastAPI dependency callable that depends on
    ``get_current_user`` and raises ``ForbiddenError`` (403) if
    ``current_user.role`` is not one of ``allowed_roles``, else returns
    ``current_user`` unchanged.
    """

    async def _dependency(
        current_user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if current_user.role not in allowed_roles:
            raise InsufficientPermissionsError()
        return current_user

    return _dependency


def require_permission(resource: str, action: str):
    """FR-E1.2/US3: the server-side enforcement point for "distinct
    permission sets enforced on every action, server-side, independent of UI
    state".

    Returns a FastAPI dependency callable that depends on
    ``get_current_user``, calls
    ``PermissionService.check(db, current_user.role, resource, action)``
    (``services/console/service.py``), and raises
    ``InsufficientPermissionsError`` (403, ``code="INSUFFICIENT_PERMISSIONS"``)
    if the check denies the action.

    US3 Alternate Flow: "Unauthorized action attempted: system denies, shows
    a clear message, and logs the attempt" — the denial path here calls
    ``AuditService.record(actor_staff_id=current_user.id,
    action_type="permission.denied", entity_type=resource, entity_id=None,
    ...)`` *before* raising, so every denied attempt is itself audit-logged,
    not just the granted ones. ``entity_id`` is required (non-null) on the
    ``audit_log`` schema (``NotNull``), so ``current_user.id`` is reused
    there too — there is no other entity to reference for a bare permission
    check.

    Never hardcodes a per-route role list beyond the ``(resource, action)``
    string pair passed in by the route file — the actual allow/deny decision
    is always delegated to ``PermissionService.check`` reading the
    data-driven ``role_permissions`` table.
    """

    async def _dependency(
        current_user: CurrentUser = Depends(get_current_user),
        db: "AsyncSession" = Depends(get_db),
    ) -> CurrentUser:
        permission_service = PermissionService(RolePermissionRepository())
        allowed = await permission_service.check(db, current_user.role, resource, action)
        if not allowed:
            audit_service = AuditService(AuditLogRepository())
            await audit_service.record(
                db,
                actor_staff_id=current_user.id,
                actor_type=ActorType.staff,
                action_type="permission.denied",
                entity_type=resource,
                entity_id=current_user.id,
                original_payload={"resource": resource, "action": action, "role": current_user.role},
            )
            raise InsufficientPermissionsError()
        return current_user

    return _dependency
