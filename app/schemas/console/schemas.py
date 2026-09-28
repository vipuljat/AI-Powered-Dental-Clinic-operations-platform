"""Pydantic v2 request/response models for every console-module endpoint
(``/auth/*``, ``/console/*``) per architecture.md §5.2.

Responsibility: request/response shape and field-level validation only. No
persistence, no business rule enforcement (e.g. duplicate-account detection,
permission checks) — those live in
``app.services.console.service.StaffAdminService`` and friends.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, field_validator

from app.common.enums import Role, StaffStatus
from app.common.validators import validate_non_empty, validate_password_complexity


class LoginRequest(BaseModel):
    """POST /auth/login request body. Unauthenticated (Auth: none)."""

    username: str
    password: str


class LoginUser(BaseModel):
    """The ``user`` object nested in ``LoginResponse``.

    Interaction contract: ``role`` must be spelled exactly as one of
    ``app.common.enums.Role``'s values (``front_office_staff`` /
    ``clinic_management`` / ``delivery_team``) — the React console's
    role-gated navigation depends on that exact string set never changing
    shape. Only ``id``/``username``/``role`` are ever exposed — never
    ``password_hash``.
    """

    id: UUID
    username: str
    role: Role


class LoginResponse(BaseModel):
    """architecture.md §5.2 login response shape, transcribed verbatim.

    ``expires_in`` is in seconds (e.g. 900 for a 15-minute access token,
    per architecture.md §11 item 5's inferred default).
    """

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: LoginUser


class PasswordResetRequestRequest(BaseModel):
    """POST /auth/password-reset/request request body. Auth: none."""

    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """POST /auth/password-reset/confirm request body. Auth: none."""

    token: str
    new_password: str

    _validate_password = field_validator("new_password")(validate_password_complexity)


class DashboardWidget(BaseModel):
    key: str
    label: str
    visible: bool


class DashboardWidgetsResponse(BaseModel):
    widgets: list[DashboardWidget]


class OverrideRequest(BaseModel):
    """FR-E1.5 / US5 override request body.

    ``reason`` is required and non-blank via ``validate_non_empty``: US5's
    Basic Flow step 3 ("Staff makes the corrected change") combined with
    FR-E1.5's "log both the original and override action" requirement makes
    a blank reason indefensible to audit later, even though architecture.md's
    example payload does not mark ``reason`` itself ``[NOT NULL]``
    explicitly.
    """

    corrected_payload: dict
    reason: str

    _validate_reason = field_validator("reason")(validate_non_empty)


class OverrideResponse(BaseModel):
    id: UUID
    original_action_id: UUID
    applied_at: datetime
    actor_id: UUID


class ProvisionStaffRequest(BaseModel):
    """FR-E1.7 / US6 staff-provisioning request body.

    Carries no explicit dedup flag: the "Duplicate account requested: system
    flags the existing account for reactivation instead of duplicating" AC
    is a service-layer concern (``StaffAdminService.provision``'s
    ``409 DUPLICATE_STAFF_ACCOUNT`` path), not a schema-level one.
    """

    name: str
    email: EmailStr
    role: Role


class StaffResponse(BaseModel):
    id: UUID
    username: str
    role: Role
    status: StaffStatus


class DeactivateStaffResponse(BaseModel):
    id: UUID
    status: StaffStatus
    sessions_revoked: int


class AuditLogItem(BaseModel):
    """A single audit-log row as returned by GET /console/audit-log.

    ``actor`` is a display string ("jane.silva") per the architecture.md
    example response, not ``actor_staff_id`` — the repository/service layer
    resolves ``staff_users.username`` (or ``null``/``"ai_agent"``/
    ``"system"`` for non-staff actors) before this schema is populated; this
    schema itself does no resolution.
    """

    id: UUID
    actor: str | None
    action_type: str
    entity_type: str
    entity_id: UUID
    timestamp: datetime


class AuditLogQueryResponse(BaseModel):
    items: list[AuditLogItem]
    page: int
    page_size: int
    total: int
