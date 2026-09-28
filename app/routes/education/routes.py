"""Thin HTTP adapter over ``EducationTriggerService``'s two read-only
wrapper methods (architecture.md §5.2).

Responsibility: this module's routes are read-only -- every write
(``EducationTriggerService.pre_appointment``/``.post_appointment``) is
invoked only by ``app/workers/recall_scanner.py``, never by a route here.
This module's ``router`` carries only its own ``/education`` sub-prefix; the
shared ``/api/v1`` prefix is added on top by ``app/main.py`` when it calls
``app.include_router`` (Interaction contract).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table
# is not already registered there -- but `app.core.dependencies` transitively
# imports that waitlist repository (via `services.console.service`) before
# this module otherwise would import the real `app.models.patients.models`.
# Importing the real patients repository first, here, ensures the real
# declarative `Patient` table claims the "patients" name before that stub
# loop ever runs, so its `if _table_name not in Base.metadata.tables` check
# correctly no-ops instead of racing a second, conflicting declaration
# (mirrors `app/routes/patients/routes.py`'s own fix for the same race).
#
# That ordering only helps when *this module* is the first of the two to be
# imported. A caller free to import `app.core.dependencies` on its own
# first (e.g. this module's own unit tests, which import
# `app.core.dependencies` before `app.routes.education.routes`) still hits
# the stub-registers-first case, so guard against that direction too: if the
# stand-in already claimed "patients" by the time this file runs, drop it
# from `Base.metadata` before the real `Patient` model below declares the
# genuine table, so the real declaration always wins regardless of which
# module a caller happens to import first.
from app.core.database import Base  # isort: skip

_existing_patients_table = Base.metadata.tables.get("patients")
if _existing_patients_table is not None and set(_existing_patients_table.columns.keys()) == {"id"}:
    Base.metadata.remove(_existing_patients_table)

from app.repositories.patients.repository import PatientRepository  # isort: skip

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.models.education.models import ContentItem
from app.repositories.console.repository import AuditLogRepository
from app.repositories.education.repository import (
    ContentDeliveryRepository,
    ContentItemRepository,
)
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    MessageTemplateRepository,
    OutreachMessageRepository,
)
from app.repositories.rules.repository import ConfigurationParameterRepository
from app.schemas.education.schemas import (
    CHANNEL_OPEN_TRACKING_CAPABILITY,
    ContentDeliveryItem,
    ContentDeliveryListResponse,
    TrackingFunnelResponse,
)
from app.services.console.service import AuditService
from app.services.education.service import EducationTriggerService
from app.services.outreach.service import OutreachService
from app.services.rules.service import ConfigurationService

router = APIRouter(prefix="/education")


def _audit_service() -> AuditService:
    # Each request builds its own thin service/repository graph over the
    # request-scoped `AsyncSession` (`Depends(get_db)`) -- no collaborator
    # here is a module-level singleton, matching the rest of this codebase's
    # `require_permission`/`AuthService` wiring (app/core/dependencies.py).
    return AuditService(AuditLogRepository())


def _outreach_service() -> OutreachService:
    # `EducationTriggerService` dispatches through the shared
    # `OutreachService.dispatch` engine (Responsibility), so this file wires
    # one up exactly the way every other campaign-trigger caller does
    # (mirrors `app/workers/outreach_retry.py`'s own construction), even
    # though this module's two GET handlers never actually reach `dispatch`.
    settings = get_settings()
    config_service = ConfigurationService(ConfigurationParameterRepository(), _audit_service())
    return OutreachService(
        message_repo=OutreachMessageRepository(),
        consent_repo=ConsentRepository(),
        template_repo=MessageTemplateRepository(),
        channel_config_repo=ChannelConfigurationRepository(),
        config_service=config_service,
        audit=_audit_service(),
        settings=settings,
    )


def _education_service() -> EducationTriggerService:
    return EducationTriggerService(
        ContentItemRepository(),
        ContentDeliveryRepository(),
        PatientRepository(),
        _outreach_service(),
    )


@router.get("/tracking", response_model=TrackingFunnelResponse)
async def get_content_tracking(
    content_item_id: "UUID | None" = None,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> TrackingFunnelResponse:
    """architecture.md §5.2 `GET /education/tracking`: 200 -- the
    delivered -> opened -> completed funnel, plus the static
    `CHANNEL_OPEN_TRACKING_CAPABILITY` table's list of channels whose opens
    cannot be tracked (FR-E10.3), optionally scoped to one
    `content_item_id`.
    """
    service = _education_service()
    funnel = await service.get_tracking_funnel(db, content_item_id)
    channel_limited = [
        channel for channel, can_track_opens in CHANNEL_OPEN_TRACKING_CAPABILITY.items() if not can_track_opens
    ]
    return TrackingFunnelResponse(
        delivered=funnel["delivered"],
        opened=funnel["opened"],
        completed=funnel["completed"],
        channel_limited=channel_limited,
    )


@router.get("/deliveries", response_model=ContentDeliveryListResponse)
async def list_content_deliveries(
    patient_id: "UUID",
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> ContentDeliveryListResponse:
    """architecture.md §5.2 `GET /education/deliveries`: 200 -- every
    `content_deliveries` row recorded for this patient.

    Watch out: the frozen `ContentDelivery` ORM model carries only a plain
    FK `content_item_id` column and no cross-module/-table ORM
    `relationship()` (its own Interaction contract), and
    `ContentItemRepository` (frozen) exposes no by-id lookup -- only
    `get_by_stage_language`/`get_general_fallback`. `ContentDeliveryItem`
    still requires `trigger_type`, so this handler issues one extra
    read-only `select()` against the frozen `ContentItem` ORM model directly
    to resolve each delivery's `trigger_type`, mirroring
    `routes/outreach/routes.py`'s own documented pattern for a case no
    repository method covers -- a documented gap against the frozen
    contract, reported in the task summary.
    """
    service = _education_service()
    deliveries = await service.list_deliveries_for_patient(db, patient_id)

    content_item_ids = {delivery.content_item_id for delivery in deliveries}
    trigger_types: dict[UUID, object] = {}
    if content_item_ids:
        result = await db.execute(select(ContentItem).where(ContentItem.id.in_(content_item_ids)))
        trigger_types = {item.id: item.trigger_type for item in result.scalars().all()}

    return ContentDeliveryListResponse(
        items=[
            ContentDeliveryItem(
                content_item_id=delivery.content_item_id,
                trigger_type=trigger_types[delivery.content_item_id],
                status=delivery.status,
                delivered_at=delivery.delivered_at,
            )
            for delivery in deliveries
        ]
    )
