"""Thin HTTP adapter over ``IdleChairPollingService``/``WaitlistService``/
``WaitlistFillService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema, delegates to the service layer, and shapes the response. This
module's ``router`` carries only its own ``/waitlist`` sub-prefix; the shared
``/api/v1`` prefix is added on top by ``app/main.py`` when it calls
``app.include_router`` (Interaction contract).
"""

from __future__ import annotations

from datetime import timezone
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy import select

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table is
# not already registered there — but `app.core.dependencies` transitively
# imports that very waitlist repository (via `services.console.service`,
# which imports `repositories.scheduling.repository` immediately before it,
# so `providers`/`chairs`/`appointments` are already real by that point —
# only `patients` is ever left as a stub) before this module otherwise would
# import the real `app.models.patients.models`. Importing the real patients
# repository first, here, ensures the real declarative `Patient` table claims
# the "patients" name before that stub loop ever runs, so its `if
# _table_name not in Base.metadata.tables` check correctly no-ops instead of
# racing a second, conflicting declaration (mirrors
# `app/routes/scheduling/routes.py` and `app/routes/patients/routes.py`'s own
# identical fix — this file needs `RecordLockRepository` for the same reason
# they do: constructing a `SchedulingService` for `WaitlistFillService`'s
# on-accept booking call).
from app.repositories.patients.repository import RecordLockRepository  # isort: skip

from app.common.enums import IdleChairStatus
from app.common.exceptions.errors import NotFoundError
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.core.messaging import get_broker
from app.models.waitlist.models import IdleChairAlert
from app.repositories.console.repository import AuditLogRepository
from app.repositories.rules.repository import RuleDefinitionRepository, RuleSetRepository
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.repositories.waitlist.repository import (
    IdleChairAlertRepository,
    WaitlistEntryRepository,
    WaitlistOfferRepository,
)
from app.schemas.waitlist.schemas import (
    AddWaitlistEntryRequest,
    FlagIdleChairRequest,
    FlagIdleChairResponse,
    IdleChairAlertItem,
    IdleChairAlertListResponse,
    RecoveryOfferStatus,
    RecoveryStatusResponse,
    RespondToOfferRequest,
    RespondToOfferResponse,
    WaitlistEntryListResponse,
    WaitlistEntryRankedItem,
    WaitlistEntryResponse,
)
from app.services.console.service import AuditService
from app.services.patients.service import RecordLockService
from app.services.rules.service import RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService
from app.services.waitlist.service import IdleChairPollingService, WaitlistFillService, WaitlistService

router = APIRouter(prefix="/waitlist")


def _aware(value):
    """Normalizes a datetime read back from the DB to tz-aware UTC for
    response serialization — project_rules.testing's SQLite pivot does not
    retain a ``DateTime(timezone=True)`` column's UTC offset the way
    Postgres does, mirroring the identical helper in
    ``services/waitlist/service.py`` and its siblings. Presentation-only:
    every timestamp in this tree is UTC, so a naive value read back is
    treated as already being UTC.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


# --- Per-request service/repository graph construction ---------------------
# Each request builds its own thin service/repository graph over the
# request-scoped `AsyncSession` (`Depends(get_db)`) -- no collaborator here is
# a module-level singleton, matching the rest of this codebase's
# `require_permission`/`AuthService` wiring (app/core/dependencies.py).


def _audit_service() -> AuditService:
    return AuditService(AuditLogRepository())


def _rule_set_service() -> RuleSetService:
    rule_def_repo = RuleDefinitionRepository()
    validator = RuleValidationService(rule_def_repo)
    return RuleSetService(RuleSetRepository(), rule_def_repo, validator, _audit_service())


def _lock_service() -> RecordLockService:
    return RecordLockService(RecordLockRepository())


def _scheduling_service() -> SchedulingService:
    settings = get_settings()
    return SchedulingService(
        AppointmentRepository(),
        AppointmentHistoryRepository(),
        AppointmentImportRepository(),
        _lock_service(),
        _rule_set_service(),
        _audit_service(),
        get_broker(settings),
    )


def _polling_service() -> IdleChairPollingService:
    return IdleChairPollingService(IdleChairAlertRepository(), _audit_service())


def _waitlist_service() -> WaitlistService:
    return WaitlistService(WaitlistEntryRepository(), _rule_set_service(), _audit_service())


def _fill_service() -> WaitlistFillService:
    settings = get_settings()
    return WaitlistFillService(
        WaitlistOfferRepository(),
        WaitlistEntryRepository(),
        IdleChairAlertRepository(),
        _scheduling_service(),
        get_broker(settings),
        _audit_service(),
    )


@router.post("/idle-chairs/flag", response_model=FlagIdleChairResponse)
async def flag_idle_chair(
    body: FlagIdleChairRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> JSONResponse:
    """architecture.md §5.2 `POST /waitlist/idle-chairs/flag`: 201 (new
    alert)/200 (merged into an already-open alert) — FR-E5.2/US24.

    `response_model=FlagIdleChairResponse` documents the OpenAPI schema; the
    actual per-call status code (201 vs. 200) can only be known after the
    service call returns, so a `JSONResponse` is constructed and returned
    directly here (bypassing FastAPI's own `response_model` serialization,
    which only ever applies a single decorator-level default) rather than
    relying on the decorator's fixed `status_code=`.
    """
    service = _polling_service()
    alert, created = await service.flag_manual(
        db, body.provider_id, body.chair_id, body.slot_start, body.slot_end, user.id
    )
    merged = not created
    payload = FlagIdleChairResponse(id=alert.id, status=alert.status, merged=merged)
    return JSONResponse(status_code=200 if merged else 201, content=jsonable_encoder(payload))


@router.get("/idle-chairs", response_model=IdleChairAlertListResponse)
async def list_idle_chair_alerts(
    status: str = "open",
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> IdleChairAlertListResponse:
    """architecture.md §5.2 `GET /waitlist/idle-chairs`: 200.

    Watch out: `IdleChairPollingService` (frozen) exposes no list/read
    method at all — only `run_cycle`/`detect_from_slot`/`flag_manual`, every
    one of which mutates. The frozen `IdleChairAlertRepository` itself only
    covers the default `status="open"` case (`list_open`); a caller passing
    any other `status` value is served by a direct, read-only `select()`
    against the frozen `IdleChairAlert` ORM model instead, since no service/
    repository method exists to call for that case. This is a documented gap
    against the frozen contract (no business logic added — filtering only),
    reported in the task summary.
    """
    if status == IdleChairStatus.open.value:
        alerts = await IdleChairAlertRepository().list_open(db)
    else:
        result = await db.execute(select(IdleChairAlert).where(IdleChairAlert.status == status))
        alerts = list(result.scalars().all())

    return IdleChairAlertListResponse(
        items=[
            IdleChairAlertItem(
                id=alert.id,
                provider_id=alert.provider_id,
                slot_start=_aware(alert.slot_start),
                source=alert.source,
                status=alert.status,
            )
            for alert in alerts
        ]
    )


@router.post("/entries", response_model=WaitlistEntryResponse, status_code=201)
async def add_waitlist_entry(
    body: AddWaitlistEntryRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> WaitlistEntryResponse:
    """architecture.md §5.2 `POST /waitlist/entries`: 201 — FR-E5.3/US25."""
    service = _waitlist_service()
    entry = await service.add_entry(db, body.model_dump(), user.id)
    return WaitlistEntryResponse(id=entry.id, priority_score=entry.priority_score, status=entry.status)


@router.get("/entries", response_model=WaitlistEntryListResponse)
async def list_waitlist_entries(
    status: str = "active",
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> WaitlistEntryListResponse:
    """architecture.md §5.2 `GET /waitlist/entries`: 200 — FR-E5.4/US26.

    Watch out: `WaitlistService` (frozen) exposes no "list the current
    waitlist" read method (only `add_entry`/`recalculate_priority`/
    `match_candidates`, every one of which either mutates or is scoped to a
    specific slot) — the frozen `WaitlistEntryRepository.list_by_priority`
    (the exact query `recalculate_priority` itself delegates to) is called
    directly here instead, a documented gap against the frozen service
    contract (no business logic added — the repository's own
    `ORDER BY priority_score DESC, created_at ASC` already produces the rank
    this handler numbers). Reported in the task summary.
    """
    entries = await WaitlistEntryRepository().list_by_priority(db, status=status)
    return WaitlistEntryListResponse(
        items=[
            WaitlistEntryRankedItem(
                id=entry.id,
                patient_id=entry.patient_id,
                priority_score=entry.priority_score,
                rank=index + 1,
                status=entry.status,
            )
            for index, entry in enumerate(entries)
        ]
    )


@router.post("/offers/{id}/respond", response_model=RespondToOfferResponse)
async def respond_to_offer(id: UUID, body: RespondToOfferRequest, db=Depends(get_db)) -> RespondToOfferResponse:
    """architecture.md §5.2 `POST /waitlist/offers/{id}/respond`: 200.

    Auth: none — this route declares no `Depends(require_role(...))`/
    `Depends(get_current_user)` at all (Behaviour; project_rules.auth's
    explicit unauthenticated list names this exact endpoint), reachable by
    the patient-facing channel webhook that carries the offer response.
    `WaitlistFillService.respond`'s own docstring documents this as the
    caller that invokes it directly and synchronously (Interaction
    contract).
    """
    service = _fill_service()
    updated = await service.respond(db, id, body.response)

    # `WaitlistOffer` (frozen model) carries no `appointment_id` column of
    # its own -- a successful accept sets it on the associated
    # `WaitlistEntry` instead (`WaitlistFillService.respond`'s accept
    # branch). This follow-up read-only lookup is presentation only (no
    # business logic), surfacing that value for `RespondToOfferResponse`.
    entry = await WaitlistEntryRepository().get_by_id(db, updated.waitlist_entry_id)
    appointment_id = entry.appointment_id if entry is not None else None

    return RespondToOfferResponse(id=updated.id, status=updated.status, appointment_id=appointment_id)


@router.get("/idle-chairs/{id}/recovery-status", response_model=RecoveryStatusResponse)
async def get_recovery_status(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> RecoveryStatusResponse:
    """architecture.md §5.2 `GET /waitlist/idle-chairs/{id}/recovery-status`:
    200; 404 (`NotFoundError`) if `id` does not exist — FR-E5.5/US27, the
    fill/cascade trail for one idle-chair alert.
    """
    alert = await IdleChairAlertRepository().get_by_id(db, id)
    if alert is None:
        raise NotFoundError("Idle chair alert not found.")

    offers = await WaitlistOfferRepository().list_by_alert(db, id)
    return RecoveryStatusResponse(
        alert_id=id,
        offers=[
            RecoveryOfferStatus(
                waitlist_entry_id=offer.waitlist_entry_id,
                status=offer.status,
                offered_at=_aware(offer.offered_at),
            )
            for offer in offers
        ],
        final_status=alert.status.value if hasattr(alert.status, "value") else alert.status,
    )
