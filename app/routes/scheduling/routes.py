"""Thin HTTP adapter over ``SchedulingService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema (or FastAPI's own required-header/query validation), delegates to the
service layer, and shapes the response. This module's ``router`` carries only
its own ``/scheduling`` sub-prefix; the shared ``/api/v1`` prefix is added on
top by ``app/main.py`` when it calls ``app.include_router`` (Interaction
contract).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, UploadFile

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table
# is not already registered there — but `app.core.dependencies` transitively
# imports that waitlist repository (via `services.console.service`) before
# this module otherwise would import the real `app.models.patients.models`.
# Whoever imports this module first controls the race: if a caller (e.g. a
# test module) already imported `app.core.dependencies` on its own, *before*
# ever importing this file, that stub "patients" Table is already sitting on
# `Base.metadata` by the time we get here, and a plain re-ordering of our own
# imports can no longer help — declaring the real `Patient` ORM class below
# would otherwise raise `InvalidRequestError: Table 'patients' is already
# defined for this MetaData instance`. So: detect that specific id-only
# placeholder (never the genuine, fully-columned `patients` table) and drop
# it from `Base.metadata` first, then import the real repository/model —
# this makes the fix hold regardless of import order, unlike a bare
# reordering (mirrors `app/routes/patients/routes.py`'s import-order fix,
# hardened here against the caller-imports-dependencies-first case).
from app.core.database import Base as _Base  # isort: skip

_stub_patients_table = _Base.metadata.tables.get("patients")
if _stub_patients_table is not None and set(_stub_patients_table.columns.keys()) == {"id"}:
    _Base.metadata.remove(_stub_patients_table)

from app.repositories.patients.repository import RecordLockRepository  # isort: skip

from app.common.exceptions.errors import ImportValidationError
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.core.messaging import get_broker
from app.repositories.console.repository import AuditLogRepository
from app.repositories.rules.repository import RuleDefinitionRepository, RuleSetRepository
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.schemas.scheduling.schemas import (
    AppointmentDetailResponse,
    AppointmentResponse,
    CancelAppointmentRequest,
    CancelAppointmentResponse,
    CreateAppointmentRequest,
    LockResponse,
    ProposeSlotsResponse,
    RescheduleAppointmentRequest,
    RescheduleAppointmentResponse,
)
from app.services.console.service import AuditService
from app.services.patients.service import RecordLockService
from app.services.rules.service import RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService

router = APIRouter(prefix="/scheduling")


def _audit_service() -> AuditService:
    # Each request builds its own thin service/repository graph over the
    # request-scoped `AsyncSession` (`Depends(get_db)`) — no collaborator
    # here is a module-level singleton, matching the rest of this codebase's
    # `require_permission`/`AuthService` wiring (app/core/dependencies.py).
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


@router.get("/slots", response_model=ProposeSlotsResponse)
async def propose_slots(
    patient_id: UUID,
    provider_id: "UUID | None" = None,
    from_: datetime = Query(alias="from"),
    to: datetime = Query(),
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> ProposeSlotsResponse:
    """FR-E4.1/US18: 200 — rule-compliant, conflict-free candidate slots plus
    nearest alternatives when the requested window has none (architecture.md
    §5.2).
    """
    service = _scheduling_service()
    result = await service.propose_slots(db, patient_id, provider_id, from_, to)
    return ProposeSlotsResponse(**result)


@router.post("/appointments", response_model=AppointmentResponse, status_code=201)
async def create_appointment(
    body: CreateAppointmentRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> AppointmentResponse:
    """RMR001/FR-E4.2/US18: 201 on create or idempotency-key replay, 409
    (``ScheduleConflictError``) on a genuinely conflicting slot
    (architecture.md §5.2: "POST /scheduling/appointments 201/409"). A
    missing ``Idempotency-Key`` header itself never reaches this body —
    FastAPI's own required-header validation turns that into a 422 before
    this handler runs (this file's spec Behaviour note).

    Watch out: the "Public surface" pseudocode for this handler shows
    ``@router.post("", ...)``, but this file's own Behaviour/Watch out
    prose spells the mounted path out in full as
    ``POST /scheduling/appointments`` (and ``GET /scheduling/appointments/{id}``
    for the sibling read below) — the literal decorator path here follows
    that full, textually-explicit architecture.md path rather than the
    seemingly-abbreviated code sample, so every appointment sub-resource
    below is nested under ``/appointments`` to match.
    """
    service = _scheduling_service()
    appointment, _replayed = await service.create(db, body.model_dump(), user.id, idempotency_key)
    return AppointmentResponse(
        id=appointment.id,
        status=appointment.status,
        risk_flag=appointment.risk_flag,
        confirmation_status=appointment.confirmation_status,
    )


@router.patch("/appointments/{id}/reschedule", response_model=RescheduleAppointmentResponse)
async def reschedule_appointment(
    id: UUID,
    body: RescheduleAppointmentRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> RescheduleAppointmentResponse:
    """FR-E4.3/FR-E4.4/US19: 200 on success, 409 (``ScheduleConflictError``)
    if the new slot conflicts (architecture.md §5.2).
    """
    service = _scheduling_service()
    appointment = await service.reschedule(db, id, body.model_dump(), user.id)
    return RescheduleAppointmentResponse(
        id=appointment.id,
        status=appointment.status,
        source_metadata=appointment.source_metadata or {},
    )


@router.post("/appointments/{id}/cancel", response_model=CancelAppointmentResponse)
async def cancel_appointment(
    id: UUID,
    body: CancelAppointmentRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> CancelAppointmentResponse:
    """FR-E4.5/US20: 200 on success, 422 (``MissingReasonCodeError``) if
    ``reason_code`` is blank (architecture.md §5.2).
    """
    service = _scheduling_service()
    appointment = await service.cancel(db, id, body.reason_code, user.id)
    return CancelAppointmentResponse(
        id=appointment.id,
        status=appointment.status,
        is_late_cancellation=appointment.is_late_cancellation,
    )


@router.get("/appointments/{id}", response_model=AppointmentDetailResponse)
async def get_appointment(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> AppointmentDetailResponse:
    """architecture.md §5.2: 200; 404 (``NotFoundError``) propagates unchanged
    if ``id`` does not exist.
    """
    service = _scheduling_service()
    appointment = await service.get_by_id(db, id)
    return AppointmentDetailResponse(
        id=appointment.id,
        patient_id=appointment.patient_id,
        status=appointment.status,
        risk_flag=appointment.risk_flag,
        risk_score_source=appointment.risk_score_source,
        confirmation_status=appointment.confirmation_status,
    )


@router.post("/appointments/{id}/lock", response_model=LockResponse)
async def lock_appointment(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> LockResponse:
    """FR-E4.6/US22: 200 on success, 409 (``RecordLockedError``) if another
    staff member currently holds this record's lock (architecture.md §5.2).
    """
    service = _scheduling_service()
    lock = await service.lock(db, id, user.id)
    return LockResponse(locked_by=str(lock.locked_by_staff_id), expires_at=lock.expires_at)


@router.post("/appointments/import", status_code=201)
async def import_appointments(
    file: UploadFile,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> dict:
    """US12/A10: 201 when the whole batch committed, 422 when it was
    rejected (architecture.md §5.2).

    Watch out: mounted at ``/scheduling/appointments/import`` (not
    ``/scheduling/import``) — since ``router`` already carries the
    ``/scheduling`` sub-prefix, this handler is declared at
    ``/appointments/import``.
    """
    service = _scheduling_service()
    file_bytes = await file.read()
    result = await service.import_csv(db, file_bytes, file.filename or "upload.csv", user.id)

    if result["status"] == "rejected":
        raise ImportValidationError(
            message=f"Import batch rejected: {result['error_count']} row(s) failed validation.",
            details=result["errors"],
        )

    return result
