"""Thin HTTP adapter over ``RecallScanService``/``RecallCampaignService``/
``TreatmentRecoveryService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema, delegates to the service layer, and shapes the response. This
module's ``router`` carries only its own ``/recall`` sub-prefix; the shared
``/api/v1`` prefix is added on top by ``app/main.py`` when it calls
``app.include_router`` (Interaction contract).

Watch out: this file's own Public surface names no endpoint that calls
``RecallScanService`` — that class's only method, ``run()``, is the
weekly-scan *mutator* (`app/workers/recall_scanner.py`'s entry point per
that service's own docstring); no read method exists on it (or on
``RecallScheduleRepository``) for "list the current queue by status", and
running the scan itself as a side effect of a `GET` would be wrong. `GET
/recall/queue` instead issues its own read-only query (see that handler's
docstring) — `RecallScanService` is not constructed anywhere in this file,
a documented gap against this file's Responsibility line, reported in the
task summary.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table is
# not already registered there -- but `app.core.dependencies` (imported
# below) transitively imports that stub-registering waitlist repository (via
# `services.console.service`) before this module otherwise would import the
# real `app.models.patients.models` (needed here for `RecordLockRepository`,
# a `SchedulingService` collaborator -- see `_scheduling_service()` below).
# Whoever imports this module first controls the race, so: detect that
# specific id-only placeholder (never the genuine, fully-columned `patients`
# table) and drop it from `Base.metadata` first, then import the real
# repository/model -- this makes the fix hold regardless of import order
# (mirrors `app/routes/scheduling/routes.py` and `app/routes/waitlist/
# routes.py`'s own identical fix).
from app.core.database import Base as _Base  # isort: skip

_stub_patients_table = _Base.metadata.tables.get("patients")
if _stub_patients_table is not None and set(_stub_patients_table.columns.keys()) == {"id"}:
    _Base.metadata.remove(_stub_patients_table)

from app.repositories.patients.repository import RecordLockRepository  # isort: skip

from app.common.exceptions.errors import NotFoundError
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.core.messaging import get_broker
from app.models.recall.models import RecallSchedule
from app.repositories.console.repository import AuditLogRepository
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    MessageTemplateRepository,
    OutreachMessageRepository,
)
from app.repositories.recall.repository import (
    RecallComplianceBaselineRepository,
    RecallScheduleRepository,
    UnscheduledTreatmentRepository,
)
from app.repositories.rules.repository import (
    ConfigurationParameterRepository,
    RuleDefinitionRepository,
    RuleSetRepository,
)
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.schemas.recall.schemas import (
    ConvertTreatmentRequest,
    ConvertTreatmentResponse,
    RecallComplianceResponse,
    RecallQueueItem,
    RecallQueueResponse,
    TriggerCampaignResponse,
    TriggerRecallCampaignRequest,
    TriggerTreatmentCampaignRequest,
    UnscheduledTreatmentItem,
    UnscheduledTreatmentListResponse,
    UpdateValuationRequest,
)
from app.services.console.service import AuditService
from app.services.outreach.service import OutreachService
from app.services.patients.service import RecordLockService
from app.services.recall.service import RecallCampaignService, TreatmentRecoveryService
from app.services.rules.service import ConfigurationService, RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService

router = APIRouter(prefix="/recall")


# --- Per-request service/repository graph construction ---------------------
# Each request builds its own thin service/repository graph over the
# request-scoped `AsyncSession` (`Depends(get_db)`) -- no collaborator here
# is a module-level singleton, matching the rest of this codebase's
# `require_permission`/`AuthService` wiring (app/core/dependencies.py) and
# `routes/outreach/routes.py`'s own factory-function pattern.


def _audit_service() -> AuditService:
    return AuditService(AuditLogRepository())


def _recall_schedule_repo() -> RecallScheduleRepository:
    return RecallScheduleRepository()


def _treatment_repo() -> UnscheduledTreatmentRepository:
    return UnscheduledTreatmentRepository()


def _baseline_repo() -> RecallComplianceBaselineRepository:
    return RecallComplianceBaselineRepository()


def _configuration_service() -> ConfigurationService:
    return ConfigurationService(ConfigurationParameterRepository(), _audit_service())


def _outreach_service() -> OutreachService:
    return OutreachService(
        OutreachMessageRepository(),
        ConsentRepository(),
        MessageTemplateRepository(),
        ChannelConfigurationRepository(),
        _configuration_service(),
        _audit_service(),
        get_settings(),
    )


def _campaign_service() -> RecallCampaignService:
    return RecallCampaignService(_recall_schedule_repo(), _baseline_repo(), _outreach_service(), _audit_service())


def _treatment_service() -> TreatmentRecoveryService:
    return TreatmentRecoveryService(_treatment_repo(), _outreach_service(), _audit_service())


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


@router.get("/queue", response_model=RecallQueueResponse)
async def get_recall_queue(
    status: str = "overdue",
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> RecallQueueResponse:
    """architecture.md §5.2 `GET /recall/queue`: 200.

    Watch out: `RecallScanService` (frozen) exposes only `run()` (US34's
    weekly-scan mutator) -- no read/list method covers "the current queue
    filtered by status". A direct, read-only `select()` against the frozen
    `RecallSchedule` ORM model is issued here instead (mirroring
    `routes/outreach/routes.py`'s own documented gap for the same reason:
    `GET /outreach/messages` without a `patient_id`), since no service/
    repository method exists to call for this case. This is a documented
    gap against the frozen `services/recall/service.py` contract (no
    business logic added -- filtering + presentation only), reported in the
    task summary. `overdue_days` is derived from today's date minus
    `due_date`, floored at zero for a not-yet-due row that a caller queries
    under a non-default `status`.
    """
    today = date.today()
    result = await db.execute(select(RecallSchedule).where(RecallSchedule.status == status))
    schedules = list(result.scalars().all())
    return RecallQueueResponse(
        items=[
            RecallQueueItem(
                patient_id=schedule.patient_id,
                risk_classification=schedule.risk_classification,
                due_date=schedule.due_date,
                overdue_days=max((today - schedule.due_date).days, 0),
                status=schedule.status,
            )
            for schedule in schedules
        ]
    )


@router.post("/campaigns", response_model=TriggerCampaignResponse, status_code=202)
async def trigger_recall_campaign(
    body: TriggerRecallCampaignRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> TriggerCampaignResponse:
    """architecture.md §5.2 `POST /recall/campaigns`: 202. FR-E7.3 (US35):
    dispatches one outreach message per `recall_schedule_id` via
    `RecallCampaignService.dispatch`; accepted (202) since dispatch fans out
    to the async outreach channel gateway per schedule.
    """
    service = _campaign_service()
    result = await service.dispatch(db, body.recall_schedule_ids, user.id)
    return TriggerCampaignResponse(**result)


@router.get("/compliance", response_model=RecallComplianceResponse)
async def get_recall_compliance(
    period: str = "weekly",
    user: CurrentUser = Depends(require_role("clinic_management", "front_office_staff")),
    db=Depends(get_db),
) -> RecallComplianceResponse:
    """architecture.md §5.2 `GET /recall/compliance`: 200. FR-E7.4 (US36):
    `RecallCampaignService.get_compliance` is itself stateless over the
    *current* ISO week regardless of its own `period` parameter (its own
    docstring: `period="weekly"` documents the calling job's cadence, not a
    value it branches on) -- this endpoint's `period` query parameter is
    accepted (per this endpoint's frozen signature) and forwarded unchanged
    for symmetry with that method's own signature, without altering which
    week is reported.
    """
    service = _campaign_service()
    result = await service.get_compliance(db, period)
    return RecallComplianceResponse(**result)


@router.get("/treatments", response_model=UnscheduledTreatmentListResponse)
async def list_unscheduled_treatments(
    sort: str = "valuation_desc",
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> UnscheduledTreatmentListResponse:
    """architecture.md §5.2 `GET /recall/treatments`: 200. FR-E7.5 (US37):
    `TreatmentRecoveryService.scan` always returns the `"unscheduled"` queue
    ordered by valuation descending with nulls last -- the only ordering its
    frozen `UnscheduledTreatmentRepository.list_by_valuation` call supports.
    `sort` is accepted (per this endpoint's frozen signature) but has no
    alternate ordering to switch to without editing that frozen repository
    method; a value other than the default is a documented no-op, reported
    in the task summary. `days_unscheduled` is derived from today's date
    minus each row's own `accepted_at` date.
    """
    service = _treatment_service()
    treatments = await service.scan(db)
    today = date.today()
    return UnscheduledTreatmentListResponse(
        items=[
            UnscheduledTreatmentItem(
                id=treatment.id,
                patient_id=treatment.patient_id,
                treatment_code=treatment.treatment_code,
                valuation_amount=treatment.valuation_amount,
                days_unscheduled=(today - treatment.accepted_at.date()).days,
                status=treatment.status,
            )
            for treatment in treatments
        ]
    )


@router.patch("/treatments/{id}", response_model=UnscheduledTreatmentItem)
async def update_treatment_valuation(
    id: "UUID",
    body: UpdateValuationRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> UnscheduledTreatmentItem:
    """architecture.md §5.2 `PATCH /recall/treatments/{id}`: 200; 404
    (`NotFoundError`) if `id` does not exist.

    Watch out: no `TreatmentRecoveryService` method covers updating
    `valuation_amount` (that class's own Public surface only documents
    `scan`/`dispatch_campaign`/`decline`/`convert`) -- this route calls
    `UnscheduledTreatmentRepository.update_valuation` directly, a documented
    gap against the frozen service contract (no business logic added),
    reported in the task summary. Existence is checked explicitly first
    (`get_by_id`) since that repository method itself resolves via
    `scalar_one` (an unhandled `NoResultFound` rather than the domain
    `NotFoundError` the registered global handler maps to 404).
    """
    repo = _treatment_repo()
    treatment = await repo.get_by_id(db, id)
    if treatment is None:
        raise NotFoundError("Unscheduled treatment not found.")
    updated = await repo.update_valuation(db, id, body.valuation_amount)
    return UnscheduledTreatmentItem(
        id=updated.id,
        patient_id=updated.patient_id,
        treatment_code=updated.treatment_code,
        valuation_amount=updated.valuation_amount,
        days_unscheduled=(date.today() - updated.accepted_at.date()).days,
        status=updated.status,
    )


@router.post("/treatment-campaigns", response_model=TriggerCampaignResponse, status_code=202)
async def trigger_treatment_campaign(
    body: TriggerTreatmentCampaignRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> TriggerCampaignResponse:
    """architecture.md §5.2 `POST /recall/treatment-campaigns`: 202. FR-E7.6
    (US38): dispatches a treatment re-engagement campaign via
    `TreatmentRecoveryService.dispatch_campaign`; accepted (202) since
    dispatch fans out to the async outreach channel gateway per treatment.
    """
    service = _treatment_service()
    result = await service.dispatch_campaign(db, body.unscheduled_treatment_ids, user.id)
    return TriggerCampaignResponse(**result)


@router.post("/treatments/{id}/convert", response_model=ConvertTreatmentResponse)
async def convert_treatment(
    id: "UUID",
    body: ConvertTreatmentRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> ConvertTreatmentResponse:
    """architecture.md §5.2 `POST /recall/treatments/{id}/convert`: 200; 404
    (`NotFoundError`) if `body.appointment_id` does not resolve to a real
    appointment.

    FR-E7.7 (US39): `services/recall/service.py`'s own Interaction contract
    documents the caller (this route) validating `appointment_id` via
    `SchedulingService.get_by_id` before calling `TreatmentRecoveryService.
    convert` -- honoured here literally: `SchedulingService.get_by_id` itself
    raises `NotFoundError` (mapped to 404 by the registered global handler)
    when `body.appointment_id` does not resolve to a real appointment, so no
    separate existence check is performed in this route body.
    """
    await _scheduling_service().get_by_id(db, body.appointment_id)

    service = _treatment_service()
    updated = await service.convert(db, id, body.appointment_id, body.partial, user.id)
    return ConvertTreatmentResponse(id=updated.id, status=updated.status, appointment_id=updated.appointment_id)
