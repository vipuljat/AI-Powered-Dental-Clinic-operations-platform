"""Thin HTTP adapter over ``RiskScoringService``/``TriageService``/
``EscalationService``/``UtilisationEngine`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema (or FastAPI's own required-query/path validation), delegates to the
service layer, and shapes the response. This module's ``router`` carries only
its own ``/intelligence`` sub-prefix; the shared ``/api/v1`` prefix is added
on top by ``app/main.py`` when it calls ``app.include_router`` (Interaction
contract).
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Query

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table is
# not already registered there — but `app.core.dependencies` transitively
# imports that waitlist repository (via `services.console.service`), and this
# module also needs a full `SchedulingService` (for `UtilisationEngine`'s one
# synchronous call into `scheduling`), which transitively imports the real,
# fully-columned `app.models.patients.models.Patient`. Whoever imports first
# controls the race: detect that specific id-only placeholder (never the
# genuine table) and drop it from `Base.metadata` first, then import the real
# patients repository/model before anything else pulls in the stub — this
# makes the fix hold regardless of import order (mirrors
# `app/routes/scheduling/routes.py`'s identical fix).
from app.core.database import Base as _Base  # isort: skip

_stub_patients_table = _Base.metadata.tables.get("patients")
if _stub_patients_table is not None and set(_stub_patients_table.columns.keys()) == {"id"}:
    _Base.metadata.remove(_stub_patients_table)

from app.repositories.patients.repository import RecordLockRepository  # isort: skip

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.core.messaging import get_broker
from app.repositories.intelligence.repository import (
    EscalationRepository,
    MlModelEvaluationRepository,
    RiskScoreRepository,
    TriageSessionRepository,
    UtilisationRecommendationRepository,
)
from app.repositories.rules.repository import RuleDefinitionRepository, RuleSetRepository
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.schemas.intelligence.schemas import (
    AcknowledgeEscalationResponse,
    AnswerTriageQuestionRequest,
    AnswerTriageQuestionResponse,
    ApplyRecommendationResponse,
    DismissRecommendationResponse,
    ResolveEscalationRequest,
    ResolveEscalationResponse,
    RiskAnalyticsResponse,
    RiskScoreResponse,
    StartTriageSessionRequest,
    StartTriageSessionResponse,
    TriageQuestion,
    UtilisationRecommendationItem,
    UtilisationRecommendationListResponse,
)
from app.common.exceptions.errors import NotFoundError
from app.services.console.service import AuditService
from app.services.intelligence.service import (
    EscalationService,
    RiskScoringService,
    TriageService,
    UtilisationEngine,
)
from app.services.patients.service import RecordLockService
from app.services.rules.service import RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService

router = APIRouter(prefix="/intelligence")


def _aware(value: "datetime | None") -> "datetime | None":
    """Same ``sqlite+aiosqlite:///:memory:`` test-pivot accommodation the
    service/repository layers make (project_rules.testing) — a
    ``DateTime(timezone=True)`` column round-trips as naive on SQLite even
    though it is always UTC by convention (architecture.md §8)."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _audit_service() -> AuditService:
    # Each request builds its own thin service/repository graph over the
    # request-scoped `AsyncSession` (`Depends(get_db)`) — no collaborator
    # here is a module-level singleton, matching the rest of this codebase's
    # `require_permission`/`AuthService` wiring (app/core/dependencies.py).
    from app.repositories.console.repository import AuditLogRepository

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


def _risk_scoring_service() -> RiskScoringService:
    return RiskScoringService(
        RiskScoreRepository(), AppointmentRepository(), MlModelEvaluationRepository(), _rule_set_service()
    )


def _escalation_service() -> EscalationService:
    settings = get_settings()
    return EscalationService(EscalationRepository(), _audit_service(), get_broker(settings))


def _triage_service() -> TriageService:
    settings = get_settings()
    return TriageService(
        TriageSessionRepository(), _rule_set_service(), _escalation_service(), get_broker(settings)
    )


def _utilisation_engine() -> UtilisationEngine:
    return UtilisationEngine(
        UtilisationRecommendationRepository(),
        RiskScoreRepository(),
        TriageSessionRepository(),
        AppointmentRepository(),
        _scheduling_service(),
        _audit_service(),
    )


def _to_triage_question(rule) -> "TriageQuestion | None":
    """Converts the active ``triage_question`` rule (``rule_key``/
    ``rule_value``) that ``TriageService.start_session``/``.answer`` hand
    back into this file's own ``TriageQuestion`` response shape (FR-E8.4/
    US43: ``options`` is always the rule's configured fixed-choice list)."""
    if rule is None:
        return None
    rule_value = rule.rule_value if isinstance(rule.rule_value, dict) else {}
    return TriageQuestion(key=rule.rule_key, options=list(rule_value.get("options", [])))


@router.get("/risk/appointments/{appointment_id}", response_model=RiskScoreResponse)
async def get_appointment_risk(
    appointment_id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> RiskScoreResponse:
    """architecture.md §5.2 `GET /intelligence/risk/appointments/{id}` — 200.

    Watch out: ``RiskScoringService`` (frozen) exposes only
    ``score_rules_based``/``score_ml``/``evaluate_precision`` — no method to
    *read back* an already-computed score for a given appointment. A direct,
    read-only call into ``RiskScoreRepository.get_latest_for_appointment`` is
    issued here instead (no business logic added, read-only), mirroring the
    same documented gap-handling already accepted in
    ``app/routes/outreach/routes.py``; reported again in this task's summary
    as a gap against the frozen service contract.
    """
    risk_repo = RiskScoreRepository()
    score = await risk_repo.get_latest_for_appointment(db, appointment_id)
    if score is None:
        raise NotFoundError("No risk score recorded for this appointment.")
    return RiskScoreResponse(
        risk_level=score.risk_level,
        score_value=score.score_value,
        source=score.source,
        model_version=score.model_version,
    )


@router.get("/risk/analytics", response_model=RiskAnalyticsResponse)
async def get_risk_analytics(
    from_: "date | None" = Query(None, alias="from"),
    to: "date | None" = None,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> RiskAnalyticsResponse:
    """architecture.md §5.2 `GET /intelligence/risk/analytics` — 200.

    Watch out: same gap as ``get_appointment_risk`` above — no
    ``RiskScoringService``/``MlModelEvaluation``-reading method exists on the
    frozen service layer, so ``RiskScoreRepository.get_distribution`` and
    ``MlModelEvaluationRepository.get_latest`` are read directly here.
    """
    date_from = datetime.combine(from_, time.min, tzinfo=timezone.utc) if from_ else None
    date_to = datetime.combine(to, time.max, tzinfo=timezone.utc) if to else None

    risk_repo = RiskScoreRepository()
    eval_repo = MlModelEvaluationRepository()
    distribution = await risk_repo.get_distribution(db, date_from, date_to)
    latest_evaluation = await eval_repo.get_latest(db)
    model_status = latest_evaluation.status if latest_evaluation is not None else None
    return RiskAnalyticsResponse(distribution=distribution, model_status=model_status)


@router.post("/triage/sessions", response_model=StartTriageSessionResponse, status_code=201)
async def start_triage_session(
    body: StartTriageSessionRequest, db=Depends(get_db)
) -> StartTriageSessionResponse:
    """architecture.md §5.2 `POST /intelligence/triage/sessions` — 201,
    `Auth: none` (project_rules.auth: patient/visitor-facing webchat/voice
    triage entry point, unauthenticated by design)."""
    service = _triage_service()
    result = await service.start_session(db, body.channel, body.patient_id)
    return StartTriageSessionResponse(
        id=result["id"], next_question=_to_triage_question(result["next_question"])
    )


@router.post("/triage/sessions/{id}/answer", response_model=AnswerTriageQuestionResponse)
async def answer_triage_question(
    id: UUID, body: AnswerTriageQuestionRequest, db=Depends(get_db)
) -> AnswerTriageQuestionResponse:
    """architecture.md §5.2 `POST /intelligence/triage/sessions/{id}/answer`
    — 200, `Auth: none` (project_rules.auth). FR-E8.4/FR-E8.5/US43-US44:
    ``TriageService.answer`` never populates ``recommended_block`` when the
    outcome is an escalation."""
    service = _triage_service()
    result = await service.answer(db, id, body.question_key, body.answer)
    return AnswerTriageQuestionResponse(
        escalated=result["escalated"],
        next_question=_to_triage_question(result["next_question"]),
        recommended_block=result["recommended_block"],
    )


@router.post("/escalations/{id}/acknowledge", response_model=AcknowledgeEscalationResponse)
async def acknowledge_escalation(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> AcknowledgeEscalationResponse:
    """architecture.md §5.2 `POST /intelligence/escalations/{id}/acknowledge`
    — 200. US44 Alternate Flow: claimed by whoever acknowledges first, via
    ``EscalationService.acknowledge``."""
    service = _escalation_service()
    escalation = await service.acknowledge(db, id, user.id)
    return AcknowledgeEscalationResponse(
        id=escalation.id,
        status=escalation.status,
        acknowledged_at=_aware(escalation.acknowledged_at),
    )


@router.post("/escalations/{id}/resolve", response_model=ResolveEscalationResponse)
async def resolve_escalation(
    id: UUID,
    body: ResolveEscalationRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> ResolveEscalationResponse:
    """architecture.md §5.2 `POST /intelligence/escalations/{id}/resolve` —
    200. US44 AC2/T110: ``EscalationService.resolve`` also records the audit
    entry (trigger reason, timestamp, resolving staff member)."""
    service = _escalation_service()
    resolved = await service.resolve(db, id, user.id, body.resolution_notes)
    return ResolveEscalationResponse(id=resolved.id, status=resolved.status)


@router.get("/utilisation/recommendations", response_model=UtilisationRecommendationListResponse)
async def list_utilisation_recommendations(
    status: str = "pending",
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> UtilisationRecommendationListResponse:
    """architecture.md §5.2 `GET /intelligence/utilisation/recommendations`
    — 200.

    Watch out: ``UtilisationEngine`` (frozen) exposes ``generate``/
    ``revalidate``/``apply``/``dismiss``/``get_adoption_rate`` but no listing
    method — a direct, read-only call into
    ``UtilisationRecommendationRepository.list_by_status`` is issued here
    instead (no business logic added), the same class of documented gap as
    the risk-read endpoints above; reported in this task's summary.
    """
    rec_repo = UtilisationRecommendationRepository()
    recommendations = await rec_repo.list_by_status(db, status)
    return UtilisationRecommendationListResponse(
        items=[
            UtilisationRecommendationItem(
                id=recommendation.id,
                rationale=recommendation.rationale,
                recommended_change=recommendation.recommended_change,
            )
            for recommendation in recommendations
        ]
    )


@router.post("/utilisation/recommendations/{id}/apply", response_model=ApplyRecommendationResponse)
async def apply_utilisation_recommendation(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> ApplyRecommendationResponse:
    """architecture.md §5.2 `POST .../apply` — 200 on success, 409
    (``RecommendationStaleError``, via the global handler) when
    ``UtilisationEngine.revalidate`` finds the proposed slot no longer
    conflict-free (US45 AC2/Alternate Flow)."""
    service = _utilisation_engine()
    recommendation = await service.apply(db, id, user.id)
    return ApplyRecommendationResponse(id=recommendation.id, status=recommendation.status)


@router.post("/utilisation/recommendations/{id}/dismiss", response_model=DismissRecommendationResponse)
async def dismiss_utilisation_recommendation(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> DismissRecommendationResponse:
    """architecture.md §5.2 `POST .../dismiss` — 200 (T112 AC/FR-E8.7)."""
    service = _utilisation_engine()
    recommendation = await service.dismiss(db, id, user.id)
    return DismissRecommendationResponse(id=recommendation.id, status=recommendation.status)
