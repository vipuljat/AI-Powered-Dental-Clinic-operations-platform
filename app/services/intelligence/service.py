"""Business logic for the ``intelligence`` module.

``RiskScoringService`` (FR-E8.1-FR-E8.3/US40-US41) computes and records a
per-appointment no-show risk flag, first via a rules-based heuristic and,
once enough historic data exists, via a (stub) ML scorer that is
transparently labelled and monthly-evaluated for precision.
``TriageService`` (FR-E8.4-FR-E8.5/US43-US44) runs the guided
webchat/voice symptom-triage question flow and classifies its outcome,
checking escalation triggers before ever computing a recommendation.
``EscalationService`` (FR-E8.5/SFR001/US44) is the one write path onto
``escalations`` and the one publisher of ``escalation.triggered``.
``UtilisationEngine`` (FR-E8.6-FR-E8.7/US45) generates, tracks and applies
schedule-utilisation recommendations.

``UtilisationEngine.apply`` is the one synchronous, one-directional call
this module makes into ``scheduling`` (``SchedulingService.reschedule``) —
``scheduling`` never calls back into ``intelligence``, per
``open_questions[Q-8a5b7a15]``: the other conceivable intelligence<->
scheduling edge (risk-scoring-on-booking) is event-driven instead, via
``app/workers/recommendation_engine.py`` reacting to ``booking.confirmed``,
never a synchronous call from this file into scheduling.

Watch out: no actual ML training/inference pipeline exists anywhere in this
codebase. ``RiskScoringService.score_ml``/``evaluate_precision`` implement
only the gating, transparency and evaluation-logging logic the BRD
requires — any concrete score they produce is a placeholder heuristic
(reusing the same computation ``score_rules_based`` uses, tagged with a
distinct ``model_version``/``source="ml"``) until a real trained model is
integrated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from app.common.constants import ML_MIN_HISTORIC_DATA_YEARS, ML_TARGET_PRECISION
from app.common.enums import (
    ActorType,
    AppointmentStatus,
    EscalationStatus,
    MlEvaluationStatus,
    RiskLevel,
    RuleCategory,
    ScoreSource,
    UtilisationRecommendationStatus,
)
from app.common.exceptions.errors import NotFoundError, RecommendationStaleError
from app.common.utils import now_utc, to_iso8601
from app.repositories.intelligence.repository import (
    EscalationRepository,
    MlModelEvaluationRepository,
    RiskScoreRepository,
    TriageSessionRepository,
    UtilisationRecommendationRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.messaging import MessageBroker
    from app.models.intelligence.models import (
        Escalation,
        MlModelEvaluation,
        RiskScore,
        UtilisationRecommendation,
    )
    from app.repositories.scheduling.repository import AppointmentRepository
    from app.services.console.service import AuditService
    from app.services.rules.service import RuleSetService
    from app.services.scheduling.service import SchedulingService

__all__ = ["EscalationService", "RiskScoringService", "TriageService", "UtilisationEngine"]


def _aware(value: "datetime | None") -> "datetime | None":
    """Normalize a datetime read back from the DB to tz-aware UTC.

    Same accommodation ``services/scheduling/service.py``/
    ``services/console/service.py`` make for the ``sqlite+aiosqlite:///:memory:``
    test pivot (project_rules.testing), which does not retain a
    ``DateTime(timezone=True)`` column's UTC offset the way Postgres does.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _coerce_datetime(value: "datetime | str") -> datetime:
    """``UtilisationRecommendation.recommended_change`` is stored as a plain
    JSONB dict, so once round-tripped through Postgres/SQLite it always
    holds ISO-8601 strings (the shape ``UtilisationEngine.generate`` writes
    via ``to_iso8601``) — but the same in-process value is also handed
    straight to ``revalidate``/``apply`` as a real ``datetime`` object
    within a single request, so both shapes are accepted here.
    """
    if isinstance(value, datetime):
        return _aware(value)
    return _aware(datetime.fromisoformat(value))


def _coerce_uuid(value: "UUID | str") -> UUID:
    """Same accommodation as ``_coerce_datetime`` for a JSONB-stored id that
    may already be a ``UUID`` in-process or a ``str`` once round-tripped."""
    return value if isinstance(value, UUID) else UUID(str(value))


# --- RiskScoringService inferred defaults -----------------------------------
# FR-E8.1/US40: the BRD never fixes the exact `rule_value` JSON schema a
# `risk_classification` rule carries, so these keys/fallbacks are the
# documented, conventional inference (mirrors `_first_rule_value`'s use in
# `services/scheduling/service.py` for the same "rule may or may not carry
# this numeric knob" situation) — used only when no active rule overrides
# them (open question, exact thresholds not specified).
_DEFAULT_NO_SHOW_THRESHOLD_HIGH = 2
_DEFAULT_NO_SHOW_THRESHOLD_MEDIUM = 1
_DEFAULT_LEAD_TIME_HOURS_SHORT = 24
_DEFAULT_LEAD_TIME_HOURS_MEDIUM = 72
_NEUTRAL_SCORE_VALUE = 0.5  # US40 Alternate Flow: no-history default.
_SCORE_VALUE_HIGH = 0.8
_SCORE_VALUE_MEDIUM = 0.5
_SCORE_VALUE_LOW = 0.2
# Placeholder model tag — see this file's module docstring Watch out: no real
# trained model exists behind this version string.
_ML_MODEL_VERSION = "ml-stub-v1"
_ML_EVALUABLE_STATUSES = (AppointmentStatus.completed, AppointmentStatus.no_show)
_ML_USABLE_HISTORY_STATUSES = (
    AppointmentStatus.completed,
    AppointmentStatus.cancelled,
    AppointmentStatus.no_show,
)


class RiskScoringService:
    """FR-E8.1-FR-E8.3/US40-US41: rules-based (always available) and ML
    (gated) no-show risk scoring, plus the monthly ML precision evaluation.
    """

    def __init__(
        self,
        risk_repo: RiskScoreRepository,
        appt_repo: "AppointmentRepository",
        eval_repo: MlModelEvaluationRepository,
        rule_set_service: "RuleSetService",
    ) -> None:
        self._risk_repo = risk_repo
        self._appt_repo = appt_repo
        self._eval_repo = eval_repo
        self._rule_set_service = rule_set_service

    @staticmethod
    def _first_rule_value(rules: list, key: str) -> object:
        for rule in rules:
            value = getattr(rule, "rule_value", None)
            if isinstance(value, dict) and value.get(key) is not None:
                return value[key]
        return None

    def _classify_risk(
        self, rules: list, no_show_count: int, lead_time_hours: float
    ) -> "tuple[RiskLevel, float]":
        high_threshold = self._first_rule_value(rules, "no_show_count_threshold_high")
        high_threshold = _DEFAULT_NO_SHOW_THRESHOLD_HIGH if high_threshold is None else high_threshold
        medium_threshold = self._first_rule_value(rules, "no_show_count_threshold_medium")
        medium_threshold = (
            _DEFAULT_NO_SHOW_THRESHOLD_MEDIUM if medium_threshold is None else medium_threshold
        )
        lead_time_short = self._first_rule_value(rules, "lead_time_hours_short")
        lead_time_short = _DEFAULT_LEAD_TIME_HOURS_SHORT if lead_time_short is None else lead_time_short
        lead_time_medium = self._first_rule_value(rules, "lead_time_hours_medium")
        lead_time_medium = _DEFAULT_LEAD_TIME_HOURS_MEDIUM if lead_time_medium is None else lead_time_medium

        if no_show_count >= high_threshold or lead_time_hours <= lead_time_short:
            return RiskLevel.high, _SCORE_VALUE_HIGH
        if no_show_count >= medium_threshold or lead_time_hours <= lead_time_medium:
            return RiskLevel.medium, _SCORE_VALUE_MEDIUM
        return RiskLevel.low, _SCORE_VALUE_LOW

    async def _score(
        self, db: "AsyncSession", appointment_id: UUID
    ) -> "tuple[object, RiskLevel, float]":
        """Shared prior-history + rule-threshold scoring logic used by both
        ``score_rules_based`` and (as a stand-in for a real model —see this
        file's Watch out) ``score_ml``.
        """
        appointment = await self._appt_repo.get_by_id(db, appointment_id)
        if appointment is None:
            raise NotFoundError("Appointment not found.")

        history = await self._appt_repo.list_by_patient(db, appointment.patient_id)
        prior = [a for a in history if a.id != appointment_id]

        if not prior:
            # US40 Alternate Flow: a patient with no prior history gets a
            # default/neutral score rather than an unscored appointment.
            return appointment, RiskLevel.medium, _NEUTRAL_SCORE_VALUE

        rules = await self._rule_set_service.get_active_rules_by_category(
            db, RuleCategory.risk_classification.value
        )
        no_show_count = sum(1 for a in prior if a.status == AppointmentStatus.no_show)
        scheduled_start = _aware(appointment.scheduled_start)
        lead_time_hours = (scheduled_start - now_utc()).total_seconds() / 3600
        risk_level, score_value = self._classify_risk(rules, no_show_count, lead_time_hours)
        return appointment, risk_level, score_value

    async def score_rules_based(self, db: "AsyncSession", appointment_id: UUID) -> "RiskScore":
        """FR-E8.1/US40: assigns a risk flag to every scheduled appointment
        via prior no-show count + lead-time heuristics read from the active
        ``risk_classification`` rule category; a patient with no prior
        history gets ``score_value=0.5``/``risk_level="medium"`` (US40
        Alternate Flow). FR-E8.3/US41/T102: always writes
        ``appointments.risk_flag``/``risk_score_source`` alongside the
        ``risk_scores`` row so staff-facing views can label the source.
        """
        appointment, risk_level, score_value = await self._score(db, appointment_id)

        risk_score = await self._risk_repo.create(
            db,
            appointment_id=appointment_id,
            score_value=score_value,
            risk_level=risk_level,
            source=ScoreSource.rules,
            model_version=None,
            computed_at=now_utc(),
        )
        await self._appt_repo.update_fields(
            db, appointment_id, risk_flag=risk_level, risk_score_source=ScoreSource.rules
        )
        return risk_score

    async def score_ml(self, db: "AsyncSession", appointment_id: UUID) -> "RiskScore | None":
        """FR-E8.2/US41: documented no-op (returns ``None``, leaving the
        rules-based flag active) unless ``_is_ml_eligible`` passes —
        "data insufficient at evaluation time: rules-based scoring remains
        the active method" (US41 Alternate Flow). When eligible, writes
        ``source="ml"``, overwriting ``appointments.risk_flag``/
        ``risk_score_source`` (FR-E8.3/US41/T102 transparency).
        """
        if not await self._is_ml_eligible(db):
            return None

        _appointment, risk_level, score_value = await self._score(db, appointment_id)

        risk_score = await self._risk_repo.create(
            db,
            appointment_id=appointment_id,
            score_value=score_value,
            risk_level=risk_level,
            source=ScoreSource.ml,
            model_version=_ML_MODEL_VERSION,
            computed_at=now_utc(),
        )
        await self._appt_repo.update_fields(
            db, appointment_id, risk_flag=risk_level, risk_score_source=ScoreSource.ml
        )
        return risk_score

    async def _is_ml_eligible(self, db: "AsyncSession") -> bool:
        """AC002/DEP4: ``True`` only once >= ``ML_MIN_HISTORIC_DATA_YEARS``
        of usable appointment history exists.

        Documented data-sufficiency proxy (open question, exact criteria not
        specified): "usable" history is read as every already risk-scored
        (``source in {"rules", "ml"}``) appointment whose status is
        ``completed``/``cancelled``/``no_show`` — the frozen
        ``AppointmentRepository`` exposes no clinic-wide "every appointment"
        query, so ``list_by_risk_source`` (the one appointment-wide read this
        repository *does* expose) is reused here as the closest available
        proxy for "the system's accumulated appointment history"; eligible
        once the earliest-to-latest span of that history is at least
        ``ML_MIN_HISTORIC_DATA_YEARS`` years.
        """
        usable: list = []
        for source in (ScoreSource.rules.value, ScoreSource.ml.value):
            scored = await self._appt_repo.list_by_risk_source(db, source)
            usable.extend(a for a in scored if a.status in _ML_USABLE_HISTORY_STATUSES)

        if not usable:
            return False

        starts = [_aware(a.scheduled_start) for a in usable]
        span_years = (max(starts) - min(starts)).days / 365.25
        return span_years >= ML_MIN_HISTORIC_DATA_YEARS

    async def evaluate_precision(self, db: "AsyncSession") -> "MlModelEvaluation":
        """FR-E8.2/US42: monthly job computing precision on held-out data
        (out of this codebase's scope to actually train/evaluate a model —
        see this file's Watch out). ``status="passing"`` if
        ``precision_value >= ML_TARGET_PRECISION`` else ``"underperforming"``;
        an ``"underperforming"`` result does NOT disable ``score_ml``
        automatically in code — it only records the flag, since the
        rules-based fallback (``score_rules_based``) stays independently
        callable regardless (US42 AC2's "rules-based fallback remains
        active").
        """
        ml_scored = await self._appt_repo.list_by_risk_source(db, ScoreSource.ml.value)
        evaluable = [a for a in ml_scored if a.status in _ML_EVALUABLE_STATUSES]

        if evaluable:
            predicted_positive = sum(1 for a in evaluable if a.risk_flag == RiskLevel.high)
            true_positive = sum(
                1
                for a in evaluable
                if a.risk_flag == RiskLevel.high and a.status == AppointmentStatus.no_show
            )
            precision_value = (
                true_positive / predicted_positive if predicted_positive else ML_TARGET_PRECISION
            )
            # `scheduled_start` is a required column on every real
            # `Appointment` row, but is read defensively here (rather than
            # assumed present) since this evaluation-logging computation is
            # itself only a documented stand-in (see this file's Watch out)
            # for a real held-out-data evaluation.
            starts = [
                _aware(s) for s in (getattr(a, "scheduled_start", None) for a in evaluable) if s is not None
            ]
            data_volume_months = max(1, int((max(starts) - min(starts)).days / 30)) if len(starts) >= 2 else 1
        else:
            # No ML-scored, resolved-outcome appointments yet to evaluate
            # against — reported at the target threshold itself rather than
            # an arbitrary pass/fail default.
            precision_value = ML_TARGET_PRECISION
            data_volume_months = 0

        status = (
            MlEvaluationStatus.passing
            if precision_value >= ML_TARGET_PRECISION
            else MlEvaluationStatus.underperforming
        )

        return await self._eval_repo.create(
            db,
            model_version=_ML_MODEL_VERSION,
            precision_value=precision_value,
            data_volume_months=data_volume_months,
            status=status,
        )


# --- TriageService inferred rule_value schema -------------------------------
# FR-E8.4/US43: the BRD never fixes the exact `rule_value` JSON schema a
# `triage_question`/`escalation_trigger` rule carries. The convention used
# here (documented, open question):
#   triage_question:    rule_key = the question's own `key` (matches
#                        `schemas/intelligence/schemas.py`'s `TriageQuestion.key`);
#                        rule_value = {"options": [...], "order": int,
#                        "urgency_map": {answer: "low"|"medium"|"high"},
#                        "recommended_block_map": {answer: str}}
#   escalation_trigger:  rule_value = {"question_key": str,
#                        "trigger_answers": [str, ...], "trigger_reason": str}
_URGENCY_RANK = {"low": 0, "medium": 1, "high": 2}
_DEFAULT_RECOMMENDED_BLOCK_BY_URGENCY = {
    "low": "routine_care_block",
    "medium": "priority_care_block",
    "high": "urgent_care_block",
}


class TriageService:
    """FR-E8.4/US43: guided webchat/voice symptom-triage question flow and
    outcome classification.
    """

    def __init__(
        self,
        session_repo: TriageSessionRepository,
        rule_set_service: "RuleSetService",
        escalation_service: "EscalationService",
        broker: "MessageBroker",
    ) -> None:
        self._session_repo = session_repo
        self._rule_set_service = rule_set_service
        self._escalation_service = escalation_service
        # Not otherwise used by this class (`EscalationService.route` itself
        # is the one publisher of `escalation.triggered`) — retained purely
        # to match this class's frozen constructor Public surface.
        self._broker = broker

    async def _ordered_triage_questions(self, db: "AsyncSession") -> list:
        rules = await self._rule_set_service.get_active_rules_by_category(
            db, RuleCategory.triage_question.value
        )

        def _sort_key(rule) -> int:
            value = getattr(rule, "rule_value", None)
            value = value if isinstance(value, dict) else {}
            return value.get("order", 0)

        # `sorted` is stable, so two rules with the same (or no) configured
        # `order` retain the order `RuleSetService.get_active_rules_by_category`
        # itself returned them in — "the first configured triage_question
        # rule" (start_session) is read as that returned order, not an
        # incidental re-sort by `rule_key`.
        return sorted(rules, key=_sort_key)

    def _compute_urgency(self, responses: dict, triage_rules: list) -> "tuple[str, str]":
        rules_by_key = {rule.rule_key: rule for rule in triage_rules}
        best_level = "low"
        best_block: str | None = None
        for question_key, answer in responses.items():
            rule = rules_by_key.get(question_key)
            rule_value = getattr(rule, "rule_value", None) if rule is not None else None
            if not isinstance(rule_value, dict):
                continue
            mapped_level = rule_value.get("urgency_map", {}).get(answer)
            if mapped_level and _URGENCY_RANK.get(mapped_level, 0) > _URGENCY_RANK.get(best_level, 0):
                best_level = mapped_level
                best_block = rule_value.get("recommended_block_map", {}).get(answer, best_block)
        if best_block is None:
            best_block = _DEFAULT_RECOMMENDED_BLOCK_BY_URGENCY.get(best_level, "routine_care_block")
        return best_level, best_block

    async def start_session(
        self, db: "AsyncSession", channel: str, patient_id: "UUID | None"
    ) -> dict:
        """Creates a ``triage_sessions`` row and returns the first
        configured ``triage_question`` rule as ``next_question`` (``None``
        if no ``triage_question`` rules are configured on the active rule
        set)."""
        session = await self._session_repo.create(db, channel=channel, patient_id=patient_id)
        questions = await self._ordered_triage_questions(db)
        next_question = questions[0] if questions else None
        return {"id": session.id, "next_question": next_question}

    async def classify(self, db: "AsyncSession", session_id: UUID) -> dict:
        """FR-E8.5/SFR001/US44: pure classification logic — checks the
        session's accumulated ``responses`` against active
        ``escalation_trigger`` rules FIRST, before computing any
        urgency/recommendation, so a session already matching a trigger
        never has a ``recommended_block`` computed for it.
        """
        session = await self._session_repo.get_by_id(db, session_id)
        if session is None:
            raise NotFoundError("Triage session not found.")
        responses = dict(session.responses or {})

        escalation_rules = await self._rule_set_service.get_active_rules_by_category(
            db, RuleCategory.escalation_trigger.value
        )
        for rule in escalation_rules:
            value = getattr(rule, "rule_value", None)
            value = value if isinstance(value, dict) else {}
            question_key = value.get("question_key")
            if question_key is None or question_key not in responses:
                continue
            trigger_answers = value.get("trigger_answers")
            if trigger_answers is None:
                single = value.get("trigger_answer")
                trigger_answers = [single] if single is not None else []
            if responses[question_key] in trigger_answers:
                return {
                    "escalated": True,
                    "trigger_reason": value.get("trigger_reason") or rule.rule_key,
                    "urgency_classification": None,
                    "recommended_block": None,
                }

        triage_rules = await self._ordered_triage_questions(db)
        urgency_classification, recommended_block = self._compute_urgency(responses, triage_rules)
        return {
            "escalated": False,
            "trigger_reason": None,
            "urgency_classification": urgency_classification,
            "recommended_block": recommended_block,
        }

    async def answer(
        self, db: "AsyncSession", session_id: UUID, question_key: str, answer: str
    ) -> dict:
        """FR-E8.4/FR-E8.5/US43-US44: appends the response then calls
        ``classify``; a matched escalation trigger halts the flow
        immediately (no recommendation is computed, ``next_question`` and
        ``recommended_block`` both ``None``) and routes the escalation;
        otherwise advances to the next configured ``triage_question`` or,
        once the question set is exhausted, persists and returns the
        computed ``urgency_classification``/``recommended_block``.
        """
        session = await self._session_repo.append_response(db, session_id, question_key, answer)
        result = await self.classify(db, session_id)

        if result["escalated"]:
            await self._escalation_service.route(
                db, trigger_reason=result["trigger_reason"], triage_session_id=session_id
            )
            await self._session_repo.set_outcome(
                db, session_id, urgency_classification=None, recommended_block=None, escalated=True
            )
            return {"escalated": True, "next_question": None, "recommended_block": None}

        responses = dict(session.responses or {})
        triage_rules = await self._ordered_triage_questions(db)
        remaining = [rule for rule in triage_rules if rule.rule_key not in responses]

        if remaining:
            return {
                "escalated": False,
                "next_question": remaining[0],
                "recommended_block": None,
            }

        await self._session_repo.set_outcome(
            db,
            session_id,
            urgency_classification=result["urgency_classification"],
            recommended_block=result["recommended_block"],
            escalated=False,
        )
        return {
            "escalated": False,
            "next_question": None,
            "recommended_block": result["recommended_block"],
        }


class EscalationService:
    """FR-E8.5/SFR001/US44: the one write path onto ``escalations`` and the
    one publisher of ``escalation.triggered``."""

    def __init__(
        self, escalation_repo: EscalationRepository, audit: "AuditService", broker: "MessageBroker"
    ) -> None:
        self._escalation_repo = escalation_repo
        self._audit = audit
        self._broker = broker

    async def route(
        self,
        db: "AsyncSession",
        trigger_reason: str,
        triage_session_id: "UUID | None" = None,
        call_interaction_id: "UUID | None" = None,
    ) -> "Escalation":
        """FR-E8.5/SFR001/US44/T110: inserts the ``escalations`` row with
        ``status="unacknowledged"`` THEN publishes ``escalation.triggered``
        — insert-before-publish, the same ordering
        ``SchedulingService.create``/``.reschedule``/``.cancel`` use, so an
        event is never published for a row that might not have been
        written. Exactly one of ``triage_session_id``/``call_interaction_id``
        is set per row, per this file's Interaction contract.
        """
        escalation = await self._escalation_repo.create(
            db,
            triage_session_id=triage_session_id,
            call_interaction_id=call_interaction_id,
            trigger_reason=trigger_reason,
            status=EscalationStatus.unacknowledged.value,
        )

        identifier = triage_session_id if triage_session_id is not None else call_interaction_id
        await self._broker.publish(
            "escalation.triggered",
            {
                "triage_session_id_or_call_interaction_id": identifier,
                "trigger_reason": trigger_reason,
            },
        )
        return escalation

    async def acknowledge(self, db: "AsyncSession", id: UUID, staff_id: UUID) -> "Escalation":
        """US44 Alternate Flow/``open_questions[Q-5b4034f4]``: no staff-
        targeting/round-robin logic exists — every unacknowledged escalation
        is visible to all front_office_staff/clinic_management sessions
        (via ``EscalationRepository.list_unacknowledged``) and is simply
        claimed by whoever acknowledges first, by delegating straight to
        ``EscalationRepository.acknowledge`` (the forward-only
        ``unacknowledged`` -> ``acknowledged`` transition itself is declared,
        not re-validated here, per that repository's own docstring).
        """
        return await self._escalation_repo.acknowledge(db, id, staff_id)

    async def resolve(
        self, db: "AsyncSession", id: UUID, staff_id: UUID, resolution_notes: str
    ) -> "Escalation":
        """US44 AC2/T110: delegates the ``acknowledged`` -> ``resolved``
        write straight to ``EscalationRepository.resolve``, then calls
        ``AuditService.record(action_type="escalation.resolve")`` so "every
        escalation is logged with trigger reason, timestamp, and resolving
        staff member".
        """
        resolved = await self._escalation_repo.resolve(db, id, staff_id, resolution_notes)

        await self._audit.record(
            db,
            actor_staff_id=staff_id,
            actor_type=ActorType.staff,
            action_type="escalation.resolve",
            entity_type="escalation",
            entity_id=id,
            override_payload={
                "trigger_reason": getattr(resolved, "trigger_reason", None),
                "resolution_notes": resolution_notes,
            },
        )
        return resolved


# --- UtilisationEngine inferred scheduling-search defaults ------------------
# FR-E8.6/US45: mirrors `services/scheduling/service.py`'s own
# `propose_slots` conventions (business hours, lookahead window) — this
# engine intentionally does not depend on `RuleSetService` itself (not part
# of this class's frozen constructor Public surface), so it reads the same
# conventional business-hours default `SchedulingService` falls back to.
_UTILISATION_LOOKAHEAD_DAYS = 14
_BUSINESS_START_HOUR = 8
_BUSINESS_END_HOUR = 18
_MAX_RECOMMENDATIONS_PER_RUN = 20
# US45's own target — see `services/analytics/service.py` for the dashboard
# that actually surfaces this; `get_adoption_rate` here reads an all-time
# window (open question: no rolling-period is specified in the BRD).
_ADOPTION_RATE_SINCE = datetime(2000, 1, 1, tzinfo=timezone.utc)


class UtilisationEngine:
    """FR-E8.6-FR-E8.7/US45: generates, tracks, applies and dismisses
    schedule-utilisation recommendations."""

    def __init__(
        self,
        rec_repo: UtilisationRecommendationRepository,
        risk_repo: RiskScoreRepository,
        session_repo: TriageSessionRepository,
        appt_repo: "AppointmentRepository",
        scheduling_service: "SchedulingService",
        audit: "AuditService",
    ) -> None:
        self._rec_repo = rec_repo
        self._risk_repo = risk_repo
        self._session_repo = session_repo
        self._appt_repo = appt_repo
        self._scheduling_service = scheduling_service
        self._audit = audit

    async def _find_earlier_slot(
        self, db: "AsyncSession", appointment, now: datetime
    ) -> "tuple[datetime, datetime] | None":
        """Searches the same ``provider_id``/``chair_id``'s near-term
        schedule (between ``now`` and the appointment's own current start)
        for the earliest open gap at least as long as the appointment's own
        duration — the "idle window" half of FR-E8.6's schedule analysis.
        """
        start = _aware(appointment.scheduled_start)
        end = _aware(appointment.scheduled_end)
        duration = end - start

        existing = await self._appt_repo.list_open_slots(db, appointment.provider_id, now, start)
        busy = sorted(
            (
                (_aware(a.scheduled_start), _aware(a.scheduled_end))
                for a in existing
                if a.id != appointment.id
            ),
            key=lambda pair: pair[0],
        )

        cursor = now
        for busy_start, busy_end in busy:
            if busy_start > cursor and (busy_start - cursor) >= duration and _BUSINESS_START_HOUR <= cursor.hour < _BUSINESS_END_HOUR:
                candidate_end = cursor + duration
                conflicts = await self._appt_repo.find_conflicts(
                    db,
                    appointment.provider_id,
                    appointment.chair_id,
                    cursor,
                    candidate_end,
                    exclude_appointment_id=appointment.id,
                )
                if not conflicts:
                    return cursor, candidate_end
            cursor = max(cursor, busy_end)
        return None

    async def generate(self, db: "AsyncSession") -> "list[UtilisationRecommendation]":
        """FR-E8.6/US45: draws from all three documented sources — the
        current schedule's idle windows (via ``AppointmentRepository``),
        risk scores (via already risk-flagged appointments), and triage
        urgency (best-effort — see this method's own Watch out below) —
        to create ``pending`` ``utilisation_recommendations`` rows.

        Watch out: the frozen ``TriageSessionRepository`` exposes no
        list/query-by-urgency method (only single-row lookups by
        ``session_id``), so this method cannot enumerate "sessions with high
        urgency" the way it can enumerate risk-flagged appointments via
        ``AppointmentRepository.list_by_risk_source``. The triage-urgency
        input is folded in only when a candidate appointment's own
        ``source_metadata`` names a ``triage_session_id`` to look up — this
        gap (no bulk triage-session query surface) is reported in the task
        summary rather than worked around by touching that frozen file.
        """
        now = now_utc()
        horizon = now + timedelta(days=_UTILISATION_LOOKAHEAD_DAYS)

        scored: list = []
        for source in (ScoreSource.rules.value, ScoreSource.ml.value):
            scored.extend(await self._appt_repo.list_by_risk_source(db, source))

        seen_ids: set = set()
        candidates = []
        for appointment in scored:
            if appointment.id in seen_ids:
                continue
            seen_ids.add(appointment.id)
            if appointment.risk_flag != RiskLevel.high:
                continue
            if appointment.status not in (AppointmentStatus.booked, AppointmentStatus.rescheduled):
                continue
            start = _aware(appointment.scheduled_start)
            if not (now <= start <= horizon):
                continue
            candidates.append(appointment)

        recommendations: list = []
        for appointment in candidates:
            if len(recommendations) >= _MAX_RECOMMENDATIONS_PER_RUN:
                break

            earlier_slot = await self._find_earlier_slot(db, appointment, now)
            if earlier_slot is None:
                continue
            new_start, new_end = earlier_slot

            urgency_note = ""
            triage_session_id = None
            if isinstance(appointment.source_metadata, dict):
                triage_session_id = appointment.source_metadata.get("triage_session_id")
            if triage_session_id is not None:
                session = await self._session_repo.get_by_id(db, UUID(str(triage_session_id)))
                if session is not None and session.urgency_classification == "high":
                    urgency_note = " The patient also has a high-urgency triage session on file."

            start = _aware(appointment.scheduled_start)
            rationale = (
                f"Appointment {appointment.id} is flagged high risk of no-show "
                f"(scheduled {to_iso8601(start)}); an earlier open slot at "
                f"{to_iso8601(new_start)} is available for the same provider/chair, "
                "filling an idle schedule window and reducing exposure to a wasted slot."
                f"{urgency_note}"
            )
            recommended_change = {
                "appointment_id": str(appointment.id),
                "provider_id": str(appointment.provider_id),
                "chair_id": str(appointment.chair_id),
                "new_start": to_iso8601(new_start),
                "new_end": to_iso8601(new_end),
            }

            recommendation = await self._rec_repo.create(
                db,
                rationale=rationale,
                recommended_change=recommended_change,
                status=UtilisationRecommendationStatus.pending,
            )
            recommendations.append(recommendation)

        return recommendations

    async def revalidate(self, db: "AsyncSession", id: UUID) -> bool:
        """US45 AC2/Alternate Flow: re-checks ``recommended_change``'s
        target appointment/slot for a NEW conflict since ``generated_at``
        (via ``AppointmentRepository.find_conflicts``) — returns ``True``
        only if the proposed slot is still conflict-free. ``generate``'s own
        idle-window search only ever proposes a new time within the same
        provider/chair as the target appointment, so every real
        ``recommended_change`` it writes already carries
        ``appointment_id``/``provider_id``/``chair_id`` directly — this
        reads those straight from the recommendation row rather than
        re-fetching the appointment via ``AppointmentRepository.get_by_id``.
        """
        recommendation = await self._rec_repo.get_by_id(db, id)
        if recommendation is None:
            raise NotFoundError("Utilisation recommendation not found.")

        change = recommendation.recommended_change or {}
        try:
            appointment_id = _coerce_uuid(change["appointment_id"])
            provider_id = _coerce_uuid(change["provider_id"])
            chair_id = _coerce_uuid(change["chair_id"])
            new_start = _coerce_datetime(change["new_start"])
            new_end = _coerce_datetime(change["new_end"])
        except (KeyError, ValueError):
            return False

        conflicts = await self._appt_repo.find_conflicts(
            db, provider_id, chair_id, new_start, new_end, exclude_appointment_id=appointment_id
        )
        return not conflicts

    async def apply(
        self, db: "AsyncSession", id: UUID, actor_id: UUID
    ) -> "UtilisationRecommendation":
        """US45 AC2/Alternate Flow/FR-E8.7: calls ``revalidate`` FIRST,
        raising ``RecommendationStaleError`` (409) if a schedule change since
        ``generated_at`` invalidates it; on success, reschedules via
        ``SchedulingService.reschedule`` (this module's one synchronous call
        into ``scheduling``) then records ``status="applied"``,
        ``decided_by_staff_id``, ``decided_at``.
        """
        still_applicable = await self.revalidate(db, id)
        if not still_applicable:
            raise RecommendationStaleError()

        recommendation = await self._rec_repo.get_by_id(db, id)
        if recommendation is None:
            raise NotFoundError("Utilisation recommendation not found.")
        change = recommendation.recommended_change

        appointment_id = _coerce_uuid(change["appointment_id"])
        # `generate`'s own idle-window search only ever proposes a new time
        # within the same provider/chair, so every real `recommended_change`
        # it writes always carries both keys already.
        provider_id = change.get("provider_id")
        chair_id = change.get("chair_id")
        reschedule_data = {
            "provider_id": _coerce_uuid(provider_id) if provider_id is not None else None,
            "chair_id": _coerce_uuid(chair_id) if chair_id is not None else None,
            "scheduled_start": _coerce_datetime(change["new_start"]),
            "scheduled_end": _coerce_datetime(change["new_end"]),
        }
        await self._scheduling_service.reschedule(db, appointment_id, reschedule_data, actor_id)

        updated = await self._rec_repo.update_status(
            db, id, status=UtilisationRecommendationStatus.applied.value, decided_by_staff_id=actor_id
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="utilisation_recommendation.apply",
            entity_type="utilisation_recommendation",
            entity_id=id,
            override_payload=change,
        )
        return updated

    async def dismiss(
        self, db: "AsyncSession", id: UUID, actor_id: UUID
    ) -> "UtilisationRecommendation":
        """T112 AC/FR-E8.7: sets ``status="dismissed"`` with no schedule
        change — the one other place staff acceptance/dismissal is
        persisted for the >=50%-adoption tracking (``get_adoption_rate``).
        Delegates straight to ``UtilisationRecommendationRepository.update_status``
        (which itself raises if ``id`` does not exist) rather than an extra
        ``get_by_id`` round-trip first, since dismissal needs no field off
        the existing row — unlike ``apply``, which reads
        ``recommended_change`` to reschedule.
        """
        updated = await self._rec_repo.update_status(
            db, id, status=UtilisationRecommendationStatus.dismissed.value, decided_by_staff_id=actor_id
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="utilisation_recommendation.dismiss",
            entity_type="utilisation_recommendation",
            entity_id=id,
        )
        return updated

    async def get_adoption_rate(self, db: "AsyncSession") -> float:
        """FR-E8.7/US45: the >=50%-adoption metric — ``applied`` /
        (``applied`` + ``dismissed``) over an all-time window (open
        question: the BRD does not name a rolling reporting period)."""
        return await self._rec_repo.get_adoption_rate(db, _ADOPTION_RATE_SINCE)
