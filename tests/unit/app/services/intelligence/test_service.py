"""Unit tests for app/services/intelligence/service.py.

These tests exercise RiskScoringService, TriageService, EscalationService, and
UtilisationEngine purely against mocked repository/service collaborators, per
the documented public surface and behaviours in
specs/app/services/intelligence/service.py.md.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.common.constants import ML_TARGET_PRECISION
from app.common.exceptions.errors import RecommendationStaleError
from app.services.intelligence.service import (
    EscalationService,
    RiskScoringService,
    TriageService,
    UtilisationEngine,
)

pytestmark = pytest.mark.asyncio


def _db():
    return MagicMock(name="AsyncSession")


def _values(call):
    """Flatten a mock call's positional+keyword arguments into one list."""
    return list(call.args) + list(call.kwargs.values())


def _get_dict_payload(call):
    if "payload" in call.kwargs:
        return call.kwargs["payload"]
    for arg in call.args:
        if isinstance(arg, dict):
            return arg
    return None


# ---------------------------------------------------------------------------
# RiskScoringService
# ---------------------------------------------------------------------------

def _risk_repo():
    repo = AsyncMock()

    async def _create(db, **fields):
        return SimpleNamespace(**fields)

    repo.create = AsyncMock(side_effect=_create)
    repo.get_latest_for_appointment = AsyncMock(return_value=None)
    repo.get_distribution = AsyncMock(return_value={})
    return repo


def _appt_repo_for_risk():
    repo = AsyncMock()
    repo.update_fields = AsyncMock(return_value=None)
    repo.list_by_patient = AsyncMock(return_value=[])
    repo.get_by_id = AsyncMock(return_value=None)
    repo.list_by_risk_source = AsyncMock(return_value=[])
    return repo


def _rule_set_service():
    svc = AsyncMock()
    svc.get_active_rules_by_category = AsyncMock(return_value=[])
    return svc


def _eval_repo():
    repo = AsyncMock()

    async def _create(db, **fields):
        return SimpleNamespace(**fields)

    repo.create = AsyncMock(side_effect=_create)
    repo.get_latest = AsyncMock(return_value=None)
    return repo


async def test_score_rules_based_new_patient_gets_neutral_default():
    appointment_id = uuid4()
    appt = SimpleNamespace(
        id=appointment_id,
        patient_id=uuid4(),
        scheduled_start=datetime.now(timezone.utc) + timedelta(days=5),
        created_at=datetime.now(timezone.utc),
        status="booked",
    )
    risk_repo = _risk_repo()
    appt_repo = _appt_repo_for_risk()
    appt_repo.get_by_id.return_value = appt
    appt_repo.list_by_patient.return_value = []  # no prior history
    rule_set_service = _rule_set_service()
    eval_repo = _eval_repo()

    service = RiskScoringService(risk_repo, appt_repo, eval_repo, rule_set_service)
    score = await service.score_rules_based(_db(), appointment_id)

    assert float(score.score_value) == pytest.approx(0.5)
    assert score.risk_level == "medium"
    assert score.source == "rules"


async def test_score_rules_based_updates_appointment_risk_fields():
    appointment_id = uuid4()
    risk_repo = _risk_repo()
    appt_repo = _appt_repo_for_risk()
    appt_repo.get_by_id.return_value = SimpleNamespace(
        id=appointment_id,
        patient_id=uuid4(),
        scheduled_start=datetime.now(timezone.utc) + timedelta(days=2),
        created_at=datetime.now(timezone.utc),
        status="booked",
    )
    rule_set_service = _rule_set_service()
    eval_repo = _eval_repo()

    service = RiskScoringService(risk_repo, appt_repo, eval_repo, rule_set_service)
    await service.score_rules_based(_db(), appointment_id)

    appt_repo.update_fields.assert_awaited()
    call = appt_repo.update_fields.call_args
    assert call.kwargs.get("risk_score_source") == "rules"
    assert "risk_flag" in call.kwargs


async def test_score_ml_is_noop_when_not_eligible():
    appointment_id = uuid4()
    risk_repo = _risk_repo()
    appt_repo = _appt_repo_for_risk()
    rule_set_service = _rule_set_service()
    eval_repo = _eval_repo()

    service = RiskScoringService(risk_repo, appt_repo, eval_repo, rule_set_service)
    service._is_ml_eligible = AsyncMock(return_value=False)

    result = await service.score_ml(_db(), appointment_id)

    assert result is None
    risk_repo.create.assert_not_awaited()
    appt_repo.update_fields.assert_not_awaited()


async def test_score_ml_writes_source_ml_when_eligible():
    appointment_id = uuid4()
    risk_repo = _risk_repo()
    appt_repo = _appt_repo_for_risk()
    appt_repo.get_by_id.return_value = SimpleNamespace(
        id=appointment_id,
        patient_id=uuid4(),
        scheduled_start=datetime.now(timezone.utc) + timedelta(days=2),
        created_at=datetime.now(timezone.utc),
        status="booked",
    )
    rule_set_service = _rule_set_service()
    eval_repo = _eval_repo()

    service = RiskScoringService(risk_repo, appt_repo, eval_repo, rule_set_service)
    service._is_ml_eligible = AsyncMock(return_value=True)

    result = await service.score_ml(_db(), appointment_id)

    assert result is not None
    assert result.source == "ml"
    call = appt_repo.update_fields.call_args
    assert call.kwargs.get("risk_score_source") == "ml"


async def test_evaluate_precision_status_matches_threshold_comparison():
    risk_repo = _risk_repo()
    appt_repo = _appt_repo_for_risk()
    rule_set_service = _rule_set_service()
    eval_repo = _eval_repo()

    service = RiskScoringService(risk_repo, appt_repo, eval_repo, rule_set_service)
    result = await service.evaluate_precision(_db())

    eval_repo.create.assert_awaited_once()
    assert result.status in ("passing", "underperforming")
    if float(result.precision_value) >= ML_TARGET_PRECISION:
        assert result.status == "passing"
    else:
        assert result.status == "underperforming"


# ---------------------------------------------------------------------------
# TriageService
# ---------------------------------------------------------------------------

def _triage_session_repo():
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=None)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.append_response = AsyncMock(return_value=None)
    repo.set_outcome = AsyncMock(return_value=None)
    repo.update_fields = AsyncMock(return_value=None)
    return repo


def _escalation_service_mock():
    svc = AsyncMock()
    svc.route = AsyncMock(return_value=SimpleNamespace(id=uuid4(), status="unacknowledged"))
    return svc


def _broker():
    b = AsyncMock()
    b.publish = AsyncMock(return_value=None)
    return b


async def test_start_session_creates_row_and_returns_first_question():
    session_id = uuid4()
    session_repo = _triage_session_repo()
    session_repo.create.return_value = SimpleNamespace(
        id=session_id, channel="webchat", patient_id=None, responses={}, escalated=False
    )
    rule_set_service = _rule_set_service()
    q1 = SimpleNamespace(rule_key="pain_level", category="triage_question", value={})
    q2 = SimpleNamespace(rule_key="duration", category="triage_question", value={})
    rule_set_service.get_active_rules_by_category.return_value = [q1, q2]
    escalation_service = _escalation_service_mock()
    broker = _broker()

    service = TriageService(session_repo, rule_set_service, escalation_service, broker)
    result = await service.start_session(_db(), "webchat", None)

    assert result["next_question"] is q1
    session_repo.create.assert_awaited_once()
    values = _values(session_repo.create.call_args)
    assert "webchat" in values


async def test_answer_escalation_short_circuits_and_routes():
    session_id = uuid4()
    session_repo = _triage_session_repo()
    session_repo.append_response.return_value = SimpleNamespace(
        id=session_id, responses={"pain_level": "10"}, escalated=False
    )
    rule_set_service = _rule_set_service()
    escalation_service = _escalation_service_mock()
    broker = _broker()

    service = TriageService(session_repo, rule_set_service, escalation_service, broker)
    service.classify = AsyncMock(return_value={"escalated": True, "trigger_reason": "severe_bleeding"})

    result = await service.answer(_db(), session_id, "pain_level", "10")

    assert result == {"escalated": True, "next_question": None, "recommended_block": None}
    escalation_service.route.assert_awaited_once()
    call = escalation_service.route.call_args
    values = _values(call)
    assert "severe_bleeding" in values
    assert session_id in values


async def test_answer_advances_to_next_question_when_not_exhausted():
    session_id = uuid4()
    q1 = SimpleNamespace(rule_key="q1", category="triage_question", value={})
    q2 = SimpleNamespace(rule_key="q2", category="triage_question", value={})
    session_repo = _triage_session_repo()
    session_repo.append_response.return_value = SimpleNamespace(
        id=session_id, responses={"q1": "yes"}, escalated=False
    )
    rule_set_service = _rule_set_service()
    rule_set_service.get_active_rules_by_category.return_value = [q1, q2]
    escalation_service = _escalation_service_mock()
    broker = _broker()

    service = TriageService(session_repo, rule_set_service, escalation_service, broker)
    service.classify = AsyncMock(return_value={"escalated": False})

    result = await service.answer(_db(), session_id, "q1", "yes")

    assert result["escalated"] is False
    assert result["next_question"] is q2
    assert result["recommended_block"] is None
    escalation_service.route.assert_not_awaited()


async def test_answer_computes_recommendation_when_question_set_exhausted():
    session_id = uuid4()
    q1 = SimpleNamespace(rule_key="q1", category="triage_question", value={})
    session_repo = _triage_session_repo()
    session_repo.append_response.return_value = SimpleNamespace(
        id=session_id, responses={"q1": "yes"}, escalated=False
    )
    session_repo.set_outcome.return_value = SimpleNamespace(
        id=session_id, urgency_classification="routine", recommended_block="standard_15min", escalated=False
    )
    rule_set_service = _rule_set_service()
    rule_set_service.get_active_rules_by_category.return_value = [q1]
    escalation_service = _escalation_service_mock()
    broker = _broker()

    service = TriageService(session_repo, rule_set_service, escalation_service, broker)
    service.classify = AsyncMock(
        return_value={"escalated": False, "urgency_classification": "routine", "recommended_block": "standard_15min"}
    )

    result = await service.answer(_db(), session_id, "q1", "yes")

    assert result == {"escalated": False, "next_question": None, "recommended_block": "standard_15min"}
    session_repo.set_outcome.assert_awaited_once()
    values = _values(session_repo.set_outcome.call_args)
    assert "routine" in values
    assert "standard_15min" in values
    assert False in values


async def test_classify_returns_not_escalated_when_no_trigger_rules_match():
    session_id = uuid4()
    session_repo = _triage_session_repo()
    session_repo.get_by_id.return_value = SimpleNamespace(
        id=session_id, responses={"q1": "no"}, channel="webchat", patient_id=None
    )
    rule_set_service = _rule_set_service()
    rule_set_service.get_active_rules_by_category.return_value = []  # no escalation_trigger rules configured
    escalation_service = _escalation_service_mock()
    broker = _broker()

    service = TriageService(session_repo, rule_set_service, escalation_service, broker)
    result = await service.classify(_db(), session_id)

    assert result["escalated"] is False


# ---------------------------------------------------------------------------
# EscalationService
# ---------------------------------------------------------------------------

def _escalation_repo():
    repo = AsyncMock()

    async def _create(db, **fields):
        return SimpleNamespace(id=uuid4(), **fields)

    repo.create = AsyncMock(side_effect=_create)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.acknowledge = AsyncMock(return_value=None)
    repo.resolve = AsyncMock(return_value=None)
    repo.list_unacknowledged = AsyncMock(return_value=[])
    return repo


def _audit_service():
    audit = AsyncMock()
    audit.record = AsyncMock(return_value=None)
    return audit


async def test_route_inserts_before_publish_for_triage_session():
    order: list[str] = []
    escalation_repo = _escalation_repo()

    async def _create(db, **fields):
        order.append("create")
        return SimpleNamespace(id=uuid4(), **fields)

    escalation_repo.create.side_effect = _create
    audit = _audit_service()
    broker = _broker()

    async def _publish(*args, **kwargs):
        order.append("publish")

    broker.publish.side_effect = _publish

    service = EscalationService(escalation_repo, audit, broker)
    triage_session_id = uuid4()
    escalation = await service.route(_db(), "chest_pain", triage_session_id=triage_session_id)

    assert escalation.status == "unacknowledged"
    assert order == ["create", "publish"]

    create_call = escalation_repo.create.call_args
    assert create_call.kwargs.get("status") == "unacknowledged"
    assert create_call.kwargs.get("trigger_reason") == "chest_pain"
    assert create_call.kwargs.get("triage_session_id") == triage_session_id

    payload = _get_dict_payload(broker.publish.call_args)
    assert payload["trigger_reason"] == "chest_pain"
    assert payload["triage_session_id_or_call_interaction_id"] == triage_session_id


async def test_route_with_call_interaction_id_only_sets_that_field():
    escalation_repo = _escalation_repo()
    audit = _audit_service()
    broker = _broker()

    service = EscalationService(escalation_repo, audit, broker)
    call_interaction_id = uuid4()
    await service.route(_db(), "fainting", call_interaction_id=call_interaction_id)

    create_call = escalation_repo.create.call_args
    assert create_call.kwargs.get("call_interaction_id") == call_interaction_id
    assert create_call.kwargs.get("triage_session_id") is None

    payload = _get_dict_payload(broker.publish.call_args)
    assert payload["triage_session_id_or_call_interaction_id"] == call_interaction_id


async def test_acknowledge_delegates_to_repository():
    escalation_repo = _escalation_repo()
    audit = _audit_service()
    broker = _broker()
    escalation_id = uuid4()
    staff_id = uuid4()
    escalation_repo.acknowledge.return_value = SimpleNamespace(
        id=escalation_id, status="acknowledged", routed_to_staff_id=staff_id
    )

    service = EscalationService(escalation_repo, audit, broker)
    result = await service.acknowledge(_db(), escalation_id, staff_id)

    assert result.status == "acknowledged"
    call = escalation_repo.acknowledge.call_args
    values = _values(call)
    assert escalation_id in values
    assert staff_id in values


async def test_resolve_records_audit_with_resolve_action_type():
    escalation_repo = _escalation_repo()
    audit = _audit_service()
    broker = _broker()
    escalation_id = uuid4()
    staff_id = uuid4()
    escalation_repo.resolve.return_value = SimpleNamespace(
        id=escalation_id, status="resolved", resolution_notes="handled on site"
    )

    service = EscalationService(escalation_repo, audit, broker)
    result = await service.resolve(_db(), escalation_id, staff_id, "handled on site")

    assert result.status == "resolved"
    audit.record.assert_awaited_once()
    call = audit.record.call_args
    assert call.kwargs.get("action_type") == "escalation.resolve"


# ---------------------------------------------------------------------------
# UtilisationEngine
# ---------------------------------------------------------------------------

def _rec_repo():
    repo = AsyncMock()
    repo.create = AsyncMock(return_value=None)
    repo.get_by_id = AsyncMock(return_value=None)
    repo.list_by_status = AsyncMock(return_value=[])
    repo.update_status = AsyncMock(return_value=None)
    repo.get_adoption_rate = AsyncMock(return_value=0.0)
    return repo


def _appt_repo_for_util():
    repo = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=None)
    repo.list_open_slots = AsyncMock(return_value=[])
    repo.find_conflicts = AsyncMock(return_value=[])
    repo.list_by_patient = AsyncMock(return_value=[])
    repo.list_completable = AsyncMock(return_value=[])
    repo.list_by_risk_source = AsyncMock(return_value=[])
    return repo


def _scheduling_service():
    svc = AsyncMock()
    svc.reschedule = AsyncMock(return_value=None)
    return svc


def _make_recommendation(rec_id, appointment_id, provider_id, chair_id, new_start, new_end):
    return SimpleNamespace(
        id=rec_id,
        generated_at=datetime.now(timezone.utc),
        rationale="idle chair window detected",
        recommended_change={
            "appointment_id": appointment_id,
            "provider_id": provider_id,
            "chair_id": chair_id,
            "new_start": new_start,
            "new_end": new_end,
        },
        status="pending",
        decided_by_staff_id=None,
        decided_at=None,
    )


async def test_generate_returns_empty_list_when_no_signal_present():
    rec_repo = _rec_repo()
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    scheduling_service = _scheduling_service()
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    result = await engine.generate(_db())

    assert result == []
    rec_repo.create.assert_not_awaited()


async def test_revalidate_true_when_no_new_conflict():
    rec_id = uuid4()
    appointment_id, provider_id, chair_id = uuid4(), uuid4(), uuid4()
    start = datetime.now(timezone.utc) + timedelta(days=1)
    end = start + timedelta(minutes=30)

    rec_repo = _rec_repo()
    rec_repo.get_by_id.return_value = _make_recommendation(rec_id, appointment_id, provider_id, chair_id, start, end)
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    appt_repo.find_conflicts.return_value = []
    scheduling_service = _scheduling_service()
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    result = await engine.revalidate(_db(), rec_id)

    assert result is True


async def test_revalidate_false_when_new_conflict_exists():
    rec_id = uuid4()
    appointment_id, provider_id, chair_id = uuid4(), uuid4(), uuid4()
    start = datetime.now(timezone.utc) + timedelta(days=1)
    end = start + timedelta(minutes=30)

    rec_repo = _rec_repo()
    rec_repo.get_by_id.return_value = _make_recommendation(rec_id, appointment_id, provider_id, chair_id, start, end)
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    appt_repo.find_conflicts.return_value = [SimpleNamespace(id=uuid4())]
    scheduling_service = _scheduling_service()
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    result = await engine.revalidate(_db(), rec_id)

    assert result is False


async def test_apply_raises_recommendation_stale_when_revalidate_fails():
    rec_id = uuid4()
    actor_id = uuid4()
    rec_repo = _rec_repo()
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    scheduling_service = _scheduling_service()
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    engine.revalidate = AsyncMock(return_value=False)

    with pytest.raises(RecommendationStaleError):
        await engine.apply(_db(), rec_id, actor_id)

    scheduling_service.reschedule.assert_not_awaited()
    rec_repo.update_status.assert_not_awaited()


async def test_apply_success_calls_reschedule_and_marks_applied():
    rec_id = uuid4()
    actor_id = uuid4()
    appointment_id, provider_id, chair_id = uuid4(), uuid4(), uuid4()
    start = datetime.now(timezone.utc) + timedelta(days=1)
    end = start + timedelta(minutes=30)

    rec_repo = _rec_repo()
    rec_repo.get_by_id.return_value = _make_recommendation(rec_id, appointment_id, provider_id, chair_id, start, end)
    rec_repo.update_status.return_value = SimpleNamespace(
        id=rec_id, status="applied", decided_by_staff_id=actor_id
    )
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    scheduling_service = _scheduling_service()
    scheduling_service.reschedule.return_value = SimpleNamespace(id=appointment_id, status="rescheduled")
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    engine.revalidate = AsyncMock(return_value=True)

    result = await engine.apply(_db(), rec_id, actor_id)

    assert result.status == "applied"
    scheduling_service.reschedule.assert_awaited_once()
    update_call = rec_repo.update_status.call_args
    values = _values(update_call)
    assert "applied" in values
    assert actor_id in values
    assert rec_id in values


async def test_dismiss_sets_status_dismissed_without_reschedule():
    rec_id = uuid4()
    actor_id = uuid4()
    rec_repo = _rec_repo()
    rec_repo.update_status.return_value = SimpleNamespace(
        id=rec_id, status="dismissed", decided_by_staff_id=actor_id
    )
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    scheduling_service = _scheduling_service()
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    result = await engine.dismiss(_db(), rec_id, actor_id)

    assert result.status == "dismissed"
    scheduling_service.reschedule.assert_not_awaited()
    update_call = rec_repo.update_status.call_args
    values = _values(update_call)
    assert "dismissed" in values
    assert actor_id in values


async def test_get_adoption_rate_delegates_to_repository():
    rec_repo = _rec_repo()
    rec_repo.get_adoption_rate.return_value = 0.6
    risk_repo = _risk_repo()
    session_repo = _triage_session_repo()
    appt_repo = _appt_repo_for_util()
    scheduling_service = _scheduling_service()
    audit = _audit_service()

    engine = UtilisationEngine(rec_repo, risk_repo, session_repo, appt_repo, scheduling_service, audit)
    result = await engine.get_adoption_rate(_db())

    assert result == 0.6
    rec_repo.get_adoption_rate.assert_awaited_once()
