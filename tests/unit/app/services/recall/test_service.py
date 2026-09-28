"""Unit tests for app/services/recall/service.py.

Derived only from specs/app/services/recall/service.py.md and the real signatures of its
declared dependencies (recall/patients repositories, RuleSetService, OutreachService,
AuditService). Dependencies are mocked with AsyncMock/MagicMock matching those signatures.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.common.exceptions.errors import NoActiveRuleSetError
from app.services.recall.service import (
    RecallCampaignService,
    RecallScanService,
    TreatmentRecoveryService,
)


def _kwarg(call_obj, name):
    """Fetch a keyword argument from a recorded mock call, or None if not passed as kwarg."""
    return call_obj.kwargs.get(name)


# ------------------------------------------------------------------------------------
# RecallScanService.run
# ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_attempts_risk_based_interval_lookup_first():
    """FR-E7.1/US34: _resolve_interval always tries the risk-based lookup first."""
    db = MagicMock(name="db")
    recall_repo = MagicMock()
    recall_repo.list_overdue = AsyncMock(return_value=[])
    recall_repo.create = AsyncMock(return_value=SimpleNamespace(id=uuid4()))
    recall_repo.update_status = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    patient_repo = MagicMock()
    patient = SimpleNamespace(id=uuid4(), status="active", risk_classification="high")
    patient_repo.list_active = AsyncMock(return_value=[patient])

    rule_set_service = MagicMock()
    rule_set_service.get_active_rules_by_category = AsyncMock(return_value=[])

    service = RecallScanService(recall_repo, patient_repo, rule_set_service)
    await service.run(db)

    assert rule_set_service.get_active_rules_by_category.await_args_list, (
        "run() must attempt the risk-based interval lookup for every active patient"
    )
    called_categories = [
        c.kwargs.get("category") if "category" in c.kwargs else (c.args[-1] if c.args else None)
        for c in rule_set_service.get_active_rules_by_category.await_args_list
    ]
    assert "recall_interval" in called_categories


@pytest.mark.asyncio
async def test_run_falls_back_to_default_interval_without_skipping_patient():
    """FR-E7.2/US34 Alt Flow: insufficient risk data -> patient still evaluated,
    interval_source='default_fallback' recorded, no exception propagates."""
    db = MagicMock(name="db")
    recall_repo = MagicMock()
    recall_repo.list_overdue = AsyncMock(return_value=[])
    recall_repo.create = AsyncMock(return_value=SimpleNamespace(id=uuid4()))
    recall_repo.update_status = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    patient_repo = MagicMock()
    patient = SimpleNamespace(
        id=uuid4(), status="active", risk_classification=None,
        last_visit_date=date(2020, 1, 1), last_visit_at=date(2020, 1, 1),
    )
    patient_repo.list_active = AsyncMock(return_value=[patient])

    rule_set_service = MagicMock()
    rule_set_service.get_active_rules_by_category = AsyncMock(side_effect=NoActiveRuleSetError())

    service = RecallScanService(recall_repo, patient_repo, rule_set_service)

    result = await service.run(db)  # must not raise

    assert isinstance(result, int)

    all_calls = list(recall_repo.create.await_args_list) + list(recall_repo.update_status.await_args_list)
    assert any(_kwarg(c, "interval_source") == "default_fallback" for c in all_calls), (
        "expected a recall_schedules write recording interval_source='default_fallback'"
    )


@pytest.mark.asyncio
async def test_run_returns_int_count_consistent_with_flagged_writes():
    """run() returns the count of patients flagged (created/updated)."""
    db = MagicMock(name="db")
    recall_repo = MagicMock()
    recall_repo.list_overdue = AsyncMock(return_value=[])
    recall_repo.create = AsyncMock(side_effect=lambda *a, **k: SimpleNamespace(id=uuid4()))
    recall_repo.update_status = AsyncMock(side_effect=lambda *a, **k: SimpleNamespace(id=uuid4()))

    patient_repo = MagicMock()
    patients = [
        SimpleNamespace(id=uuid4(), status="active", risk_classification=None,
                        last_visit_date=date(2019, 1, 1)),
        SimpleNamespace(id=uuid4(), status="active", risk_classification=None,
                        last_visit_date=date(2019, 6, 1)),
    ]
    patient_repo.list_active = AsyncMock(return_value=patients)

    rule_set_service = MagicMock()
    rule_set_service.get_active_rules_by_category = AsyncMock(return_value=[])

    service = RecallScanService(recall_repo, patient_repo, rule_set_service)
    result = await service.run(db)

    total_writes = recall_repo.create.await_count + recall_repo.update_status.await_count
    assert result == total_writes
    assert result >= 0


# ------------------------------------------------------------------------------------
# RecallCampaignService.dispatch / link_booking / get_compliance
# ------------------------------------------------------------------------------------

def _make_campaign_service(recall_repo=None, baseline_repo=None, outreach_service=None, audit=None):
    return RecallCampaignService(
        recall_repo or MagicMock(),
        baseline_repo or MagicMock(),
        outreach_service or MagicMock(),
        audit or MagicMock(),
    )


@pytest.mark.asyncio
async def test_dispatch_marks_contacted_on_successful_send():
    db = MagicMock(name="db")
    id1, id2 = uuid4(), uuid4()
    pid1, pid2 = uuid4(), uuid4()
    schedules = {
        id1: SimpleNamespace(id=id1, patient_id=pid1, language="en"),
        id2: SimpleNamespace(id=id2, patient_id=pid2, language="en"),
    }

    recall_repo = MagicMock()
    recall_repo.list_by_ids = AsyncMock(return_value=list(schedules.values()))
    recall_repo.get_by_id = AsyncMock(side_effect=lambda _db, sid: schedules.get(sid))
    recall_repo.update_status = AsyncMock(return_value=SimpleNamespace(id=id1, status="contacted"))

    outreach_service = MagicMock()
    sent_msg = SimpleNamespace(id=uuid4(), status="sent")
    outreach_service.dispatch = AsyncMock(side_effect=[sent_msg, sent_msg])

    audit = MagicMock()
    audit.record = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    service = _make_campaign_service(recall_repo=recall_repo, outreach_service=outreach_service, audit=audit)

    result = await service.dispatch(db, [id1, id2], actor_id=uuid4())

    assert isinstance(result, dict)
    assert result["status"] == "contacted"
    assert result["dispatched_count"] == 2
    assert isinstance(result["campaign_id"], UUID)

    # both consented sends should mark the schedule contacted
    assert recall_repo.update_status.await_count == 2
    for c in recall_repo.update_status.await_args_list:
        assert _kwarg(c, "status") == "contacted" or (c.args and "contacted" in c.args)

    # OutreachService.dispatch must be called with campaign_type="recall" per schedule
    for c in outreach_service.dispatch.await_args_list:
        assert _kwarg(c, "campaign_type") == "recall"
        assert _kwarg(c, "related_entity_type") == "recall_schedule"


@pytest.mark.asyncio
async def test_dispatch_skips_ids_without_consented_channel():
    db = MagicMock(name="db")
    id1, id2 = uuid4(), uuid4()
    pid1, pid2 = uuid4(), uuid4()
    schedules = {
        id1: SimpleNamespace(id=id1, patient_id=pid1, language="en"),
        id2: SimpleNamespace(id=id2, patient_id=pid2, language="en"),
    }

    recall_repo = MagicMock()
    recall_repo.list_by_ids = AsyncMock(return_value=list(schedules.values()))
    recall_repo.get_by_id = AsyncMock(side_effect=lambda _db, sid: schedules.get(sid))
    recall_repo.update_status = AsyncMock(return_value=SimpleNamespace(id=id1, status="contacted"))

    outreach_service = MagicMock()
    sent_msg = SimpleNamespace(id=uuid4(), status="sent")
    suppressed_msg = SimpleNamespace(id=uuid4(), status="suppressed")
    outreach_service.dispatch = AsyncMock(side_effect=[sent_msg, suppressed_msg])

    audit = MagicMock()
    audit.record = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    service = _make_campaign_service(recall_repo=recall_repo, outreach_service=outreach_service, audit=audit)

    result = await service.dispatch(db, [id1, id2], actor_id=uuid4())

    # only the successfully-consented id may be marked contacted
    assert result["dispatched_count"] == 1
    assert recall_repo.update_status.await_count == 1
    contacted_ids = {c.args[1] if len(c.args) > 1 else _kwarg(c, "id") for c in recall_repo.update_status.await_args_list}
    assert id2 not in contacted_ids


@pytest.mark.asyncio
async def test_link_booking_sets_completed_and_links_appointment():
    db = MagicMock(name="db")
    recall_repo = MagicMock()
    schedule_id = uuid4()
    appointment_id = uuid4()
    final = SimpleNamespace(id=schedule_id, status="completed", appointment_id=appointment_id)
    recall_repo.link_appointment = AsyncMock(return_value=final)
    recall_repo.update_status = AsyncMock(return_value=final)

    service = _make_campaign_service(recall_repo=recall_repo)

    result = await service.link_booking(db, schedule_id, appointment_id)

    recall_repo.link_appointment.assert_awaited()
    link_call = recall_repo.link_appointment.await_args
    assert schedule_id in link_call.args or _kwarg(link_call, "id") == schedule_id
    assert appointment_id in link_call.args or _kwarg(link_call, "appointment_id") == appointment_id

    assert result.status == "completed"


@pytest.mark.asyncio
async def test_get_compliance_computes_rate_and_reports_pending_baseline():
    db = MagicMock(name="db")
    recall_repo = MagicMock()
    recall_repo.count_by_status = AsyncMock(
        return_value={"due": 2, "overdue": 1, "contacted": 3, "completed": 4}
    )
    baseline_repo = MagicMock()
    baseline_repo.get_latest = AsyncMock(return_value=None)

    service = _make_campaign_service(recall_repo=recall_repo, baseline_repo=baseline_repo)

    result = await service.get_compliance(db, "weekly")

    assert result["compliance_rate"] == pytest.approx(4 / 10)
    assert result["baseline_rate"] is None
    assert result["baseline_pending"] is True

    y, w, _ = date.today().isocalendar()
    expected_period = f"{y}-W{w:02d}"
    assert result["period"] == expected_period

    baseline_repo.get_latest.assert_awaited()
    call = baseline_repo.get_latest.await_args
    assert "recall_compliance_rate" in call.args or _kwarg(call, "metric_name") == "recall_compliance_rate"


@pytest.mark.asyncio
async def test_get_compliance_reports_baseline_when_present():
    db = MagicMock(name="db")
    recall_repo = MagicMock()
    recall_repo.count_by_status = AsyncMock(
        return_value={"due": 0, "overdue": 0, "contacted": 0, "completed": 5}
    )
    baseline_repo = MagicMock()
    baseline_repo.get_latest = AsyncMock(return_value=SimpleNamespace(baseline_value=0.55))

    service = _make_campaign_service(recall_repo=recall_repo, baseline_repo=baseline_repo)

    result = await service.get_compliance(db, "weekly")

    assert result["compliance_rate"] == pytest.approx(1.0)
    assert result["baseline_rate"] == pytest.approx(0.55)
    assert result["baseline_pending"] is False


# ------------------------------------------------------------------------------------
# TreatmentRecoveryService.scan / dispatch_campaign / decline / convert
# ------------------------------------------------------------------------------------

def _make_recovery_service(treatment_repo=None, outreach_service=None, audit=None):
    return TreatmentRecoveryService(
        treatment_repo or MagicMock(),
        outreach_service or MagicMock(),
        audit or MagicMock(),
    )


@pytest.mark.asyncio
async def test_scan_returns_treatments_including_missing_valuation():
    db = MagicMock(name="db")
    t1 = SimpleNamespace(id=uuid4(), status="unscheduled", valuation_amount=None)
    t2 = SimpleNamespace(id=uuid4(), status="unscheduled", valuation_amount=500)
    treatment_repo = MagicMock()
    treatment_repo.list_by_valuation = AsyncMock(return_value=[t2, t1])

    service = _make_recovery_service(treatment_repo=treatment_repo)

    result = await service.scan(db)

    assert result == [t2, t1]
    call = treatment_repo.list_by_valuation.await_args
    assert _kwarg(call, "status") == "unscheduled" or (call.args and "unscheduled" in call.args)
    # a NULL-valuation row is still present, not filtered out
    assert any(t.valuation_amount is None for t in result)


@pytest.mark.asyncio
async def test_dispatch_campaign_sets_re_engaged_for_consenting_patients():
    db = MagicMock(name="db")
    id1, id2 = uuid4(), uuid4()
    treatments = {
        id1: SimpleNamespace(id=id1, patient_id=uuid4(), status="unscheduled"),
        id2: SimpleNamespace(id=id2, patient_id=uuid4(), status="unscheduled"),
    }
    treatment_repo = MagicMock()
    treatment_repo.list_by_ids = AsyncMock(return_value=list(treatments.values()))
    treatment_repo.get_by_id = AsyncMock(side_effect=lambda _db, tid: treatments.get(tid))
    treatment_repo.update_status = AsyncMock(return_value=SimpleNamespace(id=id1, status="re_engaged"))

    outreach_service = MagicMock()
    sent = SimpleNamespace(id=uuid4(), status="sent")
    suppressed = SimpleNamespace(id=uuid4(), status="suppressed")
    outreach_service.dispatch = AsyncMock(side_effect=[sent, suppressed])

    audit = MagicMock()
    audit.record = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    service = _make_recovery_service(treatment_repo=treatment_repo, outreach_service=outreach_service, audit=audit)

    await service.dispatch_campaign(db, [id1, id2], actor_id=uuid4())

    for c in outreach_service.dispatch.await_args_list:
        assert _kwarg(c, "campaign_type") == "treatment_reengagement"

    assert any(_kwarg(c, "status") == "re_engaged" or "re_engaged" in c.args
               for c in treatment_repo.update_status.await_args_list)


@pytest.mark.asyncio
async def test_decline_sets_declined_status():
    db = MagicMock(name="db")
    tid = uuid4()
    actor_id = uuid4()
    treatment_repo = MagicMock()
    declined = SimpleNamespace(id=tid, status="declined")
    treatment_repo.update_status = AsyncMock(return_value=declined)

    service = _make_recovery_service(treatment_repo=treatment_repo)

    result = await service.decline(db, tid, actor_id)

    call = treatment_repo.update_status.await_args
    assert _kwarg(call, "status") == "declined" or "declined" in call.args
    assert result is declined


@pytest.mark.asyncio
async def test_convert_full_booking_links_and_sets_booked():
    db = MagicMock(name="db")
    tid = uuid4()
    appointment_id = uuid4()
    actor_id = uuid4()

    final = SimpleNamespace(id=tid, status="booked", appointment_id=appointment_id)
    treatment_repo = MagicMock()
    treatment_repo.link_appointment = AsyncMock(return_value=final)
    treatment_repo.update_status = AsyncMock(return_value=final)

    audit = MagicMock()
    audit.record = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    service = _make_recovery_service(treatment_repo=treatment_repo, audit=audit)

    result = await service.convert(db, tid, appointment_id, partial=False, actor_id=actor_id)

    treatment_repo.link_appointment.assert_awaited()
    assert result.status == "booked"

    audit.record.assert_awaited()
    audit_call = audit.record.await_args
    assert _kwarg(audit_call, "action_type") == "treatment.convert"

    status_calls = treatment_repo.update_status.await_args_list
    assert any(_kwarg(c, "status") == "booked" or "booked" in c.args for c in status_calls)


@pytest.mark.asyncio
async def test_convert_partial_booking_leaves_status_unscheduled():
    db = MagicMock(name="db")
    tid = uuid4()
    appointment_id = uuid4()
    actor_id = uuid4()

    unchanged = SimpleNamespace(id=tid, status="unscheduled", appointment_id=appointment_id)
    treatment_repo = MagicMock()
    treatment_repo.link_appointment = AsyncMock(return_value=unchanged)
    treatment_repo.update_status = AsyncMock(return_value=unchanged)

    audit = MagicMock()
    audit.record = AsyncMock(return_value=SimpleNamespace(id=uuid4()))

    service = _make_recovery_service(treatment_repo=treatment_repo, audit=audit)

    result = await service.convert(db, tid, appointment_id, partial=True, actor_id=actor_id)

    # linking still happens for a partial booking
    treatment_repo.link_appointment.assert_awaited()
    # status must never be flipped to "booked" when partial=True
    for c in treatment_repo.update_status.await_args_list:
        assert _kwarg(c, "status") != "booked"
        assert "booked" not in c.args
    assert result.status == "unscheduled"

    audit.record.assert_awaited()
    assert _kwarg(audit.record.await_args, "action_type") == "treatment.convert"
