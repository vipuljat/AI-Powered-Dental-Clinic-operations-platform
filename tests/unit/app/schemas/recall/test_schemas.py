"""Unit tests for app/schemas/recall/schemas.py.

These Pydantic v2 models are the whole request/response contract for every
`/recall/*` route (architecture.md §5.2) -- `services/recall/service.py`'s
return dicts are shaped to match these field names directly, per this file's
own Interaction contract. Tests exercise construction, validation, and
serialization behaviour declared by the spec: field types/optionality,
`RecallStatus`/`UnscheduledTreatmentStatus` enum membership, the
`ConvertTreatmentRequest.partial` default, and the
`RecallComplianceResponse.baseline_pending` / `baseline_rate` pairing that
carries the US36 "baseline pending" alternate flow.
"""
from __future__ import annotations

import uuid
from datetime import date

import pytest
from pydantic import ValidationError

from app.common.enums import RecallStatus, UnscheduledTreatmentStatus
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


# ---------------------------------------------------------------------------
# RecallQueueItem
# ---------------------------------------------------------------------------

def _queue_item_kwargs(**overrides):
    kwargs = dict(
        patient_id=uuid.uuid4(),
        risk_classification="high",
        due_date=date(2026, 10, 1),
        overdue_days=5,
        status=RecallStatus.overdue,
    )
    kwargs.update(overrides)
    return kwargs


def test_recall_queue_item_accepts_full_valid_payload():
    item = RecallQueueItem(**_queue_item_kwargs())
    assert item.patient_id is not None
    assert item.risk_classification == "high"
    assert item.due_date == date(2026, 10, 1)
    assert item.overdue_days == 5
    assert item.status == RecallStatus.overdue


def test_recall_queue_item_allows_risk_classification_none():
    item = RecallQueueItem(**_queue_item_kwargs(risk_classification=None))
    assert item.risk_classification is None


def test_recall_queue_item_coerces_iso_date_string():
    item = RecallQueueItem(**_queue_item_kwargs(due_date="2026-11-15"))
    assert item.due_date == date(2026, 11, 15)


def test_recall_queue_item_coerces_uuid_string():
    patient_id = str(uuid.uuid4())
    item = RecallQueueItem(**_queue_item_kwargs(patient_id=patient_id))
    assert isinstance(item.patient_id, uuid.UUID)
    assert str(item.patient_id) == patient_id


def test_recall_queue_item_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        RecallQueueItem(**_queue_item_kwargs(status="bogus_status"))


def test_recall_queue_item_rejects_missing_required_field():
    kwargs = _queue_item_kwargs()
    del kwargs["patient_id"]
    with pytest.raises(ValidationError):
        RecallQueueItem(**kwargs)


def test_recall_queue_item_status_serializes_to_plain_string():
    item = RecallQueueItem(**_queue_item_kwargs(status=RecallStatus.dormant))
    dumped = item.model_dump(mode="json")
    assert dumped["status"] == "dormant"


# ---------------------------------------------------------------------------
# RecallQueueResponse
# ---------------------------------------------------------------------------

def test_recall_queue_response_wraps_list_of_items():
    item = RecallQueueItem(**_queue_item_kwargs())
    response = RecallQueueResponse(items=[item])
    assert response.items == [item]


def test_recall_queue_response_accepts_empty_list():
    response = RecallQueueResponse(items=[])
    assert response.items == []


def test_recall_queue_response_rejects_missing_items():
    with pytest.raises(ValidationError):
        RecallQueueResponse()


# ---------------------------------------------------------------------------
# TriggerRecallCampaignRequest
# ---------------------------------------------------------------------------

def test_trigger_recall_campaign_request_accepts_uuid_list():
    ids = [uuid.uuid4(), uuid.uuid4()]
    request = TriggerRecallCampaignRequest(recall_schedule_ids=ids)
    assert request.recall_schedule_ids == ids


def test_trigger_recall_campaign_request_coerces_uuid_strings():
    ids = [str(uuid.uuid4())]
    request = TriggerRecallCampaignRequest(recall_schedule_ids=ids)
    assert isinstance(request.recall_schedule_ids[0], uuid.UUID)


def test_trigger_recall_campaign_request_rejects_non_uuid_entry():
    with pytest.raises(ValidationError):
        TriggerRecallCampaignRequest(recall_schedule_ids=["not-a-uuid"])


def test_trigger_recall_campaign_request_rejects_missing_field():
    with pytest.raises(ValidationError):
        TriggerRecallCampaignRequest()


# ---------------------------------------------------------------------------
# TriggerCampaignResponse
# ---------------------------------------------------------------------------

def test_trigger_campaign_response_accepts_valid_payload():
    campaign_id = uuid.uuid4()
    response = TriggerCampaignResponse(
        campaign_id=campaign_id, dispatched_count=3, status="dispatched"
    )
    assert response.campaign_id == campaign_id
    assert response.dispatched_count == 3
    assert response.status == "dispatched"


def test_trigger_campaign_response_rejects_missing_dispatched_count():
    with pytest.raises(ValidationError):
        TriggerCampaignResponse(campaign_id=uuid.uuid4(), status="dispatched")


# ---------------------------------------------------------------------------
# RecallComplianceResponse -- US36 baseline-pending alternate flow.
# ---------------------------------------------------------------------------

def test_recall_compliance_response_normal_case_has_baseline_rate_and_not_pending():
    response = RecallComplianceResponse(
        compliance_rate=0.82,
        baseline_rate=0.75,
        baseline_pending=False,
        period="2026-Q3",
    )
    assert response.compliance_rate == 0.82
    assert response.baseline_rate == 0.75
    assert response.baseline_pending is False
    assert response.period == "2026-Q3"


def test_recall_compliance_response_baseline_pending_carries_null_baseline_rate():
    # US36 Alternate Flow: "baseline pending... compliance shown as raw counts" --
    # baseline_pending=True is paired with baseline_rate=None, while
    # compliance_rate remains populated as the raw rate.
    response = RecallComplianceResponse(
        compliance_rate=0.60,
        baseline_rate=None,
        baseline_pending=True,
        period="2026-Q3",
    )
    assert response.baseline_pending is True
    assert response.baseline_rate is None
    assert response.compliance_rate == 0.60


def test_recall_compliance_response_rejects_missing_baseline_pending():
    with pytest.raises(ValidationError):
        RecallComplianceResponse(
            compliance_rate=0.5, baseline_rate=None, period="2026-Q3"
        )


# ---------------------------------------------------------------------------
# UnscheduledTreatmentItem
# ---------------------------------------------------------------------------

def _treatment_item_kwargs(**overrides):
    kwargs = dict(
        id=uuid.uuid4(),
        patient_id=uuid.uuid4(),
        treatment_code="D2740",
        valuation_amount=450.0,
        days_unscheduled=30,
        status=UnscheduledTreatmentStatus.unscheduled,
    )
    kwargs.update(overrides)
    return kwargs


def test_unscheduled_treatment_item_accepts_full_valid_payload():
    item = UnscheduledTreatmentItem(**_treatment_item_kwargs())
    assert item.treatment_code == "D2740"
    assert item.valuation_amount == 450.0
    assert item.days_unscheduled == 30
    assert item.status == UnscheduledTreatmentStatus.unscheduled


def test_unscheduled_treatment_item_allows_valuation_amount_none():
    item = UnscheduledTreatmentItem(**_treatment_item_kwargs(valuation_amount=None))
    assert item.valuation_amount is None


def test_unscheduled_treatment_item_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        UnscheduledTreatmentItem(**_treatment_item_kwargs(status="not_a_status"))


def test_unscheduled_treatment_item_rejects_missing_treatment_code():
    kwargs = _treatment_item_kwargs()
    del kwargs["treatment_code"]
    with pytest.raises(ValidationError):
        UnscheduledTreatmentItem(**kwargs)


# ---------------------------------------------------------------------------
# UnscheduledTreatmentListResponse
# ---------------------------------------------------------------------------

def test_unscheduled_treatment_list_response_wraps_list_of_items():
    item = UnscheduledTreatmentItem(**_treatment_item_kwargs())
    response = UnscheduledTreatmentListResponse(items=[item])
    assert response.items == [item]


def test_unscheduled_treatment_list_response_accepts_empty_list():
    response = UnscheduledTreatmentListResponse(items=[])
    assert response.items == []


# ---------------------------------------------------------------------------
# UpdateValuationRequest
# ---------------------------------------------------------------------------

def test_update_valuation_request_accepts_float():
    request = UpdateValuationRequest(valuation_amount=612.5)
    assert request.valuation_amount == 612.5


def test_update_valuation_request_coerces_int_to_float():
    request = UpdateValuationRequest(valuation_amount=600)
    assert request.valuation_amount == 600.0
    assert isinstance(request.valuation_amount, float)


def test_update_valuation_request_rejects_missing_valuation_amount():
    with pytest.raises(ValidationError):
        UpdateValuationRequest()


# ---------------------------------------------------------------------------
# TriggerTreatmentCampaignRequest
# ---------------------------------------------------------------------------

def test_trigger_treatment_campaign_request_accepts_uuid_list():
    ids = [uuid.uuid4()]
    request = TriggerTreatmentCampaignRequest(unscheduled_treatment_ids=ids)
    assert request.unscheduled_treatment_ids == ids


def test_trigger_treatment_campaign_request_rejects_non_uuid_entry():
    with pytest.raises(ValidationError):
        TriggerTreatmentCampaignRequest(unscheduled_treatment_ids=["nope"])


def test_trigger_treatment_campaign_request_rejects_missing_field():
    with pytest.raises(ValidationError):
        TriggerTreatmentCampaignRequest()


# ---------------------------------------------------------------------------
# ConvertTreatmentRequest -- `partial` defaults to False.
# ---------------------------------------------------------------------------

def test_convert_treatment_request_partial_defaults_to_false():
    request = ConvertTreatmentRequest(appointment_id=uuid.uuid4())
    assert request.partial is False


def test_convert_treatment_request_partial_explicit_true():
    request = ConvertTreatmentRequest(appointment_id=uuid.uuid4(), partial=True)
    assert request.partial is True


def test_convert_treatment_request_rejects_missing_appointment_id():
    with pytest.raises(ValidationError):
        ConvertTreatmentRequest()


# ---------------------------------------------------------------------------
# ConvertTreatmentResponse
# ---------------------------------------------------------------------------

def test_convert_treatment_response_accepts_valid_payload():
    treatment_id = uuid.uuid4()
    appointment_id = uuid.uuid4()
    response = ConvertTreatmentResponse(
        id=treatment_id,
        status=UnscheduledTreatmentStatus.booked,
        appointment_id=appointment_id,
    )
    assert response.id == treatment_id
    assert response.status == UnscheduledTreatmentStatus.booked
    assert response.appointment_id == appointment_id


def test_convert_treatment_response_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        ConvertTreatmentResponse(
            id=uuid.uuid4(), status="unknown", appointment_id=uuid.uuid4()
        )


def test_convert_treatment_response_rejects_missing_appointment_id():
    with pytest.raises(ValidationError):
        ConvertTreatmentResponse(id=uuid.uuid4(), status=UnscheduledTreatmentStatus.booked)
