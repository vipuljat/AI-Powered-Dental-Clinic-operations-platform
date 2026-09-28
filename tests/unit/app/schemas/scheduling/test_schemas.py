"""Unit tests for app/schemas/scheduling/schemas.py.

These Pydantic v2 models are the whole request/response contract for every
`/scheduling/*` route (architecture.md §5.2). Per this file's own spec:

- `CreateAppointmentRequest` documents no 422 response, but its fields alone
  (all required, strictly typed `UUID`/`datetime`) are what triggers FastAPI's
  standard `RequestValidationError` -> 422 for a malformed body; the 409
  conflict path is a service-layer concern this file does not model.
- `CancelAppointmentRequest.reason_code` is deliberately a free `str` with no
  enum constraint -- non-emptiness is a service-layer rule, not a schema one,
  so an empty string must be *accepted* here.
- `LockResponse` must stay wire-identical to the same-named model in
  `schemas/patients/schemas.py`; that cross-file identity is not observable
  from this file's own public surface, so it is not asserted here.

Tests exercise construction, coercion, optionality, and rejection behaviour
only -- nothing about HTTP wiring, which lives in routes.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.common.enums import (
    AppointmentStatus,
    ConfirmationStatus,
    RiskLevel,
    ScoreSource,
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
    SlotOption,
)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


def _dt(hour: int = 9) -> datetime:
    return datetime(2026, 10, 1, hour, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# SlotOption
# ---------------------------------------------------------------------------

def test_slot_option_accepts_full_valid_payload():
    provider_id = _uuid()
    chair_id = _uuid()
    slot = SlotOption(
        provider_id=provider_id,
        chair_id=chair_id,
        start=_dt(9),
        end=_dt(10),
    )
    assert slot.provider_id == provider_id
    assert slot.chair_id == chair_id
    assert slot.start == _dt(9)
    assert slot.end == _dt(10)


def test_slot_option_coerces_uuid_and_datetime_strings():
    provider_id = str(_uuid())
    slot = SlotOption(
        provider_id=provider_id,
        chair_id=str(_uuid()),
        start="2026-10-01T09:00:00+00:00",
        end="2026-10-01T10:00:00+00:00",
    )
    assert isinstance(slot.provider_id, uuid.UUID)
    assert str(slot.provider_id) == provider_id
    assert isinstance(slot.start, datetime)
    assert slot.start == _dt(9)


def test_slot_option_rejects_missing_field():
    with pytest.raises(ValidationError):
        SlotOption(provider_id=_uuid(), chair_id=_uuid(), start=_dt(9))


def test_slot_option_rejects_invalid_uuid():
    with pytest.raises(ValidationError):
        SlotOption(
            provider_id="not-a-uuid",
            chair_id=_uuid(),
            start=_dt(9),
            end=_dt(10),
        )


# ---------------------------------------------------------------------------
# ProposeSlotsResponse
# ---------------------------------------------------------------------------

def test_propose_slots_response_holds_slots_and_alternatives():
    slot = SlotOption(provider_id=_uuid(), chair_id=_uuid(), start=_dt(9), end=_dt(10))
    alt = SlotOption(provider_id=_uuid(), chair_id=_uuid(), start=_dt(11), end=_dt(12))
    response = ProposeSlotsResponse(slots=[slot], nearest_alternatives=[alt])
    assert response.slots == [slot]
    assert response.nearest_alternatives == [alt]


def test_propose_slots_response_allows_empty_lists():
    response = ProposeSlotsResponse(slots=[], nearest_alternatives=[])
    assert response.slots == []
    assert response.nearest_alternatives == []


def test_propose_slots_response_rejects_non_slot_option_items():
    with pytest.raises(ValidationError):
        ProposeSlotsResponse(slots=[{"provider_id": "x"}], nearest_alternatives=[])


# ---------------------------------------------------------------------------
# CreateAppointmentRequest
# ---------------------------------------------------------------------------

def _create_request_kwargs(**overrides):
    kwargs = dict(
        patient_id=_uuid(),
        provider_id=_uuid(),
        chair_id=_uuid(),
        appointment_type="cleaning",
        scheduled_start=_dt(9),
        scheduled_end=_dt(10),
    )
    kwargs.update(overrides)
    return kwargs


def test_create_appointment_request_accepts_full_valid_payload():
    request = CreateAppointmentRequest(**_create_request_kwargs())
    assert request.appointment_type == "cleaning"
    assert request.scheduled_start == _dt(9)
    assert request.scheduled_end == _dt(10)


@pytest.mark.parametrize(
    "missing_field",
    [
        "patient_id",
        "provider_id",
        "chair_id",
        "appointment_type",
        "scheduled_start",
        "scheduled_end",
    ],
)
def test_create_appointment_request_rejects_missing_required_field(missing_field):
    kwargs = _create_request_kwargs()
    del kwargs[missing_field]
    with pytest.raises(ValidationError) as exc_info:
        CreateAppointmentRequest(**kwargs)
    assert any(err["loc"] == (missing_field,) for err in exc_info.value.errors())


def test_create_appointment_request_rejects_malformed_uuid_type():
    with pytest.raises(ValidationError):
        CreateAppointmentRequest(**_create_request_kwargs(patient_id="not-a-uuid"))


def test_create_appointment_request_rejects_malformed_datetime_type():
    with pytest.raises(ValidationError):
        CreateAppointmentRequest(**_create_request_kwargs(scheduled_start="not-a-datetime"))


# ---------------------------------------------------------------------------
# AppointmentResponse
# ---------------------------------------------------------------------------

def test_appointment_response_accepts_full_valid_payload():
    appointment_id = _uuid()
    response = AppointmentResponse(
        id=appointment_id,
        status=AppointmentStatus.booked,
        risk_flag=RiskLevel.high,
        confirmation_status=ConfirmationStatus.sent,
    )
    assert response.id == appointment_id
    assert response.status == AppointmentStatus.booked
    assert response.risk_flag == RiskLevel.high
    assert response.confirmation_status == ConfirmationStatus.sent


def test_appointment_response_allows_risk_flag_and_confirmation_status_none():
    response = AppointmentResponse(
        id=_uuid(),
        status=AppointmentStatus.cancelled,
        risk_flag=None,
        confirmation_status=None,
    )
    assert response.risk_flag is None
    assert response.confirmation_status is None


def test_appointment_response_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        AppointmentResponse(
            id=_uuid(),
            status="not_a_status",
            risk_flag=None,
            confirmation_status=None,
        )


# ---------------------------------------------------------------------------
# RescheduleAppointmentRequest / Response
# ---------------------------------------------------------------------------

def test_reschedule_appointment_request_accepts_full_valid_payload():
    provider_id = _uuid()
    chair_id = _uuid()
    request = RescheduleAppointmentRequest(
        provider_id=provider_id,
        chair_id=chair_id,
        scheduled_start=_dt(9),
        scheduled_end=_dt(10),
    )
    assert request.provider_id == provider_id
    assert request.chair_id == chair_id


def test_reschedule_appointment_request_rejects_missing_scheduled_end():
    with pytest.raises(ValidationError):
        RescheduleAppointmentRequest(
            provider_id=_uuid(),
            chair_id=_uuid(),
            scheduled_start=_dt(9),
        )


def test_reschedule_appointment_response_carries_source_metadata_dict():
    appointment_id = _uuid()
    response = RescheduleAppointmentResponse(
        id=appointment_id,
        status=AppointmentStatus.rescheduled,
        source_metadata={"rescheduled_from": str(_uuid())},
    )
    assert response.id == appointment_id
    assert response.status == AppointmentStatus.rescheduled
    assert response.source_metadata == {"rescheduled_from": response.source_metadata["rescheduled_from"]}


def test_reschedule_appointment_response_defaults_missing_source_metadata_to_error():
    with pytest.raises(ValidationError):
        RescheduleAppointmentResponse(id=_uuid(), status=AppointmentStatus.rescheduled)


def test_reschedule_appointment_response_rejects_non_dict_source_metadata():
    with pytest.raises(ValidationError):
        RescheduleAppointmentResponse(
            id=_uuid(),
            status=AppointmentStatus.rescheduled,
            source_metadata="not-a-dict",
        )


# ---------------------------------------------------------------------------
# CancelAppointmentRequest / Response
# ---------------------------------------------------------------------------

def test_cancel_appointment_request_accepts_any_non_empty_string():
    request = CancelAppointmentRequest(reason_code="patient_requested")
    assert request.reason_code == "patient_requested"


def test_cancel_appointment_request_accepts_arbitrary_unenumerated_string():
    # No enum constraint at the schema level -- FR-E4.5 leaves the closed set
    # of valid codes undefined, so any string value is schema-valid here.
    request = CancelAppointmentRequest(reason_code="some_code_not_in_any_fixed_list")
    assert request.reason_code == "some_code_not_in_any_fixed_list"


def test_cancel_appointment_request_accepts_empty_string():
    # Non-emptiness is enforced by SchedulingService.cancel, not this schema.
    request = CancelAppointmentRequest(reason_code="")
    assert request.reason_code == ""


def test_cancel_appointment_request_rejects_missing_reason_code():
    with pytest.raises(ValidationError):
        CancelAppointmentRequest()


def test_cancel_appointment_response_accepts_full_valid_payload():
    appointment_id = _uuid()
    response = CancelAppointmentResponse(
        id=appointment_id,
        status=AppointmentStatus.cancelled,
        is_late_cancellation=True,
    )
    assert response.id == appointment_id
    assert response.status == AppointmentStatus.cancelled
    assert response.is_late_cancellation is True


def test_cancel_appointment_response_rejects_non_bool_is_late_cancellation():
    with pytest.raises(ValidationError):
        CancelAppointmentResponse(
            id=_uuid(),
            status=AppointmentStatus.cancelled,
            is_late_cancellation="not-a-bool",
        )


# ---------------------------------------------------------------------------
# AppointmentDetailResponse
# ---------------------------------------------------------------------------

def _detail_kwargs(**overrides):
    kwargs = dict(
        id=_uuid(),
        patient_id=_uuid(),
        status=AppointmentStatus.completed,
        risk_flag=RiskLevel.medium,
        risk_score_source=ScoreSource.ml,
        confirmation_status=ConfirmationStatus.failed,
    )
    kwargs.update(overrides)
    return kwargs


def test_appointment_detail_response_accepts_full_valid_payload():
    response = AppointmentDetailResponse(**_detail_kwargs())
    assert response.status == AppointmentStatus.completed
    assert response.risk_flag == RiskLevel.medium
    assert response.risk_score_source == ScoreSource.ml
    assert response.confirmation_status == ConfirmationStatus.failed


def test_appointment_detail_response_allows_all_optional_fields_none():
    response = AppointmentDetailResponse(
        **_detail_kwargs(risk_flag=None, risk_score_source=None, confirmation_status=None)
    )
    assert response.risk_flag is None
    assert response.risk_score_source is None
    assert response.confirmation_status is None


def test_appointment_detail_response_rejects_invalid_risk_score_source():
    with pytest.raises(ValidationError):
        AppointmentDetailResponse(**_detail_kwargs(risk_score_source="not_a_source"))


def test_appointment_detail_response_rejects_missing_patient_id():
    kwargs = _detail_kwargs()
    del kwargs["patient_id"]
    with pytest.raises(ValidationError):
        AppointmentDetailResponse(**kwargs)


# ---------------------------------------------------------------------------
# LockResponse
# ---------------------------------------------------------------------------

def test_lock_response_accepts_full_valid_payload():
    expires_at = _dt(12)
    response = LockResponse(locked_by="staff-123", expires_at=expires_at)
    assert response.locked_by == "staff-123"
    assert response.expires_at == expires_at


def test_lock_response_rejects_missing_expires_at():
    with pytest.raises(ValidationError):
        LockResponse(locked_by="staff-123")


def test_lock_response_rejects_non_datetime_expires_at():
    with pytest.raises(ValidationError):
        LockResponse(locked_by="staff-123", expires_at="not-a-datetime")
