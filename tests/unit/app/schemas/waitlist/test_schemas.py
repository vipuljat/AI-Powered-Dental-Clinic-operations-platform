"""Unit tests for app/schemas/waitlist/schemas.py.

These Pydantic v2 models are the whole request/response contract for every
`/waitlist/*` route (architecture.md §5.2) -- `services/waitlist/service.py`'s
return values are shaped to match these field names directly, per this
file's own Interaction contract, with no separate agreement anywhere else.

Tests exercise construction, validation, and serialization behaviour
declared by the spec: field types/optionality, `IdleChairStatus` /
`IdleChairSource` / `Urgency` / `WaitlistEntryStatus` / `WaitlistOfferStatus`
enum membership, and the `FlagIdleChairResponse` "same shape serves both
201-new and 200-merged status codes, distinguished only by `merged: bool`"
behaviour from architecture.md §5.2.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from app.common.enums import (
    IdleChairSource,
    IdleChairStatus,
    Urgency,
    WaitlistEntryStatus,
    WaitlistOfferStatus,
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


UTC_NOW = datetime(2026, 9, 27, 10, 0, 0, tzinfo=timezone.utc)
UTC_LATER = datetime(2026, 9, 27, 11, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# FlagIdleChairRequest
# ---------------------------------------------------------------------------

def _flag_request_kwargs(**overrides):
    kwargs = dict(
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        slot_start=UTC_NOW,
        slot_end=UTC_LATER,
    )
    kwargs.update(overrides)
    return kwargs


def test_flag_idle_chair_request_accepts_full_valid_payload():
    request = FlagIdleChairRequest(**_flag_request_kwargs())
    assert isinstance(request.provider_id, uuid.UUID)
    assert isinstance(request.chair_id, uuid.UUID)
    assert request.slot_start == UTC_NOW
    assert request.slot_end == UTC_LATER


def test_flag_idle_chair_request_coerces_uuid_strings():
    provider_id = str(uuid.uuid4())
    request = FlagIdleChairRequest(**_flag_request_kwargs(provider_id=provider_id))
    assert isinstance(request.provider_id, uuid.UUID)
    assert str(request.provider_id) == provider_id


def test_flag_idle_chair_request_rejects_missing_provider_id():
    kwargs = _flag_request_kwargs()
    del kwargs["provider_id"]
    with pytest.raises(ValidationError):
        FlagIdleChairRequest(**kwargs)


def test_flag_idle_chair_request_rejects_missing_chair_id():
    kwargs = _flag_request_kwargs()
    del kwargs["chair_id"]
    with pytest.raises(ValidationError):
        FlagIdleChairRequest(**kwargs)


def test_flag_idle_chair_request_rejects_missing_slot_start():
    kwargs = _flag_request_kwargs()
    del kwargs["slot_start"]
    with pytest.raises(ValidationError):
        FlagIdleChairRequest(**kwargs)


def test_flag_idle_chair_request_rejects_invalid_uuid():
    with pytest.raises(ValidationError):
        FlagIdleChairRequest(**_flag_request_kwargs(provider_id="not-a-uuid"))


# ---------------------------------------------------------------------------
# FlagIdleChairResponse -- same shape serves both 201 (new) and 200 (merged).
# ---------------------------------------------------------------------------

def test_flag_idle_chair_response_new_alert_shape_has_merged_false():
    alert_id = uuid.uuid4()
    response = FlagIdleChairResponse(
        id=alert_id, status=IdleChairStatus.open, merged=False
    )
    assert response.id == alert_id
    assert response.status == IdleChairStatus.open
    assert response.merged is False


def test_flag_idle_chair_response_merged_alert_shape_has_merged_true():
    # architecture.md §5.2: 200 (merged) uses the identical response model,
    # distinguished only by merged=True -- no extra/different fields.
    alert_id = uuid.uuid4()
    response = FlagIdleChairResponse(
        id=alert_id, status=IdleChairStatus.open, merged=True
    )
    assert response.id == alert_id
    assert response.status == IdleChairStatus.open
    assert response.merged is True


def test_flag_idle_chair_response_new_and_merged_have_identical_field_set():
    new = FlagIdleChairResponse(id=uuid.uuid4(), status=IdleChairStatus.open, merged=False)
    merged = FlagIdleChairResponse(id=uuid.uuid4(), status=IdleChairStatus.open, merged=True)
    assert set(new.model_dump().keys()) == set(merged.model_dump().keys())


def test_flag_idle_chair_response_rejects_missing_merged():
    with pytest.raises(ValidationError):
        FlagIdleChairResponse(id=uuid.uuid4(), status=IdleChairStatus.open)


def test_flag_idle_chair_response_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        FlagIdleChairResponse(id=uuid.uuid4(), status="bogus_status", merged=False)


def test_flag_idle_chair_response_status_serializes_to_plain_string():
    response = FlagIdleChairResponse(
        id=uuid.uuid4(), status=IdleChairStatus.filled, merged=True
    )
    dumped = response.model_dump(mode="json")
    assert dumped["status"] == "filled"
    assert dumped["merged"] is True


# ---------------------------------------------------------------------------
# IdleChairAlertItem / IdleChairAlertListResponse
# ---------------------------------------------------------------------------

def _alert_item_kwargs(**overrides):
    kwargs = dict(
        id=uuid.uuid4(),
        provider_id=uuid.uuid4(),
        slot_start=UTC_NOW,
        source=IdleChairSource.auto_detected,
        status=IdleChairStatus.open,
    )
    kwargs.update(overrides)
    return kwargs


def test_idle_chair_alert_item_accepts_full_valid_payload():
    item = IdleChairAlertItem(**_alert_item_kwargs())
    assert item.source == IdleChairSource.auto_detected
    assert item.status == IdleChairStatus.open


def test_idle_chair_alert_item_accepts_manual_flag_source():
    item = IdleChairAlertItem(**_alert_item_kwargs(source=IdleChairSource.manual_flag))
    assert item.source == IdleChairSource.manual_flag


def test_idle_chair_alert_item_rejects_invalid_source_value():
    with pytest.raises(ValidationError):
        IdleChairAlertItem(**_alert_item_kwargs(source="not_a_source"))


def test_idle_chair_alert_item_rejects_missing_provider_id():
    kwargs = _alert_item_kwargs()
    del kwargs["provider_id"]
    with pytest.raises(ValidationError):
        IdleChairAlertItem(**kwargs)


def test_idle_chair_alert_list_response_wraps_list_of_items():
    item = IdleChairAlertItem(**_alert_item_kwargs())
    response = IdleChairAlertListResponse(items=[item])
    assert response.items == [item]


def test_idle_chair_alert_list_response_accepts_empty_list():
    response = IdleChairAlertListResponse(items=[])
    assert response.items == []


def test_idle_chair_alert_list_response_rejects_missing_items():
    with pytest.raises(ValidationError):
        IdleChairAlertListResponse()


# ---------------------------------------------------------------------------
# AddWaitlistEntryRequest
# ---------------------------------------------------------------------------

def test_add_waitlist_entry_request_accepts_full_valid_payload():
    request = AddWaitlistEntryRequest(
        patient_id=uuid.uuid4(),
        desired_provider_id=uuid.uuid4(),
        desired_timeframe_start=date(2026, 10, 1),
        desired_timeframe_end=date(2026, 10, 15),
        urgency=Urgency.high,
    )
    assert request.urgency == Urgency.high
    assert request.desired_timeframe_start == date(2026, 10, 1)


def test_add_waitlist_entry_request_optional_fields_default_to_none():
    request = AddWaitlistEntryRequest(patient_id=uuid.uuid4(), urgency=Urgency.low)
    assert request.desired_provider_id is None
    assert request.desired_timeframe_start is None
    assert request.desired_timeframe_end is None


def test_add_waitlist_entry_request_rejects_missing_patient_id():
    with pytest.raises(ValidationError):
        AddWaitlistEntryRequest(urgency=Urgency.low)


def test_add_waitlist_entry_request_rejects_missing_urgency():
    with pytest.raises(ValidationError):
        AddWaitlistEntryRequest(patient_id=uuid.uuid4())


def test_add_waitlist_entry_request_rejects_invalid_urgency_value():
    with pytest.raises(ValidationError):
        AddWaitlistEntryRequest(patient_id=uuid.uuid4(), urgency="not_a_level")


# ---------------------------------------------------------------------------
# WaitlistEntryResponse
# ---------------------------------------------------------------------------

def test_waitlist_entry_response_accepts_valid_payload():
    entry_id = uuid.uuid4()
    response = WaitlistEntryResponse(
        id=entry_id, priority_score=0.87, status=WaitlistEntryStatus.active
    )
    assert response.id == entry_id
    assert response.priority_score == 0.87
    assert response.status == WaitlistEntryStatus.active


def test_waitlist_entry_response_coerces_int_priority_score_to_float():
    response = WaitlistEntryResponse(
        id=uuid.uuid4(), priority_score=1, status=WaitlistEntryStatus.active
    )
    assert response.priority_score == 1.0
    assert isinstance(response.priority_score, float)


def test_waitlist_entry_response_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        WaitlistEntryResponse(id=uuid.uuid4(), priority_score=0.5, status="nope")


def test_waitlist_entry_response_rejects_missing_priority_score():
    with pytest.raises(ValidationError):
        WaitlistEntryResponse(id=uuid.uuid4(), status=WaitlistEntryStatus.active)


# ---------------------------------------------------------------------------
# WaitlistEntryRankedItem / WaitlistEntryListResponse
# ---------------------------------------------------------------------------

def _ranked_item_kwargs(**overrides):
    kwargs = dict(
        id=uuid.uuid4(),
        patient_id=uuid.uuid4(),
        priority_score=0.5,
        rank=1,
        status=WaitlistEntryStatus.offered,
    )
    kwargs.update(overrides)
    return kwargs


def test_waitlist_entry_ranked_item_accepts_full_valid_payload():
    item = WaitlistEntryRankedItem(**_ranked_item_kwargs())
    assert item.rank == 1
    assert item.status == WaitlistEntryStatus.offered


def test_waitlist_entry_ranked_item_rejects_missing_rank():
    kwargs = _ranked_item_kwargs()
    del kwargs["rank"]
    with pytest.raises(ValidationError):
        WaitlistEntryRankedItem(**kwargs)


def test_waitlist_entry_ranked_item_rejects_non_int_rank():
    with pytest.raises(ValidationError):
        WaitlistEntryRankedItem(**_ranked_item_kwargs(rank="first"))


def test_waitlist_entry_list_response_wraps_list_of_items():
    item = WaitlistEntryRankedItem(**_ranked_item_kwargs())
    response = WaitlistEntryListResponse(items=[item])
    assert response.items == [item]


def test_waitlist_entry_list_response_accepts_empty_list():
    response = WaitlistEntryListResponse(items=[])
    assert response.items == []


# ---------------------------------------------------------------------------
# RespondToOfferRequest / RespondToOfferResponse
# ---------------------------------------------------------------------------

def test_respond_to_offer_request_accepts_accept_value():
    request = RespondToOfferRequest(response="accept")
    assert request.response == "accept"


def test_respond_to_offer_request_accepts_decline_value():
    request = RespondToOfferRequest(response="decline")
    assert request.response == "decline"


def test_respond_to_offer_request_rejects_missing_response():
    with pytest.raises(ValidationError):
        RespondToOfferRequest()


def test_respond_to_offer_response_accepted_carries_appointment_id():
    offer_id = uuid.uuid4()
    appointment_id = uuid.uuid4()
    response = RespondToOfferResponse(
        id=offer_id, status=WaitlistOfferStatus.accepted, appointment_id=appointment_id
    )
    assert response.status == WaitlistOfferStatus.accepted
    assert response.appointment_id == appointment_id


def test_respond_to_offer_response_declined_allows_appointment_id_none():
    response = RespondToOfferResponse(
        id=uuid.uuid4(), status=WaitlistOfferStatus.declined, appointment_id=None
    )
    assert response.status == WaitlistOfferStatus.declined
    assert response.appointment_id is None


def test_respond_to_offer_response_rejects_missing_appointment_id_field():
    # appointment_id has no default -- it is required (nullable, not optional).
    with pytest.raises(ValidationError):
        RespondToOfferResponse(id=uuid.uuid4(), status=WaitlistOfferStatus.declined)


def test_respond_to_offer_response_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        RespondToOfferResponse(id=uuid.uuid4(), status="unknown", appointment_id=None)


# ---------------------------------------------------------------------------
# RecoveryOfferStatus / RecoveryStatusResponse
# ---------------------------------------------------------------------------

def test_recovery_offer_status_accepts_valid_payload():
    entry_id = uuid.uuid4()
    status = RecoveryOfferStatus(
        waitlist_entry_id=entry_id,
        status=WaitlistOfferStatus.pending,
        offered_at=UTC_NOW,
    )
    assert status.waitlist_entry_id == entry_id
    assert status.status == WaitlistOfferStatus.pending
    assert status.offered_at == UTC_NOW


def test_recovery_offer_status_rejects_missing_offered_at():
    with pytest.raises(ValidationError):
        RecoveryOfferStatus(
            waitlist_entry_id=uuid.uuid4(), status=WaitlistOfferStatus.pending
        )


def test_recovery_offer_status_rejects_invalid_status_value():
    with pytest.raises(ValidationError):
        RecoveryOfferStatus(
            waitlist_entry_id=uuid.uuid4(), status="bogus", offered_at=UTC_NOW
        )


def test_recovery_status_response_accepts_valid_payload_with_offers():
    alert_id = uuid.uuid4()
    offer = RecoveryOfferStatus(
        waitlist_entry_id=uuid.uuid4(),
        status=WaitlistOfferStatus.accepted,
        offered_at=UTC_NOW,
    )
    response = RecoveryStatusResponse(alert_id=alert_id, offers=[offer], final_status="filled")
    assert response.alert_id == alert_id
    assert response.offers == [offer]
    assert response.final_status == "filled"


def test_recovery_status_response_accepts_empty_offers_and_open_final_status():
    response = RecoveryStatusResponse(alert_id=uuid.uuid4(), offers=[], final_status="open")
    assert response.offers == []
    assert response.final_status == "open"


def test_recovery_status_response_accepts_exhausted_final_status():
    response = RecoveryStatusResponse(
        alert_id=uuid.uuid4(), offers=[], final_status="exhausted"
    )
    assert response.final_status == "exhausted"


def test_recovery_status_response_rejects_missing_final_status():
    with pytest.raises(ValidationError):
        RecoveryStatusResponse(alert_id=uuid.uuid4(), offers=[])


def test_recovery_status_response_rejects_missing_offers():
    with pytest.raises(ValidationError):
        RecoveryStatusResponse(alert_id=uuid.uuid4(), final_status="open")
