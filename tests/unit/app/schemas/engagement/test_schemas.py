"""Unit tests for app/schemas/engagement/schemas.py.

These tests exercise the Pydantic v2 request/response models declared for the
/engagement/* endpoints, driven only from the module's declared public surface
and the spec's behaviour/interaction-contract lines. Enum member values are
never hard-coded: the concrete enum classes used by ``verification_status``
and ``score_status`` are recovered from the models' own ``model_fields``
annotations, so the tests do not need to know (or assume) where those enums
live or what their members are named -- only that the schema declares that
field with *some* enum type.
"""
import uuid
from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas.engagement.schemas import (
    CallInteractionDetailResponse,
    CallInteractionItem,
    CallInteractionListResponse,
    InboundCallRequest,
    InboundCallResponse,
    SendWebchatMessageRequest,
    SendWebchatMessageResponse,
    StartWebchatSessionResponse,
    VerifyIdentityRequest,
    VerifyIdentityResponse,
)


def _first_enum_member(model, field_name):
    """Return an arbitrary valid member of the enum annotated on a field.

    This lets the tests construct valid instances without importing or
    naming the enum class directly (its home module is not part of this
    file's spec).
    """
    enum_cls = model.model_fields[field_name].annotation
    return next(iter(enum_cls))


# ---------------------------------------------------------------------------
# StartWebchatSessionResponse
# ---------------------------------------------------------------------------

class TestStartWebchatSessionResponse:
    def test_constructs_with_id_token_and_status(self):
        session_id = uuid.uuid4()
        status_member = _first_enum_member(StartWebchatSessionResponse, "verification_status")
        model = StartWebchatSessionResponse(
            id=session_id,
            visitor_token="tok-abc123",
            verification_status=status_member,
        )
        assert model.id == session_id
        assert model.visitor_token == "tok-abc123"
        assert model.verification_status == status_member

    def test_id_is_coerced_to_uuid_from_string(self):
        raw = "12345678-1234-5678-1234-567812345678"
        status_member = _first_enum_member(StartWebchatSessionResponse, "verification_status")
        model = StartWebchatSessionResponse(
            id=raw, visitor_token="tok", verification_status=status_member
        )
        assert model.id == uuid.UUID(raw)

    def test_visitor_token_is_a_plain_string_field(self):
        """The token that a client echoes back on subsequent calls is a plain
        str -- it is not itself a bearer/auth token type, and the schema
        exposes no separate auth-token field alongside it (Auth: none)."""
        assert "visitor_token" in StartWebchatSessionResponse.model_fields
        assert StartWebchatSessionResponse.model_fields["visitor_token"].annotation is str
        # No second, separate bearer/auth-token field exists on this schema.
        assert set(StartWebchatSessionResponse.model_fields) == {
            "id",
            "visitor_token",
            "verification_status",
        }

    def test_missing_visitor_token_is_rejected(self):
        status_member = _first_enum_member(StartWebchatSessionResponse, "verification_status")
        with pytest.raises(ValidationError):
            StartWebchatSessionResponse(id=uuid.uuid4(), verification_status=status_member)

    def test_invalid_verification_status_is_rejected(self):
        with pytest.raises(ValidationError):
            StartWebchatSessionResponse(
                id=uuid.uuid4(),
                visitor_token="tok",
                verification_status="not-a-real-status",
            )


# ---------------------------------------------------------------------------
# SendWebchatMessageRequest / Response
# ---------------------------------------------------------------------------

class TestSendWebchatMessageRequest:
    def test_constructs_with_body(self):
        model = SendWebchatMessageRequest(body="Hello, I need an appointment")
        assert model.body == "Hello, I need an appointment"

    def test_missing_body_is_rejected(self):
        with pytest.raises(ValidationError):
            SendWebchatMessageRequest()


class TestSendWebchatMessageResponse:
    def test_constructs_with_reply_and_verification_flag(self):
        model = SendWebchatMessageResponse(reply="Sure, let me check.", requires_verification=False)
        assert model.reply == "Sure, let me check."
        assert model.requires_verification is False

    def test_requires_verification_true_is_preserved(self):
        model = SendWebchatMessageResponse(
            reply="Please verify your identity first.", requires_verification=True
        )
        assert model.requires_verification is True

    def test_missing_required_fields_is_rejected(self):
        with pytest.raises(ValidationError):
            SendWebchatMessageResponse(reply="hi")


# ---------------------------------------------------------------------------
# VerifyIdentityRequest / Response
# ---------------------------------------------------------------------------

class TestVerifyIdentityRequest:
    def test_constructs_with_names_and_dob(self):
        model = VerifyIdentityRequest(first_name="Jane", last_name="Doe", dob=date(1990, 1, 1))
        assert model.first_name == "Jane"
        assert model.last_name == "Doe"
        assert model.dob == date(1990, 1, 1)

    def test_dob_iso_string_is_coerced_to_date(self):
        model = VerifyIdentityRequest(first_name="Jane", last_name="Doe", dob="1990-01-01")
        assert model.dob == date(1990, 1, 1)

    def test_invalid_dob_is_rejected(self):
        with pytest.raises(ValidationError):
            VerifyIdentityRequest(first_name="Jane", last_name="Doe", dob="not-a-date")

    def test_missing_required_field_is_rejected(self):
        with pytest.raises(ValidationError):
            VerifyIdentityRequest(first_name="Jane", dob=date(1990, 1, 1))


class TestVerifyIdentityResponse:
    def test_constructs_only_on_successful_match(self):
        """FR-E11.2/US51: this response shape represents a successful match
        only -- patient_id and verification_status are both required."""
        patient_id = uuid.uuid4()
        status_member = _first_enum_member(VerifyIdentityResponse, "verification_status")
        model = VerifyIdentityResponse(patient_id=patient_id, verification_status=status_member)
        assert model.patient_id == patient_id
        assert model.verification_status == status_member

    def test_patient_id_is_required_no_null_patient_id_shape(self):
        """There is no separate 200-with-null-patient_id shape: omitting
        patient_id must fail validation rather than silently defaulting."""
        status_member = _first_enum_member(VerifyIdentityResponse, "verification_status")
        with pytest.raises(ValidationError):
            VerifyIdentityResponse(verification_status=status_member)

    def test_patient_id_none_is_rejected(self):
        status_member = _first_enum_member(VerifyIdentityResponse, "verification_status")
        with pytest.raises(ValidationError):
            VerifyIdentityResponse(patient_id=None, verification_status=status_member)


# ---------------------------------------------------------------------------
# InboundCallRequest / Response
# ---------------------------------------------------------------------------

class TestInboundCallRequest:
    def test_constructs_with_caller_phone_and_call_sid(self):
        model = InboundCallRequest(caller_phone="+15551234567", call_sid="CA123abc")
        assert model.caller_phone == "+15551234567"
        assert model.call_sid == "CA123abc"

    def test_missing_call_sid_is_rejected(self):
        with pytest.raises(ValidationError):
            InboundCallRequest(caller_phone="+15551234567")


class TestInboundCallResponse:
    def test_constructs_with_call_interaction_id_and_routing(self):
        call_id = uuid.uuid4()
        model = InboundCallResponse(call_interaction_id=call_id, routing="ai_voice")
        assert model.call_interaction_id == call_id
        assert model.routing == "ai_voice"

    @pytest.mark.parametrize("routing", ["ai_voice", "staff", "callback_queue"])
    def test_documented_routing_values_are_accepted(self, routing):
        model = InboundCallResponse(call_interaction_id=uuid.uuid4(), routing=routing)
        assert model.routing == routing

    def test_missing_routing_is_rejected(self):
        with pytest.raises(ValidationError):
            InboundCallResponse(call_interaction_id=uuid.uuid4())


# ---------------------------------------------------------------------------
# CallInteractionItem / CallInteractionListResponse
# ---------------------------------------------------------------------------

class TestCallInteractionItem:
    def test_constructs_with_all_fields_populated(self):
        item_id = uuid.uuid4()
        model = CallInteractionItem(
            id=item_id,
            caller_phone="+15551234567",
            scenario_type="appointment_booking",
            outcome="booked",
            sentiment_score=0.75,
        )
        assert model.id == item_id
        assert model.caller_phone == "+15551234567"
        assert model.scenario_type == "appointment_booking"
        assert model.outcome == "booked"
        assert model.sentiment_score == 0.75

    def test_nullable_fields_accept_explicit_none(self):
        model = CallInteractionItem(
            id=uuid.uuid4(),
            caller_phone="+15551234567",
            scenario_type=None,
            outcome=None,
            sentiment_score=None,
        )
        assert model.scenario_type is None
        assert model.outcome is None
        assert model.sentiment_score is None

    def test_missing_caller_phone_is_rejected(self):
        with pytest.raises(ValidationError):
            CallInteractionItem(
                id=uuid.uuid4(), scenario_type=None, outcome=None, sentiment_score=None
            )


class TestCallInteractionListResponse:
    def test_wraps_items_in_a_list(self):
        item = CallInteractionItem(
            id=uuid.uuid4(),
            caller_phone="+15551234567",
            scenario_type=None,
            outcome=None,
            sentiment_score=None,
        )
        model = CallInteractionListResponse(items=[item])
        assert model.items == [item]

    def test_empty_items_list_is_accepted(self):
        model = CallInteractionListResponse(items=[])
        assert model.items == []

    def test_missing_items_is_rejected(self):
        with pytest.raises(ValidationError):
            CallInteractionListResponse()


# ---------------------------------------------------------------------------
# CallInteractionDetailResponse
# ---------------------------------------------------------------------------

class TestCallInteractionDetailResponse:
    def test_constructs_with_all_fields(self):
        detail_id = uuid.uuid4()
        status_member = _first_enum_member(CallInteractionDetailResponse, "score_status")
        model = CallInteractionDetailResponse(
            id=detail_id,
            sentiment_score=0.5,
            quality_score=0.9,
            score_status=status_member,
            transcript_excerpt="Patient asked about rescheduling.",
        )
        assert model.id == detail_id
        assert model.sentiment_score == 0.5
        assert model.quality_score == 0.9
        assert model.score_status == status_member
        assert model.transcript_excerpt == "Patient asked about rescheduling."

    def test_score_status_is_required(self):
        """score_status has no `| None` in its declared type, unlike the
        sentiment/quality scores and the excerpt -- omitting it must fail."""
        with pytest.raises(ValidationError):
            CallInteractionDetailResponse(
                id=uuid.uuid4(),
                sentiment_score=None,
                quality_score=None,
                transcript_excerpt=None,
            )

    def test_nullable_score_and_excerpt_fields_accept_none(self):
        status_member = _first_enum_member(CallInteractionDetailResponse, "score_status")
        model = CallInteractionDetailResponse(
            id=uuid.uuid4(),
            sentiment_score=None,
            quality_score=None,
            score_status=status_member,
            transcript_excerpt=None,
        )
        assert model.sentiment_score is None
        assert model.quality_score is None
        assert model.transcript_excerpt is None

    def test_field_is_named_transcript_excerpt_not_transcript(self):
        """DR002: the field is deliberately a short, de-identified excerpt,
        never the raw full transcript -- so the schema must expose
        `transcript_excerpt` and must NOT expose a `transcript` field."""
        fields = CallInteractionDetailResponse.model_fields
        assert "transcript_excerpt" in fields
        assert "transcript" not in fields

    def test_invalid_score_status_is_rejected(self):
        with pytest.raises(ValidationError):
            CallInteractionDetailResponse(
                id=uuid.uuid4(),
                sentiment_score=None,
                quality_score=None,
                score_status="not-a-real-status",
                transcript_excerpt=None,
            )
