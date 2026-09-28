"""Unit tests for app/schemas/outreach/schemas.py.

These are pure Pydantic v2 request/response models for every `/outreach/*`
endpoint. Each model is exercised directly here -- constructed, validated,
and (de)serialized -- without going through any route or service, and
without a database, queue, or storage of any kind.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.common.enums import (
    CampaignType,
    Channel,
    ChannelConfigStatus,
    ConsentStatus,
    Language,
    OutreachMessageStatus,
)
from app.schemas.outreach.schemas import (
    CaptureConsentRequest,
    ConfigureWhatsappRequest,
    ConfigureWhatsappResponse,
    ConsentLedgerItem,
    ConsentLedgerResponse,
    ConsentResponse,
    ManualOutcomeRequest,
    ManualOutcomeResponse,
    OutreachMessageItem,
    OutreachMessageListResponse,
    TestSendWhatsappRequest,
    TestSendWhatsappResponse,
    WithdrawConsentRequest,
    WithdrawConsentResponse,
)


UUID_A = uuid.uuid4()
UUID_B = uuid.uuid4()
NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# OutreachMessageItem / OutreachMessageListResponse
# ---------------------------------------------------------------------------

class TestOutreachMessageItem:
    def test_accepts_all_required_fields_plus_sent_at(self):
        item = OutreachMessageItem(
            id=UUID_A,
            campaign_type=CampaignType.confirmation,
            channel=Channel.whatsapp,
            language=Language.en,
            status=OutreachMessageStatus.sent,
            sent_at=NOW,
        )
        assert item.id == UUID_A
        assert item.campaign_type == CampaignType.confirmation
        assert item.channel == Channel.whatsapp
        assert item.language == Language.en
        assert item.status == OutreachMessageStatus.sent
        assert item.sent_at == NOW

    def test_sent_at_accepts_none_for_an_unsent_message(self):
        item = OutreachMessageItem(
            id=UUID_A,
            campaign_type=CampaignType.recall,
            channel=Channel.sms,
            language=Language.es,
            status=OutreachMessageStatus.queued,
            sent_at=None,
        )
        assert item.sent_at is None

    def test_rejects_invalid_campaign_type_value(self):
        with pytest.raises(ValidationError):
            OutreachMessageItem(
                id=UUID_A,
                campaign_type="not_a_real_campaign_type",
                channel=Channel.sms,
                language=Language.en,
                status=OutreachMessageStatus.queued,
                sent_at=None,
            )

    def test_rejects_invalid_channel_value(self):
        with pytest.raises(ValidationError):
            OutreachMessageItem(
                id=UUID_A,
                campaign_type=CampaignType.recall,
                channel="carrier_pigeon",
                language=Language.en,
                status=OutreachMessageStatus.queued,
                sent_at=None,
            )

    def test_rejects_invalid_status_value(self):
        with pytest.raises(ValidationError):
            OutreachMessageItem(
                id=UUID_A,
                campaign_type=CampaignType.recall,
                channel=Channel.sms,
                language=Language.en,
                status="not_a_real_status",
                sent_at=None,
            )

    def test_id_field_coerces_a_uuid_string(self):
        item = OutreachMessageItem(
            id=str(UUID_A),
            campaign_type=CampaignType.education,
            channel=Channel.email,
            language=Language.pt,
            status=OutreachMessageStatus.delivered,
            sent_at=NOW.isoformat(),
        )
        assert item.id == UUID_A
        assert item.sent_at == NOW

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            OutreachMessageItem(
                campaign_type=CampaignType.recall,
                channel=Channel.sms,
                language=Language.en,
                status=OutreachMessageStatus.queued,
                sent_at=None,
            )


class TestOutreachMessageListResponse:
    def test_accepts_empty_items_list(self):
        resp = OutreachMessageListResponse(items=[])
        assert resp.items == []

    def test_accepts_a_list_of_outreach_message_items(self):
        item = OutreachMessageItem(
            id=UUID_A,
            campaign_type=CampaignType.waitlist_offer,
            channel=Channel.whatsapp,
            language=Language.en,
            status=OutreachMessageStatus.sent,
            sent_at=NOW,
        )
        resp = OutreachMessageListResponse(items=[item])
        assert len(resp.items) == 1
        assert resp.items[0].id == UUID_A

    def test_items_is_required(self):
        with pytest.raises(ValidationError):
            OutreachMessageListResponse()


# ---------------------------------------------------------------------------
# ManualOutcomeRequest / ManualOutcomeResponse
# ---------------------------------------------------------------------------

class TestManualOutcomeRequest:
    def test_notes_defaults_to_none_when_omitted(self):
        req = ManualOutcomeRequest(outcome="patient_confirmed_by_phone")
        assert req.outcome == "patient_confirmed_by_phone"
        assert req.notes is None

    def test_accepts_explicit_notes(self):
        req = ManualOutcomeRequest(outcome="unreachable", notes="tried 3 times")
        assert req.notes == "tried 3 times"

    def test_outcome_is_required(self):
        with pytest.raises(ValidationError):
            ManualOutcomeRequest()


class TestManualOutcomeResponse:
    def test_round_trips_id_and_manual_outcome(self):
        resp = ManualOutcomeResponse(id=UUID_A, manual_outcome="patient_confirmed_by_phone")
        assert resp.id == UUID_A
        assert resp.manual_outcome == "patient_confirmed_by_phone"

    def test_manual_outcome_is_required(self):
        with pytest.raises(ValidationError):
            ManualOutcomeResponse(id=UUID_A)


# ---------------------------------------------------------------------------
# CaptureConsentRequest
# ---------------------------------------------------------------------------

class TestCaptureConsentRequest:
    def test_accepts_granted_status(self):
        req = CaptureConsentRequest(
            patient_id=UUID_A, channel=Channel.whatsapp, status=ConsentStatus.granted
        )
        assert req.status == ConsentStatus.granted
        assert req.source is None

    def test_accepts_declined_status(self):
        req = CaptureConsentRequest(
            patient_id=UUID_A, channel=Channel.sms, status=ConsentStatus.declined, source="ivr"
        )
        assert req.status == ConsentStatus.declined
        assert req.source == "ivr"

    def test_status_field_is_not_narrowed_to_exclude_withdrawn_at_the_schema_level(self):
        # Per this file's spec: the request/decline narrowing (granted/declined
        # only, never withdrawn) is enforced by the service layer, not by the
        # Pydantic field's own type -- so the schema itself must still accept
        # ConsentStatus.withdrawn without raising here.
        req = CaptureConsentRequest(
            patient_id=UUID_A, channel=Channel.whatsapp, status=ConsentStatus.withdrawn
        )
        assert req.status == ConsentStatus.withdrawn

    def test_rejects_invalid_channel(self):
        with pytest.raises(ValidationError):
            CaptureConsentRequest(patient_id=UUID_A, channel="fax", status=ConsentStatus.granted)

    def test_rejects_invalid_patient_id(self):
        with pytest.raises(ValidationError):
            CaptureConsentRequest(
                patient_id="not-a-uuid", channel=Channel.whatsapp, status=ConsentStatus.granted
            )


class TestConsentResponse:
    def test_round_trips_all_fields(self):
        resp = ConsentResponse(
            id=UUID_A, channel=Channel.whatsapp, status=ConsentStatus.granted, effective_at=NOW
        )
        assert resp.id == UUID_A
        assert resp.channel == Channel.whatsapp
        assert resp.status == ConsentStatus.granted
        assert resp.effective_at == NOW

    def test_effective_at_is_required(self):
        with pytest.raises(ValidationError):
            ConsentResponse(id=UUID_A, channel=Channel.whatsapp, status=ConsentStatus.granted)


# ---------------------------------------------------------------------------
# WithdrawConsentRequest / WithdrawConsentResponse
# ---------------------------------------------------------------------------

class TestWithdrawConsentRequest:
    def test_source_defaults_to_none(self):
        req = WithdrawConsentRequest(patient_id=UUID_A, channel=Channel.sms)
        assert req.source is None

    def test_accepts_explicit_source(self):
        req = WithdrawConsentRequest(patient_id=UUID_A, channel=Channel.sms, source="webhook")
        assert req.source == "webhook"

    def test_patient_id_and_channel_are_required(self):
        with pytest.raises(ValidationError):
            WithdrawConsentRequest(channel=Channel.sms)
        with pytest.raises(ValidationError):
            WithdrawConsentRequest(patient_id=UUID_A)


class TestWithdrawConsentResponse:
    def test_round_trips_status_and_halted_count(self):
        resp = WithdrawConsentResponse(
            id=UUID_A, status=ConsentStatus.withdrawn, queued_messages_halted=3
        )
        assert resp.id == UUID_A
        assert resp.status == ConsentStatus.withdrawn
        assert resp.queued_messages_halted == 3

    def test_queued_messages_halted_accepts_zero(self):
        resp = WithdrawConsentResponse(
            id=UUID_A, status=ConsentStatus.withdrawn, queued_messages_halted=0
        )
        assert resp.queued_messages_halted == 0

    def test_queued_messages_halted_is_required(self):
        with pytest.raises(ValidationError):
            WithdrawConsentResponse(id=UUID_A, status=ConsentStatus.withdrawn)


# ---------------------------------------------------------------------------
# ConsentLedgerItem / ConsentLedgerResponse
# ---------------------------------------------------------------------------

class TestConsentLedgerItem:
    def test_round_trips_channel_status_and_effective_at(self):
        item = ConsentLedgerItem(
            channel=Channel.email, status=ConsentStatus.declined, effective_at=NOW
        )
        assert item.channel == Channel.email
        assert item.status == ConsentStatus.declined
        assert item.effective_at == NOW

    def test_has_no_patient_or_message_identifier_field(self):
        # The ledger item is deliberately shaped as channel/status/effective_at
        # only (per its declared public surface) -- passing an id is simply
        # ignored by default Pydantic v2 behaviour, but the fields that DO
        # exist must be exactly these three.
        item = ConsentLedgerItem(
            channel=Channel.email, status=ConsentStatus.declined, effective_at=NOW
        )
        assert set(item.model_fields) == {"channel", "status", "effective_at"}


class TestConsentLedgerResponse:
    def test_accepts_empty_items_list(self):
        resp = ConsentLedgerResponse(items=[])
        assert resp.items == []

    def test_accepts_multiple_ledger_items(self):
        item1 = ConsentLedgerItem(
            channel=Channel.whatsapp, status=ConsentStatus.granted, effective_at=NOW
        )
        item2 = ConsentLedgerItem(
            channel=Channel.whatsapp, status=ConsentStatus.withdrawn, effective_at=NOW
        )
        resp = ConsentLedgerResponse(items=[item1, item2])
        assert [i.status for i in resp.items] == [ConsentStatus.granted, ConsentStatus.withdrawn]


# ---------------------------------------------------------------------------
# ConfigureWhatsappRequest / ConfigureWhatsappResponse
# ---------------------------------------------------------------------------

class TestConfigureWhatsappRequest:
    def test_accepts_all_three_required_string_fields(self):
        req = ConfigureWhatsappRequest(
            bsp_provider="meta", phone_number_id="1234567890", access_token="secret-token"
        )
        assert req.bsp_provider == "meta"
        assert req.phone_number_id == "1234567890"
        assert req.access_token == "secret-token"

    @pytest.mark.parametrize("missing", ["bsp_provider", "phone_number_id", "access_token"])
    def test_each_field_is_required(self, missing):
        fields = {
            "bsp_provider": "meta",
            "phone_number_id": "1234567890",
            "access_token": "secret-token",
        }
        del fields[missing]
        with pytest.raises(ValidationError):
            ConfigureWhatsappRequest(**fields)


class TestConfigureWhatsappResponse:
    def test_round_trips_id_and_status(self):
        resp = ConfigureWhatsappResponse(id=UUID_B, status=ChannelConfigStatus.verified)
        assert resp.id == UUID_B
        assert resp.status == ChannelConfigStatus.verified

    def test_accepts_pending_status(self):
        resp = ConfigureWhatsappResponse(id=UUID_B, status=ChannelConfigStatus.pending)
        assert resp.status == ChannelConfigStatus.pending

    def test_rejects_invalid_status_value(self):
        with pytest.raises(ValidationError):
            ConfigureWhatsappResponse(id=UUID_B, status="not_a_real_status")


# ---------------------------------------------------------------------------
# TestSendWhatsappRequest / TestSendWhatsappResponse
# ---------------------------------------------------------------------------

class TestTestSendWhatsappRequest:
    def test_round_trips_to_phone(self):
        req = TestSendWhatsappRequest(to_phone="+15551234567")
        assert req.to_phone == "+15551234567"

    def test_to_phone_is_required(self):
        with pytest.raises(ValidationError):
            TestSendWhatsappRequest()


class TestTestSendWhatsappResponse:
    def test_accepts_verified_status_the_only_status_a_successful_send_returns(self):
        # architecture.md §5.2's documented BSP-failure path (502) never produces a
        # 200 response body carrying this schema, so the one status this schema is
        # ever populated with on the success path is "verified".
        resp = TestSendWhatsappResponse(status=ChannelConfigStatus.verified)
        assert resp.status == ChannelConfigStatus.verified

    def test_schema_itself_does_not_narrow_the_status_field_to_verified_only(self):
        # The field's declared type is the full ChannelConfigStatus enum, not a
        # Literal["verified"] -- narrowing to "only ever verified" is a runtime/
        # service-layer contract (a 502 on failure, never a 200 body with
        # status="pending"), not something this Pydantic field type enforces.
        resp = TestSendWhatsappResponse(status=ChannelConfigStatus.pending)
        assert resp.status == ChannelConfigStatus.pending

    def test_rejects_invalid_status_value(self):
        with pytest.raises(ValidationError):
            TestSendWhatsappResponse(status="delivered_but_not_a_real_channel_config_status")

    def test_status_is_required(self):
        with pytest.raises(ValidationError):
            TestSendWhatsappResponse()
