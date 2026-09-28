"""Unit tests for app/services/outreach/service.py.

These tests exercise OutreachService, ConsentService, and ChannelConfigService
against fake repositories that faithfully implement the documented repository
public surface (app/repositories/outreach/repository.py.md), a fake
ConfigurationService (app/services/rules/service.py.md's get_live contract),
and a stubbed AuditService. The channel adapter factory
(app.services.outreach.channel_gateway.get_channel_adapter) is patched at its
import site inside the service module so both success and failure delivery
paths can be exercised deterministically (in real "test" environment mode the
adapter is a StubAdapter that always succeeds, which cannot exercise the
failure/retry/unreachable paths this spec requires coverage of).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.common.constants import OUTREACH_MAX_RETRY_ATTEMPTS, OUTREACH_RETRY_DELAY_MINUTES
from app.common.exceptions.errors import ChannelDeliveryError
from app.services.outreach.service import ChannelConfigService, ConsentService, OutreachService

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes over the documented repository/service public surfaces
# ---------------------------------------------------------------------------


class FakeConsentRepo:
    def __init__(self):
        self.rows: list[SimpleNamespace] = []

    async def insert(self, db, patient_id, channel, status, source):
        rec = SimpleNamespace(
            id=uuid.uuid4(),
            patient_id=patient_id,
            channel=channel,
            status=status,
            source=source,
            effective_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        self.rows.append(rec)
        return rec

    async def get_latest_by_channel(self, db, patient_id, channel):
        matches = [r for r in self.rows if r.patient_id == patient_id and r.channel == channel]
        return matches[-1] if matches else None

    async def list_latest_all_channels(self, db, patient_id):
        latest: dict[str, SimpleNamespace] = {}
        for r in self.rows:
            if r.patient_id == patient_id:
                latest[r.channel] = r
        return list(latest.values())

    async def list_ledger(self, db, patient_id):
        return [r for r in self.rows if r.patient_id == patient_id]


class FakeMessageRepo:
    def __init__(self):
        self.rows: dict[uuid.UUID, SimpleNamespace] = {}
        self.create_calls: list[dict] = []

    async def create(self, db, **fields):
        fields.setdefault("status", "queued")
        fields.setdefault("retry_count", 0)
        fields.setdefault("next_retry_at", None)
        fields.setdefault("sent_at", None)
        fields.setdefault("template_id", None)
        row = SimpleNamespace(id=uuid.uuid4(), created_at=datetime.now(timezone.utc), **fields)
        self.rows[row.id] = row
        self.create_calls.append(dict(fields))
        return row

    async def get_by_id(self, db, id):
        return self.rows.get(id)

    async def update_status(self, db, id, status, **extra):
        row = self.rows[id]
        row.status = status
        for k, v in extra.items():
            setattr(row, k, v)
        return row

    async def list_retry_due(self, db, as_of):
        return [
            r
            for r in self.rows.values()
            if r.status == "failed" and r.next_retry_at is not None and r.next_retry_at <= as_of
        ]

    async def list_queued_for_patient_channel(self, db, patient_id, channel):
        return [
            r
            for r in self.rows.values()
            if r.patient_id == patient_id and r.channel == channel and r.status in ("queued", "failed")
        ]

    async def list_by_patient(self, db, patient_id, channel=None, status=None):
        return [
            r
            for r in self.rows.values()
            if r.patient_id == patient_id
            and (channel is None or r.channel == channel)
            and (status is None or r.status == status)
        ]


class FakeTemplateRepo:
    def __init__(self):
        self.templates: dict[tuple, SimpleNamespace] = {}

    def seed(self, type_, language, channel, body="Hello"):
        t = SimpleNamespace(id=uuid.uuid4(), type=type_, language=language, channel=channel, body_template=body)
        self.templates[(type_, language, channel)] = t
        return t

    async def get_by_type_language_channel(self, db, type_, language, channel):
        return self.templates.get((type_, language, channel))


class FakeChannelConfigRepo:
    def __init__(self):
        self.configs: dict[str, SimpleNamespace] = {}

    async def upsert(self, db, channel, provider, access_token_ref, configured_by_staff_id):
        cfg = SimpleNamespace(
            id=uuid.uuid4(),
            channel=channel,
            provider=provider,
            status="pending",
            verified_at=None,
            configured_by_staff_id=configured_by_staff_id,
        )
        self.configs[channel] = cfg
        return cfg

    async def get_status(self, db, channel):
        return self.configs.get(channel)

    async def update_status(self, db, channel, status):
        cfg = self.configs.get(channel)
        if cfg is None:
            cfg = SimpleNamespace(
                id=uuid.uuid4(), channel=channel, provider=None, status=status,
                verified_at=None, configured_by_staff_id=None,
            )
            self.configs[channel] = cfg
        else:
            cfg.status = status
        if status == "verified":
            cfg.verified_at = datetime.now(timezone.utc)
        return cfg


class FakeConfigService:
    """Mirrors ConfigurationService.get_live: returns `default` when unconfigured."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.get_live = AsyncMock(side_effect=self._get_live)

    async def _get_live(self, db, name, default=None):
        self.calls.append((name, default))
        return default


def make_audit():
    return SimpleNamespace(record=AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4())))


def make_adapter(success=True, failure_reason=None, provider_message_id="stub-id"):
    result = SimpleNamespace(
        success=success,
        provider_message_id=provider_message_id if success else None,
        failure_reason=failure_reason,
    )
    return SimpleNamespace(send=AsyncMock(return_value=result))


def make_outreach_service():
    message_repo = FakeMessageRepo()
    consent_repo = FakeConsentRepo()
    template_repo = FakeTemplateRepo()
    channel_config_repo = FakeChannelConfigRepo()
    config_service = FakeConfigService()
    audit = make_audit()
    settings = SimpleNamespace(environment="test")
    service = OutreachService(
        message_repo=message_repo,
        consent_repo=consent_repo,
        template_repo=template_repo,
        channel_config_repo=channel_config_repo,
        config_service=config_service,
        audit=audit,
        settings=settings,
    )
    return service, message_repo, consent_repo, template_repo, channel_config_repo, config_service, audit


DB = SimpleNamespace()  # opaque AsyncSession stand-in; fakes never inspect it


# ---------------------------------------------------------------------------
# OutreachService.dispatch
# ---------------------------------------------------------------------------


async def test_dispatch_idempotent_returns_existing_row_unchanged():
    service, message_repo, consent_repo, template_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    existing = SimpleNamespace(
        id=uuid.uuid4(), patient_id=patient_id, campaign_type="confirmation", channel="whatsapp",
        language="en", template_id=None, status="queued", retry_count=0, next_retry_at=None,
        related_entity_type="appointment", related_entity_id=entity_id, sent_at=None,
        created_at=datetime.now(timezone.utc),
    )
    message_repo.rows[existing.id] = existing

    with patch("app.services.outreach.service.get_channel_adapter") as mock_adapter_factory:
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="confirmation", language="en",
            related_entity_type="appointment", related_entity_id=entity_id,
            idempotency_key="dedupe-key",
        )

    assert result.id == existing.id
    assert not message_repo.create_calls
    mock_adapter_factory.assert_not_called()


async def test_dispatch_idempotency_ignores_failed_prior_row():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    entity_id = uuid.uuid4()
    failed_prior = SimpleNamespace(
        id=uuid.uuid4(), patient_id=patient_id, campaign_type="confirmation", channel="whatsapp",
        language="en", template_id=None, status="failed", retry_count=0, next_retry_at=None,
        related_entity_type="appointment", related_entity_id=entity_id, sent_at=None,
        created_at=datetime.now(timezone.utc),
    )
    message_repo.rows[failed_prior.id] = failed_prior
    template_repo.seed("confirmation", "en", "whatsapp")
    await consent_repo.insert(DB, patient_id, "whatsapp", "granted", "staff_manual", )
    channel_config_repo.configs["whatsapp"] = SimpleNamespace(
        id=uuid.uuid4(), channel="whatsapp", provider="meta", status="verified",
        verified_at=datetime.now(timezone.utc), configured_by_staff_id=uuid.uuid4(),
    )

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="confirmation", language="en",
            related_entity_type="appointment", related_entity_id=entity_id,
            idempotency_key="dedupe-key",
        )

    assert result.id != failed_prior.id
    assert len(message_repo.create_calls) == 1


async def test_dispatch_preferred_channel_used_when_granted():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None, preferred_channel="sms",
        )

    assert result.channel == "sms"


async def test_dispatch_preferred_whatsapp_unverified_falls_back_to_alternate():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "whatsapp")
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "whatsapp", "granted", "staff_manual")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    channel_config_repo.configs["whatsapp"] = SimpleNamespace(
        id=uuid.uuid4(), channel="whatsapp", provider="meta", status="pending",
        verified_at=None, configured_by_staff_id=uuid.uuid4(),
    )

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None, preferred_channel="whatsapp",
        )

    assert result.channel != "whatsapp"
    assert result.channel == "sms"


async def test_dispatch_channel_priority_order_without_preference():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    template_repo.seed("recall", "en", "email")
    # No WhatsApp consent at all; consent granted on sms and email -> sms wins (higher priority).
    await consent_repo.insert(DB, patient_id, "email", "granted", "staff_manual")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None,
        )

    assert result.channel == "sms"


async def test_dispatch_no_consent_returns_suppressed_without_sending():
    service, message_repo, consent_repo, template_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()

    with patch("app.services.outreach.service.get_channel_adapter") as mock_adapter_factory:
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None,
        )

    assert result.status == "suppressed"
    mock_adapter_factory.assert_not_called()


async def test_dispatch_template_falls_back_to_english():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    # Only an English template exists; patient's language is Spanish.
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="es",
            related_entity_type=None, related_entity_id=None,
        )

    assert result.status in ("sent", "delivered")
    en_template = template_repo.templates[("recall", "en", "sms")]
    assert result.template_id == en_template.id


async def test_dispatch_send_success_sets_sent_status_and_sent_at():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None,
        )

    assert result.status in ("sent", "delivered")
    assert result.sent_at is not None


async def test_dispatch_send_failure_sets_failed_and_schedules_retry():
    service, message_repo, consent_repo, template_repo, channel_config_repo, config_service, _ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    before = datetime.now(timezone.utc)

    with patch(
        "app.services.outreach.service.get_channel_adapter",
        return_value=make_adapter(success=False, failure_reason="vendor down"),
    ):
        result = await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None,
        )

    assert result.status == "failed"
    assert result.retry_count == 0
    assert result.next_retry_at is not None
    assert result.next_retry_at > before
    assert ("outreach_retry_delay_minutes", OUTREACH_RETRY_DELAY_MINUTES) in config_service.calls


async def test_dispatch_calls_audit_record_with_expected_action():
    service, message_repo, consent_repo, template_repo, channel_config_repo, config_service, audit = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        await service.dispatch(
            DB, patient_id=patient_id, campaign_type="recall", language="en",
            related_entity_type=None, related_entity_id=None,
        )

    audit.record.assert_awaited_once()
    kwargs = audit.record.await_args.kwargs
    assert kwargs.get("action_type") == "outreach.dispatch"
    assert kwargs.get("actor_type") == "ai_agent"


# ---------------------------------------------------------------------------
# OutreachService.retry / mark_unreachable
# ---------------------------------------------------------------------------


async def test_retry_prefers_alternate_channel_over_failed_one():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "whatsapp")
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "whatsapp", "granted", "staff_manual")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    channel_config_repo.configs["whatsapp"] = SimpleNamespace(
        id=uuid.uuid4(), channel="whatsapp", provider="meta", status="verified",
        verified_at=datetime.now(timezone.utc), configured_by_staff_id=uuid.uuid4(),
    )
    failed_msg = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="whatsapp", language="en",
        template_id=None, status="failed", retry_count=0, next_retry_at=datetime.now(timezone.utc),
        related_entity_type=None, related_entity_id=None,
    )

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.retry(DB, failed_msg.id)

    assert result.channel == "sms"
    assert result.status in ("sent", "delivered")


async def test_retry_failure_increments_retry_count_and_reschedules_when_under_max():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    failed_msg = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="sms", language="en",
        template_id=None, status="failed", retry_count=0, next_retry_at=datetime.now(timezone.utc),
        related_entity_type=None, related_entity_id=None,
    )

    with patch(
        "app.services.outreach.service.get_channel_adapter",
        return_value=make_adapter(success=False, failure_reason="still down"),
    ):
        result = await service.retry(DB, failed_msg.id)

    assert result.status != "unreachable"
    assert result.retry_count == 1
    assert result.next_retry_at is not None


async def test_retry_marks_unreachable_when_max_attempts_reached():
    service, message_repo, consent_repo, template_repo, channel_config_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    template_repo.seed("recall", "en", "sms")
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    # retry_count + 1 == OUTREACH_MAX_RETRY_ATTEMPTS (default 3) -> exhausted this attempt.
    failed_msg = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="sms", language="en",
        template_id=None, status="failed", retry_count=OUTREACH_MAX_RETRY_ATTEMPTS - 1,
        next_retry_at=datetime.now(timezone.utc), related_entity_type=None, related_entity_id=None,
    )

    with patch(
        "app.services.outreach.service.get_channel_adapter",
        return_value=make_adapter(success=False, failure_reason="still down"),
    ):
        result = await service.retry(DB, failed_msg.id)

    assert result.status == "unreachable"


async def test_mark_unreachable_sets_status_only():
    service, message_repo, *_ = make_outreach_service()
    patient_id = uuid.uuid4()
    msg = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="sms", language="en",
        template_id=None, status="failed", retry_count=2, next_retry_at=None,
        related_entity_type="waitlist_entry", related_entity_id=uuid.uuid4(),
    )

    result = await service.mark_unreachable(DB, msg.id)

    assert result.status == "unreachable"


# ---------------------------------------------------------------------------
# ConsentService
# ---------------------------------------------------------------------------


def make_consent_service():
    consent_repo = FakeConsentRepo()
    message_repo = FakeMessageRepo()
    audit = make_audit()
    return ConsentService(consent_repo=consent_repo, message_repo=message_repo, audit=audit), consent_repo, message_repo, audit


async def test_grant_inserts_granted_consent_record():
    service, consent_repo, message_repo, audit = make_consent_service()
    patient_id = uuid.uuid4()

    record = await service.grant(DB, patient_id, "sms", "staff_manual", uuid.uuid4())

    assert record.status == "granted"
    assert record.channel == "sms"
    assert len(consent_repo.rows) == 1


async def test_decline_inserts_declined_consent_record():
    service, consent_repo, message_repo, audit = make_consent_service()
    patient_id = uuid.uuid4()

    record = await service.decline(DB, patient_id, "email", "patient_reply", None)

    assert record.status == "declined"
    assert record.channel == "email"


async def test_withdraw_inserts_withdrawn_row_and_halts_queued_messages():
    service, consent_repo, message_repo, audit = make_consent_service()
    patient_id = uuid.uuid4()
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    queued = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="sms", language="en",
        template_id=None, status="queued", retry_count=0, next_retry_at=None,
        related_entity_type=None, related_entity_id=None,
    )

    record, halted_count = await service.withdraw(DB, patient_id, "sms", "keyword_stop")

    assert record.status == "withdrawn"
    assert halted_count == 1
    assert message_repo.rows[queued.id].status == "suppressed"
    # append-only: the prior "granted" row must still be present, unmodified.
    statuses = [r.status for r in consent_repo.rows]
    assert "granted" in statuses and "withdrawn" in statuses


async def test_halt_queued_suppresses_only_matching_channel_messages():
    service, consent_repo, message_repo, audit = make_consent_service()
    patient_id = uuid.uuid4()
    sms_msg = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="sms", language="en",
        template_id=None, status="queued", retry_count=0, next_retry_at=None,
        related_entity_type=None, related_entity_id=None,
    )
    email_msg = await message_repo.create(
        DB, patient_id=patient_id, campaign_type="recall", channel="email", language="en",
        template_id=None, status="queued", retry_count=0, next_retry_at=None,
        related_entity_type=None, related_entity_id=None,
    )

    count = await service.halt_queued(DB, patient_id, "sms")

    assert count == 1
    assert message_repo.rows[sms_msg.id].status == "suppressed"
    assert message_repo.rows[email_msg.id].status == "queued"


async def test_get_ledger_returns_full_chronological_history():
    service, consent_repo, message_repo, audit = make_consent_service()
    patient_id = uuid.uuid4()
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    await consent_repo.insert(DB, patient_id, "sms", "withdrawn", "keyword_stop")

    ledger = await service.get_ledger(DB, patient_id)

    assert len(ledger) == 2
    assert [r.status for r in ledger] == ["granted", "withdrawn"]


async def test_export_ledger_returns_bytes_and_audit_logs_export():
    service, consent_repo, message_repo, audit = make_consent_service()
    patient_id = uuid.uuid4()
    await consent_repo.insert(DB, patient_id, "sms", "granted", "staff_manual")
    actor_id = uuid.uuid4()

    output = await service.export_ledger(DB, patient_id, actor_id)

    assert isinstance(output, bytes)
    audit.record.assert_awaited_once()
    assert audit.record.await_args.kwargs.get("action_type") == "consent.export"


# ---------------------------------------------------------------------------
# ChannelConfigService
# ---------------------------------------------------------------------------


def make_channel_config_service():
    config_repo = FakeChannelConfigRepo()
    audit = make_audit()
    return ChannelConfigService(config_repo=config_repo, audit=audit), config_repo, audit


async def test_configure_always_sets_status_pending():
    service, config_repo, audit = make_channel_config_service()

    result = await service.configure(DB, "whatsapp", "meta_bsp", "***", uuid.uuid4())

    assert result.status == "pending"


async def test_test_send_success_marks_verified():
    service, config_repo, audit = make_channel_config_service()
    await service.configure(DB, "whatsapp", "meta_bsp", "***", uuid.uuid4())

    with patch("app.services.outreach.service.get_channel_adapter", return_value=make_adapter(success=True)):
        result = await service.test_send(DB, "+15551234567", uuid.uuid4())

    assert result.status == "verified"


async def test_test_send_failure_raises_and_stays_pending():
    service, config_repo, audit = make_channel_config_service()
    await service.configure(DB, "whatsapp", "meta_bsp", "***", uuid.uuid4())

    with patch(
        "app.services.outreach.service.get_channel_adapter",
        return_value=make_adapter(success=False, failure_reason="Meta 400"),
    ):
        with pytest.raises(ChannelDeliveryError) as exc_info:
            await service.test_send(DB, "+15551234567", uuid.uuid4())

    assert exc_info.value.status_code == 502
    assert config_repo.configs["whatsapp"].status == "pending"


async def test_update_status_updates_channel_status():
    service, config_repo, audit = make_channel_config_service()
    await service.configure(DB, "sms", "twilio", "***", uuid.uuid4())

    result = await service.update_status(DB, "sms", "verified")

    assert result.status == "verified"
    assert config_repo.configs["sms"].status == "verified"
