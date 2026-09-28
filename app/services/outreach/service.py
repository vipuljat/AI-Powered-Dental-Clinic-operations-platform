"""Business logic for the `outreach` module: the shared dispatch engine every
other module's campaign trigger calls into (architecture.md §4.2/§5.3
outreach module).

``OutreachService`` (dispatch/retry/mark_unreachable) is the **only** send
path any other module ever calls (Interaction contract) — confirmation
(scheduling), waitlist-offer (waitlist), recall, treatment re-engagement, and
education campaigns all route through ``OutreachService.dispatch`` rather
than touching ``channel_gateway.py`` or the ``outreach_messages`` table
directly. This module never imports another module's service (Responsibility)
— it only ever depends on its own repositories, ``app.services.rules.service
.ConfigurationService`` (the live-config read path), and
``app.services.console.service.AuditService`` (the audit write path every
other module's service is, in turn, free to depend on).

``ConsentService`` owns the append-only consent ledger (FR-E6.6) and the
"halt everything queued the instant consent is withdrawn" guarantee
(FR-E6.5). ``ChannelConfigService`` owns the delivery_team-only outbound
channel configuration/verification workflow (architecture.md §5.2's
``configure``/``test-send``/``update-status`` calls).
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from app.common.constants import OUTREACH_MAX_RETRY_ATTEMPTS, OUTREACH_RETRY_DELAY_MINUTES
from app.common.enums import (
    ActorType,
    Channel,
    ChannelConfigStatus,
    ConsentStatus,
    Language,
    OutreachMessageStatus,
)
from app.common.exceptions.errors import ChannelDeliveryError, NotFoundError
from app.common.utils import now_utc, to_iso8601
from app.core.config import get_settings
from app.services.outreach.channel_gateway import DeliveryResult, get_channel_adapter

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.config import Settings
    from app.models.outreach.models import ChannelConfiguration, ConsentRecord, OutreachMessage
    from app.repositories.outreach.repository import (
        ChannelConfigurationRepository,
        ConsentRepository,
        MessageTemplateRepository,
        OutreachMessageRepository,
    )
    from app.services.console.service import AuditService
    from app.services.rules.service import ConfigurationService

__all__ = ["ChannelConfigService", "ConsentService", "OutreachService"]


# --- Documented default channel priority ordering (open question) ----------
# WhatsApp > SMS > Email > Voice: the fallback order `dispatch`/`retry` walk
# when no `preferred_channel` is given (or it isn't usable) — the spec names
# this exact ordering explicitly rather than leaving it to be invented here.
_CHANNEL_PRIORITY: list[str] = [
    Channel.whatsapp.value,
    Channel.sms.value,
    Channel.email.value,
    Channel.voice.value,
]


def _enum_value(value: object) -> object:
    """Normalizes an attribute read back off an ORM row that may be either a
    Python ``Enum`` member (the native in-process shape) or its plain string
    value (the shape a SQLite-backed test row can come back as) — both are
    treated identically throughout this file, the same accommodation
    ``services/console/service.py`` and ``services/rules/service.py`` make
    for their own enum-typed columns.
    """
    return value.value if hasattr(value, "value") else value


# --- Minimal, dependency-free PDF rendering ---------------------------------
# No PDF library is declared in pyproject.toml's dependency set, so
# `ConsentService.export_ledger`'s "renders the full ledger as a PDF byte
# stream" (T82 AC) is satisfied by hand-assembling a minimal, syntactically
# valid single-page PDF (header, a handful of indirect objects, an xref
# table, and a trailer) rather than reaching for a vendor library this
# project never installed.


def _pdf_escape(text: str) -> str:
    """Escapes the three characters PDF's literal-string syntax reserves
    (``\\``, ``(``, ``)``) so arbitrary ledger text can never break out of a
    ``(...) Tj`` operator.
    """
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_minimal_pdf(lines: list[str]) -> bytes:
    """Assembles a single-page, US-Letter, Helvetica-12 PDF whose content
    stream prints `lines` top-to-bottom — a complete, independently-openable
    PDF byte stream, not a fragment.
    """
    content_ops = ["BT", "/F1 12 Tf", "72 750 Td"]
    for index, line in enumerate(lines):
        if index > 0:
            content_ops.append("0 -16 Td")
        content_ops.append(f"({_pdf_escape(line)}) Tj")
    content_ops.append("ET")
    content_stream = "\n".join(content_ops).encode("latin-1", errors="replace")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n"
        + content_stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    buffer = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer += f"{index} 0 obj\n".encode("ascii")
        buffer += obj
        buffer += b"\nendobj\n"

    xref_offset = len(buffer)
    buffer += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    buffer += b"0000000000 65535 f \n"
    for offset in offsets:
        buffer += f"{offset:010d} 00000 n \n".encode("ascii")
    buffer += b"trailer\n"
    buffer += f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("ascii")
    buffer += b"startxref\n"
    buffer += f"{xref_offset}\n".encode("ascii")
    buffer += b"%%EOF"
    return bytes(buffer)


def _render_consent_ledger_pdf(patient_id: "UUID", records: "list[ConsentRecord]") -> bytes:
    """FR-E6.6/T82: renders the full, chronological, append-only ledger — one
    line per row, no row omitted or reordered.
    """
    lines = [f"Consent Ledger - Patient {patient_id}", ""]
    if not records:
        lines.append("No consent records on file.")
    for record in records:
        channel = _enum_value(record.channel)
        status = _enum_value(record.status)
        effective_at = to_iso8601(record.effective_at) if record.effective_at is not None else ""
        source_suffix = f"  (source: {record.source})" if record.source else ""
        lines.append(f"{effective_at}  {channel}  {status}{source_suffix}")
    return _build_minimal_pdf(lines)


class OutreachService:
    """FR-E6.1-FR-E6.4/FR-E6.7: the shared dispatch engine (Responsibility).
    Never imports `scheduling`/`waitlist`/`recall`/`education` services —
    every campaign trigger in those modules calls `dispatch` instead.
    """

    def __init__(
        self,
        message_repo: "OutreachMessageRepository",
        consent_repo: "ConsentRepository",
        template_repo: "MessageTemplateRepository",
        channel_config_repo: "ChannelConfigurationRepository",
        config_service: "ConfigurationService",
        audit: "AuditService",
        settings: "Settings",
    ) -> None:
        self._message_repo = message_repo
        self._consent_repo = consent_repo
        self._template_repo = template_repo
        self._channel_config_repo = channel_config_repo
        self._config_service = config_service
        self._audit = audit
        self._settings = settings

    async def _channel_usable(self, db: "AsyncSession", patient_id: "UUID", channel: str) -> bool:
        """FR-E6.1/FR-E6.7: a channel is usable only with a `"granted"`
        `ConsentRecord` (checked at send-time, T72 AC) and, for WhatsApp
        specifically, a `"verified"` `ChannelConfiguration` — an unverified
        WhatsApp configuration is treated as not usable, falling back to
        another consented channel (FR-E6.7).
        """
        consent = await self._consent_repo.get_latest_by_channel(db, patient_id, channel)
        if consent is None or _enum_value(consent.status) != ConsentStatus.granted.value:
            return False
        if channel == Channel.whatsapp.value:
            config = await self._channel_config_repo.get_status(db, channel)
            if config is None or _enum_value(config.status) != ChannelConfigStatus.verified.value:
                return False
        return True

    async def _resolve_channel(
        self,
        db: "AsyncSession",
        patient_id: "UUID",
        preferred_channel: str | None,
        exclude: frozenset[str] = frozenset(),
    ) -> str | None:
        """FR-E6.1/FR-E6.7: `preferred_channel` wins only if it is itself
        usable; otherwise (or if excluded, per `retry`'s alternate-channel
        rule) falls back to the highest-priority usable channel in
        `_CHANNEL_PRIORITY`. Returns `None` when no channel is usable at all
        (US21 Alternate Flow) — never raises.
        """
        if preferred_channel and preferred_channel not in exclude:
            if await self._channel_usable(db, patient_id, preferred_channel):
                return preferred_channel
        for channel in _CHANNEL_PRIORITY:
            if channel in exclude:
                continue
            if await self._channel_usable(db, patient_id, channel):
                return channel
        return None

    async def _resolve_template(self, db: "AsyncSession", campaign_type: str, language: str, channel: str):
        """FR-E6.2/T71: falls back to `language="en"` only when no template
        exists for the patient's own language — never the reverse.
        """
        template = await self._template_repo.get_by_type_language_channel(db, campaign_type, language, channel)
        if template is None and language != Language.en.value:
            template = await self._template_repo.get_by_type_language_channel(
                db, campaign_type, Language.en.value, channel
            )
        return template

    async def _find_existing_dispatch(
        self,
        db: "AsyncSession",
        patient_id: "UUID",
        related_entity_type: str,
        related_entity_id: "UUID",
        campaign_type: str,
    ):
        """Watch out: idempotency here is `(related_entity_type,
        related_entity_id, campaign_type)`-keyed, not the HTTP
        `Idempotency-Key` header mechanism — no dedicated repository lookup
        exists for that triple, so every message row for this patient is
        listed via the frozen `OutreachMessageRepository.list_by_patient`
        (channel/status unfiltered) and scanned in-memory for a matching,
        non-`failed` row. A worker redelivering the same
        confirmation/waitlist-offer event is thereby prevented from
        double-sending it.
        """
        candidates = await self._message_repo.list_by_patient(db, patient_id, channel=None, status=None)
        for candidate in candidates:
            if (
                candidate.related_entity_type == related_entity_type
                and candidate.related_entity_id == related_entity_id
                and _enum_value(candidate.campaign_type) == campaign_type
                and _enum_value(candidate.status) != OutreachMessageStatus.failed.value
            ):
                return candidate
        return None

    async def dispatch(
        self,
        db: "AsyncSession",
        patient_id: "UUID",
        campaign_type: str,
        language: str,
        related_entity_type: str | None,
        related_entity_id: "UUID | None",
        preferred_channel: str | None = None,
        idempotency_key: str | None = None,
    ) -> "OutreachMessage":
        # Step 1 (Watch out): idempotency replay check.
        if idempotency_key and related_entity_type is not None and related_entity_id is not None:
            existing = await self._find_existing_dispatch(
                db, patient_id, related_entity_type, related_entity_id, campaign_type
            )
            if existing is not None:
                return existing

        # Step 2 (FR-E6.1/FR-E6.7): resolve a consented+usable channel.
        channel = await self._resolve_channel(db, patient_id, preferred_channel)

        if channel is None:
            # US21 Alternate Flow: no consented channel at all — a
            # `status="suppressed"` row is created and returned immediately,
            # never an exception, so the caller (e.g.
            # `app/workers/outreach_retry.py`'s confirmation path) can flag
            # the originating entity for manual follow-up.
            placeholder_channel = preferred_channel or _CHANNEL_PRIORITY[0]
            message = await self._message_repo.create(
                db,
                patient_id=patient_id,
                campaign_type=campaign_type,
                channel=placeholder_channel,
                language=language,
                template_id=None,
                status=OutreachMessageStatus.suppressed.value,
                related_entity_type=related_entity_type,
                related_entity_id=related_entity_id,
            )
            await self._audit.record(
                db,
                actor_staff_id=None,
                actor_type=ActorType.ai_agent,
                action_type="outreach.dispatch",
                entity_type="outreach_message",
                entity_id=message.id,
                original_payload={
                    "campaign_type": campaign_type,
                    "channel": None,
                    "status": OutreachMessageStatus.suppressed.value,
                },
            )
            return message

        # Step 3 (FR-E6.2/T71): resolve template, falling back to English.
        template = await self._resolve_template(db, campaign_type, language, channel)
        body = (
            template.body_template
            if template is not None
            else f"You have a new {campaign_type} update from your clinic. Please contact us for details."
        )

        # Step 4: create the queued row, then attempt the send.
        message = await self._message_repo.create(
            db,
            patient_id=patient_id,
            campaign_type=campaign_type,
            channel=channel,
            language=language,
            template_id=template.id if template is not None else None,
            status=OutreachMessageStatus.queued.value,
            related_entity_type=related_entity_type,
            related_entity_id=related_entity_id,
        )

        channel_config = await self._channel_config_repo.get_status(db, channel)
        adapter = get_channel_adapter(channel, self._settings, channel_config)

        # Watch out: this constructor injects no patient-contact repository
        # (per this file's frozen Public surface), so the real phone/email
        # address a production adapter would dial cannot be resolved here.
        # `str(patient_id)` is used as the adapter's `to` argument — every
        # adapter in test mode resolves to `StubAdapter` regardless of this
        # value, and this is documented in the task summary as a real gap
        # against the frozen constructor signature.
        to_address = str(patient_id)

        try:
            result: DeliveryResult = await adapter.send(to_address, body)
        except ChannelDeliveryError as exc:
            result = DeliveryResult(success=False, provider_message_id=None, failure_reason=str(exc))

        if result.success:
            message = await self._message_repo.update_status(
                db, message.id, OutreachMessageStatus.sent.value, sent_at=now_utc()
            )
        else:
            delay_minutes = await self._config_service.get_live(
                db, "outreach_retry_delay_minutes", default=OUTREACH_RETRY_DELAY_MINUTES
            )
            message = await self._message_repo.update_status(
                db,
                message.id,
                OutreachMessageStatus.failed.value,
                retry_count=0,
                next_retry_at=now_utc() + timedelta(minutes=delay_minutes),
            )

        # Step 5.
        await self._audit.record(
            db,
            actor_staff_id=None,
            actor_type=ActorType.ai_agent,
            action_type="outreach.dispatch",
            entity_type="outreach_message",
            entity_id=message.id,
            original_payload={
                "campaign_type": campaign_type,
                "channel": channel,
                "status": _enum_value(message.status),
            },
        )
        return message

    async def retry(self, db: "AsyncSession", message_id: "UUID") -> "OutreachMessage":
        """FR-E6.3/FR-E6.4/T75: prefers an alternate consented channel over
        repeating the one that just failed; falls back to repeating the
        original channel only if no different consented channel exists.
        Once `max_attempts` is (or would be) exhausted, hands off to
        `mark_unreachable` instead of scheduling yet another attempt
        (US29/US30).
        """
        message = await self._message_repo.get_by_id(db, message_id)
        if message is None:
            raise NotFoundError("Outreach message not found.")

        failed_channel = _enum_value(message.channel)
        alternate = await self._resolve_channel(
            db, message.patient_id, None, exclude=frozenset({failed_channel})
        )
        if alternate is None and await self._channel_usable(db, message.patient_id, failed_channel):
            # T75 AC: repeating the same channel is only a last resort, when
            # nothing else is consented.
            alternate = failed_channel

        if alternate is None:
            # No consented channel remains at all — nothing further to try.
            return await self.mark_unreachable(db, message_id)

        campaign_type = _enum_value(message.campaign_type)
        language = _enum_value(message.language)

        template = await self._resolve_template(db, campaign_type, language, alternate)
        body = (
            template.body_template
            if template is not None
            else f"You have a new {campaign_type} update from your clinic. Please contact us for details."
        )

        channel_config = await self._channel_config_repo.get_status(db, alternate)
        adapter = get_channel_adapter(alternate, self._settings, channel_config)
        to_address = str(message.patient_id)

        try:
            result: DeliveryResult = await adapter.send(to_address, body)
        except ChannelDeliveryError as exc:
            result = DeliveryResult(success=False, provider_message_id=None, failure_reason=str(exc))

        if result.success:
            return await self._message_repo.update_status(
                db,
                message_id,
                OutreachMessageStatus.sent.value,
                channel=alternate,
                template_id=template.id if template is not None else None,
                sent_at=now_utc(),
            )

        max_attempts = await self._config_service.get_live(
            db, "outreach_max_retry_attempts", default=OUTREACH_MAX_RETRY_ATTEMPTS
        )
        new_retry_count = message.retry_count + 1
        if new_retry_count >= max_attempts:
            return await self.mark_unreachable(db, message_id)

        delay_minutes = await self._config_service.get_live(
            db, "outreach_retry_delay_minutes", default=OUTREACH_RETRY_DELAY_MINUTES
        )
        return await self._message_repo.update_status(
            db,
            message_id,
            OutreachMessageStatus.failed.value,
            channel=alternate,
            retry_count=new_retry_count,
            next_retry_at=now_utc() + timedelta(minutes=delay_minutes),
        )

    async def mark_unreachable(self, db: "AsyncSession", message_id: "UUID") -> "OutreachMessage":
        """US30 Alternate Flow: only the `outreach_messages` row itself is
        closed out — the underlying `waitlist_entry`/`recall_schedule` (or
        any other related entity) is never touched/archived from here.
        """
        return await self._message_repo.update_status(db, message_id, OutreachMessageStatus.unreachable.value)


class ConsentService:
    """FR-E6.5/FR-E6.6: the append-only consent ledger and the
    "immediately halt everything queued" guarantee.
    """

    def __init__(
        self,
        consent_repo: "ConsentRepository",
        message_repo: "OutreachMessageRepository",
        audit: "AuditService",
    ) -> None:
        self._consent_repo = consent_repo
        self._message_repo = message_repo
        self._audit = audit

    async def grant(
        self, db: "AsyncSession", patient_id: "UUID", channel: str, source: str | None, actor_id: "UUID | None"
    ) -> "ConsentRecord":
        """FR-E6.6: inserts a brand-new row — never updates a prior one."""
        record = await self._consent_repo.insert(db, patient_id, channel, ConsentStatus.granted.value, source)
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff if actor_id is not None else ActorType.system,
            action_type="consent.grant",
            entity_type="consent_record",
            entity_id=record.id,
            override_payload={"channel": channel, "source": source},
        )
        return record

    async def decline(
        self, db: "AsyncSession", patient_id: "UUID", channel: str, source: str | None, actor_id: "UUID | None"
    ) -> "ConsentRecord":
        """FR-E6.6: inserts a brand-new row — never updates a prior one."""
        record = await self._consent_repo.insert(db, patient_id, channel, ConsentStatus.declined.value, source)
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff if actor_id is not None else ActorType.system,
            action_type="consent.decline",
            entity_type="consent_record",
            entity_id=record.id,
            override_payload={"channel": channel, "source": source},
        )
        return record

    async def withdraw(
        self, db: "AsyncSession", patient_id: "UUID", channel: str, source: str | None
    ) -> "tuple[ConsentRecord, int]":
        """FR-E6.5/FR-E6.6/US31/US32: inserts a NEW `"withdrawn"` row (never
        updates a prior one), then immediately (synchronously, in this same
        call — not deferred to a worker sweep) halts every queued/failed
        message on this exact channel via `halt_queued`. No `actor_id`
        parameter is accepted (Public surface) — this path is reachable from
        the unauthenticated, patient-facing `POST /outreach/consent/withdraw`
        webhook (project_rules.auth), so the audit entry is attributed to
        `ActorType.system` rather than a staff member.
        """
        record = await self._consent_repo.insert(db, patient_id, channel, ConsentStatus.withdrawn.value, source)
        await self._audit.record(
            db,
            actor_staff_id=None,
            actor_type=ActorType.system,
            action_type="consent.withdraw",
            entity_type="consent_record",
            entity_id=record.id,
            override_payload={"channel": channel, "source": source},
        )
        halted = await self.halt_queued(db, patient_id, channel)
        return record, halted

    async def halt_queued(self, db: "AsyncSession", patient_id: "UUID", channel: str) -> int:
        """FR-E6.5/US31 Alternate Flow: suppresses every queued/failed row on
        this one channel; other channels for the same patient are left
        completely untouched.
        """
        queued = await self._message_repo.list_queued_for_patient_channel(db, patient_id, channel)
        for message in queued:
            await self._message_repo.update_status(db, message.id, OutreachMessageStatus.suppressed.value)
        return len(queued)

    async def get_ledger(self, db: "AsyncSession", patient_id: "UUID") -> "list[ConsentRecord]":
        """FR-E6.6: every row ever inserted, chronological, unfiltered."""
        return await self._consent_repo.list_ledger(db, patient_id)

    async def export_ledger(self, db: "AsyncSession", patient_id: "UUID", actor_id: "UUID") -> bytes:
        """T82 AC: the export request is itself audit-logged, in addition to
        rendering the full ledger as a PDF byte stream.
        """
        records = await self.get_ledger(db, patient_id)
        pdf_bytes = _render_consent_ledger_pdf(patient_id, records)
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="consent.export",
            entity_type="patient",
            entity_id=patient_id,
            override_payload={"record_count": len(records)},
        )
        return pdf_bytes


class ChannelConfigService:
    """delivery_team-only outbound channel configuration/verification
    workflow (architecture.md §5.2).
    """

    def __init__(self, config_repo: "ChannelConfigurationRepository", audit: "AuditService") -> None:
        self._config_repo = config_repo
        self._audit = audit

    async def configure(
        self, db: "AsyncSession", channel: str, provider: str, access_token: str, actor_id: "UUID"
    ) -> "ChannelConfiguration":
        """`status="pending"` always — only `test_send`'s verification can
        flip it to `"verified"`. Watch out: the frozen
        `ChannelConfigurationRepository.upsert`/`ChannelConfiguration` ORM
        model persist no raw secret value at all (no `access_token`/
        `access_token_ref` column exists) — the real secret belongs in the
        deployment's secret store (architecture.md §9), never in this table
        or in an audit-log payload. No secrets-manager client is injected
        into this constructor (Public surface), so a stable, non-secret
        pointer name is passed through as `access_token_ref` purely to keep
        the call's intent explicit; `access_token` itself is read here and
        then discarded, never logged or persisted by this method.
        """
        access_token_ref = f"secretsmanager://outreach/channel-config/{channel}"
        config = await self._config_repo.upsert(
            db,
            channel=channel,
            provider=provider,
            access_token_ref=access_token_ref,
            configured_by_staff_id=actor_id,
        )
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="channel_config.configure",
            entity_type="channel_configuration",
            entity_id=config.id,
            override_payload={"channel": channel, "provider": provider, "status": _enum_value(config.status)},
        )
        return config

    async def test_send(self, db: "AsyncSession", to_phone: str, actor_id: "UUID") -> "ChannelConfiguration":
        """architecture.md §5.2 `POST /outreach/channels/whatsapp/test-send`:
        the only channel this endpoint (and therefore this method) applies
        to is WhatsApp (`schemas/outreach/schemas.py`'s
        `TestSendWhatsappRequest`/`Response` are the only configure/test-send
        schemas declared). On success, flips the configuration to
        `"verified"`; on failure the row STAYS `"pending"` and a
        `ChannelDeliveryError` (502) is raised rather than a degraded 200.
        """
        channel = Channel.whatsapp.value
        config = await self._config_repo.get_status(db, channel)
        if config is None:
            raise NotFoundError("WhatsApp channel has not been configured yet.")

        settings = get_settings()
        adapter = get_channel_adapter(channel, settings, config)
        result: DeliveryResult = await adapter.send(to_phone, "This is a test message from your clinic.")
        if not result.success:
            raise ChannelDeliveryError(
                "WhatsApp test message delivery failed.",
                details=[{"failure_reason": result.failure_reason}],
            )

        updated = await self.update_status(db, channel, ChannelConfigStatus.verified.value)
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="channel_config.test_send",
            entity_type="channel_configuration",
            entity_id=updated.id,
            override_payload={"channel": channel, "status": ChannelConfigStatus.verified.value},
        )
        return updated

    async def update_status(self, db: "AsyncSession", channel: str, status: str) -> "ChannelConfiguration":
        return await self._config_repo.update_status(db, channel, status)
