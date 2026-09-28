"""Data-access classes over ``consent_records``, ``outreach_messages``,
``message_templates``, ``channel_configurations`` (architecture.md §4.2
outreach module).

Interaction contract: every class here is used only from
``services/outreach/service.py`` (``OutreachService``, ``ConsentService``,
``ChannelConfigService``) -- with two documented cross-module exceptions per
the console layering rule: ``services/console/service.py``'s
``OverrideService`` depends directly on ``OutreachMessageRepository`` (never
on ``services/outreach/service.py``), and ``services/waitlist/service.py``'s
``WaitlistService.match_candidates`` depends directly on
``ConsentRepository.get_latest_by_channel`` for its consent filter -- a
cross-module repository dependency, acceptable since outreach's own service
never depends back on waitlist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.common.constants import OUTREACH_MAX_RETRY_ATTEMPTS
from app.common.enums import ChannelConfigStatus, OutreachMessageStatus
from app.common.utils import now_utc
from app.models.outreach.models import (
    ChannelConfiguration,
    ConsentRecord,
    MessageTemplate,
    OutreachMessage,
)


def _normalize_aware_utc(obj: object, *fields: str) -> None:
    """Coerce the named datetime attributes on ``obj`` to tz-aware UTC.

    SQLite (used per project_rules.testing's ENVIRONMENT=test pivot) does
    not retain a ``DateTime(timezone=True)`` column's UTC offset the way
    Postgres does -- a value round-tripped through it comes back naive.
    Every timestamp in this tree is UTC (project_rules "all timestamps are
    timezone-aware UTC"), so a naive value read back from the DB is
    re-stamped as UTC before it is handed to a caller.
    """
    for field in fields:
        value = getattr(obj, field, None)
        if value is not None and value.tzinfo is None:
            setattr(obj, field, value.replace(tzinfo=timezone.utc))


class ConsentRepository:
    """Read/insert access over the append-only ``consent_records`` table.

    FR-E6.6: no ``update``/``delete`` method is exposed here -- every state
    change (grant/withdraw/decline) is a brand-new row, so the full
    chronological history is always reconstructable and never mutated.
    """

    async def insert(
        self,
        db: AsyncSession,
        patient_id: UUID,
        channel: str,
        status: str,
        source: str | None,
    ) -> ConsentRecord:
        record = ConsentRecord(
            patient_id=patient_id,
            channel=channel,
            status=status,
            source=source,
            effective_at=now_utc(),
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        _normalize_aware_utc(record, "effective_at", "created_at")
        return record

    async def get_latest_by_channel(
        self, db: AsyncSession, patient_id: UUID, channel: str
    ) -> ConsentRecord | None:
        """Latest row for ``(patient_id, channel)`` -- equivalent to
        ``SELECT DISTINCT ON (channel) ... ORDER BY channel, effective_at
        DESC`` narrowed to a single channel, expressed portably as an
        ``ORDER BY ... LIMIT 1`` so it runs unmodified on both Postgres and
        the SQLite test engine (``DISTINCT ON`` is Postgres-only syntax).
        """
        result = await db.execute(
            select(ConsentRecord)
            .where(
                ConsentRecord.patient_id == patient_id,
                ConsentRecord.channel == channel,
            )
            .order_by(ConsentRecord.effective_at.desc())
            .limit(1)
        )
        record = result.scalars().first()
        if record is not None:
            _normalize_aware_utc(record, "effective_at", "created_at")
        return record

    async def list_latest_all_channels(
        self, db: AsyncSession, patient_id: UUID
    ) -> list[ConsentRecord]:
        """Latest row per channel for a patient -- the ``SELECT DISTINCT ON
        (channel) ... ORDER BY channel, effective_at DESC`` semantics,
        expressed with a ``ROW_NUMBER() OVER (PARTITION BY channel ORDER BY
        effective_at DESC)`` window function instead of Postgres-only
        ``DISTINCT ON`` so the same query runs against the SQLite test
        engine too (window functions are supported by both).
        """
        ranked = (
            select(
                ConsentRecord,
                func.row_number()
                .over(
                    partition_by=ConsentRecord.channel,
                    order_by=ConsentRecord.effective_at.desc(),
                )
                .label("rn"),
            )
            .where(ConsentRecord.patient_id == patient_id)
            .subquery()
        )
        ranked_consent = aliased(ConsentRecord, ranked)
        result = await db.execute(select(ranked_consent).where(ranked.c.rn == 1))
        records = list(result.scalars().all())
        for record in records:
            _normalize_aware_utc(record, "effective_at", "created_at")
        return records

    async def list_ledger(self, db: AsyncSession, patient_id: UUID) -> list[ConsentRecord]:
        """FR-E6.6: full chronological history, every row ever inserted for
        this patient, no filtering/deletion.
        """
        result = await db.execute(
            select(ConsentRecord)
            .where(ConsentRecord.patient_id == patient_id)
            .order_by(ConsentRecord.effective_at.asc())
        )
        records = list(result.scalars().all())
        for record in records:
            _normalize_aware_utc(record, "effective_at", "created_at")
        return records


class OutreachMessageRepository:
    """Read/write access over ``outreach_messages``."""

    _DATETIME_FIELDS = ("next_retry_at", "sent_at", "created_at")

    async def create(self, db: AsyncSession, **fields) -> OutreachMessage:
        message = OutreachMessage(**fields)
        db.add(message)
        await db.commit()
        await db.refresh(message)
        _normalize_aware_utc(message, *self._DATETIME_FIELDS)
        return message

    async def get_by_id(self, db: AsyncSession, id: UUID) -> OutreachMessage | None:
        result = await db.execute(select(OutreachMessage).where(OutreachMessage.id == id))
        message = result.scalars().first()
        if message is not None:
            _normalize_aware_utc(message, *self._DATETIME_FIELDS)
        return message

    async def update_status(
        self, db: AsyncSession, id: UUID, status: str, **extra_fields
    ) -> OutreachMessage:
        result = await db.execute(select(OutreachMessage).where(OutreachMessage.id == id))
        message = result.scalar_one()
        message.status = status
        for key, value in extra_fields.items():
            setattr(message, key, value)
        await db.commit()
        await db.refresh(message)
        _normalize_aware_utc(message, *self._DATETIME_FIELDS)
        return message

    async def list_retry_due(self, db: AsyncSession, as_of: datetime) -> list[OutreachMessage]:
        """``WHERE status='failed' AND next_retry_at <= as_of AND
        retry_count < OUTREACH_MAX_RETRY_ATTEMPTS`` -- the query
        ``app/workers/outreach_retry.py`` polls to pick up failed
        dispatches due for another attempt.
        """
        result = await db.execute(
            select(OutreachMessage).where(
                OutreachMessage.status == OutreachMessageStatus.failed,
                OutreachMessage.next_retry_at <= as_of,
                OutreachMessage.retry_count < OUTREACH_MAX_RETRY_ATTEMPTS,
            )
        )
        messages = list(result.scalars().all())
        for message in messages:
            _normalize_aware_utc(message, *self._DATETIME_FIELDS)
        return messages

    async def list_queued_for_patient_channel(
        self, db: AsyncSession, patient_id: UUID, channel: str
    ) -> list[OutreachMessage]:
        """FR-E6.5: ``ConsentService.halt_queued`` runs this the instant a
        withdrawal is recorded to "immediately halt all queued and
        scheduled outreach on that channel" (FR-E6.5).
        """
        result = await db.execute(
            select(OutreachMessage).where(
                OutreachMessage.patient_id == patient_id,
                OutreachMessage.channel == channel,
                OutreachMessage.status.in_(
                    [OutreachMessageStatus.queued, OutreachMessageStatus.failed]
                ),
            )
        )
        messages = list(result.scalars().all())
        for message in messages:
            _normalize_aware_utc(message, *self._DATETIME_FIELDS)
        return messages

    async def list_by_patient(
        self,
        db: AsyncSession,
        patient_id: UUID,
        channel: str | None,
        status: str | None,
    ) -> list[OutreachMessage]:
        query = select(OutreachMessage).where(OutreachMessage.patient_id == patient_id)
        if channel is not None:
            query = query.where(OutreachMessage.channel == channel)
        if status is not None:
            query = query.where(OutreachMessage.status == status)
        result = await db.execute(query.order_by(OutreachMessage.created_at.desc()))
        messages = list(result.scalars().all())
        for message in messages:
            _normalize_aware_utc(message, *self._DATETIME_FIELDS)
        return messages


class MessageTemplateRepository:
    """Read access over ``message_templates``."""

    async def get_by_type_language_channel(
        self, db: AsyncSession, type_: str, language: str, channel: str
    ) -> MessageTemplate | None:
        result = await db.execute(
            select(MessageTemplate).where(
                MessageTemplate.type == type_,
                MessageTemplate.language == language,
                MessageTemplate.channel == channel,
            )
        )
        return result.scalars().first()


class ChannelConfigurationRepository:
    """Read/write access over ``channel_configurations``.

    Watch out: ``upsert`` never persists the raw ``access_token`` -- the
    real secret is expected to live in the deployment's secret store (AWS
    Secrets Manager per architecture.md §9), never in a Postgres column.
    The ``ChannelConfiguration`` ORM model (frozen, already built) has no
    ``access_token_ref`` column at all, so ``access_token_ref`` is accepted
    here only to keep the caller's contract explicit about what must never
    reach this table, and is deliberately never written to any column.
    """

    async def upsert(
        self,
        db: AsyncSession,
        channel: str,
        provider: str,
        access_token_ref: str,
        configured_by_staff_id: UUID,
    ) -> ChannelConfiguration:
        result = await db.execute(
            select(ChannelConfiguration).where(ChannelConfiguration.channel == channel)
        )
        config = result.scalars().first()
        if config is None:
            config = ChannelConfiguration(
                channel=channel,
                provider=provider,
                status=ChannelConfigStatus.pending,
                configured_by_staff_id=configured_by_staff_id,
            )
            db.add(config)
        else:
            config.provider = provider
            config.configured_by_staff_id = configured_by_staff_id
            config.status = ChannelConfigStatus.pending
            config.verified_at = None
        await db.commit()
        await db.refresh(config)
        _normalize_aware_utc(config, "verified_at")
        return config

    async def get_status(self, db: AsyncSession, channel: str) -> ChannelConfiguration | None:
        result = await db.execute(
            select(ChannelConfiguration).where(ChannelConfiguration.channel == channel)
        )
        config = result.scalars().first()
        if config is not None:
            _normalize_aware_utc(config, "verified_at")
        return config

    async def update_status(
        self, db: AsyncSession, channel: str, status: str
    ) -> ChannelConfiguration:
        result = await db.execute(
            select(ChannelConfiguration).where(ChannelConfiguration.channel == channel)
        )
        config = result.scalar_one()
        config.status = status
        if status == ChannelConfigStatus.verified:
            config.verified_at = now_utc()
        await db.commit()
        await db.refresh(config)
        _normalize_aware_utc(config, "verified_at")
        return config
