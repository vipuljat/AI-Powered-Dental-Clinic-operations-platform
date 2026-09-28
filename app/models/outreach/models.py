"""SQLAlchemy ORM declarations for the outreach module's four tables — the
shared communication backbone every other module dispatches through
(architecture.md §4.2 outreach module).

Pure schema — no query methods, no business logic. This file declares only
columns and constraints; the append-only enforcement on ``ConsentRecord``
(FR-E6.6) is enforced at the repository layer (``repositories/outreach/
repository.py`` exposes inserts/reads only, never an update to a prior
consent row), not by a DB trigger — same pattern as ``AuditLog`` in
``app/models/console/models.py``.

Interaction contract: ``patient_id``/``template_id`` are plain FK columns,
no cross-module ORM ``relationship()`` — ``repositories/outreach/
repository.py`` is the only file that imports these classes;
``services/console/service.py``'s ``OverrideService`` depends directly on
``OutreachMessageRepository`` (not this ORM class) per the console layering
rule, and ``services/recall/service.py``/``services/education/service.py``/
``app/workers/outreach_retry.py`` all reach these tables only via
``services/outreach/service.py``'s ``OutreachService.dispatch``, never a
direct repository/ORM import.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import (
    CampaignType,
    Channel,
    ChannelConfigStatus,
    ConsentStatus,
    Language,
    OutreachMessageStatus,
)
from app.core.database import Base


class ConsentRecord(Base):
    """FR-E6.6: an append-only per-channel consent state change (US32).

    Every grant/withdraw/decline is inserted as a NEW row — never an UPDATE
    to a prior row — so the full chronological history is always
    reconstructable by reading all rows for a ``(patient_id, channel)`` pair
    ordered by ``effective_at``.
    """

    __tablename__ = "consent_records"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), nullable=False)
    channel: Mapped[Channel] = mapped_column(SAEnum(Channel, native_enum=False), nullable=False)
    status: Mapped[ConsentStatus] = mapped_column(SAEnum(ConsentStatus, native_enum=False), nullable=False)
    # e.g. "keyword_stop", "staff_manual".
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MessageTemplate(Base):
    """A reusable body template for a given ``(type, language, channel)``."""

    __tablename__ = "message_templates"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    language: Mapped[Language] = mapped_column(SAEnum(Language, native_enum=False), nullable=False)
    channel: Mapped[Channel] = mapped_column(SAEnum(Channel, native_enum=False), nullable=False)
    body_template: Mapped[str] = mapped_column(Text, nullable=False)


class OutreachMessage(Base):
    """A single outbound (or attempted) outreach dispatch.

    ``related_entity_type``/``related_entity_id`` is polymorphic exactly like
    ``audit_log.entity_type``/``entity_id`` — no DB-level FK, plain table-name
    string + UUID. Every writer (confirmation, waitlist-offer, recall,
    treatment-reengagement, education dispatch) must set both consistently
    so ``GET /outreach/messages`` and the outreach log panel can resolve
    context.
    """

    __tablename__ = "outreach_messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), nullable=False)
    campaign_type: Mapped[CampaignType] = mapped_column(SAEnum(CampaignType, native_enum=False), nullable=False)
    channel: Mapped[Channel] = mapped_column(SAEnum(Channel, native_enum=False), nullable=False)
    language: Mapped[Language] = mapped_column(SAEnum(Language, native_enum=False), nullable=False)
    template_id: Mapped[UUID | None] = mapped_column(ForeignKey("message_templates.id"), nullable=True)
    status: Mapped[OutreachMessageStatus] = mapped_column(
        SAEnum(OutreachMessageStatus, native_enum=False),
        nullable=False,
        default=OutreachMessageStatus.queued,
        server_default=OutreachMessageStatus.queued.value,
    )
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Polymorphic link (appointment, waitlist_entry, ...) — no DB-level FK, see docstring.
    related_entity_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    related_entity_id: Mapped[UUID | None] = mapped_column(nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class ChannelConfiguration(Base):
    """A staff-configured outbound channel (delivery_team only)."""

    __tablename__ = "channel_configurations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Note: unlike every other `channel` column in this module, this one has
    # no `webchat` member -- webchat has no provider/channel config row of
    # its own (it is a fully in-app, staff-config-free surface), so the
    # enum here is deliberately a strict subset of the shared `Channel` enum.
    channel: Mapped[Channel] = mapped_column(
        SAEnum(
            *(member.value for member in Channel if member is not Channel.webchat),
            name="channel_configurations_channel",
            native_enum=False,
        ),
        nullable=False,
    )
    # e.g. Meta BSP name.
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[ChannelConfigStatus] = mapped_column(
        SAEnum(ChannelConfigStatus, native_enum=False),
        nullable=False,
        default=ChannelConfigStatus.pending,
        server_default=ChannelConfigStatus.pending.value,
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    configured_by_staff_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
