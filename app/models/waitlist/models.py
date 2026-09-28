"""SQLAlchemy ORM declarations for `idle_chair_alerts`, `waitlist_entries`,
`waitlist_offers` (architecture.md §4.2 waitlist module).

Pure schema — no query methods, no business logic. Query/mutation behaviour
lives in ``repositories/waitlist/repository.py``; this file only declares
columns and constraints.

Interaction contract: ``patient_id``/``desired_provider_id``/
``appointment_id``/``provider_id``/``chair_id`` are plain FK columns, with no
cross-module ORM ``relationship()`` declared here.
``repositories/waitlist/repository.py`` is the only file that imports these
classes directly; ``services/console/service.py``'s ``OverrideService``
depends directly on ``WaitlistEntryRepository`` (not this ORM class) per the
console layering rule that forbids console depending on another module's
service layer.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Numeric
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import IdleChairSource, IdleChairStatus, Urgency, WaitlistEntryStatus, WaitlistOfferStatus
from app.core.database import Base


class IdleChairAlert(Base):
    """An open (or resolved) idle-chair window, either detected automatically
    from a cancellation or flagged manually by staff.

    FR-E5.2: no ``UNIQUE`` constraint on
    ``(provider_id, chair_id, slot_start, slot_end)`` — deduplication between
    an auto-detected and a manually-flagged alert for the same slot is an
    application-layer check (``IdleChairPollingService.detect_from_slot``'s
    "find matching before insert" logic in
    ``repositories/waitlist/repository.py``), not a DB constraint, because
    two genuinely distinct idle windows can legitimately share a
    provider/chair.
    """

    __tablename__ = "idle_chair_alerts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Originating cancelled appointment; null for a purely manual flag.
    appointment_id: Mapped[UUID | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    provider_id: Mapped[UUID] = mapped_column(ForeignKey("providers.id"), nullable=False)
    chair_id: Mapped[UUID] = mapped_column(ForeignKey("chairs.id"), nullable=False)
    slot_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    slot_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[IdleChairSource] = mapped_column(SAEnum(IdleChairSource, native_enum=False), nullable=False)
    status: Mapped[IdleChairStatus] = mapped_column(
        SAEnum(IdleChairStatus, native_enum=False),
        nullable=False,
        default=IdleChairStatus.open,
        server_default=IdleChairStatus.open.value,
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class WaitlistEntry(Base):
    """A patient's standing request to be offered an earlier slot, ranked by
    ``priority_score`` (recalculated on trigger events; ``created_at`` is the
    documented tiebreaker)."""

    __tablename__ = "waitlist_entries"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), nullable=False)
    desired_provider_id: Mapped[UUID | None] = mapped_column(ForeignKey("providers.id"), nullable=True)
    desired_timeframe_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    desired_timeframe_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    urgency: Mapped[Urgency] = mapped_column(SAEnum(Urgency, native_enum=False), nullable=False)
    # Recalculated on trigger events (e.g. urgency change, elapsed wait time).
    priority_score: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    # Active offer, if any; null when not currently being offered a slot.
    current_alert_id: Mapped[UUID | None] = mapped_column(ForeignKey("idle_chair_alerts.id"), nullable=True)
    # Set on successful fill.
    appointment_id: Mapped[UUID | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    status: Mapped[WaitlistEntryStatus] = mapped_column(
        SAEnum(WaitlistEntryStatus, native_enum=False),
        nullable=False,
        default=WaitlistEntryStatus.active,
        server_default=WaitlistEntryStatus.active.value,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class WaitlistOffer(Base):
    """One offer of a specific ``IdleChairAlert`` slot to a specific
    ``WaitlistEntry``, with a bounded response window."""

    __tablename__ = "waitlist_offers"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    waitlist_entry_id: Mapped[UUID] = mapped_column(ForeignKey("waitlist_entries.id"), nullable=False)
    idle_chair_alert_id: Mapped[UUID] = mapped_column(ForeignKey("idle_chair_alerts.id"), nullable=False)
    offered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    response_window_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[WaitlistOfferStatus] = mapped_column(
        SAEnum(WaitlistOfferStatus, native_enum=False),
        nullable=False,
        default=WaitlistOfferStatus.pending,
        server_default=WaitlistOfferStatus.pending.value,
    )
