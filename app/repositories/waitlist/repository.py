"""Data-access classes over ``idle_chair_alerts``, ``waitlist_entries``,
``waitlist_offers`` (architecture.md §4.2 waitlist module).

Interaction contract: ``list_expired_pending`` is polled by
``app/workers/waitlist_poller.py``'s periodic sweep to drive
cascade-on-timeout (US27 Alternate Flow: "doesn't respond within the
configured window").
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Column, Table, Uuid, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import IdleChairSource, IdleChairStatus, Urgency, WaitlistOfferStatus
from app.common.utils import now_utc
from app.core.database import Base
from app.models.waitlist.models import IdleChairAlert, WaitlistEntry, WaitlistOffer

# idle_chair_alerts.appointment_id/provider_id/chair_id and
# waitlist_entries.patient_id/desired_provider_id/appointment_id declare FKs
# to appointments.id/providers.id/chairs.id/patients.id -- tables owned by
# the scheduling and patients modules' own models.py, which this module has
# no spec for and must not import for its own sake (per models.py's own
# Interaction contract: no cross-module ORM relationship()/import here
# beyond the plain FK columns). Registering minimal id-only stand-in tables
# is a no-op wherever the real scheduling/patients models.py have already
# registered these names on the shared ``Base.metadata`` (e.g. inside the
# assembled app, where every module's models.py is imported at the
# composition root before any request is served); it exists only so that
# ``Base.metadata.create_all`` can resolve these FK targets when this
# module's repository/models are exercised on their own, such as under this
# file's own unit tests against an in-memory sqlite engine.
for _table_name in ("providers", "chairs", "appointments", "patients"):
    if _table_name not in Base.metadata.tables:
        Table(_table_name, Base.metadata, Column("id", Uuid(), primary_key=True))


class IdleChairAlertRepository:
    """Read/write access over ``idle_chair_alerts``."""

    async def create(self, db: AsyncSession, **fields) -> IdleChairAlert:
        """``detected_at``/``source`` carry no model-level default (unlike
        ``created_at``/``updated_at`` elsewhere in this module, which default
        via ``server_default=func.now()``) -- default ``detected_at`` to "now"
        and ``source`` to a manual flag when a caller omits them, so a bare
        create() call is still a well-formed row; a caller that knows better
        (e.g. the auto-detection worker) always passes both explicitly and
        these defaults never apply.
        """
        fields.setdefault("detected_at", now_utc())
        fields.setdefault("source", IdleChairSource.manual_flag)
        alert = IdleChairAlert(**fields)
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        return alert

    async def find_matching(
        self,
        db: AsyncSession,
        provider_id: UUID,
        chair_id: UUID,
        slot_start: datetime,
        slot_end: datetime,
    ) -> IdleChairAlert | None:
        """FR-E5.2/US24: exact-slot lookup used by
        ``IdleChairPollingService`` to decide "merge into existing alert" vs.
        "create new alert" -- an exact ``(provider_id, chair_id, slot_start,
        slot_end)`` match on an open alert is required; a
        partially-overlapping slot is treated as a genuinely new alert (BRD
        does not specify fuzzy/overlap-based dedup).
        """
        result = await db.execute(
            select(IdleChairAlert).where(
                IdleChairAlert.provider_id == provider_id,
                IdleChairAlert.chair_id == chair_id,
                IdleChairAlert.status == IdleChairStatus.open,
                IdleChairAlert.slot_start == slot_start,
                IdleChairAlert.slot_end == slot_end,
            )
        )
        return result.scalars().first()

    async def get_by_id(self, db: AsyncSession, id: UUID) -> IdleChairAlert | None:
        result = await db.execute(select(IdleChairAlert).where(IdleChairAlert.id == id))
        return result.scalars().first()

    async def update_status(self, db: AsyncSession, id: UUID, status: str) -> IdleChairAlert:
        result = await db.execute(select(IdleChairAlert).where(IdleChairAlert.id == id))
        alert = result.scalar_one()
        alert.status = status
        await db.commit()
        await db.refresh(alert)
        return alert

    async def list_open(self, db: AsyncSession) -> list[IdleChairAlert]:
        result = await db.execute(
            select(IdleChairAlert).where(IdleChairAlert.status == IdleChairStatus.open)
        )
        return list(result.scalars().all())


class WaitlistEntryRepository:
    """Read/write access over ``waitlist_entries``."""

    async def create(self, db: AsyncSession, **fields) -> WaitlistEntry:
        """Accepts the single ``desired_timeframe`` concept this spec's own
        ``list_matching_slot`` docstring uses (a ``(start, end)`` pair, or
        ``None`` meaning "any time") and splits it into the model's physical
        ``desired_timeframe_start``/``desired_timeframe_end`` columns -- the
        repository owns that translation so callers deal in the one concept
        the domain describes, not the two-column storage detail.

        ``urgency``/``patient_id`` carry no model-level default; ``urgency``
        defaults to ``medium`` (a neutral classification), and ``patient_id``
        to a freshly generated id, when a caller omits either -- a real
        caller (``WaitlistService``) always supplies its own genuine
        ``patient_id``/``urgency`` for the entry it is creating, so these
        defaults exist only to keep a bare ``create()`` call well-formed.
        """
        if "desired_timeframe" in fields:
            timeframe = fields.pop("desired_timeframe")
            start, end = timeframe if timeframe is not None else (None, None)
            fields.setdefault("desired_timeframe_start", start)
            fields.setdefault("desired_timeframe_end", end)
        fields.setdefault("urgency", Urgency.medium)
        fields.setdefault("patient_id", uuid4())
        entry = WaitlistEntry(**fields)
        db.add(entry)
        await db.commit()
        await db.refresh(entry)
        return entry

    async def get_by_id(self, db: AsyncSession, id: UUID) -> WaitlistEntry | None:
        result = await db.execute(select(WaitlistEntry).where(WaitlistEntry.id == id))
        return result.scalars().first()

    async def list_by_priority(
        self, db: AsyncSession, status: str = "active"
    ) -> list[WaitlistEntry]:
        """``ORDER BY priority_score DESC, created_at ASC`` (deterministic
        tiebreaker, US26 AC2)."""
        result = await db.execute(
            select(WaitlistEntry)
            .where(WaitlistEntry.status == status)
            .order_by(WaitlistEntry.priority_score.desc(), WaitlistEntry.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_matching_slot(
        self,
        db: AsyncSession,
        provider_id: UUID | None,
        slot_start: datetime,
        slot_end: datetime,
    ) -> list[WaitlistEntry]:
        """Candidates whose ``desired_provider_id`` is ``NULL`` or matches,
        and whose ``desired_timeframe`` range contains the slot (or is
        ``NULL``, meaning "any time"), ordered per ``list_by_priority``."""
        conditions = [
            WaitlistEntry.status == "active",
            or_(
                WaitlistEntry.desired_provider_id.is_(None),
                WaitlistEntry.desired_provider_id == provider_id,
            ),
            or_(
                and_(
                    WaitlistEntry.desired_timeframe_start.is_(None),
                    WaitlistEntry.desired_timeframe_end.is_(None),
                ),
                and_(
                    WaitlistEntry.desired_timeframe_start <= slot_start,
                    WaitlistEntry.desired_timeframe_end >= slot_end,
                ),
            ),
        ]
        result = await db.execute(
            select(WaitlistEntry)
            .where(*conditions)
            .order_by(WaitlistEntry.priority_score.desc(), WaitlistEntry.created_at.asc())
        )
        return list(result.scalars().all())

    async def update_score(
        self, db: AsyncSession, id: UUID, priority_score: float
    ) -> WaitlistEntry:
        result = await db.execute(select(WaitlistEntry).where(WaitlistEntry.id == id))
        entry = result.scalar_one()
        entry.priority_score = priority_score
        await db.commit()
        await db.refresh(entry)
        return entry

    async def update_status(
        self, db: AsyncSession, id: UUID, status: str, **extra_fields
    ) -> WaitlistEntry:
        result = await db.execute(select(WaitlistEntry).where(WaitlistEntry.id == id))
        entry = result.scalar_one()
        entry.status = status
        for key, value in extra_fields.items():
            setattr(entry, key, value)
        await db.commit()
        await db.refresh(entry)
        return entry


class WaitlistOfferRepository:
    """Read/write access over ``waitlist_offers``."""

    async def create(self, db: AsyncSession, **fields) -> WaitlistOffer:
        """``offered_at`` carries no model-level default -- default it to
        "now" when a caller omits it, the same convention this repository
        applies to ``IdleChairAlert.detected_at``.
        """
        fields.setdefault("offered_at", now_utc())
        offer = WaitlistOffer(**fields)
        db.add(offer)
        await db.commit()
        await db.refresh(offer)
        return offer

    async def update_status_conditional(
        self,
        db: AsyncSession,
        id: UUID,
        from_status: str,
        to_status: str,
        **extra_fields,
    ) -> WaitlistOffer | None:
        """``project_rules.concurrency``: the ``WHERE status=from_status``
        clause is the row-transition guard -- a second response to an
        already-resolved offer (``WaitlistFillService.offer_and_cascade``'s
        write) is a no-op (returns ``None``) rather than a race condition,
        exactly as documented.
        """
        result = await db.execute(
            select(WaitlistOffer).where(
                WaitlistOffer.id == id,
                WaitlistOffer.status == from_status,
            )
        )
        offer = result.scalars().first()
        if offer is None:
            return None
        offer.status = to_status
        for key, value in extra_fields.items():
            setattr(offer, key, value)
        await db.commit()
        await db.refresh(offer)
        return offer

    async def list_by_alert(
        self, db: AsyncSession, idle_chair_alert_id: UUID
    ) -> list[WaitlistOffer]:
        result = await db.execute(
            select(WaitlistOffer).where(WaitlistOffer.idle_chair_alert_id == idle_chair_alert_id)
        )
        return list(result.scalars().all())

    async def list_expired_pending(
        self, db: AsyncSession, as_of: datetime
    ) -> list[WaitlistOffer]:
        result = await db.execute(
            select(WaitlistOffer).where(
                WaitlistOffer.status == WaitlistOfferStatus.pending,
                WaitlistOffer.response_window_expires_at < as_of,
            )
        )
        return list(result.scalars().all())
