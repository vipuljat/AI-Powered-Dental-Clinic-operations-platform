"""Business logic for the ``waitlist`` module: idle-chair detection (both
poll-driven and event-driven), the standing waitlist itself, and the
offer/cascade fill workflow (architecture.md §4.2 waitlist module).

``IdleChairPollingService`` (FR-E5.1/US23, FR-E5.2/US24) owns idle-chair
alert detection/dedup. ``WaitlistService`` (FR-E5.3/US25, FR-E5.4/US26) owns
standing-entry creation and priority scoring. ``WaitlistFillService``
(FR-E5.5/US27) owns the offer-to-top-candidate-then-cascade workflow.

Responsibility: this file never issues a raw SQL/ORM statement against
another module's tables — every read/write goes through this module's own
``repositories/waitlist/repository.py`` classes or, per the project-wide
layering rule (service -> another module's *repository*, never another
module's *service*), ``repositories/outreach/repository.py``'s
``ConsentRepository`` for the consent filter. The one documented exception is
``WaitlistFillService.respond``'s on-accept booking call into
``services/scheduling/service.py``'s ``SchedulingService.create`` (this
file's own Interaction contract section spells out why that one edge is
one-directional and therefore does not cycle).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select

from app.common.constants import IDLE_CHAIR_POLL_INTERVAL_MINUTES, WAITLIST_RESPONSE_WINDOW_MINUTES
from app.common.enums import (
    ActorType,
    Channel,
    ConsentStatus,
    IdleChairSource,
    IdleChairStatus,
    RuleCategory,
    Urgency,
    WaitlistEntryStatus,
    WaitlistOfferStatus,
)
from app.common.exceptions.errors import NoActiveRuleSetError, NotFoundError
from app.common.utils import now_utc
from app.models.waitlist.models import WaitlistOffer
from app.repositories.outreach.repository import ConsentRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.messaging import MessageBroker
    from app.models.waitlist.models import IdleChairAlert, WaitlistEntry
    from app.repositories.waitlist.repository import (
        IdleChairAlertRepository,
        WaitlistEntryRepository,
        WaitlistOfferRepository,
    )
    from app.services.console.service import AuditService
    from app.services.rules.service import RuleSetService
    from app.services.scheduling.service import SchedulingService

__all__ = ["IdleChairPollingService", "WaitlistFillService", "WaitlistService"]

_logger = logging.getLogger(__name__)

# --- _score inferred defaults ------------------------------------------------
# US25/US26 never fix the exact weighting formula behind "urgency weight, risk
# weight if available, wait time" — these are the documented conventional
# inference: a rule-configurable multiplier per contributor (read from an
# active `scheduling_priority` rule's `rule_value`, keyed the same way
# `services/scheduling/service.py`'s own `_resolve_slot_parameters` reads its
# own `rule_value` keys), falling back to these fixed defaults when no active
# rule set carries them (or none is active at all — see `_score` below).
_DEFAULT_URGENCY_WEIGHT: float = 10.0
_DEFAULT_RISK_WEIGHT: float = 5.0
_DEFAULT_WAIT_TIME_WEIGHT_PER_HOUR: float = 0.1
_URGENCY_NUMERIC: dict[str, int] = {"low": 1, "medium": 2, "high": 3}

# WaitlistEntry (frozen models.py) carries no `appointment_type` column of its
# own — `respond()`'s on-accept booking must still supply one to
# `SchedulingService.create`'s `data["appointment_type"]`. This fixed,
# self-documenting value is the conventional inference used where the BRD
# never names a per-entry appointment type (open_questions).
_WAITLIST_FILL_APPOINTMENT_TYPE = "waitlist_fill"


def _aware(value: "datetime | None") -> "datetime | None":
    """Normalize a datetime read back from the DB to tz-aware UTC.

    project_rules.testing's SQLite pivot does not retain a
    ``DateTime(timezone=True)`` column's UTC offset the way Postgres does —
    see the identical helper in ``services/scheduling/service.py`` and
    ``services/console/service.py``. Every timestamp in this tree is UTC, so
    a naive value read back is treated as already being UTC.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _enum_value(value: object) -> "str | None":
    """Normalizes an ``Urgency``/``RiskLevel``-shaped value to its plain
    string, accepted here as either the enum member or its raw string value
    — the same accommodation ``services/rules/service.py``'s
    ``_category_value`` makes.
    """
    if value is None:
        return None
    return value.value if hasattr(value, "value") else value  # type: ignore[return-value]


def _to_urgency(value: object) -> Urgency:
    if isinstance(value, Urgency):
        return value
    return Urgency(_enum_value(value) or Urgency.medium.value)


class IdleChairPollingService:
    """FR-E5.1/US23 (poll-driven detection), FR-E5.2/US24 (dedup/merge)."""

    def __init__(
        self,
        alert_repo: "IdleChairAlertRepository",
        audit: "AuditService | None" = None,
    ) -> None:
        # Watch out: `services/scheduling/service.py`'s frozen
        # `_notify_freed_slot` (already built, cannot be edited) constructs
        # this class with a single positional argument
        # (`IdleChairPollingService(IdleChairAlertRepository())`), omitting
        # `audit` entirely. `audit` therefore defaults to `None` here (rather
        # than being a strictly required collaborator) so that already-frozen
        # call site keeps working; every method below only calls
        # `self._audit.record(...)` when `self._audit is not None`.
        self._alert_repo = alert_repo
        self._audit = audit

    async def run_cycle(self, db: "AsyncSession") -> int:
        """FR-E5.1/US23: scans ``appointments`` for ``status='cancelled'``
        rows and routes each one through :meth:`detect_from_slot` (which
        itself dedups via ``find_matching`` — see FR-E5.2/US24 below) so a
        slot already alerted from the event-driven cancellation path is
        never double-alerted by this poll path. A clean cycle — nothing new
        raised — is logged and returns 0 (US23 Alternate Flow), satisfying
        the ``IDLE_CHAIR_POLL_INTERVAL_MINUTES`` (2 min default)
        detection-to-alert latency this method is invoked against.

        Gap: this class's frozen constructor (see ``__init__`` above) takes
        no ``AppointmentRepository`` collaborator of its own, and the frozen
        ``repositories/scheduling/repository.py``'s ``AppointmentRepository``
        exposes no "cancelled appointments not yet alerted" query to depend
        on instead — only ``list_open_slots``/``find_conflicts``/
        ``list_completable``/``list_by_risk_source``, none of which fit. The
        ``Appointment`` ORM class is therefore imported here, narrowly and
        lazily (never at module import time, mirroring
        ``SchedulingService._notify_freed_slot``'s own lazy cross-module
        import), for this one read-only scan; ``detect_from_slot``'s own
        dedup already makes re-scanning an already-alerted cancellation a
        safe no-op, so no separate "not yet alerted" filter is needed here.
        """
        from app.common.enums import AppointmentStatus
        from app.models.scheduling.models import Appointment

        result = await db.execute(
            select(Appointment).where(Appointment.status == AppointmentStatus.cancelled)
        )
        cancelled_appointments = list(result.scalars().all())

        new_alert_count = 0
        for appointment in cancelled_appointments:
            _alert, created = await self.detect_from_slot(
                db,
                appointment.provider_id,
                appointment.chair_id,
                _aware(appointment.scheduled_start),
                _aware(appointment.scheduled_end),
                source=IdleChairSource.auto_detected.value,
                appointment_id=appointment.id,
            )
            if created:
                new_alert_count += 1

        if new_alert_count == 0:
            _logger.info(
                "waitlist.idle_chair_poll_cycle_clean",
                extra={
                    "scanned": len(cancelled_appointments),
                    "poll_interval_minutes": IDLE_CHAIR_POLL_INTERVAL_MINUTES,
                },
            )
        return new_alert_count

    async def detect_from_slot(
        self,
        db: "AsyncSession",
        provider_id: "UUID",
        chair_id: "UUID",
        slot_start: "datetime",
        slot_end: "datetime",
        source: str,
        appointment_id: "UUID | None" = None,
    ) -> "tuple[IdleChairAlert, bool]":
        """FR-E5.1 (Flow 3)/FR-E5.2/US24: the single shared entry point both
        the poll path (:meth:`run_cycle`) and the event-driven cancellation
        path (``app/workers/waitlist_poller.py`` consuming
        ``appointment.cancelled``) call. Dedups an exact
        ``(provider_id, chair_id, slot_start, slot_end)`` match on an open
        alert via ``find_matching`` first ("merges the manual flag rather
        than duplicating", T62 AC) — only when none exists is a new alert
        row created.
        """
        existing = await self._alert_repo.find_matching(db, provider_id, chair_id, slot_start, slot_end)
        if existing is not None:
            return existing, False

        alert = await self._alert_repo.create(
            db,
            appointment_id=appointment_id,
            provider_id=provider_id,
            chair_id=chair_id,
            slot_start=slot_start,
            slot_end=slot_end,
            source=source,
            detected_at=now_utc(),
        )
        return alert, True

    async def flag_manual(
        self,
        db: "AsyncSession",
        provider_id: "UUID",
        chair_id: "UUID",
        slot_start: "datetime",
        slot_end: "datetime",
        actor_id: "UUID",
    ) -> "tuple[IdleChairAlert, bool]":
        """FR-E5.2/US24: thin wrapper over ``detect_from_slot(source="manual_flag")``
        — a manual flag that turns out to match an already-open auto-detected
        alert is merged into it (``created`` is ``False``) rather than
        duplicated, then this manual action is itself audit-logged.
        """
        alert, created = await self.detect_from_slot(
            db, provider_id, chair_id, slot_start, slot_end, source=IdleChairSource.manual_flag.value
        )
        if self._audit is not None:
            await self._audit.record(
                db,
                actor_staff_id=actor_id,
                actor_type=ActorType.staff,
                action_type="idle_chair_alert.manual_flag",
                entity_type="idle_chair_alert",
                entity_id=alert.id,
                override_payload={"created": created},
            )
        return alert, created


class WaitlistService:
    """FR-E5.3/US25 (entry creation + initial score), FR-E5.4/US26
    (re-scoring on trigger events)."""

    def __init__(
        self,
        entry_repo: "WaitlistEntryRepository",
        rule_set_service: "RuleSetService | None",
        audit: "AuditService | None" = None,
    ) -> None:
        # `rule_set_service` defaults to `None` (rather than being a strictly
        # required collaborator) so `WaitlistFillService` can construct a
        # private `WaitlistService` purely for `match_candidates` (which
        # never reads rule sets) without itself owning a `RuleSetService` —
        # its own frozen Public surface constructor has none. `_score` treats
        # a missing rule-set collaborator identically to "no active rule set"
        # (falls back to the documented default weights below).
        self._entry_repo = entry_repo
        self._rule_set_service = rule_set_service
        self._audit = audit
        # Stateless, session-free collaborator instantiated directly here —
        # the same accommodation `services/scheduling/service.py`'s
        # `SchedulingService.__init__` makes for `ProviderRepository`/
        # `ChairRepository` (not part of this class's frozen constructor
        # Public surface).
        self._consent_repo = ConsentRepository()

    @staticmethod
    def _first_rule_value(rules: list, key: str, default: float) -> float:
        for rule in rules:
            value = getattr(rule, "rule_value", None)
            if isinstance(value, dict) and value.get(key) is not None:
                return float(value[key])
        return default

    async def _active_scheduling_priority_rules(self, db: "AsyncSession") -> list:
        if self._rule_set_service is None:
            return []
        try:
            return await self._rule_set_service.get_active_rules_by_category(
                db, RuleCategory.scheduling_priority.value
            )
        except NoActiveRuleSetError:
            # FR-E5.3/US25: a waitlist entry can be created (and re-scored)
            # even before any clinical rule set has ever been approved — the
            # documented default weights below stand in until one is active,
            # rather than blocking waitlist usage on rule-set approval.
            return []

    async def _score(
        self,
        db: "AsyncSession",
        urgency: object,
        risk_level: object,
        wait_hours: float,
    ) -> float:
        """FR-E5.3/US25, FR-E5.4/US26: ``priority_score = urgency_weight *
        urgency_numeric + risk_weight * risk_numeric (if risk_level is
        available) + wait_time_weight_per_hour * wait_hours`` — every weight
        is read from the active ``scheduling_priority`` rule set's
        ``rule_value`` when present, else the module-level defaults above.
        """
        rules = await self._active_scheduling_priority_rules(db)
        urgency_weight = self._first_rule_value(rules, "urgency_weight", _DEFAULT_URGENCY_WEIGHT)
        risk_weight = self._first_rule_value(rules, "risk_weight", _DEFAULT_RISK_WEIGHT)
        wait_weight = self._first_rule_value(
            rules, "wait_time_weight_per_hour", _DEFAULT_WAIT_TIME_WEIGHT_PER_HOUR
        )

        urgency_numeric = _URGENCY_NUMERIC.get(_enum_value(urgency), _URGENCY_NUMERIC[Urgency.medium.value])
        score = urgency_weight * urgency_numeric

        if risk_level is not None:
            risk_numeric = _URGENCY_NUMERIC.get(_enum_value(risk_level), 0)
            score += risk_weight * risk_numeric

        score += wait_weight * max(wait_hours, 0.0)
        return float(score)

    async def add_entry(self, db: "AsyncSession", data: dict, actor_id: "UUID") -> "WaitlistEntry":
        """FR-E5.3/US25: computes the initial ``priority_score`` (wait-time
        contribution is 0 at creation, per this method's own spec), inserts
        the entry, audit-logs it, then re-scores the whole active waitlist
        (``trigger="new_entry"``, FR-E5.4/US26) so a newly-added high-urgency
        entry can immediately outrank lower-priority existing entries.
        """
        entry_fields = dict(data)
        risk_level = entry_fields.pop("risk_level", None)
        urgency = _to_urgency(entry_fields.get("urgency", Urgency.medium))
        entry_fields["urgency"] = urgency

        entry_fields["priority_score"] = await self._score(db, urgency, risk_level, wait_hours=0.0)

        entry = await self._entry_repo.create(db, **entry_fields)

        if self._audit is not None:
            await self._audit.record(
                db,
                actor_staff_id=actor_id,
                actor_type=ActorType.staff,
                action_type="waitlist_entry.create",
                entity_type="waitlist_entry",
                entity_id=entry.id,
                override_payload={"priority_score": entry_fields["priority_score"]},
            )

        # `recalculate_priority` re-scores every active entry (including this
        # one) via `WaitlistEntryRepository.update_score`, which mutates the
        # row in place under the same SQLAlchemy identity map `entry` was
        # created from -- `entry` already reflects the freshest score once
        # that call returns, so no separate re-fetch is needed (or correct:
        # a repository this file has no signature control over could return
        # a distinct object for the same row, and this method's only
        # documented return value is "the created WaitlistEntry").
        await self.recalculate_priority(db, trigger="new_entry")

        return entry

    async def recalculate_priority(self, db: "AsyncSession", trigger: str) -> int:
        """FR-E5.4/US26: re-scores every ``status='active'`` entry.

        Triggered on cancellation (a fresh idle-chair alert), rule-set
        activation, and new-waitlist-entry events (``trigger`` in
        ``{"cancellation", "rule_change", "new_entry"}``); the deterministic
        tiebreaker on equal scores is ``created_at ASC``, already guaranteed
        by ``WaitlistEntryRepository.list_by_priority``'s own ``ORDER BY``.
        """
        entries = await self._entry_repo.list_by_priority(db, status=WaitlistEntryStatus.active.value)
        now = now_utc()
        for entry in entries:
            # Only a genuine `datetime` is fed through `_aware` -- a row read
            # back before its `server_default=func.now()` populated
            # `created_at` (or, equivalently here, any collaborator that
            # cannot yet supply one) is treated the same as "unknown wait
            # time", falling back to a 0-hour contribution rather than
            # raising on a non-comparable value.
            raw_created_at = getattr(entry, "created_at", None)
            created_at = _aware(raw_created_at) if isinstance(raw_created_at, datetime) else None
            wait_hours = (now - created_at).total_seconds() / 3600.0 if created_at is not None else 0.0
            score = await self._score(db, entry.urgency, None, wait_hours)
            await self._entry_repo.update_score(db, entry.id, score)
        return len(entries)

    async def match_candidates(self, db: "AsyncSession", alert: "IdleChairAlert") -> list["WaitlistEntry"]:
        """FR-E5.5/US27: ``list_matching_slot`` results filtered down to
        patients carrying an active (``status="granted"``) ``ConsentRecord``
        on at least one channel — see this file's Interaction contract.
        """
        candidates = await self._entry_repo.list_matching_slot(db, alert.provider_id, alert.slot_start, alert.slot_end)
        consenting: list["WaitlistEntry"] = []
        for entry in candidates:
            if await self._has_active_consent(db, entry.patient_id):
                consenting.append(entry)
        return consenting

    async def _has_active_consent(self, db: "AsyncSession", patient_id: "UUID") -> bool:
        for channel in Channel:
            record = await self._consent_repo.get_latest_by_channel(db, patient_id, channel.value)
            if record is not None and record.status == ConsentStatus.granted:
                return True
        return False


class WaitlistFillService:
    """FR-E5.5/US27: offers the top-priority consenting candidate the freed
    slot, cascading to the next on decline or on-timeout."""

    def __init__(
        self,
        offer_repo: "WaitlistOfferRepository",
        entry_repo: "WaitlistEntryRepository",
        alert_repo: "IdleChairAlertRepository",
        scheduling_service: "SchedulingService",
        broker: "MessageBroker",
        audit: "AuditService | None" = None,
    ) -> None:
        self._offer_repo = offer_repo
        self._entry_repo = entry_repo
        self._alert_repo = alert_repo
        self._scheduling_service = scheduling_service
        self._broker = broker
        self._audit = audit
        # `match_candidates` never reads rule sets (only `add_entry`/
        # `recalculate_priority` do), so this private `WaitlistService`
        # instance is built with `rule_set_service=None` — see that class's
        # own `__init__` docstring for why that is safe here.
        self._waitlist_service = WaitlistService(entry_repo, None, audit)

    async def _get_offer_by_id(self, db: "AsyncSession", offer_id: "UUID") -> "WaitlistOffer | None":
        """Gap: the frozen ``WaitlistOfferRepository`` exposes no single-row
        ``get_by_id`` — only ``create``/``update_status_conditional``/
        ``list_by_alert``/``list_expired_pending``. ``respond()``'s
        idempotent-replay path ("returns the CURRENT row unchanged" once
        ``update_status_conditional`` reports the offer already resolved)
        needs exactly that lookup, so this narrow read-only query is issued
        directly against this module's own ``waitlist_offers`` table (this
        module's own frozen repository simply missing one read method, not a
        bypass into another module's data) as the closest-correct
        implementation available without editing the frozen file.
        """
        return await db.get(WaitlistOffer, offer_id)

    async def offer_and_cascade(self, db: "AsyncSession", alert_id: "UUID") -> "WaitlistOffer | None":
        """FR-E5.5/US27: offers the next unoffered, consenting, top-priority
        candidate the slot; when none remain, marks the alert
        ``status="exhausted"`` (US27 Alternate Flow — "remains idle and is
        reported on the dashboard" via that same ``status`` field, no
        separate reporting table).
        """
        alert = await self._alert_repo.get_by_id(db, alert_id)
        if alert is None:
            raise NotFoundError("Idle chair alert not found.")

        candidates = await self._waitlist_service.match_candidates(db, alert)
        existing_offers = await self._offer_repo.list_by_alert(db, alert_id)
        excluded_entry_ids = {
            offer.waitlist_entry_id
            for offer in existing_offers
            if offer.status in (WaitlistOfferStatus.declined, WaitlistOfferStatus.expired)
        }

        next_candidate = next((entry for entry in candidates if entry.id not in excluded_entry_ids), None)

        if next_candidate is None:
            await self._alert_repo.update_status(db, alert_id, IdleChairStatus.exhausted.value)
            return None

        offer = await self._offer_repo.create(
            db,
            waitlist_entry_id=next_candidate.id,
            idle_chair_alert_id=alert_id,
            offered_at=now_utc(),
            response_window_expires_at=now_utc() + timedelta(minutes=WAITLIST_RESPONSE_WINDOW_MINUTES),
            status=WaitlistOfferStatus.pending.value,
        )
        await self._entry_repo.update_status(
            db, next_candidate.id, WaitlistEntryStatus.offered.value, current_alert_id=alert_id
        )

        await self._broker.publish(
            "waitlist_offer.created",
            {"waitlist_entry_id": str(next_candidate.id), "idle_chair_alert_id": str(alert_id)},
        )
        return offer

    async def respond(self, db: "AsyncSession", offer_id: "UUID", response: str) -> "WaitlistOffer":
        """``POST /waitlist/offers/{id}/respond`` (Auth: none, patient-facing/
        webhook) calls this directly and synchronously — see this file's
        Interaction contract.

        project_rules.concurrency: the sole guard against a double-response
        race is ``update_status_conditional``'s conditional
        ``WHERE status='pending'`` update — a second response to an
        already-resolved offer is treated as an idempotent-safe webhook
        replay: the CURRENT row is returned unchanged rather than re-applying
        the booking/decline a second time.
        """
        to_status = (
            WaitlistOfferStatus.accepted.value if response == "accept" else WaitlistOfferStatus.declined.value
        )
        updated = await self._offer_repo.update_status_conditional(
            db, offer_id, from_status=WaitlistOfferStatus.pending.value, to_status=to_status
        )
        if updated is None:
            current = await self._get_offer_by_id(db, offer_id)
            if current is None:
                raise NotFoundError("Waitlist offer not found.")
            return current

        entry = await self._entry_repo.get_by_id(db, updated.waitlist_entry_id)
        if entry is None:
            raise NotFoundError("Waitlist entry not found.")
        alert = await self._alert_repo.get_by_id(db, updated.idle_chair_alert_id)
        if alert is None:
            raise NotFoundError("Idle chair alert not found.")

        if response == "accept":
            # Interaction contract: this is the one place `waitlist` depends
            # synchronously on `scheduling`'s SERVICE (not just its
            # repository) — `actor_id=None` since this booking is
            # AI-Agent-originated (`appointments.created_by_staff_id` is
            # nullable for exactly this case), with a fresh idempotency key
            # derived from the offer id (RMR001).
            appointment_data = {
                "patient_id": entry.patient_id,
                "provider_id": alert.provider_id,
                "chair_id": alert.chair_id,
                "appointment_type": _WAITLIST_FILL_APPOINTMENT_TYPE,
                "scheduled_start": alert.slot_start,
                "scheduled_end": alert.slot_end,
            }
            appointment, _is_replay = await self._scheduling_service.create(
                db, appointment_data, actor_id=None, idempotency_key=f"waitlist-offer-{offer_id}"
            )
            # `updated.idle_chair_alert_id` (already known from the offer row
            # itself) rather than the separately re-fetched `alert.id` --
            # both name the same row in production, but the former is the
            # canonical id this method was handed, with no extra dependency
            # on the fetched alert object round-tripping its own `id`.
            await self._alert_repo.update_status(db, updated.idle_chair_alert_id, IdleChairStatus.filled.value)
            await self._entry_repo.update_status(
                db, entry.id, WaitlistEntryStatus.booked.value, appointment_id=appointment.id
            )
            if self._audit is not None:
                await self._audit.record(
                    db,
                    actor_staff_id=None,
                    actor_type=ActorType.ai_agent,
                    action_type="waitlist_offer.accepted",
                    entity_type="waitlist_offer",
                    entity_id=updated.id,
                    override_payload={"appointment_id": str(appointment.id)},
                )
        else:
            # Watch out: `waitlist_entries.status` is one flat field with no
            # separate "declined this one offer" vs. "opted out entirely"
            # distinction (architecture.md §4.2) — declining is therefore
            # terminal for THIS entry row; a patient who wants to remain on
            # the waitlist needs a fresh `POST /waitlist/entries` call. This
            # is a documented inference from the schema's flat status field,
            # not an explicit BRD statement (open_questions).
            await self._entry_repo.update_status(db, entry.id, WaitlistEntryStatus.declined.value)
            if self._audit is not None:
                await self._audit.record(
                    db,
                    actor_staff_id=None,
                    actor_type=ActorType.ai_agent,
                    action_type="waitlist_offer.declined",
                    entity_type="waitlist_offer",
                    entity_id=updated.id,
                )
            # See the accept branch above: `updated.idle_chair_alert_id`,
            # not the separately re-fetched `alert.id`.
            await self.offer_and_cascade(db, updated.idle_chair_alert_id)

        return updated

    async def sweep_expired_offers(self, db: "AsyncSession") -> int:
        """FR-E5.5/US27: cascades to the next candidate once
        ``WAITLIST_RESPONSE_WINDOW_MINUTES`` elapses unanswered.

        For each expired-pending offer: the conditional
        ``update_status_conditional`` transition to ``"expired"`` is this
        sweep's own concurrency guard (a patient responding in the same
        instant this sweep runs simply loses the race harmlessly, since
        ``update_status_conditional`` only actually mutates a row still
        ``"pending"``). Once expired, the entry is reset to
        ``status="active"`` so it can be re-offered a *different* future
        slot (an entry excluded from being re-offered THIS same alert again
        via ``offer_and_cascade``'s own ``('declined', 'expired')`` filter),
        and — since the slot itself may still be fillable — the same alert
        is immediately cascaded to its next candidate.
        """
        expired_pending = await self._offer_repo.list_expired_pending(db, now_utc())
        count = 0
        for offer in expired_pending:
            updated = await self._offer_repo.update_status_conditional(
                db,
                offer.id,
                from_status=WaitlistOfferStatus.pending.value,
                to_status=WaitlistOfferStatus.expired.value,
            )
            if updated is None:
                continue
            count += 1

            # `offer` (the row already in hand from `list_expired_pending`)
            # rather than `updated` for these two ids -- both name the same
            # persisted row in production, but `offer` is the canonical
            # reference this loop was already iterating, with no extra
            # dependency on `update_status_conditional`'s return value
            # round-tripping every column of the row it mutated.
            entry = await self._entry_repo.get_by_id(db, offer.waitlist_entry_id)
            if entry is not None and entry.status == WaitlistEntryStatus.offered:
                await self._entry_repo.update_status(
                    db, entry.id, WaitlistEntryStatus.active.value, current_alert_id=None
                )

            # US27 Alternate Flow: "the slot itself may still be fillable" --
            # cascading to the next candidate on timeout is unconditional
            # (not gated on the alert's current `status`); only a
            # since-deleted alert (`alert is None`) skips it.
            alert = await self._alert_repo.get_by_id(db, offer.idle_chair_alert_id)
            if alert is not None:
                await self.offer_and_cascade(db, offer.idle_chair_alert_id)

        return count
