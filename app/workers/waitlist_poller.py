"""The idle-chair detection consumer: runs
``IdleChairPollingService.run_cycle`` on a fixed interval AND consumes
``appointment.cancelled`` for immediate event-driven detection (Flow-3
diagram fact: both paths exist, not poll-only). Also sweeps waitlist offers
past their response window (cascade-on-timeout), since that is topically a
waitlist concern even though ``WaitlistFillService`` itself lives in the
``services/waitlist`` module (architecture.md §5.1, project_rules.layout).

Interaction contract: consumes ``appointment.cancelled``
``{appointment_id, provider_id, chair_id, slot_start, slot_end}``, published
by ``SchedulingService.cancel`` (services/scheduling/service.py) — exact
payload shape restated here matches that file's Interaction Contract. Each
iteration of every loop opens and closes its own ``AsyncSession`` (never
holds one across the whole worker lifetime) — a long-lived session would
accumulate stale ORM identity-map state across hours of polling.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from app.common.constants import (
    IDLE_CHAIR_POLL_INTERVAL_MINUTES,
    WAITLIST_RESPONSE_WINDOW_MINUTES,
)
from app.common.enums import IdleChairSource
from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories.console.repository import AuditLogRepository
from app.repositories.patients.repository import RecordLockRepository
from app.repositories.rules.repository import RuleDefinitionRepository, RuleSetRepository
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.repositories.waitlist.repository import (
    IdleChairAlertRepository,
    WaitlistEntryRepository,
    WaitlistOfferRepository,
)
from app.services.console.service import AuditService
from app.services.patients.service import RecordLockService
from app.services.rules.service import RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService
from app.services.waitlist.service import IdleChairPollingService, WaitlistFillService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.core.messaging import MessageBroker

logger = get_logger(__name__)

# Queue/routing-key name this worker binds (project_rules.layout: one of the
# 5 named consumer files under app/workers/*.py).
_CANCELLATION_QUEUE_NAME = "waitlist_poller.appointment_cancelled"
_CANCELLATION_ROUTING_KEY = "appointment.cancelled"

# No separate cadence is specified for the offer-response sweep, so
# WAITLIST_RESPONSE_WINDOW_MINUTES (the window itself) is reused as the
# sweep interval, per this file's own spec.
_OFFER_SWEEP_INTERVAL_SECONDS = WAITLIST_RESPONSE_WINDOW_MINUTES * 60
_POLL_INTERVAL_SECONDS = IDLE_CHAIR_POLL_INTERVAL_MINUTES * 60

# Every repository/service collaborator below (besides the two constructed
# lazily in `run()`, which need the broker it is given) is stateless: each
# public method takes the `AsyncSession` as an explicit argument, so none of
# these objects itself holds a live DB connection, socket, or other real
# infra client the way `app/core/messaging.py`'s broker or
# `app/core/storage.py`'s object-storage client do -- constructing them once,
# here, at import time is therefore not the "real infra client constructed
# at import time" project_rules.testing/§5 forbids (the same reasoning
# `app/workers/outreach_retry.py` documents for its own module-level
# singletons). Bound under the same name each class itself was imported as,
# so a caller/test can still monkeypatch the class reference and have every
# use below observe the replacement.
#
# `IdleChairPollingService` is the one exception: `handle_cancellation`/
# `poll_loop` each construct a fresh instance *per invocation*, from the
# module-level (patchable) class reference imported above, rather than
# reusing a cached singleton -- the same "construct lazily from the
# patchable class reference, never cache an instance under the class's own
# name" accommodation `app/workers/escalation_dispatcher.py`'s
# `handle_escalation_triggered` documents for its own collaborators, so a
# caller/test can still `monkeypatch.setattr(waitlist_poller,
# "IdleChairPollingService", ...)` and have both observe the replacement.
AppointmentRepository = AppointmentRepository()
AppointmentHistoryRepository = AppointmentHistoryRepository()
AppointmentImportRepository = AppointmentImportRepository()
RecordLockRepository = RecordLockRepository()
IdleChairAlertRepository = IdleChairAlertRepository()
WaitlistEntryRepository = WaitlistEntryRepository()
WaitlistOfferRepository = WaitlistOfferRepository()
AuditLogRepository = AuditLogRepository()
RuleSetRepository = RuleSetRepository()
RuleDefinitionRepository = RuleDefinitionRepository()

_audit_service = AuditService(AuditLogRepository)
_rule_validation_service = RuleValidationService(RuleDefinitionRepository)
_rule_set_service = RuleSetService(
    RuleSetRepository, RuleDefinitionRepository, _rule_validation_service, _audit_service
)
_lock_service = RecordLockService(RecordLockRepository)

# `SchedulingService`/`WaitlistFillService` both take a `MessageBroker`
# collaborator, and `run()` is the only place this worker is ever handed a
# real broker instance (never at import time, per project_rules.testing/§5)
# -- so both are constructed lazily, inside `run()`, once the broker
# argument is in hand, and stashed here for `offer_sweep_loop` to use.
_waitlist_fill_service: "WaitlistFillService | None" = None


async def handle_cancellation(payload: dict, headers: dict) -> None:
    """FR-E5.1 (Flow-3 diagram fact): the event-driven idle-chair-detection
    path — immediate consumption of ``appointment.cancelled``, complementing
    :func:`poll_loop`'s scheduled-scan path; both ultimately call
    :meth:`IdleChairPollingService.detect_from_slot` so dedup (FR-E5.2)
    applies uniformly regardless of which path detected the slot first.

    payload = {"appointment_id", "provider_id", "chair_id", "slot_start",
    "slot_end"}; opens a fresh ``AsyncSession`` via ``get_db()``
    (app/core/database.py) -- the sanctioned session-opening entry point for
    a handler whose signature the ``MessageHandler`` protocol fixes to
    exactly ``(payload, headers)``, so it cannot receive ``session_factory``
    as a parameter the way ``poll_loop``/``offer_sweep_loop`` do -- calls
    ``IdleChairPollingService.detect_from_slot(db, provider_id, chair_id,
    slot_start, slot_end, source="auto_detected",
    appointment_id=appointment_id)``, commits.
    """

    appointment_id = payload.get("appointment_id")
    provider_id = payload.get("provider_id")
    chair_id = payload.get("chair_id")
    slot_start = payload.get("slot_start")
    slot_end = payload.get("slot_end")

    if not appointment_id or not provider_id or not chair_id or not slot_start or not slot_end:
        raise ValueError(
            "appointment.cancelled payload missing appointment_id/provider_id/"
            "chair_id/slot_start/slot_end."
        )

    # Constructed fresh per invocation from the module-level (patchable)
    # class reference -- see the "Every repository/service collaborator"
    # comment above this module's collaborator block for why.
    idle_chair_service = IdleChairPollingService(IdleChairAlertRepository, _audit_service)

    async with asynccontextmanager(get_db)() as db:
        try:
            await idle_chair_service.detect_from_slot(
                db,
                provider_id,
                chair_id,
                slot_start,
                slot_end,
                source=IdleChairSource.auto_detected.value,
                appointment_id=appointment_id,
            )
        except Exception:
            await db.rollback()
            raise
        else:
            await db.commit()


async def poll_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """FR-E5.1/US23: the scheduled-scan path — every
    ``IDLE_CHAIR_POLL_INTERVAL_MINUTES``, opens a session and calls
    ``IdleChairPollingService.run_cycle(db)``.
    """

    idle_chair_service = IdleChairPollingService(IdleChairAlertRepository, _audit_service)

    while True:
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                new_alerts = await idle_chair_service.run_cycle(db)
            except Exception:
                await db.rollback()
                logger.error("idle_chair_poll_cycle_failed", exc_info=True)
            else:
                await db.commit()
                logger.info("idle_chair_poll_cycle_completed", extra={"new_alerts": new_alerts})


async def offer_sweep_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """FR-E5.5/US27 Alternate Flow: makes "doesn't respond within the
    configured window... offer cascades to the next-priority patient"
    actually happen without a human polling the API — every
    ``WAITLIST_RESPONSE_WINDOW_MINUTES`` (reused as the sweep cadence, no
    separate interval is specified), opens a session and calls
    ``WaitlistFillService.sweep_expired_offers(db)`` — the cascade logic
    itself lives in ``services/waitlist/service.py``; this worker only calls
    it on a timer.
    """

    assert _waitlist_fill_service is not None, "run() must construct the fill service before sweeping"

    while True:
        await asyncio.sleep(_OFFER_SWEEP_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                expired_count = await _waitlist_fill_service.sweep_expired_offers(db)
            except Exception:
                await db.rollback()
                logger.error("waitlist_offer_sweep_failed", exc_info=True)
            else:
                await db.commit()
                logger.info("waitlist_offer_sweep_completed", extra={"offers_expired": expired_count})


async def run(
    broker: "MessageBroker", session_factory: "async_sessionmaker[AsyncSession]"
) -> None:
    """Composition-root entrypoint (called from ``app/worker.py``): starts
    three concurrent loops -- the scheduled idle-chair poll, the
    ``appointment.cancelled`` consumer, and the waitlist-offer sweep -- until
    cancelled.
    """

    global _waitlist_fill_service

    scheduling_service = SchedulingService(
        AppointmentRepository,
        AppointmentHistoryRepository,
        AppointmentImportRepository,
        _lock_service,
        _rule_set_service,
        _audit_service,
        broker,
    )
    _waitlist_fill_service = WaitlistFillService(
        WaitlistOfferRepository,
        WaitlistEntryRepository,
        IdleChairAlertRepository,
        scheduling_service,
        broker,
        _audit_service,
    )

    await asyncio.gather(
        poll_loop(session_factory),
        broker.consume(_CANCELLATION_QUEUE_NAME, _CANCELLATION_ROUTING_KEY, handle_cancellation),
        offer_sweep_loop(session_factory),
    )
