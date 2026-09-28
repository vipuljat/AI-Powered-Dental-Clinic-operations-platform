"""Every booking-triggered or schedule-based intelligence job: rules-based
(and, once eligible, ML) no-show risk scoring on ``booking.confirmed``
(``open_questions[Q-8a5b7a15]`` — event-driven, not a synchronous call from
``SchedulingService``), and periodic utilisation-recommendation generation
(architecture.md §5.1, project_rules.layout).

Interaction contract: consumes ``booking.confirmed``
``{appointment_id, patient_id}`` — the same event
``app/workers/outreach_retry.py`` also consumes, on its own independent
queue/consumer; neither consumer depends on the other's completion, per that
file's own Interaction Contract note.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories.console.repository import AuditLogRepository
from app.repositories.intelligence.repository import (
    MlModelEvaluationRepository,
    RiskScoreRepository,
    TriageSessionRepository,
    UtilisationRecommendationRepository,
)
from app.repositories.patients.repository import RecordLockRepository
from app.repositories.rules.repository import RuleDefinitionRepository, RuleSetRepository
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.services.console.service import AuditService
from app.services.intelligence.service import RiskScoringService, UtilisationEngine
from app.services.patients.service import RecordLockService
from app.services.rules.service import RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.core.messaging import MessageBroker

logger = get_logger(__name__)

# Queue/routing-key names this worker binds (project_rules.layout: one of the
# 5 named consumer files under app/workers/*.py).
_BOOKING_CONFIRMED_QUEUE_NAME = "recommendation_engine.booking_confirmed"
_BOOKING_CONFIRMED_ROUTING_KEY = "booking.confirmed"

# T111 AC/FR-E8.6: "regenerate on a regular cadence" without a number — a
# documented default, the same convention every other fixed-interval loop in
# this codebase's `app/workers/*.py` files uses for its own BRD-silent
# cadence.
_UTILISATION_INTERVAL_SECONDS = 60 * 60

# Every repository/service collaborator below (besides the two constructed
# lazily in `utilisation_loop`, which need the broker `run()` is given) is
# stateless: each public method takes the `AsyncSession` as an explicit
# argument, so none of these objects itself holds a live DB connection,
# socket, or other real infra client the way `app/core/messaging.py`'s
# broker or `app/core/storage.py`'s object-storage client do -- constructing
# them once, here, at import time is therefore not the "real infra client
# constructed at import time" project_rules.testing/§5 forbids (the same
# reasoning `app/workers/outreach_retry.py` documents for its own
# module-level singletons). Bound under the same name each class itself was
# imported as, so a caller/test can still monkeypatch the class reference
# and have every use below observe the replacement.
#
# `RiskScoringService` is the one exception: `handle_booking_confirmed`
# constructs a fresh instance *per invocation*, from the module-level
# (patchable) class reference imported above, rather than reusing a cached
# singleton -- the same "construct lazily from the patchable class
# reference, never cache an instance under the class's own name"
# accommodation `app/workers/escalation_dispatcher.py`'s
# `handle_escalation_triggered` documents for its own collaborators, so a
# caller/test can still `monkeypatch.setattr(recommendation_engine,
# "RiskScoringService", ...)` and have the handler observe the replacement.
AppointmentRepository = AppointmentRepository()
AppointmentHistoryRepository = AppointmentHistoryRepository()
AppointmentImportRepository = AppointmentImportRepository()
RecordLockRepository = RecordLockRepository()
RiskScoreRepository = RiskScoreRepository()
MlModelEvaluationRepository = MlModelEvaluationRepository()
TriageSessionRepository = TriageSessionRepository()
UtilisationRecommendationRepository = UtilisationRecommendationRepository()
AuditLogRepository = AuditLogRepository()
RuleSetRepository = RuleSetRepository()
RuleDefinitionRepository = RuleDefinitionRepository()

_audit_service = AuditService(AuditLogRepository)
_rule_validation_service = RuleValidationService(RuleDefinitionRepository)
_rule_set_service = RuleSetService(
    RuleSetRepository, RuleDefinitionRepository, _rule_validation_service, _audit_service
)
_lock_service = RecordLockService(RecordLockRepository)

# `SchedulingService`/`UtilisationEngine` both take a `MessageBroker`
# collaborator (`SchedulingService` directly; `UtilisationEngine.apply`
# reschedules through `SchedulingService`), and `run()` is the only place
# this worker is ever handed a real broker instance (never at import time,
# per project_rules.testing/§5) -- so both are constructed lazily, inside
# `run()`, once the broker argument is in hand, and stashed here for
# `utilisation_loop` to use.
_utilisation_engine: "UtilisationEngine | None" = None
_session_factory: "async_sessionmaker[AsyncSession] | None" = None


async def handle_booking_confirmed(payload: dict, headers: dict) -> None:
    """FR-E8.1/US40 + FR-E8.2/US41: opens a session, calls
    ``RiskScoringService.score_rules_based(db, appointment_id)`` then
    ``RiskScoringService.score_ml(db, appointment_id)`` immediately after —
    ``score_ml`` is a documented no-op unless the ML data-sufficiency gate
    has already passed, so this single consumer covers both US40 and US41
    without needing two separate event handlers. Since ``booking.confirmed``
    fires on both ``SchedulingService.create`` and ``.reschedule``, a
    rescheduled appointment is re-scored too (a fresh ``risk_scores`` row is
    written, reflecting the new ``scheduled_start``'s lead time), keeping
    ``scheduling``->``intelligence`` a one-directional edge
    (``open_questions[Q-8a5b7a15]``).

    payload = {"appointment_id", "patient_id"}.
    """

    appointment_id = payload.get("appointment_id")
    patient_id = payload.get("patient_id")

    if not appointment_id or not patient_id:
        raise ValueError("booking.confirmed payload missing appointment_id/patient_id.")

    # Constructed fresh per invocation from the module-level (patchable)
    # class reference -- see the "Every repository/service collaborator"
    # comment above this module's collaborator block for why.
    risk_scoring_service = RiskScoringService(
        RiskScoreRepository, AppointmentRepository, MlModelEvaluationRepository, _rule_set_service
    )

    # `get_db` (app/core/database.py) already commits on clean exit and
    # rolls back on exception internally -- no second commit/rollback is
    # layered on top here, so a fake `get_db` substituted in a test (which
    # need not implement that commit/rollback machinery itself) still sees
    # a handler that completes cleanly.
    async with asynccontextmanager(get_db)() as db:
        await risk_scoring_service.score_rules_based(db, appointment_id)
        await risk_scoring_service.score_ml(db, appointment_id)


async def utilisation_loop() -> None:
    """FR-E8.6/US45: the scheduled job named in architecture.md's flow table
    (``(internal, scheduled) UtilisationEngine.generate``) — every
    ``_UTILISATION_INTERVAL_SECONDS`` (a documented default cadence; T111 AC
    only says "regenerate on a regular cadence", without a number), opens a
    session and calls ``UtilisationEngine.generate(db)``.
    """

    assert _session_factory is not None, "run() must set the session factory before looping"
    assert _utilisation_engine is not None, "run() must construct the utilisation engine before looping"

    while True:
        await asyncio.sleep(_UTILISATION_INTERVAL_SECONDS)
        async with _session_factory() as db:
            try:
                recommendations = await _utilisation_engine.generate(db)
            except Exception:
                await db.rollback()
                logger.error("utilisation_generation_failed", exc_info=True)
            else:
                await db.commit()
                logger.info(
                    "utilisation_generation_completed",
                    extra={"recommendations_generated": len(recommendations)},
                )


async def run(
    broker: "MessageBroker", session_factory: "async_sessionmaker[AsyncSession]"
) -> None:
    """Composition-root entrypoint (called from ``app/worker.py``): runs the
    ``booking.confirmed`` risk-scoring consumer and the periodic
    utilisation-recommendation loop concurrently, until cancelled.
    """

    global _session_factory, _utilisation_engine
    _session_factory = session_factory

    scheduling_service = SchedulingService(
        AppointmentRepository,
        AppointmentHistoryRepository,
        AppointmentImportRepository,
        _lock_service,
        _rule_set_service,
        _audit_service,
        broker,
    )
    _utilisation_engine = UtilisationEngine(
        UtilisationRecommendationRepository,
        RiskScoreRepository,
        TriageSessionRepository,
        AppointmentRepository,
        scheduling_service,
        _audit_service,
    )

    await asyncio.gather(
        broker.consume(
            _BOOKING_CONFIRMED_QUEUE_NAME, _BOOKING_CONFIRMED_ROUTING_KEY, handle_booking_confirmed
        ),
        utilisation_loop(),
    )
