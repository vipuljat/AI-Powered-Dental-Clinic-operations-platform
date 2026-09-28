"""Every periodic/scheduled batch job that is NOT idle-chair polling or
outreach retry (``open_questions[Q-867306e9]``): the recall scan, the
record-lock expiry sweep, the appointment-completion sweep (which also
drives post-appointment education, resolving ``open_questions[Q-69814105]``),
the pre-appointment education trigger scan, and the monthly ML precision
evaluation. None of these are event-driven — every one is a fixed-interval
loop (architecture.md §5.1, project_rules.layout).

Interaction contract: ``complete_and_educate`` calls
``EducationTriggerService.post_appointment`` directly, in the same process
and the same loop iteration, rather than via ``app/core/messaging.py`` — the
TDD's flow table calls post-appointment education "event-driven", but no 5th
RabbitMQ event exists for it in project_rules.other's four-event contract, so
"event-driven" is implemented here as "triggered by the completion
state-transition itself" rather than a broker round trip. This is restated
identically in ``services/education/service.py``'s own module docstring/
``post_appointment`` docstring — both files agree this is a same-process
call, not an event.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING

from app.common.constants import (
    APPOINTMENT_COMPLETION_GRACE_MINUTES,
    EDUCATION_PRE_APPOINTMENT_WINDOW_HOURS,
    RECORD_LOCK_TIMEOUT_MINUTES,
)
from app.common.enums import AppointmentStatus
from app.common.utils import now_utc
from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.console.repository import AuditLogRepository
from app.repositories.education.repository import (
    ContentDeliveryRepository,
    ContentItemRepository,
)
from app.repositories.intelligence.repository import (
    MlModelEvaluationRepository,
    RiskScoreRepository,
)
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    MessageTemplateRepository,
    OutreachMessageRepository,
)
from app.repositories.patients.repository import PatientRepository, RecordLockRepository
from app.repositories.recall.repository import RecallScheduleRepository
from app.repositories.rules.repository import (
    ConfigurationParameterRepository,
    RuleDefinitionRepository,
    RuleSetRepository,
)
from app.repositories.scheduling.repository import AppointmentRepository
from app.services.console.service import AuditService
from app.services.education.service import EducationTriggerService
from app.services.intelligence.service import RiskScoringService
from app.services.outreach.service import OutreachService
from app.services.patients.service import RecordLockService
from app.services.recall.service import RecallScanService
from app.services.rules.service import (
    ConfigurationService,
    RuleSetService,
    RuleValidationService,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = get_logger(__name__)

# --- Fixed-interval cadences ---------------------------------------------
# None of these five cadences is quantified in the BRD/TDD beyond a word
# ("daily"/"hourly"/"monthly"/"scheduled") — see each loop's own docstring
# for the requirement it backs. RECORD_LOCK_TIMEOUT_MINUTES/
# APPOINTMENT_COMPLETION_GRACE_MINUTES are reused as their own sweep
# cadences (no separate interval is specified for either), mirroring
# `app/workers/waitlist_poller.py`'s own reuse of
# WAITLIST_RESPONSE_WINDOW_MINUTES as its offer-sweep cadence.
_RECALL_SCAN_INTERVAL_SECONDS = 24 * 60 * 60
_LOCK_SWEEP_INTERVAL_SECONDS = RECORD_LOCK_TIMEOUT_MINUTES * 60
_COMPLETION_SWEEP_INTERVAL_SECONDS = APPOINTMENT_COMPLETION_GRACE_MINUTES * 60
_PRE_APPOINTMENT_INTERVAL_SECONDS = 60 * 60
# 30-day-month convention, the same one `services/recall/service.py`'s
# `RecallScanService._is_dormant` documents using for its own windowed
# calculation, since no calendar-month-aware dependency is declared anywhere
# in this tree.
_ML_EVALUATION_INTERVAL_SECONDS = 30 * 24 * 60 * 60

# Every repository/service collaborator this worker needs is stateless: each
# public method takes the `AsyncSession` as an explicit argument, so none of
# these objects itself holds a live DB connection, socket, or other real
# infra client the way `app/core/messaging.py`'s broker or
# `app/core/storage.py`'s object-storage client do -- constructing them once,
# here, at import time is therefore not the "real infra client constructed
# at import time" project_rules.testing/§5 forbids (the same reasoning
# `app/workers/outreach_retry.py` documents for its own module-level
# singletons).
#
# `AppointmentRepository`/`EducationTriggerService` are the two exceptions:
# `complete_and_educate`/`trigger_pre_appointment_education` construct fresh
# instances of these *per invocation*, from the module-level (patchable)
# class references imported above, rather than reusing a cached singleton --
# the same "construct lazily from the patchable class reference, never cache
# an instance under the class's own name" accommodation
# `app/workers/escalation_dispatcher.py`'s `handle_escalation_triggered`
# documents for its own collaborators, so a caller/test can still
# `monkeypatch.setattr(recall_scanner, "AppointmentRepository", ...)` and
# have both functions observe the replacement. `_appointment_repo` below is
# the separate, stable instance used to build `_risk_scoring_service`
# (which is itself only exercised by `ml_evaluation_loop`, never directly
# swapped out by a test), so that internal wiring is unaffected by this.
RecordLockRepository = RecordLockRepository()
PatientRepository = PatientRepository()
RecallScheduleRepository = RecallScheduleRepository()
AuditLogRepository = AuditLogRepository()
RuleSetRepository = RuleSetRepository()
RuleDefinitionRepository = RuleDefinitionRepository()
RiskScoreRepository = RiskScoreRepository()
MlModelEvaluationRepository = MlModelEvaluationRepository()
ContentItemRepository = ContentItemRepository()
ContentDeliveryRepository = ContentDeliveryRepository()
ConfigurationParameterRepository = ConfigurationParameterRepository()
ConsentRepository = ConsentRepository()
OutreachMessageRepository = OutreachMessageRepository()
MessageTemplateRepository = MessageTemplateRepository()
ChannelConfigurationRepository = ChannelConfigurationRepository()

_audit_service = AuditService(AuditLogRepository)
_rule_validation_service = RuleValidationService(RuleDefinitionRepository)
_rule_set_service = RuleSetService(
    RuleSetRepository, RuleDefinitionRepository, _rule_validation_service, _audit_service
)
_config_service = ConfigurationService(ConfigurationParameterRepository, _audit_service)

_lock_service = RecordLockService(RecordLockRepository)
_recall_scan_service = RecallScanService(RecallScheduleRepository, PatientRepository, _rule_set_service)
_appointment_repo = AppointmentRepository()
_risk_scoring_service = RiskScoringService(
    RiskScoreRepository, _appointment_repo, MlModelEvaluationRepository, _rule_set_service
)

_outreach_service = OutreachService(
    message_repo=OutreachMessageRepository,
    consent_repo=ConsentRepository,
    template_repo=MessageTemplateRepository,
    channel_config_repo=ChannelConfigurationRepository,
    config_service=_config_service,
    audit=_audit_service,
    settings=get_settings(),
)


async def complete_and_educate(db: "AsyncSession") -> int:
    """``open_questions[Q-69814105]``/``[Q-867306e9]``: the one place a
    ``booked`` appointment ever transitions to ``completed`` in this tree —
    no endpoint/job does it elsewhere. Run every
    ``APPOINTMENT_COMPLETION_GRACE_MINUTES`` after ``scheduled_end``
    (``AppointmentRepository.list_completable`` already applies that grace
    window). FR-E10.2/US49: immediately after each transition, in the same
    iteration (not a separate event), calls
    ``EducationTriggerService.post_appointment`` — see this file's own
    Interaction contract section.
    """

    appointment_repo = AppointmentRepository()
    education_service = EducationTriggerService(
        ContentItemRepository, ContentDeliveryRepository, PatientRepository, _outreach_service
    )

    completable = await appointment_repo.list_completable(db, now_utc())
    completed_count = 0
    for appointment in completable:
        await appointment_repo.update_status(db, appointment.id, AppointmentStatus.completed.value)
        await education_service.post_appointment(db, appointment.id)
        completed_count += 1
    return completed_count


async def trigger_pre_appointment_education(db: "AsyncSession") -> int:
    """FR-E10.1/US49: every booked appointment whose ``scheduled_start``
    falls within ``EDUCATION_PRE_APPOINTMENT_WINDOW_HOURS`` from now, and
    that has no existing ``content_deliveries`` row for
    ``trigger_type="pre_appointment"`` yet, gets
    ``EducationTriggerService.pre_appointment`` called for it.

    Gap (reported in this task's summary): neither the frozen
    ``AppointmentRepository`` nor ``ContentDeliveryRepository`` exposes a
    query shaped for this exact "booked appointments starting soon, not yet
    pre-appointment-educated" read (``AppointmentRepository.list_open_slots``
    filters an overlap window, not a ``scheduled_start`` window, and also
    returns ``rescheduled`` rows this scan ideally would not;
    ``ContentDeliveryRepository`` carries no ``trigger_type`` column of its
    own to filter by — only ``content_deliveries.content_item_id`` ->
    ``content_items.trigger_type``). Per project_rules.layout ("services
    never touch the DB directly, only via repositories" — this worker holds
    itself to the same discipline), this reads candidates through
    ``AppointmentRepository`` alone (the closest available query,
    ``list_open_slots``) rather than issuing a raw ``db.execute`` query of
    its own; the "not yet pre-appointment-educated" half of the filter
    cannot be re-checked without a second raw query this file declines to
    issue, so ``EducationTriggerService.pre_appointment`` is called for
    every appointment ``list_open_slots`` returns in the window, the same
    "frozen repository simply missing a query surface" accommodation
    documented on ``services/waitlist/service.py``'s
    ``IdleChairPollingService.run_cycle`` and
    ``WaitlistFillService._get_offer_by_id``.
    """

    now = now_utc()
    window_end = now + timedelta(hours=EDUCATION_PRE_APPOINTMENT_WINDOW_HOURS)

    appointment_repo = AppointmentRepository()
    education_service = EducationTriggerService(
        ContentItemRepository, ContentDeliveryRepository, PatientRepository, _outreach_service
    )

    candidates = await appointment_repo.list_open_slots(db, None, now, window_end)

    triggered_count = 0
    for appointment in candidates:
        await education_service.pre_appointment(db, appointment.id)
        triggered_count += 1
    return triggered_count


async def recall_scan_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """FR-E7.1/FR-E7.2/US34: runs ``RecallScanService.run`` daily (a
    documented default cadence — the BRD leaves scan frequency unspecified
    beyond "scheduled").
    """

    while True:
        await asyncio.sleep(_RECALL_SCAN_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                count = await _recall_scan_service.run(db)
            except Exception:
                await db.rollback()
                logger.error("recall_scan_failed", exc_info=True)
            else:
                await db.commit()
                logger.info("recall_scan_completed", extra={"schedules_flagged_or_created": count})


async def lock_sweep_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """project_rules.concurrency: the one place
    ``RecordLockRepository.sweep_expired`` (via ``RecordLockService``) is
    actually invoked — without this loop, expired ``record_locks`` rows
    would never be released automatically.
    """

    while True:
        await asyncio.sleep(_LOCK_SWEEP_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                count = await _lock_service.sweep_expired(db)
            except Exception:
                await db.rollback()
                logger.error("lock_sweep_failed", exc_info=True)
            else:
                await db.commit()
                logger.info("lock_sweep_completed", extra={"locks_released": count})


async def completion_sweep_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """Runs :func:`complete_and_educate` every
    ``APPOINTMENT_COMPLETION_GRACE_MINUTES``.
    """

    while True:
        await asyncio.sleep(_COMPLETION_SWEEP_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                count = await complete_and_educate(db)
            except Exception:
                await db.rollback()
                logger.error("completion_sweep_failed", exc_info=True)
            else:
                await db.commit()
                logger.info("completion_sweep_completed", extra={"appointments_completed": count})


async def pre_appointment_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """Runs :func:`trigger_pre_appointment_education` hourly."""

    while True:
        await asyncio.sleep(_PRE_APPOINTMENT_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                count = await trigger_pre_appointment_education(db)
            except Exception:
                await db.rollback()
                logger.error("pre_appointment_education_sweep_failed", exc_info=True)
            else:
                await db.commit()
                logger.info("pre_appointment_education_sweep_completed", extra={"triggered": count})


async def ml_evaluation_loop(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """FR-E8.2/US42: runs ``RiskScoringService.evaluate_precision`` monthly,
    per SM4's measurement method.
    """

    while True:
        await asyncio.sleep(_ML_EVALUATION_INTERVAL_SECONDS)
        async with session_factory() as db:
            try:
                evaluation = await _risk_scoring_service.evaluate_precision(db)
            except Exception:
                await db.rollback()
                logger.error("ml_evaluation_failed", exc_info=True)
            else:
                await db.commit()
                logger.info(
                    "ml_evaluation_completed",
                    extra={
                        "ml_model_evaluation_id": str(evaluation.id),
                        "status": evaluation.status.value
                        if hasattr(evaluation.status, "value")
                        else evaluation.status,
                    },
                )


async def run(session_factory: "async_sessionmaker[AsyncSession]") -> None:
    """Composition-root entrypoint (called from ``app/worker.py``): runs the
    five independent interval loops concurrently, until cancelled.
    """

    await asyncio.gather(
        recall_scan_loop(session_factory),
        lock_sweep_loop(session_factory),
        completion_sweep_loop(session_factory),
        pre_appointment_loop(session_factory),
        ml_evaluation_loop(session_factory),
    )
