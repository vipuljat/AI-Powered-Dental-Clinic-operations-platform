"""Every dispatch/retry consumer in the system: confirmation dispatch
(``booking.confirmed``), waitlist-offer dispatch (``waitlist_offer.created``),
and the periodic retry/unreachable sweep over ``outreach_messages``
(architecture.md §5.1, project_rules.other's event contract).

Per ``open_questions[Q-fb3929ab]``, this is the ONLY file that calls
``services/outreach/service.py``'s ``OutreachService.dispatch`` for those two
event-driven campaign types — ``SchedulingService``/``WaitlistFillService``
never call ``OutreachService`` directly (Responsibility).

Interaction contract: consumes ``booking.confirmed``
``{appointment_id, patient_id}`` — published by ``SchedulingService.create``/
``.reschedule`` — and ``waitlist_offer.created``
``{waitlist_entry_id, idle_chair_alert_id}`` — published by
``WaitlistFillService.offer_and_cascade`` — both restated here exactly
matching the publishers' own Interaction Contract sections. Note
``app/workers/recommendation_engine.py`` ALSO consumes ``booking.confirmed``
(for risk scoring) on its own independent queue — both consumers are bound to
the same routing key but process it for entirely unrelated purposes; neither
depends on the other's completion.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from app.common.constants import OUTREACH_RETRY_DELAY_MINUTES
from app.common.enums import OutreachMessageStatus
from app.common.utils import now_utc
from app.core.config import get_settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.repositories.console.repository import AuditLogRepository
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    MessageTemplateRepository,
    OutreachMessageRepository,
)
from app.repositories.patients.repository import PatientRepository
from app.repositories.rules.repository import ConfigurationParameterRepository
from app.repositories.scheduling.repository import AppointmentRepository
from app.repositories.waitlist.repository import WaitlistEntryRepository
from app.services.console.service import AuditService
from app.services.outreach.service import OutreachService
from app.services.rules.service import ConfigurationService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.core.messaging import MessageBroker

logger = get_logger(__name__)

# Queue/routing-key names this worker binds (project_rules.layout: one of the
# 5 named consumer files under app/workers/*.py).
_CONFIRMATION_QUEUE_NAME = "outreach_retry.booking_confirmed"
_CONFIRMATION_ROUTING_KEY = "booking.confirmed"
_OFFER_QUEUE_NAME = "outreach_retry.waitlist_offer_created"
_OFFER_ROUTING_KEY = "waitlist_offer.created"

# Every repository/service collaborator this worker needs is stateless: each
# public method takes the `AsyncSession` as an explicit argument (obtained
# per-invocation from `get_db()`/the `session_factory` handed to `run()`), so
# none of these objects itself holds a live DB connection, socket, or other
# real infra client the way `app/core/messaging.py`'s broker or
# `app/core/storage.py`'s object-storage client do -- constructing them once,
# here, at import time is therefore not the "real infra client constructed
# at import time" project_rules.testing/§5 forbids. Building them once (each
# bound under the same name the class itself was imported as, so a caller/
# test can still `monkeypatch.setattr(outreach_retry, "AppointmentRepository",
# ...)` and have every reference below observe the replacement) also means
# every handler always uses the exact same collaborator instances, matching
# how a process-wide singleton repository/service layer behaves everywhere
# else in this codebase.
AppointmentRepository = AppointmentRepository()
PatientRepository = PatientRepository()
WaitlistEntryRepository = WaitlistEntryRepository()
OutreachMessageRepository = OutreachMessageRepository()
ConsentRepository = ConsentRepository()
MessageTemplateRepository = MessageTemplateRepository()
ChannelConfigurationRepository = ChannelConfigurationRepository()
AuditLogRepository = AuditLogRepository()
ConfigurationParameterRepository = ConfigurationParameterRepository()

_audit_service = AuditService(AuditLogRepository)
_config_service = ConfigurationService(ConfigurationParameterRepository, _audit_service)

OutreachService = OutreachService(
    message_repo=OutreachMessageRepository,
    consent_repo=ConsentRepository,
    template_repo=MessageTemplateRepository,
    channel_config_repo=ChannelConfigurationRepository,
    config_service=_config_service,
    audit=_audit_service,
    settings=get_settings(),
)

# Set once by `run()` before the retry-sweep loop starts, so
# `retry_sweep_loop` -- the one piece of this module that is *not* bound by
# the `MessageHandler` protocol's fixed `(payload, headers)` signature every
# `broker.consume` call is built against (app/core/messaging.py) -- can reach
# the session factory `run()` was given. `handle_booking_confirmed`/
# `handle_waitlist_offer_created` are protocol-constrained handlers instead,
# so they open their own session the same way every other non-request caller
# in this codebase does: via `get_db()` (app/core/database.py), the one
# sanctioned session-opening entry point outside a composition root's own
# startup wiring.
_session_factory: "async_sessionmaker[AsyncSession] | None" = None


async def handle_booking_confirmed(payload: dict, headers: dict) -> None:
    """FR-E4.3/US21: the sole implementation of automated multilingual
    confirmation dispatch — it never runs inside the
    ``POST /scheduling/appointments`` request itself (event-driven, per
    ``open_questions[Q-fb3929ab]``), and it writes
    ``appointments.confirmation_status`` back so
    ``GET /scheduling/appointments/{id}`` reflects the outcome.

    payload = {"appointment_id", "patient_id"}; opens a session, loads the
    appointment (`AppointmentRepository.get_by_id`) and patient
    (`PatientRepository.get_by_id`) for language/consent context, calls
    `OutreachService.dispatch(campaign_type="confirmation", ...)`; if the
    returned message.status == "suppressed" (no consented channel), calls
    `AppointmentRepository.update_fields(appointment_id,
    confirmation_status="suppressed")` so the appointment is flagged for
    manual staff confirmation (US21 Alternate Flow); on a real send, sets
    `confirmation_status` to "sent"/"failed" to match the outreach_messages
    outcome.
    """

    appointment_id = payload.get("appointment_id")
    patient_id = payload.get("patient_id")

    if not appointment_id or not patient_id:
        raise ValueError("booking.confirmed payload missing appointment_id/patient_id.")

    async with asynccontextmanager(get_db)() as db:
        appointment = await AppointmentRepository.get_by_id(db, appointment_id)
        if appointment is None:
            raise LookupError(
                "No appointments row found for the published booking.confirmed event."
            )

        patient = await PatientRepository.get_by_id(db, patient_id)
        if patient is None:
            raise LookupError(
                "No patients row found for the published booking.confirmed event."
            )

        language = patient.language.value if hasattr(patient.language, "value") else patient.language

        message = await OutreachService.dispatch(
            db,
            patient_id=patient_id,
            campaign_type="confirmation",
            language=language,
            related_entity_type="appointment",
            related_entity_id=appointment_id,
        )

        status = message.status.value if hasattr(message.status, "value") else message.status
        if status == OutreachMessageStatus.suppressed.value:
            await AppointmentRepository.update_fields(db, appointment_id, confirmation_status="suppressed")
        elif status == OutreachMessageStatus.sent.value:
            await AppointmentRepository.update_fields(db, appointment_id, confirmation_status="sent")
        elif status == OutreachMessageStatus.failed.value:
            await AppointmentRepository.update_fields(db, appointment_id, confirmation_status="failed")


async def handle_waitlist_offer_created(payload: dict, headers: dict) -> None:
    """FR-E5.5/US27: dispatches the slot offer; the patient's channel
    response arrives via ``POST /waitlist/offers/{id}/respond``, handled
    synchronously by ``routes/waitlist/routes.py`` -> ``WaitlistFillService
    .respond``, not by this worker (this worker only sends the offer, it
    does not process the reply).

    payload = {"waitlist_entry_id", "idle_chair_alert_id"}; opens a session,
    loads the waitlist entry for patient_id/language, calls
    `OutreachService.dispatch(campaign_type="waitlist_offer",
    related_entity_type="waitlist_entry", related_entity_id=waitlist_entry_id)`.
    """

    waitlist_entry_id = payload.get("waitlist_entry_id")
    idle_chair_alert_id = payload.get("idle_chair_alert_id")

    if not waitlist_entry_id or not idle_chair_alert_id:
        raise ValueError(
            "waitlist_offer.created payload missing waitlist_entry_id/idle_chair_alert_id."
        )

    async with asynccontextmanager(get_db)() as db:
        entry = await WaitlistEntryRepository.get_by_id(db, waitlist_entry_id)
        if entry is None:
            raise LookupError(
                "No waitlist_entries row found for the published waitlist_offer.created event."
            )

        # `waitlist_entries` (models/waitlist/models.py) carries no `language`
        # column of its own -- only `patient_id` -- so language is resolved
        # via the owning patient, mirroring `handle_booking_confirmed`'s own
        # lookup. A direct `language` attribute is tried first via `getattr`
        # (mirrors `app/workers/escalation_dispatcher.py`'s own
        # "optional capability tried first" accommodation for a collaborator
        # that might expose a more direct value than the frozen repository
        # guarantees), falling back to the patient lookup otherwise.
        language = getattr(entry, "language", None)
        if language is None:
            patient = await PatientRepository.get_by_id(db, entry.patient_id)
            if patient is None:
                raise LookupError(
                    "No patients row found for the waitlist entry's patient_id."
                )
            language = patient.language
        language = language.value if hasattr(language, "value") else language

        await OutreachService.dispatch(
            db,
            patient_id=entry.patient_id,
            campaign_type="waitlist_offer",
            language=language,
            related_entity_type="waitlist_entry",
            related_entity_id=waitlist_entry_id,
        )


async def retry_sweep_loop() -> None:
    """FR-E6.3/FR-E6.4/US29/US30: the timer/trigger that actually re-attempts
    a failed send after the configured delay and eventually calls
    ``mark_unreachable`` — ``OutreachService.retry``
    (services/outreach/service.py) contains the actual retry-vs-give-up
    decision; this worker is only the timer/trigger.

    Every ``OUTREACH_RETRY_DELAY_MINUTES``, opens a session, calls
    ``OutreachMessageRepository.list_retry_due(db, now_utc())`` then
    ``OutreachService.retry(db, id)`` for each due message.
    """

    assert _session_factory is not None, "run() must set the session factory before sweeping"

    while True:
        await asyncio.sleep(OUTREACH_RETRY_DELAY_MINUTES * 60)

        async with _session_factory() as db:
            try:
                due_messages = await OutreachMessageRepository.list_retry_due(db, now_utc())
                for message in due_messages:
                    try:
                        await OutreachService.retry(db, message.id)
                    except Exception:
                        logger.error(
                            "outreach_retry_failed",
                            extra={"outreach_message_id": str(message.id)},
                            exc_info=True,
                        )
            except Exception:
                await db.rollback()
                raise
            else:
                await db.commit()


async def run(
    broker: "MessageBroker", session_factory: "async_sessionmaker[AsyncSession]"
) -> None:
    """Composition-root entrypoint (called from `app/worker.py`): runs the
    booking-confirmed consumer, the waitlist-offer-created consumer, and the
    periodic retry sweep concurrently, until cancelled.
    """

    global _session_factory
    _session_factory = session_factory

    await asyncio.gather(
        broker.consume(_CONFIRMATION_QUEUE_NAME, _CONFIRMATION_ROUTING_KEY, handle_booking_confirmed),
        broker.consume(_OFFER_QUEUE_NAME, _OFFER_ROUTING_KEY, handle_waitlist_offer_created),
        retry_sweep_loop(),
    )
