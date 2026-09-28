"""Business logic for the `education` module (architecture.md §4.2/§5.3
education module).

`EducationTriggerService` (Responsibility) selects patient-education content
by developmental stage/language and dispatches it through the shared
`OutreachService.dispatch` engine (`campaign_type="education"`) -- it never
talks to a channel adapter or writes to `outreach_messages`/`content_items`
directly, only through its injected repositories and `OutreachService`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.common.enums import (
    AppointmentStatus,
    ContentDeliveryStatus,
    EducationTriggerType,
    OutreachMessageStatus,
)
from app.repositories.scheduling.repository import AppointmentRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models.education.models import ContentDelivery, ContentItem
    from app.models.patients.models import Patient
    from app.repositories.education.repository import (
        ContentDeliveryRepository,
        ContentItemRepository,
    )
    from app.repositories.patients.repository import PatientRepository
    from app.services.outreach.service import OutreachService

__all__ = ["EducationTriggerService"]

_logger = logging.getLogger(__name__)


def _enum_value(value: object) -> object:
    """Normalizes an attribute read back off an ORM row that may be either a
    Python ``Enum`` member (the native in-process shape) or its plain string
    value (the shape a SQLite-backed test row can come back as) -- the same
    accommodation ``services/outreach/service.py`` makes for its own
    enum-typed columns.
    """
    return value.value if hasattr(value, "value") else value


class EducationTriggerService:
    """FR-E10.1-FR-E10.3/US49-US50: content selection by developmental
    stage/language and dispatch via ``OutreachService.dispatch``.
    """

    def __init__(
        self,
        content_repo: "ContentItemRepository",
        delivery_repo: "ContentDeliveryRepository",
        patient_repo: "PatientRepository",
        outreach_service: "OutreachService",
    ) -> None:
        self._content_repo = content_repo
        self._delivery_repo = delivery_repo
        self._patient_repo = patient_repo
        self._outreach_service = outreach_service
        # Watch out (spec gap, reported in the task summary): this file's
        # declared Public surface takes no scheduling repository, yet both
        # `pre_appointment`/`post_appointment` must "resolve the
        # appointment's patient" (and, for `post_appointment`, the
        # appointment's current `status`) from an `appointment_id` alone.
        # `AppointmentRepository` carries no constructor state of its own --
        # every method takes `db` explicitly, per its own frozen signature --
        # so it is instantiated locally here rather than added as a 5th
        # constructor parameter. This is a cross-module service -> repository
        # dependency, the same shape the layering rule already allows for
        # `services/scheduling/service.py` -> `RecordLockRepository` and
        # `services/waitlist/service.py` -> `ConsentRepository`.
        self._appointment_repo = AppointmentRepository()

    async def _resolve_content(
        self,
        db: "AsyncSession",
        developmental_stage: str | None,
        language: str,
        trigger_type: str,
    ) -> "ContentItem | None":
        """FR-E10.1/US49: reads the stage/language straight off the patient
        record (`patients.developmental_stage`/`patients.language`), falling
        back to the `developmental_stage="general"` content item when no
        exact stage-specific match exists (T122 AC), and logging that gap
        -- a content-library gap, not a patient-affecting action, so this is
        a plain WARNING-level `app.core.logging` line rather than an
        `AuditService` row.
        """
        content: "ContentItem | None" = None
        if developmental_stage is not None:
            content = await self._content_repo.get_by_stage_language(
                db, developmental_stage, language, trigger_type
            )
        if content is None:
            _logger.warning(
                "education.content_gap",
                extra={
                    "developmental_stage": developmental_stage,
                    "language": language,
                    "trigger_type": trigger_type,
                },
            )
            content = await self._content_repo.get_general_fallback(db, language, trigger_type)
        return content

    async def _dispatch_and_record(
        self,
        db: "AsyncSession",
        patient: "Patient",
        appointment_id: "UUID",
        content: "ContentItem",
    ) -> "ContentDelivery | None":
        """Dispatches the resolved content via `OutreachService.dispatch`
        and records the delivery.

        `OutreachService.dispatch` is itself the sole owner of consent
        resolution (FR-E6.1/FR-E6.7) -- this service has no
        `ConsentRepository` of its own (Public surface), so "if consent
        exists on the patient's preferred channel" is determined from
        `dispatch`'s own outcome: a `status="suppressed"` result means no
        consented/usable channel existed at all (US21 Alternate Flow), which
        this method treats as the "no consent exists" no-op case and never
        turns into a `content_deliveries` row.

        `preferred_channel` is passed as `None`: `patients.*` carries no
        `preferred_channel` column (only `language`/`developmental_stage`),
        so "the patient's preferred channel" is deferred entirely to
        `OutreachService`'s own consented-channel priority order
        (WhatsApp > SMS > Email > Voice) rather than invented here.
        """
        message = await self._outreach_service.dispatch(
            db,
            patient_id=patient.id,
            campaign_type="education",
            language=_enum_value(patient.language),
            related_entity_type="appointment",
            related_entity_id=appointment_id,
            preferred_channel=None,
        )
        if _enum_value(message.status) == OutreachMessageStatus.suppressed.value:
            return None

        # FR-E10.3/US50: every record_delivery call sets status to exactly
        # what the channel can support at dispatch time -- "delivered" for
        # SMS (no open-tracking capability) and "delivered" (later
        # promotable to "opened"/"completed" via webhook events handled
        # elsewhere, e.g. `ContentDeliveryRepository.record_open`/
        # `.record_completed`) for WhatsApp/email.
        return await self._delivery_repo.record_delivery(
            db,
            patient_id=patient.id,
            appointment_id=appointment_id,
            content_item_id=content.id,
            channel=_enum_value(message.channel),
            status=ContentDeliveryStatus.delivered.value,
        )

    async def pre_appointment(self, db: "AsyncSession", appointment_id: "UUID") -> "ContentDelivery | None":
        """FR-E10.1/US49: never raises for a missing-content/missing-consent
        case -- returns `None` (no-op) whenever there is no channel-capable
        content or no consent, per the Public surface.
        """
        appointment = await self._appointment_repo.get_by_id(db, appointment_id)
        if appointment is None:
            return None
        patient = await self._patient_repo.get_by_id(db, appointment.patient_id)
        if patient is None:
            return None

        content = await self._resolve_content(
            db,
            patient.developmental_stage,
            _enum_value(patient.language),
            EducationTriggerType.pre_appointment.value,
        )
        if content is None:
            return None

        return await self._dispatch_and_record(db, patient, appointment_id, content)

    async def post_appointment(self, db: "AsyncSession", appointment_id: "UUID") -> "ContentDelivery | None":
        """FR-E10.2/US49 Alternate Flow: only fires when the appointment's
        current status is exactly `"completed"` -- a `"no_show"` (or any
        other non-`"completed"`) status suppresses it entirely. The caller
        (`app/workers/recall_scanner.py`'s sweep, per this file's
        Interaction contract) guarantees this by invoking `post_appointment`
        immediately after transitioning an appointment to `"completed"`, but
        this method re-checks defensively rather than trusting that.
        """
        appointment = await self._appointment_repo.get_by_id(db, appointment_id)
        if appointment is None:
            return None
        if _enum_value(appointment.status) != AppointmentStatus.completed.value:
            return None

        patient = await self._patient_repo.get_by_id(db, appointment.patient_id)
        if patient is None:
            return None

        content = await self._resolve_content(
            db,
            patient.developmental_stage,
            _enum_value(patient.language),
            EducationTriggerType.post_appointment.value,
        )
        if content is None:
            return None

        return await self._dispatch_and_record(db, patient, appointment_id, content)

    async def get_tracking_funnel(self, db: "AsyncSession", content_item_id: "UUID | None") -> dict:
        """Thin wrapper over `ContentDeliveryRepository.get_funnel`; backs
        `GET /education/tracking`.
        """
        return await self._delivery_repo.get_funnel(db, content_item_id)

    async def list_deliveries_for_patient(
        self, db: "AsyncSession", patient_id: "UUID"
    ) -> "list[ContentDelivery]":
        """Thin wrapper over `ContentDeliveryRepository.list_by_patient`;
        backs `GET /education/deliveries`.
        """
        return await self._delivery_repo.list_by_patient(db, patient_id)
