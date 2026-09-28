"""Pydantic v2 request/response models for every ``/scheduling/*`` endpoint
(architecture.md §5.2).

Contains no behaviour beyond field-level type validation: conflict detection
(``ScheduleConflictError`` -> 409), reason-code non-emptiness, and every other
business rule live in ``services/scheduling/service.py`` — this module only
shapes the wire format.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import (
    AppointmentStatus,
    ConfirmationStatus,
    RiskLevel,
    ScoreSource,
)


class SlotOption(BaseModel):
    provider_id: UUID
    chair_id: UUID
    start: datetime
    end: datetime


class ProposeSlotsResponse(BaseModel):
    slots: list[SlotOption]
    nearest_alternatives: list[SlotOption]


class CreateAppointmentRequest(BaseModel):
    """POST /scheduling/appointments body.

    architecture.md §5.2: 422 is not documented for this endpoint (unlike
    cancel) — the required, strictly-typed (UUID/datetime) fields below are
    enough on their own to trigger FastAPI's standard RequestValidationError
    -> 422 via the global handler for a malformed body; the 409 conflict path
    is a service-layer ScheduleConflictError, not a schema-level concern.
    """

    patient_id: UUID
    provider_id: UUID
    chair_id: UUID
    appointment_type: str
    scheduled_start: datetime
    scheduled_end: datetime


class AppointmentResponse(BaseModel):
    id: UUID
    status: AppointmentStatus
    risk_flag: RiskLevel | None
    confirmation_status: ConfirmationStatus | None


class RescheduleAppointmentRequest(BaseModel):
    provider_id: UUID
    chair_id: UUID
    scheduled_start: datetime
    scheduled_end: datetime


class RescheduleAppointmentResponse(BaseModel):
    id: UUID
    status: AppointmentStatus
    source_metadata: dict


class CancelAppointmentRequest(BaseModel):
    """FR-E4.5 requires a reason code on cancellation, but the BRD does not
    enumerate a fixed set of valid codes — ``reason_code`` is therefore a free
    ``str`` here with no enum constraint; ``SchedulingService.cancel`` only
    enforces non-emptiness, not membership in a closed list (open question).
    """

    reason_code: str


class CancelAppointmentResponse(BaseModel):
    id: UUID
    status: AppointmentStatus
    is_late_cancellation: bool


class AppointmentDetailResponse(BaseModel):
    id: UUID
    patient_id: UUID
    status: AppointmentStatus
    risk_flag: RiskLevel | None
    risk_score_source: ScoreSource | None
    confirmation_status: ConfirmationStatus | None


class LockResponse(BaseModel):
    """Wire-identical to ``schemas/patients/schemas.py``'s ``LockResponse`` —
    both represent the same underlying ``record_locks`` row shape and must be
    kept in sync (see that file's own note).
    """

    locked_by: str
    expires_at: datetime
