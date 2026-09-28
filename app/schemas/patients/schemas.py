"""Pydantic v2 request/response models for every ``/patients/*`` endpoint
(architecture.md §5.2).

Contains no behaviour beyond field-level type/format validation (delegated to
``app.common.validators``) — dedup detection, the record-lock lifecycle, and
every other business rule live in ``services/patients/service.py``; this
module only shapes the wire format.
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.common.enums import Channel, ConsentStatus, Language, PatientStatus
from app.common.validators import validate_non_empty, validate_phone


class CreatePatientRequest(BaseModel):
    first_name: str
    last_name: str
    phone: str
    dob: date | None = None
    language: Language = Language.pt
    demographics: dict | None = None

    _validate_first_name = field_validator("first_name")(validate_non_empty)
    _validate_last_name = field_validator("last_name")(validate_non_empty)
    _validate_phone = field_validator("phone")(validate_phone)


class PatientResponse(BaseModel):
    id: UUID
    patient_code: str
    first_name: str
    last_name: str
    status: PatientStatus


class PatientSummary(BaseModel):
    id: UUID
    patient_code: str


class DuplicateCandidate(BaseModel):
    """architecture.md §5.2 ``POST /patients`` 409 body:
    ``{ "duplicate_candidate": { "id": "uuid", "patient_code": "PT-00118" } }``
    — the exact envelope the route returns nested inside the standard error
    envelope's ``details`` (project_rules.errors).
    """

    duplicate_candidate: PatientSummary


class PatientSearchItem(BaseModel):
    id: UUID
    patient_code: str
    first_name: str
    last_name: str
    phone: str


class PatientSearchResponse(BaseModel):
    items: list[PatientSearchItem]


class PatientAppointmentSummary(BaseModel):
    id: UUID
    scheduled_start: datetime
    status: str


class PatientConsentSummary(BaseModel):
    channel: Channel
    status: ConsentStatus


class PatientProfileResponse(BaseModel):
    id: UUID
    patient_code: str
    first_name: str
    last_name: str
    status: PatientStatus
    appointments: list[PatientAppointmentSummary]
    consent: list[PatientConsentSummary]
    recall_status: str | None


class LockResponse(BaseModel):
    """FR-E2.4: returned by both ``POST /patients/{id}/lock`` and
    ``POST /scheduling/appointments/{id}/lock``. Wire-identical to
    ``schemas/scheduling/schemas.py``'s ``LockResponse`` — both represent the
    same underlying ``record_locks`` row shape and must be kept in sync (see
    that file's own note).
    """

    locked_by: str
    expires_at: datetime


class UpdatePatientRequest(BaseModel):
    phone: str | None = None
    language: Language | None = None
    email: str | None = None
    demographics: dict | None = None

    @field_validator("phone")
    @classmethod
    def _validate_phone(cls, value: str | None) -> str | None:
        # Only runs when phone is not None — validate_phone assumes a str.
        if value is None:
            return value
        return validate_phone(value)


class ArchivePatientRequest(BaseModel):
    reason: str

    _validate_reason = field_validator("reason")(validate_non_empty)


class ArchivePatientResponse(BaseModel):
    id: UUID
    status: PatientStatus
    archived_reason: str


class DateRange(BaseModel):
    from_: date = Field(alias="from")
    to: date

    model_config = {"populate_by_name": True}


class ExportPatientDataRequest(BaseModel):
    entity: str  # "patients" | "appointments"
    date_range: DateRange
    de_identified: bool = False


class ExportResponse(BaseModel):
    export_id: UUID
    status: str
    download_url: str | None
