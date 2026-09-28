"""Unit tests for app/schemas/patients/schemas.py.

These are pure Pydantic v2 models with no DB/HTTP access, so each is
constructed directly and checked for the exact shape/validation behaviour
its own spec declares.

`Language`, `PatientStatus`, `Channel`, `ConsentStatus` are not defined by
this spec but are its own dependency (app/common/enums.py); those tests
exercise the real enum members via introspection rather than hard-coding a
guessed member name that might drift from the source of truth.

`validate_non_empty`/`validate_phone` are likewise not defined by this spec
(app/common/validators.py) but are wired in verbatim as this file's own
field validators, so their pass/fail boundary is exercised here through the
schema's own public surface (constructing `CreatePatientRequest` etc.),
never by importing/calling the validator function directly.

The Interaction contract note that `LockResponse` must stay wire-identical
to `schemas/scheduling/schemas.py`'s own `LockResponse` describes a
cross-file invariant that this file's own public surface cannot observe (it
has no visibility into the scheduling module's schema); it is exercised here
only by pinning `LockResponse`'s own two-field shape, and noted again in the
summary.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from app.common.enums import Channel, ConsentStatus, Language, PatientStatus
from app.schemas.patients.schemas import (
    ArchivePatientRequest,
    ArchivePatientResponse,
    CreatePatientRequest,
    DateRange,
    DuplicateCandidate,
    ExportPatientDataRequest,
    ExportResponse,
    LockResponse,
    PatientAppointmentSummary,
    PatientConsentSummary,
    PatientProfileResponse,
    PatientResponse,
    PatientSearchItem,
    PatientSearchResponse,
    PatientSummary,
    UpdatePatientRequest,
)


VALID_PHONE = "+1 (555) 123-4567"  # 11 digits after stripping separators


# ---------------------------------------------------------------------------
# CreatePatientRequest
# ---------------------------------------------------------------------------

class TestCreatePatientRequest:
    def test_constructs_with_required_fields_and_documented_defaults(self):
        model = CreatePatientRequest(
            first_name="Maria", last_name="Silva", phone=VALID_PHONE
        )
        assert model.first_name == "Maria"
        assert model.last_name == "Silva"
        assert model.phone == VALID_PHONE
        assert model.dob is None
        assert model.language == Language.pt
        assert model.demographics is None

    def test_accepts_optional_dob_and_demographics(self):
        dob = date(1990, 5, 20)
        model = CreatePatientRequest(
            first_name="Maria",
            last_name="Silva",
            phone=VALID_PHONE,
            dob=dob,
            language=Language.en,
            demographics={"insurance": "none"},
        )
        assert model.dob == dob
        assert model.language == Language.en
        assert model.demographics == {"insurance": "none"}

    def test_missing_first_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(last_name="Silva", phone=VALID_PHONE)

    def test_missing_last_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(first_name="Maria", phone=VALID_PHONE)

    def test_missing_phone_raises_validation_error(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(first_name="Maria", last_name="Silva")

    def test_blank_first_name_is_rejected_by_validate_non_empty(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(first_name="   ", last_name="Silva", phone=VALID_PHONE)

    def test_blank_last_name_is_rejected_by_validate_non_empty(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(first_name="Maria", last_name="", phone=VALID_PHONE)

    def test_phone_with_letters_is_rejected_by_validate_phone(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(
                first_name="Maria", last_name="Silva", phone="call-me-maybe"
            )

    def test_phone_too_short_after_stripping_separators_is_rejected(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(first_name="Maria", last_name="Silva", phone="123-45")

    def test_language_rejects_a_value_outside_the_language_enum(self):
        with pytest.raises(ValidationError):
            CreatePatientRequest(
                first_name="Maria",
                last_name="Silva",
                phone=VALID_PHONE,
                language="fr",
            )


# ---------------------------------------------------------------------------
# PatientResponse
# ---------------------------------------------------------------------------

class TestPatientResponse:
    def test_constructs_with_all_declared_fields(self):
        patient_id = uuid.uuid4()
        model = PatientResponse(
            id=patient_id,
            patient_code="PT-00118",
            first_name="Maria",
            last_name="Silva",
            status=PatientStatus.active,
        )
        assert model.id == patient_id
        assert model.patient_code == "PT-00118"
        assert model.status == PatientStatus.active

    def test_status_rejects_a_value_outside_the_patient_status_enum(self):
        with pytest.raises(ValidationError):
            PatientResponse(
                id=uuid.uuid4(),
                patient_code="PT-00118",
                first_name="Maria",
                last_name="Silva",
                status="deleted",
            )

    def test_id_must_be_a_uuid(self):
        with pytest.raises(ValidationError):
            PatientResponse(
                id="not-a-uuid",
                patient_code="PT-00118",
                first_name="Maria",
                last_name="Silva",
                status=PatientStatus.active,
            )


# ---------------------------------------------------------------------------
# DuplicateCandidate / PatientSummary -- the exact 409 body shape
# (architecture.md §5.2 POST /patients 409)
# ---------------------------------------------------------------------------

class TestPatientSummary:
    def test_constructs_with_id_and_patient_code(self):
        summary_id = uuid.uuid4()
        model = PatientSummary(id=summary_id, patient_code="PT-00118")
        assert model.id == summary_id
        assert model.patient_code == "PT-00118"


class TestDuplicateCandidate:
    def test_matches_the_documented_409_envelope_shape(self):
        summary_id = uuid.uuid4()
        candidate = DuplicateCandidate(
            duplicate_candidate=PatientSummary(id=summary_id, patient_code="PT-00118")
        )
        dumped = candidate.model_dump()
        assert dumped == {
            "duplicate_candidate": {"id": summary_id, "patient_code": "PT-00118"}
        }

    def test_coerces_a_plain_dict_into_a_patient_summary(self):
        summary_id = uuid.uuid4()
        candidate = DuplicateCandidate(
            duplicate_candidate={"id": summary_id, "patient_code": "PT-00118"}
        )
        assert isinstance(candidate.duplicate_candidate, PatientSummary)
        assert candidate.duplicate_candidate.id == summary_id

    def test_missing_duplicate_candidate_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DuplicateCandidate()


# ---------------------------------------------------------------------------
# PatientSearchItem / PatientSearchResponse
# ---------------------------------------------------------------------------

class TestPatientSearchItem:
    def test_constructs_with_all_declared_fields(self):
        item_id = uuid.uuid4()
        item = PatientSearchItem(
            id=item_id,
            patient_code="PT-00118",
            first_name="Maria",
            last_name="Silva",
            phone=VALID_PHONE,
        )
        assert item.id == item_id
        assert item.phone == VALID_PHONE

    def test_missing_phone_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PatientSearchItem(
                id=uuid.uuid4(),
                patient_code="PT-00118",
                first_name="Maria",
                last_name="Silva",
            )


class TestPatientSearchResponse:
    def test_wraps_a_list_of_search_items(self):
        item = PatientSearchItem(
            id=uuid.uuid4(),
            patient_code="PT-00118",
            first_name="Maria",
            last_name="Silva",
            phone=VALID_PHONE,
        )
        response = PatientSearchResponse(items=[item])
        assert response.items == [item]

    def test_an_empty_list_is_a_valid_search_response(self):
        response = PatientSearchResponse(items=[])
        assert response.items == []

    def test_missing_items_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PatientSearchResponse()


# ---------------------------------------------------------------------------
# PatientAppointmentSummary / PatientConsentSummary / PatientProfileResponse
# ---------------------------------------------------------------------------

class TestPatientAppointmentSummary:
    def test_constructs_with_id_scheduled_start_and_status(self):
        appt_id = uuid.uuid4()
        scheduled_start = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
        model = PatientAppointmentSummary(
            id=appt_id, scheduled_start=scheduled_start, status="booked"
        )
        assert model.id == appt_id
        assert model.scheduled_start == scheduled_start
        assert model.status == "booked"

    def test_missing_scheduled_start_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PatientAppointmentSummary(id=uuid.uuid4(), status="booked")


class TestPatientConsentSummary:
    def test_constructs_with_real_channel_and_consent_status_members(self):
        model = PatientConsentSummary(
            channel=Channel.whatsapp, status=ConsentStatus.granted
        )
        assert model.channel == Channel.whatsapp
        assert model.status == ConsentStatus.granted

    def test_channel_rejects_a_value_outside_the_channel_enum(self):
        with pytest.raises(ValidationError):
            PatientConsentSummary(channel="carrier_pigeon", status=ConsentStatus.granted)

    def test_status_rejects_a_value_outside_the_consent_status_enum(self):
        with pytest.raises(ValidationError):
            PatientConsentSummary(channel=Channel.sms, status="unknown")


class TestPatientProfileResponse:
    def _build(self, **overrides):
        defaults = dict(
            id=uuid.uuid4(),
            patient_code="PT-00118",
            first_name="Maria",
            last_name="Silva",
            status=PatientStatus.active,
            appointments=[],
            consent=[],
            recall_status=None,
        )
        defaults.update(overrides)
        return PatientProfileResponse(**defaults)

    def test_constructs_with_nested_appointments_and_consent_lists(self):
        appt = PatientAppointmentSummary(
            id=uuid.uuid4(),
            scheduled_start=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
            status="booked",
        )
        consent = PatientConsentSummary(channel=Channel.email, status=ConsentStatus.granted)
        model = self._build(appointments=[appt], consent=[consent], recall_status="due")
        assert model.appointments == [appt]
        assert model.consent == [consent]
        assert model.recall_status == "due"

    def test_recall_status_may_be_none(self):
        model = self._build(recall_status=None)
        assert model.recall_status is None

    def test_coerces_plain_dicts_in_appointments_and_consent_lists(self):
        model = self._build(
            appointments=[
                {
                    "id": uuid.uuid4(),
                    "scheduled_start": datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
                    "status": "booked",
                }
            ],
            consent=[{"channel": "sms", "status": "withdrawn"}],
        )
        assert isinstance(model.appointments[0], PatientAppointmentSummary)
        assert isinstance(model.consent[0], PatientConsentSummary)
        assert model.consent[0].status == ConsentStatus.withdrawn

    def test_missing_required_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PatientProfileResponse(
                id=uuid.uuid4(),
                patient_code="PT-00118",
                first_name="Maria",
                last_name="Silva",
                appointments=[],
                consent=[],
                recall_status=None,
            )


# ---------------------------------------------------------------------------
# LockResponse -- shared shape with POST /patients/{id}/lock and
# POST /scheduling/appointments/{id}/lock (FR-E2.4)
# ---------------------------------------------------------------------------

class TestLockResponse:
    def test_constructs_with_locked_by_and_expires_at(self):
        expires_at = datetime(2026, 9, 27, 10, 5, tzinfo=timezone.utc)
        model = LockResponse(locked_by="staff-42", expires_at=expires_at)
        assert model.locked_by == "staff-42"
        assert model.expires_at == expires_at

    def test_declares_exactly_the_two_fields_of_the_shared_lock_shape(self):
        assert set(LockResponse.model_fields) == {"locked_by", "expires_at"}

    def test_missing_expires_at_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LockResponse(locked_by="staff-42")


# ---------------------------------------------------------------------------
# UpdatePatientRequest
# ---------------------------------------------------------------------------

class TestUpdatePatientRequest:
    def test_all_fields_are_optional_and_default_to_none(self):
        model = UpdatePatientRequest()
        assert model.phone is None
        assert model.language is None
        assert model.email is None
        assert model.demographics is None

    def test_phone_validator_does_not_run_when_phone_is_omitted(self):
        # No ValidationError even though an omitted/None phone would never
        # pass validate_phone's own shape check -- the validator is only
        # wired to run when phone is not None.
        model = UpdatePatientRequest(language=Language.en)
        assert model.phone is None

    def test_a_provided_phone_is_still_validated_for_shape(self):
        with pytest.raises(ValidationError):
            UpdatePatientRequest(phone="not-a-phone!!")

    def test_a_provided_valid_phone_is_accepted(self):
        model = UpdatePatientRequest(phone=VALID_PHONE)
        assert model.phone == VALID_PHONE

    def test_accepts_email_and_demographics(self):
        model = UpdatePatientRequest(email="maria@example.com", demographics={"a": 1})
        assert model.email == "maria@example.com"
        assert model.demographics == {"a": 1}


# ---------------------------------------------------------------------------
# ArchivePatientRequest / ArchivePatientResponse
# ---------------------------------------------------------------------------

class TestArchivePatientRequest:
    def test_constructs_with_reason(self):
        model = ArchivePatientRequest(reason="Patient deceased")
        assert model.reason == "Patient deceased"

    def test_blank_reason_is_rejected_by_validate_non_empty(self):
        with pytest.raises(ValidationError):
            ArchivePatientRequest(reason="   ")

    def test_missing_reason_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ArchivePatientRequest()


class TestArchivePatientResponse:
    def test_constructs_with_id_status_and_archived_reason(self):
        patient_id = uuid.uuid4()
        model = ArchivePatientResponse(
            id=patient_id, status=PatientStatus.archived, archived_reason="duplicate"
        )
        assert model.id == patient_id
        assert model.status == PatientStatus.archived
        assert model.archived_reason == "duplicate"

    def test_status_rejects_a_value_outside_the_patient_status_enum(self):
        with pytest.raises(ValidationError):
            ArchivePatientResponse(
                id=uuid.uuid4(), status="deleted", archived_reason="duplicate"
            )


# ---------------------------------------------------------------------------
# DateRange / ExportPatientDataRequest / ExportResponse
# ---------------------------------------------------------------------------

class TestDateRange:
    def test_constructs_via_the_from_alias(self):
        model = DateRange(**{"from": date(2026, 1, 1), "to": date(2026, 6, 30)})
        assert model.from_ == date(2026, 1, 1)
        assert model.to == date(2026, 6, 30)

    def test_dumping_by_alias_uses_from_not_from_(self):
        model = DateRange(**{"from": date(2026, 1, 1), "to": date(2026, 6, 30)})
        dumped = model.model_dump(by_alias=True)
        assert dumped == {"from": date(2026, 1, 1), "to": date(2026, 6, 30)}

    def test_missing_to_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DateRange(**{"from": date(2026, 1, 1)})


class TestExportPatientDataRequest:
    def test_constructs_with_defaults(self):
        date_range = DateRange(**{"from": date(2026, 1, 1), "to": date(2026, 6, 30)})
        model = ExportPatientDataRequest(entity="patients", date_range=date_range)
        assert model.entity == "patients"
        assert model.date_range == date_range
        assert model.de_identified is False

    def test_accepts_appointments_entity_and_de_identified_true(self):
        date_range = DateRange(**{"from": date(2026, 1, 1), "to": date(2026, 6, 30)})
        model = ExportPatientDataRequest(
            entity="appointments", date_range=date_range, de_identified=True
        )
        assert model.entity == "appointments"
        assert model.de_identified is True

    def test_coerces_a_plain_dict_date_range(self):
        model = ExportPatientDataRequest(
            entity="patients",
            date_range={"from": "2026-01-01", "to": "2026-06-30"},
        )
        assert isinstance(model.date_range, DateRange)
        assert model.date_range.from_ == date(2026, 1, 1)

    def test_missing_date_range_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExportPatientDataRequest(entity="patients")


class TestExportResponse:
    def test_constructs_with_all_declared_fields(self):
        export_id = uuid.uuid4()
        model = ExportResponse(
            export_id=export_id, status="pending", download_url=None
        )
        assert model.export_id == export_id
        assert model.status == "pending"
        assert model.download_url is None

    def test_download_url_may_be_a_populated_string(self):
        model = ExportResponse(
            export_id=uuid.uuid4(),
            status="complete",
            download_url="https://storage.example/exports/1.csv",
        )
        assert model.download_url == "https://storage.example/exports/1.csv"

    def test_missing_status_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ExportResponse(export_id=uuid.uuid4(), download_url=None)


# ---------------------------------------------------------------------------
# Every declared model really is a BaseModel (schemas.py's stated public
# surface: "Pydantic v2 request/response models for every /patients/*
# endpoint")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_cls",
    [
        CreatePatientRequest,
        PatientResponse,
        DuplicateCandidate,
        PatientSummary,
        PatientSearchItem,
        PatientSearchResponse,
        PatientAppointmentSummary,
        PatientConsentSummary,
        PatientProfileResponse,
        LockResponse,
        UpdatePatientRequest,
        ArchivePatientRequest,
        ArchivePatientResponse,
        ExportPatientDataRequest,
        DateRange,
        ExportResponse,
    ],
)
def test_every_public_model_is_a_pydantic_base_model(model_cls):
    assert issubclass(model_cls, BaseModel)
