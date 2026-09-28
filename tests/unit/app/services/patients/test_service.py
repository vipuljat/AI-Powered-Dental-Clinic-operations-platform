"""Unit tests for app/services/patients/service.py.

These tests exercise PatientService, PatientImportService, RecordLockService and
ExportService purely against their documented public surface, mocking the repository/
audit/storage collaborators listed in the spec's "what this code calls" section.
"""
import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from app.common.constants import CSV_IMPORT_MAX_ROWS, RECORD_LOCK_TIMEOUT_MINUTES
from app.common.exceptions.errors import (
    DuplicatePatientError,
    ImportValidationError,
    NotFoundError,
    NotLockHolderError,
    RecordLockedError,
)
from app.services.patients.service import (
    ExportService,
    PatientImportService,
    PatientService,
    RecordLockService,
)

pytestmark = pytest.mark.asyncio


class _Row(dict):
    """A dict that also supports attribute access, so a fake repository can stand
    in whether the code under test treats rows as ORM objects or as plain dicts."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _AnyMethodRepo:
    """Generic fake repository: any attribute access resolves to an async callable
    that returns the configured rows. ExportService.generate's exact internal method
    for pulling export rows isn't named in its spec, so this avoids guessing it."""

    def __init__(self, rows):
        self._rows = rows

    def __getattr__(self, name):
        async def _method(*args, **kwargs):
            return self._rows

        return _method


@pytest.fixture
def db():
    return MagicMock(name="AsyncSession")


# ---------------------------------------------------------------------------
# PatientService
# ---------------------------------------------------------------------------


async def test_create_success_calls_audit_and_returns_tuple(db):
    patient_repo = AsyncMock()
    patient_repo.find_duplicate_by_phone = AsyncMock(return_value=None)
    created_patient = SimpleNamespace(id=uuid4(), first_name="Jane", last_name="Doe", phone="5551112222")
    patient_repo.create = AsyncMock(return_value=created_patient)
    lock_repo = AsyncMock()
    audit = AsyncMock()
    audit.record = AsyncMock(return_value=SimpleNamespace(id=uuid4()))
    service = PatientService(patient_repo, lock_repo, audit)

    result = await service.create(
        db, {"first_name": "Jane", "last_name": "Doe", "phone": "5551112222"}, uuid4()
    )

    assert result == (created_patient, None)
    patient_repo.find_duplicate_by_phone.assert_awaited_once()
    patient_repo.create.assert_awaited_once()
    audit.record.assert_awaited_once()


async def test_create_duplicate_phone_raises_without_creating(db):
    patient_repo = AsyncMock()
    duplicate = SimpleNamespace(id=uuid4(), first_name="Jane", last_name="Doe", phone="5551112222")
    patient_repo.find_duplicate_by_phone = AsyncMock(return_value=duplicate)
    patient_repo.create = AsyncMock()
    lock_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    with pytest.raises(DuplicatePatientError) as exc_info:
        await service.create(
            db, {"first_name": "Jane", "last_name": "Doe", "phone": "5551112222"}, uuid4()
        )

    assert exc_info.value.status_code == 409
    assert "duplicate_candidate" in exc_info.value.details[0]
    patient_repo.create.assert_not_awaited()
    audit.record.assert_not_awaited()


async def test_search_delegates_to_repository(db):
    patient_repo = AsyncMock()
    expected = [SimpleNamespace(id=uuid4(), first_name="Jane")]
    patient_repo.search = AsyncMock(return_value=expected)
    lock_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    result = await service.search(db, "Jane")

    assert result == expected
    patient_repo.search.assert_awaited_once()
    args, kwargs = patient_repo.search.call_args
    assert "Jane" in args or kwargs.get("q") == "Jane"


async def test_get_profile_raises_not_found_when_missing(db):
    patient_repo = AsyncMock()
    patient_repo.get_by_id = AsyncMock(return_value=None)
    lock_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    with pytest.raises(NotFoundError):
        await service.get_profile(db, uuid4())


async def test_update_raises_not_lock_holder_when_no_active_lock(db):
    patient_repo = AsyncMock()
    lock_repo = AsyncMock()
    lock_repo.find_active = AsyncMock(return_value=None)
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    with pytest.raises(NotLockHolderError) as exc_info:
        await service.update(db, uuid4(), {"phone": "5550001111"}, uuid4())

    assert exc_info.value.status_code == 423
    patient_repo.update.assert_not_awaited()


async def test_update_raises_not_lock_holder_when_locked_by_other_staff(db):
    patient_repo = AsyncMock()
    lock_repo = AsyncMock()
    lock_repo.find_active = AsyncMock(return_value=SimpleNamespace(locked_by_staff_id=uuid4()))
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    with pytest.raises(NotLockHolderError):
        await service.update(db, uuid4(), {"phone": "5550001111"}, uuid4())

    patient_repo.update.assert_not_awaited()


async def test_update_success_when_actor_holds_lock(db):
    actor_id = uuid4()
    patient_id = uuid4()
    patient_repo = AsyncMock()
    updated_patient = SimpleNamespace(id=patient_id, phone="5559998888")
    patient_repo.update = AsyncMock(return_value=updated_patient)
    lock_repo = AsyncMock()
    lock_repo.find_active = AsyncMock(return_value=SimpleNamespace(locked_by_staff_id=actor_id))
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    result = await service.update(db, patient_id, {"phone": "5559998888"}, actor_id)

    assert result is updated_patient
    patient_repo.update.assert_awaited_once()
    audit.record.assert_awaited_once()


async def test_archive_sets_status_and_records_audit(db):
    patient_id = uuid4()
    patient_repo = AsyncMock()
    archived_patient = SimpleNamespace(id=patient_id, status="archived", archived_reason="duplicate record")
    patient_repo.archive = AsyncMock(return_value=archived_patient)
    lock_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientService(patient_repo, lock_repo, audit)

    result = await service.archive(db, patient_id, "duplicate record", uuid4())

    assert result.status == "archived"
    patient_repo.archive.assert_awaited_once()
    audit.record.assert_awaited_once()


# ---------------------------------------------------------------------------
# PatientImportService
# ---------------------------------------------------------------------------


async def test_validate_raises_on_missing_required_columns(db):
    import_repo = AsyncMock()
    patient_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientImportService(import_repo, patient_repo, audit)

    csv_bytes = b"foo,bar\nval1,val2\n"

    with pytest.raises(ImportValidationError) as exc_info:
        await service.validate(db, csv_bytes, "patients.csv")

    assert exc_info.value.status_code == 422


async def test_validate_raises_on_exceeding_max_rows(db):
    import_repo = AsyncMock()
    patient_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientImportService(import_repo, patient_repo, audit)

    header = "first_name,last_name,phone"
    rows = [f"Jane{i},Doe,{5550000000 + i}" for i in range(CSV_IMPORT_MAX_ROWS + 1)]
    csv_bytes = (header + "\n" + "\n".join(rows) + "\n").encode()

    with pytest.raises(ImportValidationError):
        await service.validate(db, csv_bytes, "patients.csv")


async def test_validate_returns_valid_and_error_rows_for_mixed_csv(db):
    import_repo = AsyncMock()
    patient_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientImportService(import_repo, patient_repo, audit)

    csv_bytes = (
        "first_name,last_name,phone\n"
        "Jane,Doe,5551112222\n"
        "John,Smith,\n"
    ).encode()

    valid_rows, errors = await service.validate(db, csv_bytes, "patients.csv")

    assert isinstance(valid_rows, list)
    assert isinstance(errors, list)
    assert len(errors) >= 1
    assert len(valid_rows) >= 1
    assert "column_name" in errors[0]
    assert "error_message" in errors[0]


async def test_commit_rejects_entire_file_when_any_row_invalid(db):
    import_repo = AsyncMock()
    patient_repo = AsyncMock()
    audit = AsyncMock()
    service = PatientImportService(import_repo, patient_repo, audit)
    service.validate = AsyncMock(
        return_value=([], [{"row_number": 2, "column_name": "phone", "error_message": "required"}])
    )

    result = await service.commit(db, b"irrelevant", "patients.csv", uuid4())

    import_repo.bulk_insert.assert_not_awaited()
    assert isinstance(result, dict)


async def test_commit_inserts_when_all_rows_valid(db):
    import_repo = AsyncMock()
    patient_repo = AsyncMock()
    audit = AsyncMock()
    valid_row = {"first_name": "Jane", "last_name": "Doe", "phone": "5551112222"}
    import_repo.bulk_insert = AsyncMock(return_value=[SimpleNamespace(id=uuid4(), **valid_row)])
    service = PatientImportService(import_repo, patient_repo, audit)
    service.validate = AsyncMock(return_value=([valid_row], []))

    result = await service.commit(db, b"irrelevant", "patients.csv", uuid4())

    import_repo.bulk_insert.assert_awaited_once()
    assert isinstance(result, dict)
    audit.record.assert_awaited()


# ---------------------------------------------------------------------------
# RecordLockService
# ---------------------------------------------------------------------------


async def test_acquire_success_uses_configured_ttl(db):
    lock_repo = AsyncMock()
    fake_lock = SimpleNamespace(id=uuid4(), entity_type="patient")
    lock_repo.acquire = AsyncMock(return_value=fake_lock)
    service = RecordLockService(lock_repo)
    entity_id = uuid4()
    staff_id = uuid4()

    result = await service.acquire(db, "patient", entity_id, staff_id)

    assert result is fake_lock
    args, kwargs = lock_repo.acquire.call_args
    ttl = kwargs.get("ttl_minutes", args[-1] if args else None)
    assert ttl == RECORD_LOCK_TIMEOUT_MINUTES


async def test_acquire_conflict_raises_record_locked_error(db):
    lock_repo = AsyncMock()
    lock_repo.acquire = AsyncMock(
        side_effect=IntegrityError("INSERT INTO record_locks", {}, Exception("duplicate key"))
    )
    lock_repo.find_active = AsyncMock(
        return_value=SimpleNamespace(id=uuid4(), locked_by_staff_id=uuid4())
    )
    service = RecordLockService(lock_repo)

    with pytest.raises(RecordLockedError) as exc_info:
        await service.acquire(db, "patient", uuid4(), uuid4())

    assert exc_info.value.status_code == 409
    assert exc_info.value.details
    assert "locked_by" in exc_info.value.details[0]


async def test_release_delegates_to_repository(db):
    lock_repo = AsyncMock()
    lock_repo.release = AsyncMock(return_value=None)
    service = RecordLockService(lock_repo)
    entity_id = uuid4()
    staff_id = uuid4()

    result = await service.release(db, "patient", entity_id, staff_id)

    assert result is None
    lock_repo.release.assert_awaited_once()


async def test_sweep_expired_returns_count_from_repository(db):
    lock_repo = AsyncMock()
    lock_repo.sweep_expired = AsyncMock(return_value=7)
    service = RecordLockService(lock_repo)

    result = await service.sweep_expired(db)

    assert result == 7
    lock_repo.sweep_expired.assert_awaited_once_with(db)


# ---------------------------------------------------------------------------
# ExportService
# ---------------------------------------------------------------------------


async def test_generate_returns_processing_envelope_immediately(db):
    patient_repo = _AnyMethodRepo([])
    appointment_repo = _AnyMethodRepo([])
    storage = MagicMock()
    storage.put_object = AsyncMock(return_value="s3://bucket/key")
    service = ExportService(patient_repo, appointment_repo, storage)

    result = await service.generate(
        db, "patients", date(2026, 1, 1), date(2026, 1, 31), False, uuid4()
    )

    assert result["status"] == "processing"
    assert result["download_url"] is None
    assert isinstance(result["export_id"], UUID)
    # let any fire-and-forget background task drain quietly before the test ends
    await asyncio.sleep(0.1)


async def test_generate_does_not_block_on_slow_upload(db):
    async def slow_put_object(key, data, content_type):
        await asyncio.sleep(0.3)
        return "s3://bucket/key"

    patient_repo = _AnyMethodRepo([])
    appointment_repo = _AnyMethodRepo([])
    storage = MagicMock()
    storage.put_object = slow_put_object
    service = ExportService(patient_repo, appointment_repo, storage)

    result = await asyncio.wait_for(
        service.generate(db, "patients", date(2026, 1, 1), date(2026, 1, 31), False, uuid4()),
        timeout=0.1,
    )

    assert result["status"] == "processing"
    # drain the background upload before the test/event loop tears down
    await asyncio.sleep(0.35)


async def test_generate_raw_export_includes_patient_pii(db):
    captured = {}

    async def capture_put_object(key, data, content_type):
        captured["data"] = data
        return "s3://bucket/key"

    rows = [_Row(id=uuid4(), first_name="Jane", last_name="Doe", phone="5551234567", status="active")]
    patient_repo = _AnyMethodRepo(rows)
    appointment_repo = _AnyMethodRepo([])
    storage = MagicMock()
    storage.put_object = capture_put_object
    service = ExportService(patient_repo, appointment_repo, storage)

    await service.generate(db, "patients", date(2026, 1, 1), date(2026, 1, 31), False, uuid4())
    await asyncio.sleep(0.1)

    assert "data" in captured
    assert b"5551234567" in captured["data"]


async def test_generate_deidentified_export_strips_patient_pii(db):
    captured = {}

    async def capture_put_object(key, data, content_type):
        captured["data"] = data
        return "s3://bucket/key"

    rows = [_Row(id=uuid4(), first_name="Jane", last_name="Doe", phone="5551234567", status="active")]
    patient_repo = _AnyMethodRepo(rows)
    appointment_repo = _AnyMethodRepo([])
    storage = MagicMock()
    storage.put_object = capture_put_object
    service = ExportService(patient_repo, appointment_repo, storage)

    await service.generate(db, "patients", date(2026, 1, 1), date(2026, 1, 31), True, uuid4())
    await asyncio.sleep(0.1)

    assert "data" in captured
    assert b"5551234567" not in captured["data"]
