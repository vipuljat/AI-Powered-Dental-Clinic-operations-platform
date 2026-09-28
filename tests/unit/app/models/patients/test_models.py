"""Unit tests for app/models/patients/models.py.

Exercises the four ORM tables declared here (patients, patient_import_batches,
patient_import_errors, record_locks) directly against an in-memory SQLite
database, mirroring the ENVIRONMENT=test substitution described in
project_rules.testing (app/core/database.py switches to
sqlite+aiosqlite:///:memory: with a StaticPool at the composition root).

patient_import_batches.uploaded_by_staff_id and record_locks.locked_by_staff_id
are plain FK columns pointing at staff_users.id (owned by the console module's
own models.py, per this file's spec -- the console module is not part of this
task). Every ORM model shares one Base.metadata (per project_rules.testing),
so that FK target must be resolvable for Base.metadata.create_all to compile
this file's own tables' DDL. Registering a minimal stand-in table here --
rather than importing the console module's ORM classes -- keeps this test
scoped to this file's own spec; the guard is a no-op if the real console
models already registered the name on the shared metadata (e.g. when this
file runs alongside the full suite).
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.patients.models import (
    Patient,
    PatientImportBatch,
    PatientImportError,
    RecordLock,
)

pytestmark = pytest.mark.asyncio

if "staff_users" not in Base.metadata.tables:
    Table("staff_users", Base.metadata, Column("id", Uuid, primary_key=True))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


def make_patient(**overrides):
    defaults = dict(
        patient_code=f"PT-{uuid.uuid4().hex[:10]}",
        first_name="Ana",
        last_name="Silva",
        phone="+15551234567",
    )
    defaults.update(overrides)
    return Patient(**defaults)


def make_batch(**overrides):
    defaults = dict(
        uploaded_by_staff_id=uuid.uuid4(),
        filename="patients_import.csv",
        status="validating",
    )
    defaults.update(overrides)
    return PatientImportBatch(**defaults)


# --------------------------------------------------------------------------
# table identity
# --------------------------------------------------------------------------


def test_table_names():
    assert Patient.__tablename__ == "patients"
    assert PatientImportBatch.__tablename__ == "patient_import_batches"
    assert PatientImportError.__tablename__ == "patient_import_errors"
    assert RecordLock.__tablename__ == "record_locks"


# --------------------------------------------------------------------------
# patients
# --------------------------------------------------------------------------


async def test_patient_id_is_a_generated_uuid(session):
    patient = make_patient()
    session.add(patient)
    await session.flush()
    assert isinstance(patient.id, uuid.UUID)


async def test_patient_first_name_required(session):
    # FR-E2.1: first_name is NOT NULL
    patient = Patient(
        patient_code=f"PT-{uuid.uuid4().hex[:10]}",
        last_name="Silva",
        phone="+15551234567",
    )
    session.add(patient)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_patient_last_name_required(session):
    # FR-E2.1: last_name is NOT NULL
    patient = Patient(
        patient_code=f"PT-{uuid.uuid4().hex[:10]}",
        first_name="Ana",
        phone="+15551234567",
    )
    session.add(patient)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_patient_phone_required(session):
    # FR-E2.1: phone is NOT NULL (the duplicate-detection key)
    patient = Patient(
        patient_code=f"PT-{uuid.uuid4().hex[:10]}",
        first_name="Ana",
        last_name="Silva",
    )
    session.add(patient)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_patient_code_required(session):
    patient = Patient(first_name="Ana", last_name="Silva", phone="+15551234567")
    session.add(patient)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_patient_code_uniqueness_enforced(session):
    dupe_code = f"PT-{uuid.uuid4().hex[:10]}"
    session.add(make_patient(patient_code=dupe_code, phone="+15550000001"))
    await session.commit()

    session.add(make_patient(patient_code=dupe_code, phone="+15550000002"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_patient_language_defaults_to_pt(session):
    patient = make_patient()
    session.add(patient)
    await session.commit()
    await session.refresh(patient)
    assert patient.language == "pt"


async def test_patient_status_defaults_to_active(session):
    patient = make_patient()
    session.add(patient)
    await session.commit()
    await session.refresh(patient)
    assert patient.status == "active"


async def test_patient_dob_and_archived_fields_default_to_none(session):
    patient = make_patient()
    session.add(patient)
    await session.commit()
    await session.refresh(patient)
    assert patient.dob is None
    assert patient.archived_reason is None
    assert patient.archived_at is None


async def test_patient_demographics_jsonb_roundtrips_through_fresh_session(
    session, session_factory
):
    demographics = {"insurance": "private", "household_size": 4}
    patient = make_patient(demographics=demographics, developmental_stage="adult")
    session.add(patient)
    await session.commit()
    patient_id = patient.id

    # re-query from a fresh session to force a real round trip through the
    # JSONB/JSON-fallback TypeDecorator rather than relying on the identity map
    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(Patient, patient_id)
        assert reloaded.demographics == demographics
        assert reloaded.developmental_stage == "adult"


async def test_patient_created_and_updated_at_populated(session):
    patient = make_patient()
    session.add(patient)
    await session.commit()
    await session.refresh(patient)
    assert patient.created_at is not None
    assert patient.updated_at is not None


# --------------------------------------------------------------------------
# patient_import_batches
# --------------------------------------------------------------------------


async def test_batch_uploaded_by_staff_id_required(session):
    batch = PatientImportBatch(filename="import.csv", status="validating")
    session.add(batch)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_batch_filename_required(session):
    batch = PatientImportBatch(uploaded_by_staff_id=uuid.uuid4(), status="validating")
    session.add(batch)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_batch_status_required(session):
    batch = PatientImportBatch(uploaded_by_staff_id=uuid.uuid4(), filename="import.csv")
    session.add(batch)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_batch_row_counts_default_to_zero(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()
    await session.refresh(batch)
    assert batch.total_rows == 0
    assert batch.success_count == 0
    assert batch.error_count == 0


async def test_batch_status_values_committed_and_rejected_roundtrip(session):
    committed = make_batch(status="committed")
    rejected = make_batch(status="rejected")
    session.add_all([committed, rejected])
    await session.commit()
    await session.refresh(committed)
    await session.refresh(rejected)
    assert committed.status == "committed"
    assert rejected.status == "rejected"


# --------------------------------------------------------------------------
# patient_import_errors
# --------------------------------------------------------------------------


async def test_error_batch_id_required(session):
    error = PatientImportError(
        row_number=3, column_name="phone", error_message="Missing phone number"
    )
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_row_number_required(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = PatientImportError(
        batch_id=batch.id, column_name="phone", error_message="Missing phone number"
    )
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_column_name_required(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = PatientImportError(
        batch_id=batch.id, row_number=3, error_message="Missing phone number"
    )
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_message_required(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = PatientImportError(batch_id=batch.id, row_number=3, column_name="phone")
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_fields_roundtrip(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = PatientImportError(
        batch_id=batch.id,
        row_number=7,
        column_name="dob",
        error_message="Unparseable date of birth value",
    )
    session.add(error)
    await session.commit()
    await session.refresh(error)
    assert error.batch_id == batch.id
    assert error.row_number == 7
    assert error.column_name == "dob"
    assert error.error_message == "Unparseable date of birth value"


# --------------------------------------------------------------------------
# record_locks
# --------------------------------------------------------------------------


def make_lock(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        entity_type="patient",
        entity_id=uuid.uuid4(),
        locked_by_staff_id=uuid.uuid4(),
        locked_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    defaults.update(overrides)
    return RecordLock(**defaults)


async def test_lock_entity_type_required(session):
    now = datetime.now(timezone.utc)
    lock = RecordLock(
        entity_id=uuid.uuid4(),
        locked_by_staff_id=uuid.uuid4(),
        locked_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    session.add(lock)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_lock_entity_id_required(session):
    now = datetime.now(timezone.utc)
    lock = RecordLock(
        entity_type="patient",
        locked_by_staff_id=uuid.uuid4(),
        locked_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    session.add(lock)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_lock_locked_by_staff_id_required(session):
    now = datetime.now(timezone.utc)
    lock = RecordLock(
        entity_type="patient",
        entity_id=uuid.uuid4(),
        locked_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    session.add(lock)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_lock_locked_at_required(session):
    lock = RecordLock(
        entity_type="patient",
        entity_id=uuid.uuid4(),
        locked_by_staff_id=uuid.uuid4(),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    session.add(lock)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_lock_expires_at_required(session):
    lock = RecordLock(
        entity_type="patient",
        entity_id=uuid.uuid4(),
        locked_by_staff_id=uuid.uuid4(),
        locked_at=datetime.now(timezone.utc),
    )
    session.add(lock)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_lock_supports_appointment_entity_type(session):
    # record_locks is shared by patients (US10) and scheduling (US22): the
    # entity_type enum must accept "appointment", not just "patient".
    lock = make_lock(entity_type="appointment", entity_id=uuid.uuid4())
    session.add(lock)
    await session.commit()
    await session.refresh(lock)
    assert lock.entity_type == "appointment"


async def test_lock_unique_constraint_on_entity_type_and_entity_id_enforced(session):
    # FR-E2.4: UNIQUE(entity_type, entity_id) is the DB-level constraint
    # RecordLockService.acquire relies on for "only one active lock per record".
    entity_id = uuid.uuid4()
    session.add(make_lock(entity_type="patient", entity_id=entity_id))
    await session.commit()

    session.add(make_lock(entity_type="patient", entity_id=entity_id))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_lock_unique_constraint_is_composite_allows_same_entity_id_different_type(
    session,
):
    # the same UUID may legitimately identify a patient row and, separately,
    # an appointment row -- the uniqueness must be on the (type, id) pair,
    # not on entity_id alone.
    shared_id = uuid.uuid4()
    session.add(make_lock(entity_type="patient", entity_id=shared_id))
    session.add(make_lock(entity_type="appointment", entity_id=shared_id))
    await session.commit()  # must not raise


async def test_lock_unique_constraint_allows_different_entity_id_same_type(session):
    session.add(make_lock(entity_type="patient", entity_id=uuid.uuid4()))
    session.add(make_lock(entity_type="patient", entity_id=uuid.uuid4()))
    await session.commit()  # must not raise
