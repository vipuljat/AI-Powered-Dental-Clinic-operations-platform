"""Unit tests for app/repositories/patients/repository.py.

Exercises the three data-access classes (PatientRepository,
PatientImportRepository, RecordLockRepository) against an in-memory SQLite
database, mirroring the ENVIRONMENT=test substitution described in
project_rules.testing.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# The real composition roots set this before ever importing app.core.database;
# do the same here defensively so importing it never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from sqlalchemy import Column, Table  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.common.utils import new_patient_code  # noqa: E402
from app.core.database import Base  # noqa: E402
from app.models.patients.models import (  # noqa: E402
    Patient,
    PatientImportBatch,
    PatientImportError,
    RecordLock,
)
from app.repositories.patients.repository import (  # noqa: E402
    PatientImportRepository,
    PatientRepository,
    RecordLockRepository,
)

# patient_import_batches.uploaded_by_staff_id and record_locks.locked_by_staff_id
# declare an FK to staff_users.id (see app/models/patients/models.py), a table
# owned by the console module that this task has no spec for and must not
# import for its own sake. If the console module happens to already be
# importable (and so already registered `staff_users` on the shared
# Base.metadata) reuse it as-is; otherwise register a minimal stand-in table
# with just the `id` primary key FK resolution needs, purely so this
# in-memory engine can create *this* spec's own four tables.
try:  # pragma: no cover - exercised implicitly by the fixtures below
    import app.models.console.models  # noqa: F401
except ModuleNotFoundError:
    pass

if "staff_users" not in Base.metadata.tables:
    Table(
        "staff_users",
        Base.metadata,
        Column("id", Patient.__table__.c.id.type, primary_key=True),
    )

_PATIENTS_TABLES = [
    Patient.__table__,
    PatientImportBatch.__table__,
    PatientImportError.__table__,
    RecordLock.__table__,
]

pytestmark = pytest.mark.asyncio


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
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=_PATIENTS_TABLES)
        )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def db(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


@pytest.fixture
def patient_repo():
    return PatientRepository()


@pytest.fixture
def import_repo():
    return PatientImportRepository()


@pytest.fixture
def lock_repo():
    return RecordLockRepository()


def make_patient_fields(**overrides):
    defaults = dict(
        patient_code=f"PT-{uuid.uuid4().hex[:8]}",
        first_name="Maria",
        last_name="Silva",
        phone="+15551234567",
    )
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------
# PatientRepository.create / get_by_id
# --------------------------------------------------------------------------


async def test_create_persists_patient_with_given_fields(db, patient_repo):
    patient = await patient_repo.create(
        db, **make_patient_fields(patient_code="PT-00001", first_name="Ana", last_name="Costa", phone="+15550000001")
    )
    assert isinstance(patient, Patient)
    assert patient.patient_code == "PT-00001"
    assert patient.first_name == "Ana"
    assert patient.last_name == "Costa"
    assert patient.phone == "+15550000001"


async def test_get_by_id_returns_created_patient(db, patient_repo):
    created = await patient_repo.create(db, **make_patient_fields(patient_code="PT-00002"))
    fetched = await patient_repo.get_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.patient_code == "PT-00002"


async def test_get_by_id_returns_none_for_unknown_id(db, patient_repo):
    assert await patient_repo.get_by_id(db, uuid.uuid4()) is None


# --------------------------------------------------------------------------
# PatientRepository.search
# --------------------------------------------------------------------------


async def test_search_matches_full_name_case_insensitive_substring(db, patient_repo):
    await patient_repo.create(
        db, **make_patient_fields(patient_code="PT-10001", first_name="Maria", last_name="Silva", phone="+15551111111")
    )
    await patient_repo.create(
        db, **make_patient_fields(patient_code="PT-10002", first_name="Joao", last_name="Souza", phone="+15552222222")
    )

    results = await patient_repo.search(db, "MARIA SIL")

    assert len(results) == 1
    assert results[0].first_name == "Maria"


async def test_search_matches_phone_substring(db, patient_repo):
    await patient_repo.create(
        db, **make_patient_fields(patient_code="PT-10003", phone="+15559998888")
    )
    await patient_repo.create(
        db, **make_patient_fields(patient_code="PT-10004", phone="+15550001111")
    )

    results = await patient_repo.search(db, "9998888")

    assert len(results) == 1
    assert results[0].phone == "+15559998888"


async def test_search_matches_patient_code_substring(db, patient_repo):
    await patient_repo.create(db, **make_patient_fields(patient_code="PT-77777"))
    await patient_repo.create(db, **make_patient_fields(patient_code="PT-88888"))

    results = await patient_repo.search(db, "77777")

    assert len(results) == 1
    assert results[0].patient_code == "PT-77777"


async def test_search_excludes_archived_by_default(db, patient_repo):
    archived = await patient_repo.create(db, **make_patient_fields(patient_code="PT-ARCH1", first_name="Archie"))
    await patient_repo.archive(db, archived.id, reason="duplicate record")

    results = await patient_repo.search(db, "Archie")

    assert results == []


async def test_search_includes_archived_when_flag_set(db, patient_repo):
    archived = await patient_repo.create(db, **make_patient_fields(patient_code="PT-ARCH2", first_name="Archibald"))
    await patient_repo.archive(db, archived.id, reason="duplicate record")

    results = await patient_repo.search(db, "Archibald", include_archived=True)

    assert len(results) == 1
    assert results[0].id == archived.id


# --------------------------------------------------------------------------
# PatientRepository.update / archive
# --------------------------------------------------------------------------


async def test_update_persists_given_fields(db, patient_repo):
    created = await patient_repo.create(db, **make_patient_fields(patient_code="PT-20001"))

    updated = await patient_repo.update(db, created.id, email="new@example.com", phone="+15550009999")

    assert updated.email == "new@example.com"
    assert updated.phone == "+15550009999"

    reloaded = await patient_repo.get_by_id(db, created.id)
    assert reloaded.email == "new@example.com"
    assert reloaded.phone == "+15550009999"


async def test_archive_sets_archived_status_and_reason(db, patient_repo):
    created = await patient_repo.create(db, **make_patient_fields(patient_code="PT-20002"))

    archived = await patient_repo.archive(db, created.id, reason="patient deceased")

    assert archived.status == "archived"
    assert archived.archived_reason == "patient deceased"

    reloaded = await patient_repo.get_by_id(db, created.id)
    assert reloaded.status == "archived"
    assert reloaded.archived_reason == "patient deceased"


# --------------------------------------------------------------------------
# PatientRepository.find_duplicate_by_phone
# --------------------------------------------------------------------------


async def test_find_duplicate_by_phone_returns_matching_patient(db, patient_repo):
    created = await patient_repo.create(db, **make_patient_fields(patient_code="PT-30001", phone="+15553334444"))

    found = await patient_repo.find_duplicate_by_phone(db, "+15553334444")

    assert found is not None
    assert found.id == created.id


async def test_find_duplicate_by_phone_returns_none_when_no_match(db, patient_repo):
    await patient_repo.create(db, **make_patient_fields(patient_code="PT-30002", phone="+15553334444"))

    found = await patient_repo.find_duplicate_by_phone(db, "+15550000000")

    assert found is None


# --------------------------------------------------------------------------
# PatientRepository.next_patient_code
# --------------------------------------------------------------------------


async def test_next_patient_code_starts_at_one_when_no_patients_exist(db, patient_repo):
    code = await patient_repo.next_patient_code(db)
    assert code == new_patient_code(1)


async def test_next_patient_code_reflects_existing_patient_count(db, patient_repo):
    await patient_repo.create(db, **make_patient_fields(patient_code="PT-40001"))
    await patient_repo.create(db, **make_patient_fields(patient_code="PT-40002"))

    code = await patient_repo.next_patient_code(db)

    assert code == new_patient_code(3)


# --------------------------------------------------------------------------
# PatientRepository.list_active
# --------------------------------------------------------------------------


async def test_list_active_excludes_archived_patients(db, patient_repo):
    active = await patient_repo.create(db, **make_patient_fields(patient_code="PT-50001"))
    archived = await patient_repo.create(db, **make_patient_fields(patient_code="PT-50002"))
    await patient_repo.archive(db, archived.id, reason="left clinic")

    results = await patient_repo.list_active(db)

    ids = {p.id for p in results}
    assert active.id in ids
    assert archived.id not in ids


async def test_list_active_filters_by_given_ids(db, patient_repo):
    a = await patient_repo.create(db, **make_patient_fields(patient_code="PT-50003"))
    b = await patient_repo.create(db, **make_patient_fields(patient_code="PT-50004"))

    results = await patient_repo.list_active(db, ids=[a.id])

    ids = {p.id for p in results}
    assert ids == {a.id}
    assert b.id not in ids


async def test_list_active_with_ids_still_excludes_archived(db, patient_repo):
    archived = await patient_repo.create(db, **make_patient_fields(patient_code="PT-50005"))
    await patient_repo.archive(db, archived.id, reason="left clinic")

    results = await patient_repo.list_active(db, ids=[archived.id])

    assert results == []


# --------------------------------------------------------------------------
# PatientImportRepository
# --------------------------------------------------------------------------


async def test_create_batch_persists_upload_metadata(db, import_repo):
    staff_id = uuid.uuid4()

    batch = await import_repo.create_batch(db, uploaded_by_staff_id=staff_id, filename="patients.csv", total_rows=10)

    assert isinstance(batch, PatientImportBatch)
    assert batch.uploaded_by_staff_id == staff_id
    assert batch.filename == "patients.csv"
    assert batch.total_rows == 10
    assert batch.success_count == 0
    assert batch.error_count == 0


async def test_bulk_insert_persists_all_rows_in_one_call(db, import_repo):
    rows = [
        dict(patient_code="PT-60001", first_name="Rows", last_name="One", phone="+15551110000"),
        dict(patient_code="PT-60002", first_name="Rows", last_name="Two", phone="+15551110001"),
        dict(patient_code="PT-60003", first_name="Rows", last_name="Three", phone="+15551110002"),
    ]

    inserted = await import_repo.bulk_insert(db, rows)

    assert len(inserted) == 3
    assert all(isinstance(p, Patient) for p in inserted)
    assert {p.patient_code for p in inserted} == {"PT-60001", "PT-60002", "PT-60003"}


async def test_bulk_insert_rolls_back_entire_batch_on_mid_batch_failure(db, import_repo):
    # FR-E2.2/US9/A10: a duplicate patient_code triggers the DB-level
    # unique-constraint violation on the second row -- the whole call must be
    # all-or-nothing, so even the first, individually-valid row must not
    # survive.
    rows = [
        dict(patient_code="PT-70001", first_name="Ok", last_name="Row", phone="+15552220000"),
        dict(patient_code="PT-70001", first_name="Dup", last_name="Row", phone="+15552220001"),
    ]

    with pytest.raises(IntegrityError):
        await import_repo.bulk_insert(db, rows)

    await db.rollback()

    patient_repo = PatientRepository()
    survivors = await patient_repo.search(db, "Row", include_archived=True)
    assert survivors == []


async def test_log_errors_persists_each_error_linked_to_batch(db, import_repo):
    staff_id = uuid.uuid4()
    batch = await import_repo.create_batch(db, uploaded_by_staff_id=staff_id, filename="bad.csv", total_rows=2)

    await import_repo.log_errors(
        db,
        batch.id,
        [
            {"row_number": 1, "column_name": "phone", "error_message": "missing phone"},
            {"row_number": 2, "column_name": "first_name", "error_message": "missing first name"},
        ],
    )

    from sqlalchemy import select

    result = await db.execute(select(PatientImportError).where(PatientImportError.batch_id == batch.id))
    errors = result.scalars().all()

    assert len(errors) == 2
    assert {e.column_name for e in errors} == {"phone", "first_name"}
    assert {e.row_number for e in errors} == {1, 2}


async def test_update_batch_status_persists_counts_and_status(db, import_repo):
    staff_id = uuid.uuid4()
    batch = await import_repo.create_batch(db, uploaded_by_staff_id=staff_id, filename="ok.csv", total_rows=5)

    updated = await import_repo.update_batch_status(db, batch.id, status="committed", success_count=4, error_count=1)

    assert updated.status == "committed"
    assert updated.success_count == 4
    assert updated.error_count == 1


# --------------------------------------------------------------------------
# RecordLockRepository
# --------------------------------------------------------------------------


async def test_acquire_persists_lock_with_expiry_from_ttl_minutes(db, lock_repo):
    entity_id = uuid.uuid4()
    staff_id = uuid.uuid4()

    lock = await lock_repo.acquire(db, "patient", entity_id, staff_id, ttl_minutes=15)

    assert isinstance(lock, RecordLock)
    assert lock.entity_type == "patient"
    assert lock.entity_id == entity_id
    assert lock.locked_by_staff_id == staff_id
    delta = lock.expires_at - lock.locked_at
    assert timedelta(minutes=14, seconds=55) <= delta <= timedelta(minutes=15, seconds=30)


async def test_acquire_raises_integrity_error_when_already_locked(db, lock_repo):
    entity_id = uuid.uuid4()
    first_staff = uuid.uuid4()
    second_staff = uuid.uuid4()

    await lock_repo.acquire(db, "patient", entity_id, first_staff, ttl_minutes=15)

    with pytest.raises(IntegrityError):
        await lock_repo.acquire(db, "patient", entity_id, second_staff, ttl_minutes=15)


async def test_find_active_returns_none_when_no_lock_exists(db, lock_repo):
    assert await lock_repo.find_active(db, "patient", uuid.uuid4()) is None


async def test_find_active_returns_the_acquired_lock(db, lock_repo):
    entity_id = uuid.uuid4()
    staff_id = uuid.uuid4()
    await lock_repo.acquire(db, "appointment", entity_id, staff_id, ttl_minutes=10)

    found = await lock_repo.find_active(db, "appointment", entity_id)

    assert found is not None
    assert found.entity_id == entity_id
    assert found.locked_by_staff_id == staff_id


async def test_release_deletes_matching_lock(db, lock_repo):
    entity_id = uuid.uuid4()
    staff_id = uuid.uuid4()
    await lock_repo.acquire(db, "patient", entity_id, staff_id, ttl_minutes=10)

    await lock_repo.release(db, "patient", entity_id, staff_id)

    assert await lock_repo.find_active(db, "patient", entity_id) is None


async def test_release_is_a_no_op_when_staff_id_does_not_match(db, lock_repo):
    entity_id = uuid.uuid4()
    owner_staff_id = uuid.uuid4()
    other_staff_id = uuid.uuid4()
    await lock_repo.acquire(db, "patient", entity_id, owner_staff_id, ttl_minutes=10)

    # Must not raise, and must not release a lock owned by someone else.
    await lock_repo.release(db, "patient", entity_id, other_staff_id)

    still_locked = await lock_repo.find_active(db, "patient", entity_id)
    assert still_locked is not None
    assert still_locked.locked_by_staff_id == owner_staff_id


async def test_sweep_expired_removes_only_expired_locks_and_returns_count(db, lock_repo):
    expired_entity = uuid.uuid4()
    active_entity = uuid.uuid4()
    staff_id = uuid.uuid4()

    expired_lock = await lock_repo.acquire(db, "patient", expired_entity, staff_id, ttl_minutes=10)
    # Force this lock into the past directly, since acquire() always computes
    # a future expiry from ttl_minutes.
    expired_lock.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await db.flush()

    await lock_repo.acquire(db, "patient", active_entity, staff_id, ttl_minutes=10)

    released_count = await lock_repo.sweep_expired(db)

    assert released_count == 1
    assert await lock_repo.find_active(db, "patient", expired_entity) is None
    assert await lock_repo.find_active(db, "patient", active_entity) is not None
