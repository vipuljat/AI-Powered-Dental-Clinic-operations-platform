"""Unit tests for app/repositories/scheduling/repository.py.

Exercises the five data-access classes (ProviderRepository, ChairRepository,
AppointmentRepository, AppointmentHistoryRepository, AppointmentImportRepository)
against an in-memory SQLite database, mirroring the ENVIRONMENT=test
substitution described in project_rules.testing (app/core/database.py switches
the DB URL to sqlite+aiosqlite:///:memory: so the same ORM models run
unmodified against both Postgres and SQLite).
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

from sqlalchemy import Column, Table, Uuid  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.common.constants import APPOINTMENT_COMPLETION_GRACE_MINUTES  # noqa: E402
from app.common.enums import AppointmentStatus, ImportBatchStatus, ScoreSource  # noqa: E402
from app.core.database import Base  # noqa: E402
from app.models.scheduling.models import (  # noqa: E402
    Appointment,
    AppointmentHistory,
    AppointmentImportBatch,
    AppointmentImportError,
    Chair,
    Provider,
)
from app.repositories.scheduling.repository import (  # noqa: E402
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
    ChairRepository,
    ProviderRepository,
)

# appointments.patient_id/created_by_staff_id, appointment_history
# .changed_by_staff_id and appointment_import_batches.uploaded_by_staff_id are
# plain FK columns pointing at patients.id / staff_users.id -- tables owned by
# other modules' own models.py that this task has no spec for and must not
# import. Register minimal stand-in tables (id column only) so SQLAlchemy can
# resolve those FK targets when compiling this module's own tables. This is a
# no-op if the real patients/console models already registered these names on
# the shared metadata.
for _name in ("patients", "staff_users"):
    if _name not in Base.metadata.tables:
        Table(_name, Base.metadata, Column("id", Uuid(), primary_key=True))

_SCHEDULING_TABLES = [
    Provider.__table__,
    Chair.__table__,
    Appointment.__table__,
    AppointmentHistory.__table__,
    AppointmentImportBatch.__table__,
    AppointmentImportError.__table__,
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
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=_SCHEDULING_TABLES)
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
def provider_repo():
    return ProviderRepository()


@pytest.fixture
def chair_repo():
    return ChairRepository()


@pytest.fixture
def appt_repo():
    return AppointmentRepository()


@pytest.fixture
def history_repo():
    return AppointmentHistoryRepository()


@pytest.fixture
def import_repo():
    return AppointmentImportRepository()


async def seed_provider(db, **overrides) -> Provider:
    defaults = dict(name="Dr. Smith", specialty="General")
    defaults.update(overrides)
    provider = Provider(**defaults)
    db.add(provider)
    await db.commit()
    await db.refresh(provider)
    return provider


async def seed_chair(db, **overrides) -> Chair:
    defaults = dict(room="Room 1", label="Chair A")
    defaults.update(overrides)
    chair = Chair(**defaults)
    db.add(chair)
    await db.commit()
    await db.refresh(chair)
    return chair


def appt_fields(**overrides):
    start = overrides.pop("scheduled_start", datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc))
    end = overrides.pop("scheduled_end", start + timedelta(minutes=30))
    defaults = dict(
        patient_id=uuid.uuid4(),
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        appointment_type="cleaning",
        scheduled_start=start,
        scheduled_end=end,
        status=AppointmentStatus.booked,
    )
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------
# ProviderRepository
# --------------------------------------------------------------------------


async def test_provider_list_returns_all_providers(db, provider_repo):
    a = await seed_provider(db, name="Dr. Alpha")
    b = await seed_provider(db, name="Dr. Beta")

    results = await provider_repo.list(db)

    assert {p.id for p in results} == {a.id, b.id}


async def test_provider_get_by_id_returns_matching_provider(db, provider_repo):
    created = await seed_provider(db)

    fetched = await provider_repo.get_by_id(db, created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == created.name


async def test_provider_get_by_id_returns_none_for_unknown_id(db, provider_repo):
    assert await provider_repo.get_by_id(db, uuid.uuid4()) is None


# --------------------------------------------------------------------------
# ChairRepository
# --------------------------------------------------------------------------


async def test_chair_list_returns_all_chairs(db, chair_repo):
    a = await seed_chair(db, label="Chair A")
    b = await seed_chair(db, label="Chair B")

    results = await chair_repo.list(db)

    assert {c.id for c in results} == {a.id, b.id}


async def test_chair_get_by_id_returns_matching_chair(db, chair_repo):
    created = await seed_chair(db)

    fetched = await chair_repo.get_by_id(db, created.id)

    assert fetched is not None
    assert fetched.id == created.id


async def test_chair_get_by_id_returns_none_for_unknown_id(db, chair_repo):
    assert await chair_repo.get_by_id(db, uuid.uuid4()) is None


# --------------------------------------------------------------------------
# AppointmentRepository.create / get_by_id
# --------------------------------------------------------------------------


async def test_create_persists_appointment_with_given_fields(db, appt_repo):
    patient_id = uuid.uuid4()

    created = await appt_repo.create(db, **appt_fields(patient_id=patient_id))

    assert isinstance(created, Appointment)
    assert created.patient_id == patient_id
    assert created.appointment_type == "cleaning"
    assert created.status == AppointmentStatus.booked


async def test_get_by_id_returns_created_appointment(db, appt_repo):
    created = await appt_repo.create(db, **appt_fields())

    fetched = await appt_repo.get_by_id(db, created.id)

    assert fetched is not None
    assert fetched.id == created.id


async def test_get_by_id_returns_none_for_unknown_id(db, appt_repo):
    assert await appt_repo.get_by_id(db, uuid.uuid4()) is None


# --------------------------------------------------------------------------
# RMR001: get_by_idempotency_key
# --------------------------------------------------------------------------


async def test_get_by_idempotency_key_returns_existing_row(db, appt_repo):
    created = await appt_repo.create(db, **appt_fields(idempotency_key="idem-key-1"))

    found = await appt_repo.get_by_idempotency_key(db, "idem-key-1")

    assert found is not None
    assert found.id == created.id


async def test_get_by_idempotency_key_returns_none_when_key_unused(db, appt_repo):
    await appt_repo.create(db, **appt_fields(idempotency_key="idem-key-1"))

    assert await appt_repo.get_by_idempotency_key(db, "never-used-key") is None


async def test_repeated_idempotency_key_is_not_inserted_twice(db, appt_repo):
    # RMR001: SchedulingService.create checks get_by_idempotency_key BEFORE
    # calling create -- simulate that flow and confirm the second lookup
    # yields the SAME row as the first insert rather than a duplicate.
    created = await appt_repo.create(db, **appt_fields(idempotency_key="idem-key-2"))

    existing = await appt_repo.get_by_idempotency_key(db, "idem-key-2")
    assert existing is not None
    assert existing.id == created.id

    all_matching = [
        row
        for row in (await db.execute(select(Appointment))).scalars().all()
        if row.idempotency_key == "idem-key-2"
    ]
    assert len(all_matching) == 1


# --------------------------------------------------------------------------
# FR-E4.2/US18-20: find_conflicts
# --------------------------------------------------------------------------


async def test_find_conflicts_detects_overlap_on_same_provider_different_chair(db, appt_repo):
    provider_id = uuid.uuid4()
    existing = await appt_repo.create(
        db,
        **appt_fields(
            provider_id=provider_id,
            chair_id=uuid.uuid4(),
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=provider_id,
        chair_id=uuid.uuid4(),  # different chair -- still a conflict via provider
        start=datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 9, 45, tzinfo=timezone.utc),
    )

    assert {c.id for c in conflicts} == {existing.id}


async def test_find_conflicts_detects_overlap_on_same_chair_different_provider(db, appt_repo):
    chair_id = uuid.uuid4()
    existing = await appt_repo.create(
        db,
        **appt_fields(
            provider_id=uuid.uuid4(),
            chair_id=chair_id,
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=uuid.uuid4(),  # different provider -- still a conflict via chair
        chair_id=chair_id,
        start=datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 9, 45, tzinfo=timezone.utc),
    )

    assert {c.id for c in conflicts} == {existing.id}


async def test_find_conflicts_no_conflict_when_neither_provider_nor_chair_match(db, appt_repo):
    await appt_repo.create(
        db,
        **appt_fields(
            provider_id=uuid.uuid4(),
            chair_id=uuid.uuid4(),
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        start=datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 9, 45, tzinfo=timezone.utc),
    )

    assert conflicts == []


async def test_find_conflicts_no_conflict_when_time_ranges_do_not_overlap(db, appt_repo):
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    await appt_repo.create(
        db,
        **appt_fields(
            provider_id=provider_id,
            chair_id=chair_id,
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=provider_id,
        chair_id=chair_id,
        start=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 10, 30, tzinfo=timezone.utc),
    )

    assert conflicts == []


async def test_find_conflicts_ignores_cancelled_appointments(db, appt_repo):
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    await appt_repo.create(
        db,
        **appt_fields(
            provider_id=provider_id,
            chair_id=chair_id,
            status=AppointmentStatus.cancelled,
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=provider_id,
        chair_id=chair_id,
        start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
    )

    assert conflicts == []


async def test_find_conflicts_includes_rescheduled_appointments(db, appt_repo):
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    existing = await appt_repo.create(
        db,
        **appt_fields(
            provider_id=provider_id,
            chair_id=chair_id,
            status=AppointmentStatus.rescheduled,
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=provider_id,
        chair_id=chair_id,
        start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
    )

    assert {c.id for c in conflicts} == {existing.id}


async def test_find_conflicts_excludes_given_exclude_appointment_id(db, appt_repo):
    # The reschedule path calls find_conflicts against the appointment's own
    # unmodified row -- it must not conflict with itself.
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    existing = await appt_repo.create(
        db,
        **appt_fields(
            provider_id=provider_id,
            chair_id=chair_id,
            scheduled_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    conflicts = await appt_repo.find_conflicts(
        db,
        provider_id=provider_id,
        chair_id=chair_id,
        start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        exclude_appointment_id=existing.id,
    )

    assert conflicts == []


# --------------------------------------------------------------------------
# AppointmentRepository.update_status / update_fields
# --------------------------------------------------------------------------


async def test_update_status_persists_new_status(db, appt_repo):
    created = await appt_repo.create(db, **appt_fields(status=AppointmentStatus.booked))

    updated = await appt_repo.update_status(db, created.id, AppointmentStatus.completed.value)

    assert updated.status == AppointmentStatus.completed
    reloaded = await appt_repo.get_by_id(db, created.id)
    assert reloaded.status == AppointmentStatus.completed


async def test_update_status_persists_extra_fields(db, appt_repo):
    created = await appt_repo.create(db, **appt_fields(status=AppointmentStatus.booked))

    updated = await appt_repo.update_status(
        db, created.id, AppointmentStatus.cancelled.value, cancellation_reason="patient request"
    )

    assert updated.status == AppointmentStatus.cancelled
    assert updated.cancellation_reason == "patient request"
    reloaded = await appt_repo.get_by_id(db, created.id)
    assert reloaded.cancellation_reason == "patient request"


async def test_update_fields_persists_reschedule_fields(db, appt_repo):
    # FR-E4.4: reschedule updates the SAME row's scheduled_start/end/provider/
    # chair in place rather than inserting a new row.
    created = await appt_repo.create(db, **appt_fields())
    new_provider_id = uuid.uuid4()
    new_chair_id = uuid.uuid4()
    new_start = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
    new_end = datetime(2026, 6, 1, 14, 30, tzinfo=timezone.utc)

    updated = await appt_repo.update_fields(
        db,
        created.id,
        provider_id=new_provider_id,
        chair_id=new_chair_id,
        scheduled_start=new_start,
        scheduled_end=new_end,
        status=AppointmentStatus.rescheduled,
    )

    assert updated.id == created.id
    assert updated.provider_id == new_provider_id
    assert updated.chair_id == new_chair_id
    assert updated.status == AppointmentStatus.rescheduled

    reloaded = await appt_repo.get_by_id(db, created.id)
    assert reloaded.provider_id == new_provider_id
    assert reloaded.scheduled_start.replace(tzinfo=None) == new_start.replace(tzinfo=None)


# --------------------------------------------------------------------------
# AppointmentRepository.list_by_patient
# --------------------------------------------------------------------------


async def test_list_by_patient_returns_only_that_patients_appointments(db, appt_repo):
    patient_id = uuid.uuid4()
    a = await appt_repo.create(db, **appt_fields(patient_id=patient_id))
    b = await appt_repo.create(db, **appt_fields(patient_id=patient_id))
    await appt_repo.create(db, **appt_fields(patient_id=uuid.uuid4()))  # other patient

    results = await appt_repo.list_by_patient(db, patient_id)

    assert {r.id for r in results} == {a.id, b.id}


async def test_list_by_patient_returns_empty_for_patient_with_no_appointments(db, appt_repo):
    await appt_repo.create(db, **appt_fields())

    results = await appt_repo.list_by_patient(db, uuid.uuid4())

    assert results == []


# --------------------------------------------------------------------------
# AppointmentRepository.list_completable
# --------------------------------------------------------------------------


async def test_list_completable_includes_booked_appointment_past_grace_window(db, appt_repo):
    as_of = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    # Ended well before as_of minus the grace period -- eligible for auto-complete.
    ended_at = as_of - timedelta(minutes=APPOINTMENT_COMPLETION_GRACE_MINUTES + 60)
    created = await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus.booked,
            scheduled_start=ended_at - timedelta(minutes=30),
            scheduled_end=ended_at,
        ),
    )

    results = await appt_repo.list_completable(db, as_of)

    assert {r.id for r in results} == {created.id}


async def test_list_completable_excludes_appointment_still_within_grace_window(db, appt_repo):
    as_of = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    # Ended only 5 minutes ago -- well inside the grace period, not yet completable.
    ended_at = as_of - timedelta(minutes=5)
    await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus.booked,
            scheduled_start=ended_at - timedelta(minutes=30),
            scheduled_end=ended_at,
        ),
    )

    results = await appt_repo.list_completable(db, as_of)

    assert results == []


@pytest.mark.parametrize("status", ["cancelled", "completed", "no_show", "rescheduled"])
async def test_list_completable_excludes_non_booked_statuses(db, appt_repo, status):
    as_of = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ended_at = as_of - timedelta(minutes=APPOINTMENT_COMPLETION_GRACE_MINUTES + 60)
    await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus(status),
            scheduled_start=ended_at - timedelta(minutes=30),
            scheduled_end=ended_at,
        ),
    )

    results = await appt_repo.list_completable(db, as_of)

    assert results == []


async def test_list_completable_does_not_itself_change_status(db, appt_repo):
    # Interaction contract: this repository only surfaces candidates -- the
    # booked -> completed transition is performed by recall_scanner.py's
    # sweep, not by list_completable itself.
    as_of = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    ended_at = as_of - timedelta(minutes=APPOINTMENT_COMPLETION_GRACE_MINUTES + 60)
    created = await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus.booked,
            scheduled_start=ended_at - timedelta(minutes=30),
            scheduled_end=ended_at,
        ),
    )

    await appt_repo.list_completable(db, as_of)

    reloaded = await appt_repo.get_by_id(db, created.id)
    assert reloaded.status == AppointmentStatus.booked


# --------------------------------------------------------------------------
# AppointmentRepository.list_open_slots
# --------------------------------------------------------------------------


async def test_list_open_slots_returns_booked_appointments_within_range(db, appt_repo):
    created = await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus.booked,
            scheduled_start=datetime(2026, 5, 10, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 10, 9, 30, tzinfo=timezone.utc),
        ),
    )

    results = await appt_repo.list_open_slots(
        db,
        provider_id=None,
        date_from=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        date_to=datetime(2026, 5, 10, 23, 59, tzinfo=timezone.utc),
    )

    assert created.id in {r.id for r in results}


async def test_list_open_slots_excludes_appointments_outside_range(db, appt_repo):
    await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus.booked,
            scheduled_start=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc),
        ),
    )

    results = await appt_repo.list_open_slots(
        db,
        provider_id=None,
        date_from=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        date_to=datetime(2026, 5, 10, 23, 59, tzinfo=timezone.utc),
    )

    assert results == []


async def test_list_open_slots_filters_by_provider_id_when_given(db, appt_repo):
    provider_id = uuid.uuid4()
    created = await appt_repo.create(
        db,
        **appt_fields(
            provider_id=provider_id,
            status=AppointmentStatus.booked,
            scheduled_start=datetime(2026, 5, 10, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 10, 9, 30, tzinfo=timezone.utc),
        ),
    )
    await appt_repo.create(
        db,
        **appt_fields(
            provider_id=uuid.uuid4(),  # different provider
            status=AppointmentStatus.booked,
            scheduled_start=datetime(2026, 5, 10, 10, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 10, 10, 30, tzinfo=timezone.utc),
        ),
    )

    results = await appt_repo.list_open_slots(
        db,
        provider_id=provider_id,
        date_from=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        date_to=datetime(2026, 5, 10, 23, 59, tzinfo=timezone.utc),
    )

    assert {r.id for r in results} == {created.id}


async def test_list_open_slots_excludes_cancelled_appointments(db, appt_repo):
    await appt_repo.create(
        db,
        **appt_fields(
            status=AppointmentStatus.cancelled,
            scheduled_start=datetime(2026, 5, 10, 9, 0, tzinfo=timezone.utc),
            scheduled_end=datetime(2026, 5, 10, 9, 30, tzinfo=timezone.utc),
        ),
    )

    results = await appt_repo.list_open_slots(
        db,
        provider_id=None,
        date_from=datetime(2026, 5, 10, 0, 0, tzinfo=timezone.utc),
        date_to=datetime(2026, 5, 10, 23, 59, tzinfo=timezone.utc),
    )

    assert results == []


# --------------------------------------------------------------------------
# AppointmentRepository.list_by_risk_source
# --------------------------------------------------------------------------


async def test_list_by_risk_source_filters_by_given_source(db, appt_repo):
    rules_appt = await appt_repo.create(
        db, **appt_fields(risk_score_source=ScoreSource.rules)
    )
    await appt_repo.create(db, **appt_fields(risk_score_source=ScoreSource.ml))

    results = await appt_repo.list_by_risk_source(db, "rules")

    assert {r.id for r in results} == {rules_appt.id}


async def test_list_by_risk_source_returns_empty_when_no_matches(db, appt_repo):
    await appt_repo.create(db, **appt_fields(risk_score_source=ScoreSource.ml))

    results = await appt_repo.list_by_risk_source(db, "rules")

    assert results == []


# --------------------------------------------------------------------------
# AppointmentHistoryRepository.insert
# --------------------------------------------------------------------------


async def test_history_insert_persists_change_record(db, history_repo, appt_repo):
    appointment = await appt_repo.create(db, **appt_fields())
    staff_id = uuid.uuid4()

    record = await history_repo.insert(
        db,
        appointment_id=appointment.id,
        changed_field="scheduled_start",
        old_value="2026-05-01T09:00:00+00:00",
        new_value="2026-06-01T14:00:00+00:00",
        changed_by_staff_id=staff_id,
    )

    assert isinstance(record, AppointmentHistory)
    assert record.appointment_id == appointment.id
    assert record.changed_field == "scheduled_start"
    assert record.old_value == "2026-05-01T09:00:00+00:00"
    assert record.new_value == "2026-06-01T14:00:00+00:00"
    assert record.changed_by_staff_id == staff_id


async def test_history_insert_allows_null_old_value_and_staff_id(db, history_repo, appt_repo):
    # AI-Agent-originated changes have no staff actor, and a brand-new field
    # (e.g. first confirmation_status write) has no prior value.
    appointment = await appt_repo.create(db, **appt_fields())

    record = await history_repo.insert(
        db,
        appointment_id=appointment.id,
        changed_field="confirmation_status",
        old_value=None,
        new_value="sent",
        changed_by_staff_id=None,
    )

    assert record.old_value is None
    assert record.changed_by_staff_id is None


async def test_history_insert_sets_changed_at_automatically(db, history_repo, appt_repo):
    # insert()'s own public surface takes no changed_at argument, so it must
    # stamp one itself rather than leaving the NOT NULL column unset.
    appointment = await appt_repo.create(db, **appt_fields())
    before = datetime.now(timezone.utc).replace(tzinfo=None)

    record = await history_repo.insert(
        db,
        appointment_id=appointment.id,
        changed_field="status",
        old_value="booked",
        new_value="rescheduled",
        changed_by_staff_id=None,
    )

    assert isinstance(record.changed_at, datetime)
    assert record.changed_at.replace(tzinfo=None) >= before - timedelta(seconds=5)


# --------------------------------------------------------------------------
# AppointmentImportRepository
# --------------------------------------------------------------------------


async def test_create_batch_persists_given_fields(db, import_repo):
    staff_id = uuid.uuid4()

    batch = await import_repo.create_batch(
        db, uploaded_by_staff_id=staff_id, filename="import.csv", total_rows=10
    )

    assert isinstance(batch, AppointmentImportBatch)
    assert batch.uploaded_by_staff_id == staff_id
    assert batch.filename == "import.csv"
    assert batch.total_rows == 10


async def test_create_batch_starts_success_and_error_counts_at_zero(db, import_repo):
    # create_batch's own signature takes no success_count/error_count -- the
    # model's own NOT NULL columns default to 0, and a freshly-created batch
    # must not already report any outcome.
    batch = await import_repo.create_batch(
        db, uploaded_by_staff_id=uuid.uuid4(), filename="import.csv", total_rows=10
    )

    assert batch.success_count == 0
    assert batch.error_count == 0


async def test_bulk_insert_creates_one_appointment_per_row(db, import_repo, appt_repo):
    rows = [appt_fields(patient_id=uuid.uuid4()), appt_fields(patient_id=uuid.uuid4())]

    created = await import_repo.bulk_insert(db, rows)

    assert len(created) == 2
    for appointment, row in zip(created, rows):
        assert isinstance(appointment, Appointment)
        assert appointment.patient_id == row["patient_id"]
    # Each row is independently retrievable afterwards through the ordinary
    # AppointmentRepository, confirming the rows were actually committed.
    for appointment in created:
        assert (await appt_repo.get_by_id(db, appointment.id)) is not None


async def test_bulk_insert_returns_empty_list_for_empty_input(db, import_repo):
    assert await import_repo.bulk_insert(db, []) == []


async def test_log_errors_persists_error_rows_linked_to_batch(db, import_repo):
    batch = await import_repo.create_batch(
        db, uploaded_by_staff_id=uuid.uuid4(), filename="import.csv", total_rows=2
    )

    result = await import_repo.log_errors(
        db,
        batch.id,
        [
            {
                "row_number": 2,
                "column_name": "patient_code",
                "error_message": "unknown patient_code",
                "unmatched_patient": True,
            },
            {
                "row_number": 3,
                "column_name": "scheduled_start",
                "error_message": "invalid date format",
                "unmatched_patient": False,
            },
        ],
    )

    assert result is None
    persisted = (
        (await db.execute(select(AppointmentImportError).where(AppointmentImportError.batch_id == batch.id)))
        .scalars()
        .all()
    )
    assert len(persisted) == 2
    row_numbers = {e.row_number for e in persisted}
    assert row_numbers == {2, 3}
    unmatched = {e.row_number: e.unmatched_patient for e in persisted}
    assert unmatched[2] is True
    assert unmatched[3] is False


async def test_update_batch_status_persists_final_counts(db, import_repo):
    batch = await import_repo.create_batch(
        db, uploaded_by_staff_id=uuid.uuid4(), filename="import.csv", total_rows=10
    )

    updated = await import_repo.update_batch_status(
        db, batch.id, ImportBatchStatus.committed.value, success_count=8, error_count=2
    )

    assert updated.status == ImportBatchStatus.committed
    assert updated.success_count == 8
    assert updated.error_count == 2
