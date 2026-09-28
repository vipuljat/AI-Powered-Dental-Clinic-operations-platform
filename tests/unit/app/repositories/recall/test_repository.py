"""Unit tests for app/repositories/recall/repository.py.

Exercises the three data-access classes (RecallScheduleRepository,
UnscheduledTreatmentRepository, RecallComplianceBaselineRepository) against an
in-memory SQLite database, mirroring the ENVIRONMENT=test substitution
described in project_rules.testing (app/core/database.py switches the DB URL
to sqlite+aiosqlite:///:memory: so the same ORM models run unmodified against
both Postgres and SQLite).
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio

# The real composition roots set this before ever importing app.core.database;
# do the same here defensively so importing it never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from sqlalchemy import Column, Table, Uuid  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.database import Base  # noqa: E402
from app.models.recall.models import (  # noqa: E402
    RecallComplianceBaseline,
    RecallSchedule,
    UnscheduledTreatment,
)
from app.repositories.recall.repository import (  # noqa: E402
    RecallComplianceBaselineRepository,
    RecallScheduleRepository,
    UnscheduledTreatmentRepository,
)

# recall_schedules.patient_id/appointment_id and unscheduled_treatments
# .patient_id/appointment_id are plain FK columns pointing at patients.id /
# appointments.id -- tables owned by other modules' own models.py that this
# task has no spec for and must not import. Register minimal stand-in tables
# (id column only) so `create_all` can resolve those FK targets. This is a
# no-op if the real patients/scheduling models already registered these names
# on the shared metadata.
for _name in ("patients", "appointments"):
    if _name not in Base.metadata.tables:
        Table(_name, Base.metadata, Column("id", Uuid(), primary_key=True))

_RECALL_TABLES = [
    RecallSchedule.__table__,
    UnscheduledTreatment.__table__,
    RecallComplianceBaseline.__table__,
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
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=_RECALL_TABLES)
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
def schedule_repo():
    return RecallScheduleRepository()


@pytest.fixture
def treatment_repo():
    return UnscheduledTreatmentRepository()


@pytest.fixture
def baseline_repo():
    return RecallComplianceBaselineRepository()


def schedule_fields(**overrides):
    defaults = dict(
        patient_id=uuid.uuid4(),
        due_date=date(2026, 4, 1),
        status="due",
        interval_source="risk_based",
    )
    defaults.update(overrides)
    return defaults


def treatment_fields(**overrides):
    defaults = dict(
        patient_id=uuid.uuid4(),
        treatment_code="D2140",
        accepted_at=datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return defaults


async def seed_treatment(db, **overrides):
    # UnscheduledTreatmentRepository's own public surface declares no
    # `create` -- these rows originate elsewhere (e.g. a treatment-acceptance
    # flow this task has no spec for). Seed directly via the ORM, exactly as
    # app/models/recall/models.py's own test does, so these tests exercise
    # only the repository methods this spec actually declares.
    treatment = UnscheduledTreatment(**treatment_fields(**overrides))
    db.add(treatment)
    await db.commit()
    await db.refresh(treatment)
    return treatment


# --------------------------------------------------------------------------
# RecallScheduleRepository.create / get_by_id
# --------------------------------------------------------------------------


async def test_create_persists_recall_schedule_with_given_fields(db, schedule_repo):
    patient_id = uuid.uuid4()
    schedule = await schedule_repo.create(
        db, **schedule_fields(patient_id=patient_id, status="overdue", due_date=date(2026, 5, 1))
    )

    assert isinstance(schedule, RecallSchedule)
    assert schedule.patient_id == patient_id
    assert schedule.status == "overdue"
    assert schedule.due_date == date(2026, 5, 1)


async def test_get_by_id_returns_created_schedule(db, schedule_repo):
    created = await schedule_repo.create(db, **schedule_fields())
    fetched = await schedule_repo.get_by_id(db, created.id)

    assert fetched is not None
    assert fetched.id == created.id


async def test_get_by_id_returns_none_for_unknown_id(db, schedule_repo):
    assert await schedule_repo.get_by_id(db, uuid.uuid4()) is None


# --------------------------------------------------------------------------
# RecallScheduleRepository.list_overdue
# --------------------------------------------------------------------------


async def test_list_overdue_includes_due_status_before_as_of(db, schedule_repo):
    as_of = date(2026, 4, 10)
    created = await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 4, 1), status="due")
    )

    results = await schedule_repo.list_overdue(db, as_of)

    assert {r.id for r in results} == {created.id}


async def test_list_overdue_includes_overdue_status_before_as_of(db, schedule_repo):
    as_of = date(2026, 4, 10)
    created = await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 3, 1), status="overdue")
    )

    results = await schedule_repo.list_overdue(db, as_of)

    assert {r.id for r in results} == {created.id}


async def test_list_overdue_excludes_due_date_equal_to_as_of(db, schedule_repo):
    # WHERE due_date < as_of is documented as a strict inequality.
    as_of = date(2026, 4, 10)
    await schedule_repo.create(db, **schedule_fields(due_date=as_of, status="due"))

    results = await schedule_repo.list_overdue(db, as_of)

    assert results == []


async def test_list_overdue_excludes_due_date_after_as_of(db, schedule_repo):
    as_of = date(2026, 4, 10)
    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 4, 20), status="due")
    )

    results = await schedule_repo.list_overdue(db, as_of)

    assert results == []


@pytest.mark.parametrize("status", ["dormant", "contacted", "completed"])
async def test_list_overdue_excludes_statuses_outside_due_or_overdue(db, schedule_repo, status):
    as_of = date(2026, 4, 10)
    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 3, 1), status=status)
    )

    results = await schedule_repo.list_overdue(db, as_of)

    assert results == []


async def test_list_overdue_does_not_mutate_status_itself(db, schedule_repo):
    # Interaction contract: reclassification to "overdue" is applied by the
    # caller/service (and the dormant threshold too) -- this read-only method
    # must not itself flip the row's stored status as a side effect.
    created = await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 3, 1), status="due")
    )

    await schedule_repo.list_overdue(db, date(2026, 4, 10))

    reloaded = await schedule_repo.get_by_id(db, created.id)
    assert reloaded.status == "due"


# --------------------------------------------------------------------------
# RecallScheduleRepository.list_by_ids
# --------------------------------------------------------------------------


async def test_list_by_ids_returns_only_requested_schedules(db, schedule_repo):
    a = await schedule_repo.create(db, **schedule_fields())
    b = await schedule_repo.create(db, **schedule_fields())
    await schedule_repo.create(db, **schedule_fields())  # not requested

    results = await schedule_repo.list_by_ids(db, [a.id, b.id])

    assert {r.id for r in results} == {a.id, b.id}


async def test_list_by_ids_ignores_unknown_ids(db, schedule_repo):
    a = await schedule_repo.create(db, **schedule_fields())

    results = await schedule_repo.list_by_ids(db, [a.id, uuid.uuid4()])

    assert {r.id for r in results} == {a.id}


async def test_list_by_ids_returns_empty_list_for_empty_input(db, schedule_repo):
    await schedule_repo.create(db, **schedule_fields())

    results = await schedule_repo.list_by_ids(db, [])

    assert results == []


# --------------------------------------------------------------------------
# RecallScheduleRepository.update_status / link_appointment
# --------------------------------------------------------------------------


async def test_update_status_persists_new_status(db, schedule_repo):
    created = await schedule_repo.create(db, **schedule_fields(status="due"))

    updated = await schedule_repo.update_status(db, created.id, "contacted")

    assert updated.status == "contacted"
    reloaded = await schedule_repo.get_by_id(db, created.id)
    assert reloaded.status == "contacted"


async def test_update_status_persists_extra_fields(db, schedule_repo):
    created = await schedule_repo.create(db, **schedule_fields(status="due"))
    last_recall_at = datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc)

    updated = await schedule_repo.update_status(
        db, created.id, "completed", last_recall_at=last_recall_at
    )

    assert updated.status == "completed"
    # Compared with tzinfo normalised away: the sqlite+aiosqlite fallback
    # this test suite runs against (per project_rules.testing) is not
    # guaranteed to round-trip a TIMESTAMPTZ column's UTC offset bit-for-bit,
    # only its wall-clock value -- app/models/recall/models.py's own test
    # makes the same accommodation for this column.
    assert updated.last_recall_at.replace(tzinfo=None) == last_recall_at.replace(tzinfo=None)
    reloaded = await schedule_repo.get_by_id(db, created.id)
    assert reloaded.last_recall_at.replace(tzinfo=None) == last_recall_at.replace(tzinfo=None)


async def test_link_appointment_sets_appointment_id_on_schedule(db, schedule_repo):
    created = await schedule_repo.create(db, **schedule_fields(appointment_id=None))
    appointment_id = uuid.uuid4()

    updated = await schedule_repo.link_appointment(db, created.id, appointment_id)

    assert updated.appointment_id == appointment_id
    reloaded = await schedule_repo.get_by_id(db, created.id)
    assert reloaded.appointment_id == appointment_id


# --------------------------------------------------------------------------
# RecallScheduleRepository.count_by_status
# --------------------------------------------------------------------------


async def test_count_by_status_counts_only_requested_statuses_in_period(db, schedule_repo):
    period_start = date(2026, 4, 1)
    period_end = date(2026, 4, 7)

    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 4, 2), status="due")
    )
    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 4, 3), status="overdue")
    )
    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 4, 4), status="completed")
    )
    # In-period but not one of the requested statuses -- must not be counted.
    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 4, 5), status="dormant")
    )

    counts = await schedule_repo.count_by_status(
        db, ["due", "overdue", "completed"], period_start, period_end
    )

    assert counts["due"] == 1
    assert counts["overdue"] == 1
    assert counts["completed"] == 1
    assert counts.get("dormant", 0) == 0
    assert sum(counts.values()) == 3


async def test_count_by_status_excludes_rows_outside_the_period(db, schedule_repo):
    period_start = date(2026, 4, 1)
    period_end = date(2026, 4, 7)

    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 3, 20), status="completed")
    )
    await schedule_repo.create(
        db, **schedule_fields(due_date=date(2026, 5, 1), status="completed")
    )

    counts = await schedule_repo.count_by_status(
        db, ["completed"], period_start, period_end
    )

    assert counts.get("completed", 0) == 0


# --------------------------------------------------------------------------
# UnscheduledTreatmentRepository.list_by_valuation
# --------------------------------------------------------------------------


async def test_list_by_valuation_orders_descending_by_amount(db, treatment_repo):
    low = await seed_treatment(db, valuation_amount=Decimal("100.00"))
    high = await seed_treatment(db, valuation_amount=Decimal("900.00"))
    mid = await seed_treatment(db, valuation_amount=Decimal("500.00"))

    results = await treatment_repo.list_by_valuation(db)

    ids_in_order = [r.id for r in results]
    assert ids_in_order == [high.id, mid.id, low.id]


async def test_list_by_valuation_sorts_null_valuation_last(db, treatment_repo):
    valued = await seed_treatment(db, valuation_amount=Decimal("100.00"))
    unvalued = await seed_treatment(db, valuation_amount=None)

    results = await treatment_repo.list_by_valuation(db)

    ids_in_order = [r.id for r in results]
    assert ids_in_order == [valued.id, unvalued.id]


async def test_list_by_valuation_defaults_to_unscheduled_status(db, treatment_repo):
    unscheduled = await seed_treatment(
        db, valuation_amount=Decimal("200.00"), status="unscheduled"
    )
    await seed_treatment(db, valuation_amount=Decimal("999.00"), status="booked")

    results = await treatment_repo.list_by_valuation(db)

    assert [r.id for r in results] == [unscheduled.id]


async def test_list_by_valuation_filters_by_given_status(db, treatment_repo):
    booked = await seed_treatment(db, valuation_amount=Decimal("999.00"), status="booked")
    await seed_treatment(db, valuation_amount=Decimal("200.00"), status="unscheduled")

    results = await treatment_repo.list_by_valuation(db, status="booked")

    assert [r.id for r in results] == [booked.id]


# --------------------------------------------------------------------------
# UnscheduledTreatmentRepository.get_by_id / update_valuation / update_status
# / link_appointment / list_by_ids
# --------------------------------------------------------------------------


async def test_treatment_get_by_id_returns_created_treatment(db, treatment_repo):
    created = await seed_treatment(db)
    fetched = await treatment_repo.get_by_id(db, created.id)

    assert fetched is not None
    assert fetched.id == created.id


async def test_treatment_get_by_id_returns_none_for_unknown_id(db, treatment_repo):
    assert await treatment_repo.get_by_id(db, uuid.uuid4()) is None


async def test_update_valuation_persists_new_amount(db, treatment_repo):
    created = await seed_treatment(db, valuation_amount=None)

    updated = await treatment_repo.update_valuation(db, created.id, 950.00)

    assert float(updated.valuation_amount) == pytest.approx(950.00)
    reloaded = await treatment_repo.get_by_id(db, created.id)
    assert float(reloaded.valuation_amount) == pytest.approx(950.00)


async def test_treatment_update_status_persists_new_status(db, treatment_repo):
    created = await seed_treatment(db, status="unscheduled")

    updated = await treatment_repo.update_status(db, created.id, "re_engaged")

    assert updated.status == "re_engaged"
    reloaded = await treatment_repo.get_by_id(db, created.id)
    assert reloaded.status == "re_engaged"


async def test_treatment_link_appointment_sets_appointment_id(db, treatment_repo):
    created = await seed_treatment(db, appointment_id=None)
    appointment_id = uuid.uuid4()

    updated = await treatment_repo.link_appointment(db, created.id, appointment_id)

    assert updated.appointment_id == appointment_id
    reloaded = await treatment_repo.get_by_id(db, created.id)
    assert reloaded.appointment_id == appointment_id


async def test_treatment_list_by_ids_returns_only_requested_treatments(db, treatment_repo):
    a = await seed_treatment(db)
    b = await seed_treatment(db)
    await seed_treatment(db)  # not requested

    results = await treatment_repo.list_by_ids(db, [a.id, b.id])

    assert {r.id for r in results} == {a.id, b.id}


async def test_treatment_list_by_ids_ignores_unknown_ids(db, treatment_repo):
    a = await seed_treatment(db)

    results = await treatment_repo.list_by_ids(db, [a.id, uuid.uuid4()])

    assert {r.id for r in results} == {a.id}


# --------------------------------------------------------------------------
# RecallComplianceBaselineRepository
# --------------------------------------------------------------------------


async def test_get_latest_returns_none_when_no_baseline_for_metric(db, baseline_repo):
    assert await baseline_repo.get_latest(db, "recall_compliance_rate") is None


async def test_create_persists_metric_name_and_baseline_value(db, baseline_repo):
    baseline = await baseline_repo.create(db, "recall_compliance_rate", 55.00)

    assert isinstance(baseline, RecallComplianceBaseline)
    assert baseline.metric_name == "recall_compliance_rate"
    assert float(baseline.baseline_value) == pytest.approx(55.00)


async def test_create_sets_captured_at_automatically(db, baseline_repo):
    # create()'s own public surface takes no captured_at argument, so it must
    # stamp one itself rather than leaving the NOT NULL column unset.
    before = datetime.now(timezone.utc).replace(tzinfo=None)

    baseline = await baseline_repo.create(db, "recall_compliance_rate", 55.00)

    assert isinstance(baseline.captured_at, datetime)
    assert baseline.captured_at.replace(tzinfo=None) >= before - timedelta(seconds=5)


async def test_get_latest_returns_most_recently_captured_baseline_for_metric(db, session_factory, baseline_repo):
    older = RecallComplianceBaseline(
        metric_name="recall_compliance_rate",
        baseline_value=Decimal("40.00"),
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    newer = RecallComplianceBaseline(
        metric_name="recall_compliance_rate",
        baseline_value=Decimal("60.00"),
        captured_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    async with session_factory() as s:
        s.add_all([older, newer])
        await s.commit()

    latest = await baseline_repo.get_latest(db, "recall_compliance_rate")

    assert latest is not None
    assert latest.baseline_value == Decimal("60.00")


async def test_get_latest_filters_by_metric_name(db, baseline_repo):
    await baseline_repo.create(db, "recall_compliance_rate", 55.00)
    await baseline_repo.create(db, "some_other_metric", 10.00)

    latest = await baseline_repo.get_latest(db, "some_other_metric")

    assert latest is not None
    assert latest.metric_name == "some_other_metric"
    assert float(latest.baseline_value) == pytest.approx(10.00)
