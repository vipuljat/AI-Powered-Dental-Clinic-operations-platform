"""Unit tests for app/models/recall/models.py.

Exercises the three recall ORM tables (recall_schedules,
unscheduled_treatments, recall_compliance_baselines) directly against an
in-memory SQLite database, mirroring the ENVIRONMENT=test substitution
described in project_rules.testing (app/core/database.py switches to
sqlite+aiosqlite:///:memory: so the same ORM models run unmodified against
both Postgres and SQLite).
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# The real composition roots set this before ever importing app.core.database;
# do the same here defensively so importing it never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from app.core.database import Base  # noqa: E402
from app.models.recall.models import (  # noqa: E402
    RecallComplianceBaseline,
    RecallSchedule,
    UnscheduledTreatment,
)

pytestmark = pytest.mark.asyncio

# recall_schedules.patient_id/appointment_id and unscheduled_treatments
# .patient_id/appointment_id are plain FK columns pointing at patients.id /
# appointments.id -- tables owned by other modules' own models.py, per the
# Interaction contract ("no cross-module relationship()"), which this task
# has no spec for and must not import. Register minimal stand-in tables
# (id column only) so `create_all` can resolve those FK targets and create
# *this* module's tables without requiring any other module's implementation
# to exist. This is a no-op if the real patients/scheduling models already
# registered these names on the shared metadata.
for _name in ("patients", "appointments"):
    if _name not in Base.metadata.tables:
        Table(_name, Base.metadata, Column("id", Uuid(), primary_key=True))


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


def make_recall_schedule(**overrides):
    defaults = dict(
        patient_id=uuid.uuid4(),
        due_date=date(2026, 4, 1),
        status="due",
        interval_source="risk_based",
    )
    defaults.update(overrides)
    return RecallSchedule(**defaults)


def make_unscheduled_treatment(**overrides):
    defaults = dict(
        patient_id=uuid.uuid4(),
        treatment_code="D2140",
        accepted_at=datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return UnscheduledTreatment(**defaults)


def make_recall_compliance_baseline(**overrides):
    defaults = dict(
        metric_name="recall_compliance_rate",
        baseline_value=Decimal("55.00"),
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return RecallComplianceBaseline(**defaults)


# --------------------------------------------------------------------------
# table identity / no cross-module relationship (Interaction contract)
# --------------------------------------------------------------------------


def test_table_names():
    assert RecallSchedule.__tablename__ == "recall_schedules"
    assert UnscheduledTreatment.__tablename__ == "unscheduled_treatments"
    assert RecallComplianceBaseline.__tablename__ == "recall_compliance_baselines"


@pytest.mark.parametrize(
    "model", [RecallSchedule, UnscheduledTreatment, RecallComplianceBaseline]
)
def test_no_cross_module_orm_relationships_declared(model):
    assert list(inspect(model).relationships) == []


@pytest.mark.parametrize(
    "model", [RecallSchedule, UnscheduledTreatment, RecallComplianceBaseline]
)
def test_id_is_the_primary_key(model):
    assert [c.name for c in inspect(model).primary_key] == ["id"]


# --------------------------------------------------------------------------
# recall_schedules
# --------------------------------------------------------------------------


async def test_recall_schedule_round_trips_all_fields(session, session_factory):
    patient_id = uuid.uuid4()
    appointment_id = uuid.uuid4()
    last_recall_at = datetime(2025, 10, 1, 8, 0, tzinfo=timezone.utc)
    schedule = make_recall_schedule(
        patient_id=patient_id,
        appointment_id=appointment_id,
        risk_classification="high_risk",
        due_date=date(2026, 4, 1),
        last_recall_at=last_recall_at,
        status="overdue",
        interval_source="risk_based",
    )
    session.add(schedule)
    await session.commit()
    row_id = schedule.id

    async with session_factory() as fresh_session:
        fetched = await fresh_session.get(RecallSchedule, row_id)
        assert fetched.patient_id == patient_id
        assert fetched.appointment_id == appointment_id
        assert fetched.risk_classification == "high_risk"
        assert fetched.due_date == date(2026, 4, 1)
        assert fetched.status == "overdue"
        assert fetched.interval_source == "risk_based"
        assert fetched.created_at is not None
        assert fetched.updated_at is not None


async def test_recall_schedule_appointment_id_is_nullable(session):
    # "set on completion" -- must be omittable before the recall appointment exists.
    schedule = make_recall_schedule(appointment_id=None)
    session.add(schedule)
    await session.commit()
    await session.refresh(schedule)
    assert schedule.appointment_id is None


async def test_recall_schedule_risk_classification_is_nullable(session):
    schedule = make_recall_schedule(risk_classification=None)
    session.add(schedule)
    await session.commit()
    await session.refresh(schedule)
    assert schedule.risk_classification is None


async def test_recall_schedule_last_recall_at_is_nullable(session):
    schedule = make_recall_schedule(last_recall_at=None)
    session.add(schedule)
    await session.commit()
    await session.refresh(schedule)
    assert schedule.last_recall_at is None


async def test_recall_schedule_accepts_every_documented_status(session):
    for status in ("due", "overdue", "dormant", "contacted", "completed"):
        schedule = make_recall_schedule(patient_id=uuid.uuid4(), status=status)
        session.add(schedule)
        await session.commit()
        await session.refresh(schedule)
        assert schedule.status == status


async def test_recall_schedule_accepts_every_documented_interval_source(session):
    for interval_source in ("risk_based", "default_fallback"):
        schedule = make_recall_schedule(
            patient_id=uuid.uuid4(), interval_source=interval_source
        )
        session.add(schedule)
        await session.commit()
        await session.refresh(schedule)
        assert schedule.interval_source == interval_source


async def test_recall_schedule_patient_id_required(session):
    schedule = RecallSchedule(
        due_date=date(2026, 4, 1), status="due", interval_source="risk_based"
    )
    session.add(schedule)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_schedule_due_date_required(session):
    schedule = RecallSchedule(
        patient_id=uuid.uuid4(), status="due", interval_source="risk_based"
    )
    session.add(schedule)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_schedule_status_required(session):
    schedule = RecallSchedule(
        patient_id=uuid.uuid4(), due_date=date(2026, 4, 1), interval_source="risk_based"
    )
    session.add(schedule)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_schedule_interval_source_required(session):
    schedule = RecallSchedule(
        patient_id=uuid.uuid4(), due_date=date(2026, 4, 1), status="due"
    )
    session.add(schedule)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_schedule_appointment_id_foreign_key_targets_appointments(session):
    fks = RecallSchedule.__table__.c.appointment_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "appointments.id"


async def test_recall_schedule_patient_id_foreign_key_targets_patients(session):
    fks = RecallSchedule.__table__.c.patient_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "patients.id"


# --------------------------------------------------------------------------
# unscheduled_treatments
# --------------------------------------------------------------------------


async def test_unscheduled_treatment_round_trips_all_fields(session, session_factory):
    patient_id = uuid.uuid4()
    appointment_id = uuid.uuid4()
    accepted_at = datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc)
    treatment = make_unscheduled_treatment(
        patient_id=patient_id,
        treatment_code="D2740",
        description="Crown - porcelain/ceramic",
        accepted_at=accepted_at,
        valuation_amount=Decimal("1200.00"),
        status="re_engaged",
        appointment_id=appointment_id,
    )
    session.add(treatment)
    await session.commit()
    row_id = treatment.id

    async with session_factory() as fresh_session:
        fetched = await fresh_session.get(UnscheduledTreatment, row_id)
        assert fetched.patient_id == patient_id
        assert fetched.treatment_code == "D2740"
        assert fetched.description == "Crown - porcelain/ceramic"
        assert fetched.valuation_amount == Decimal("1200.00")
        assert fetched.status == "re_engaged"
        assert fetched.appointment_id == appointment_id
        assert fetched.created_at is not None
        assert fetched.updated_at is not None


async def test_unscheduled_treatment_status_defaults_to_unscheduled(session):
    treatment = make_unscheduled_treatment()
    session.add(treatment)
    await session.commit()
    await session.refresh(treatment)
    assert treatment.status == "unscheduled"


async def test_unscheduled_treatment_accepts_every_documented_status(session):
    for status in ("unscheduled", "re_engaged", "booked", "declined"):
        treatment = make_unscheduled_treatment(patient_id=uuid.uuid4(), status=status)
        session.add(treatment)
        await session.commit()
        await session.refresh(treatment)
        assert treatment.status == status


async def test_unscheduled_treatment_description_is_nullable(session):
    treatment = make_unscheduled_treatment(description=None)
    session.add(treatment)
    await session.commit()
    await session.refresh(treatment)
    assert treatment.description is None


async def test_unscheduled_treatment_appointment_id_is_nullable(session):
    treatment = make_unscheduled_treatment(appointment_id=None)
    session.add(treatment)
    await session.commit()
    await session.refresh(treatment)
    assert treatment.appointment_id is None


async def test_unscheduled_treatment_valuation_amount_is_nullable(session):
    # Data shapes note: null valuation_amount is what flags the row incomplete.
    treatment = make_unscheduled_treatment(valuation_amount=None)
    session.add(treatment)
    await session.commit()
    await session.refresh(treatment)
    assert treatment.valuation_amount is None


async def test_unscheduled_treatment_patient_id_required(session):
    treatment = UnscheduledTreatment(
        treatment_code="D2140", accepted_at=datetime(2026, 1, 15, tzinfo=timezone.utc)
    )
    session.add(treatment)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_unscheduled_treatment_treatment_code_required(session):
    treatment = UnscheduledTreatment(
        patient_id=uuid.uuid4(), accepted_at=datetime(2026, 1, 15, tzinfo=timezone.utc)
    )
    session.add(treatment)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_unscheduled_treatment_accepted_at_required(session):
    treatment = UnscheduledTreatment(patient_id=uuid.uuid4(), treatment_code="D2140")
    session.add(treatment)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_unscheduled_treatment_appointment_id_foreign_key_targets_appointments(session):
    fks = UnscheduledTreatment.__table__.c.appointment_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "appointments.id"


async def test_unscheduled_treatment_patient_id_foreign_key_targets_patients(session):
    fks = UnscheduledTreatment.__table__.c.patient_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "patients.id"


# --------------------------------------------------------------------------
# recall_compliance_baselines
# --------------------------------------------------------------------------


async def test_recall_compliance_baseline_round_trips_all_fields(session, session_factory):
    captured_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    baseline = make_recall_compliance_baseline(
        metric_name="recall_compliance_rate",
        baseline_value=Decimal("55.00"),
        captured_at=captured_at,
    )
    session.add(baseline)
    await session.commit()
    row_id = baseline.id

    async with session_factory() as fresh_session:
        fetched = await fresh_session.get(RecallComplianceBaseline, row_id)
        assert fetched.metric_name == "recall_compliance_rate"
        assert fetched.baseline_value == Decimal("55.00")


async def test_recall_compliance_baseline_metric_name_required(session):
    baseline = RecallComplianceBaseline(
        baseline_value=Decimal("55.00"), captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    session.add(baseline)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_compliance_baseline_baseline_value_required(session):
    baseline = RecallComplianceBaseline(
        metric_name="recall_compliance_rate",
        captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    session.add(baseline)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_compliance_baseline_captured_at_required(session):
    baseline = RecallComplianceBaseline(
        metric_name="recall_compliance_rate", baseline_value=Decimal("55.00")
    )
    session.add(baseline)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_recall_compliance_baseline_is_readable_by_metric_name_when_captured(session):
    # FR-E7.4/US36: RecallCampaignService's compliance endpoint reads this row
    # by metric_name="recall_compliance_rate" to report a rate-vs-baseline.
    session.add(
        make_recall_compliance_baseline(
            metric_name="recall_compliance_rate", baseline_value=Decimal("42.50")
        )
    )
    await session.commit()

    result = await session.execute(
        select(RecallComplianceBaseline).where(
            RecallComplianceBaseline.metric_name == "recall_compliance_rate"
        )
    )
    found = result.scalar_one()
    assert found.baseline_value == Decimal("42.50")


async def test_recall_compliance_baseline_absent_for_metric_name_signals_baseline_pending(session):
    # US36 Alternate Flow: with no row for "recall_compliance_rate" yet, the
    # lookup this model's own public surface exposes must yield nothing --
    # the raw-count/"baseline pending" branch, not a fabricated rate.
    result = await session.execute(
        select(RecallComplianceBaseline).where(
            RecallComplianceBaseline.metric_name == "recall_compliance_rate"
        )
    )
    assert result.scalar_one_or_none() is None
