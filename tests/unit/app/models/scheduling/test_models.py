"""Unit tests for app/models/scheduling/models.py.

Exercises the six ORM tables declared here (providers, chairs, appointments,
appointment_history, appointment_import_batches, appointment_import_errors)
directly against an in-memory SQLite database, mirroring the ENVIRONMENT=test
substitution described in project_rules.testing (app/core/database.py
switches to sqlite+aiosqlite:///:memory: with a StaticPool at the composition
root).

appointments.patient_id and appointment_history/appointment_import_batches'
*_staff_id columns are plain FK columns pointing at patients.id / staff_users.id
(owned by other modules' own models.py, per this file's Interaction contract --
those modules are not part of this task). Every ORM model shares one
Base.metadata (per project_rules.testing), so those FK targets must be
resolvable for Base.metadata.create_all to compile this file's own tables'
DDL. Registering minimal stand-in tables here -- rather than importing another
module's ORM classes -- keeps this test scoped to this file's own spec; the
guard is a no-op if the real patient/console models already registered these
names on the shared metadata (e.g. when this file runs alongside the full
suite).
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.scheduling.models import (
    Appointment,
    AppointmentHistory,
    AppointmentImportBatch,
    AppointmentImportError,
    Chair,
    Provider,
)

pytestmark = pytest.mark.asyncio

if "patients" not in Base.metadata.tables:
    Table("patients", Base.metadata, Column("id", Uuid, primary_key=True))
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


def _appointment_kwargs(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        patient_id=uuid.uuid4(),
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        appointment_type="cleaning",
        scheduled_start=now + timedelta(days=1),
        scheduled_end=now + timedelta(days=1, hours=1),
    )
    defaults.update(overrides)
    return defaults


def make_appointment(**overrides):
    return Appointment(**_appointment_kwargs(**overrides))


def make_batch(**overrides):
    defaults = dict(
        uploaded_by_staff_id=uuid.uuid4(),
        filename="appointments_import.csv",
        status="validating",
    )
    defaults.update(overrides)
    return AppointmentImportBatch(**defaults)


# --------------------------------------------------------------------------
# table identity
# --------------------------------------------------------------------------


def test_table_names():
    assert Provider.__tablename__ == "providers"
    assert Chair.__tablename__ == "chairs"
    assert Appointment.__tablename__ == "appointments"
    assert AppointmentHistory.__tablename__ == "appointment_history"
    assert AppointmentImportBatch.__tablename__ == "appointment_import_batches"
    assert AppointmentImportError.__tablename__ == "appointment_import_errors"


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


async def test_provider_id_is_generated_uuid(session):
    provider = Provider(name="Dr. Souza", specialty="Orthodontics")
    session.add(provider)
    await session.flush()
    assert isinstance(provider.id, uuid.UUID)


async def test_provider_name_required(session):
    provider = Provider(specialty="Orthodontics")
    session.add(provider)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_provider_specialty_nullable(session):
    provider = Provider(name="Dr. Souza")
    session.add(provider)
    await session.commit()
    await session.refresh(provider)
    assert provider.specialty is None


# --------------------------------------------------------------------------
# chairs
# --------------------------------------------------------------------------


async def test_chair_room_required(session):
    chair = Chair(label="Chair 1")
    session.add(chair)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_chair_label_required(session):
    chair = Chair(room="Room A")
    session.add(chair)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_chair_fields_roundtrip(session):
    chair = Chair(room="Room A", label="Chair 1")
    session.add(chair)
    await session.commit()
    await session.refresh(chair)
    assert chair.room == "Room A"
    assert chair.label == "Chair 1"


# --------------------------------------------------------------------------
# appointments -- required fields
# --------------------------------------------------------------------------


async def test_appointment_patient_id_required(session):
    kwargs = _appointment_kwargs()
    del kwargs["patient_id"]
    session.add(Appointment(**kwargs))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_provider_id_required(session):
    kwargs = _appointment_kwargs()
    del kwargs["provider_id"]
    session.add(Appointment(**kwargs))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_chair_id_required(session):
    kwargs = _appointment_kwargs()
    del kwargs["chair_id"]
    session.add(Appointment(**kwargs))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_type_required(session):
    kwargs = _appointment_kwargs()
    del kwargs["appointment_type"]
    session.add(Appointment(**kwargs))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_scheduled_start_required(session):
    kwargs = _appointment_kwargs()
    del kwargs["scheduled_start"]
    session.add(Appointment(**kwargs))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_scheduled_end_required(session):
    kwargs = _appointment_kwargs()
    del kwargs["scheduled_end"]
    session.add(Appointment(**kwargs))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_created_by_staff_id_nullable_for_ai_originated(session):
    # created_by_staff_id is NULLABLE -- null when the appointment was
    # AI-Agent-originated rather than staff-originated.
    appointment = make_appointment(created_by_staff_id=None)
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.created_by_staff_id is None


# --------------------------------------------------------------------------
# appointments -- defaults / enums
# --------------------------------------------------------------------------


async def test_appointment_status_defaults_to_booked(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.status == "booked"


@pytest.mark.parametrize(
    "status_value",
    ["booked", "rescheduled", "cancelled", "completed", "no_show"],
)
async def test_appointment_status_accepts_all_documented_values(session, status_value):
    appointment = make_appointment(status=status_value)
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.status == status_value


async def test_appointment_is_late_cancellation_defaults_to_false(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.is_late_cancellation is False


async def test_appointment_cancellation_reason_nullable(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.cancellation_reason is None


async def test_appointment_risk_flag_and_source_nullable_by_default(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.risk_flag is None
    assert appointment.risk_score_source is None


@pytest.mark.parametrize(
    ("risk_flag_value", "risk_score_source_value"),
    [("low", "rules"), ("medium", "ml"), ("high", "rules")],
)
async def test_appointment_risk_flag_and_source_roundtrip(
    session, risk_flag_value, risk_score_source_value
):
    # US41 transparency: risk_score_source records which mechanism (rules vs
    # ml) produced the risk_flag.
    appointment = make_appointment(
        risk_flag=risk_flag_value, risk_score_source=risk_score_source_value
    )
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.risk_flag == risk_flag_value
    assert appointment.risk_score_source == risk_score_source_value


async def test_appointment_confirmation_status_nullable_by_default(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.confirmation_status is None


@pytest.mark.parametrize("confirmation_status_value", ["sent", "failed", "suppressed"])
async def test_appointment_confirmation_status_roundtrip(session, confirmation_status_value):
    appointment = make_appointment(confirmation_status=confirmation_status_value)
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.confirmation_status == confirmation_status_value


async def test_appointment_source_metadata_nullable_and_roundtrips_through_fresh_session(
    session, session_factory
):
    lineage = {"rescheduled_from": str(uuid.uuid4()), "reason": "provider_unavailable"}
    appointment = make_appointment(source_metadata=lineage)
    session.add(appointment)
    await session.commit()
    appointment_id = appointment.id

    # re-query from a fresh session to force a real round trip through the
    # JSONB/JSON-fallback TypeDecorator rather than relying on the identity map
    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(Appointment, appointment_id)
        assert reloaded.source_metadata == lineage

    other = make_appointment()
    session.add(other)
    await session.commit()
    await session.refresh(other)
    assert other.source_metadata is None


async def test_appointment_created_and_updated_at_populated(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()
    await session.refresh(appointment)
    assert appointment.created_at is not None
    assert appointment.updated_at is not None


# --------------------------------------------------------------------------
# appointments -- RMR001 idempotency_key
# --------------------------------------------------------------------------


async def test_appointment_idempotency_key_uniqueness_enforced(session):
    # RMR001: idempotency_key is UNIQUE -- the mechanism SchedulingService.create
    # relies on to make a retried POST /scheduling/appointments safe to replay
    # rather than double-booking.
    dupe_key = f"idem-{uuid.uuid4().hex[:12]}"
    session.add(make_appointment(idempotency_key=dupe_key))
    await session.commit()

    session.add(make_appointment(idempotency_key=dupe_key))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_appointment_idempotency_key_nullable_multiple_nulls_allowed(session):
    # idempotency_key is NULLABLE -- appointments created without a supplied
    # key (e.g. legacy import) must not collide with one another.
    session.add(make_appointment(idempotency_key=None))
    session.add(make_appointment(idempotency_key=None))
    await session.commit()

    result = await session.execute(select(Appointment))
    assert len(result.scalars().all()) == 2


# --------------------------------------------------------------------------
# appointment_history -- required fields
# --------------------------------------------------------------------------


async def test_history_appointment_id_required(session):
    history = AppointmentHistory(changed_field="scheduled_start", old_value="a", new_value="b")
    session.add(history)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_history_changed_field_required(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()

    history = AppointmentHistory(
        appointment_id=appointment.id, old_value="a", new_value="b"
    )
    session.add(history)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_history_old_and_new_value_nullable(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()

    history = AppointmentHistory(
        appointment_id=appointment.id,
        changed_field="status",
        changed_at=datetime.now(timezone.utc),
    )
    session.add(history)
    await session.commit()
    await session.refresh(history)
    assert history.old_value is None
    assert history.new_value is None


async def test_history_changed_by_staff_id_nullable(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()

    history = AppointmentHistory(
        appointment_id=appointment.id,
        changed_field="status",
        changed_by_staff_id=None,
        changed_at=datetime.now(timezone.utc),
    )
    session.add(history)
    await session.commit()
    await session.refresh(history)
    assert history.changed_by_staff_id is None


async def test_history_changed_at_required(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()

    history = AppointmentHistory(appointment_id=appointment.id, changed_field="status")
    session.add(history)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_history_changed_at_roundtrips(session):
    appointment = make_appointment()
    session.add(appointment)
    await session.commit()

    changed_at = datetime.now(timezone.utc)
    history = AppointmentHistory(
        appointment_id=appointment.id, changed_field="status", changed_at=changed_at
    )
    session.add(history)
    await session.commit()
    await session.refresh(history)
    # compared tz-naive: the SQLite fallback used in test mode does not
    # preserve the UTC offset on round trip the way native Postgres TIMESTAMPTZ
    # does, but the wall-clock value itself must be preserved exactly.
    assert history.changed_at.replace(tzinfo=None) == changed_at.replace(tzinfo=None)


# --------------------------------------------------------------------------
# FR-E4.4: reschedule updates the same appointments row in place
# --------------------------------------------------------------------------


async def test_reschedule_updates_same_row_and_history_preserves_old_and_new_values(
    session,
):
    original_start = datetime.now(timezone.utc) + timedelta(days=2)
    original_end = original_start + timedelta(hours=1)
    original_provider_id = uuid.uuid4()
    original_chair_id = uuid.uuid4()

    appointment = make_appointment(
        provider_id=original_provider_id,
        chair_id=original_chair_id,
        scheduled_start=original_start,
        scheduled_end=original_end,
    )
    session.add(appointment)
    await session.commit()
    appointment_id = appointment.id

    new_start = original_start + timedelta(days=1)
    new_end = original_end + timedelta(days=1)
    new_provider_id = uuid.uuid4()
    new_chair_id = uuid.uuid4()

    # FR-E4.4: the SAME appointments row is updated in place -- never deleted
    # and re-inserted -- so every FK pointing at this appointment id from
    # other modules keeps resolving to it.
    appointment.scheduled_start = new_start
    appointment.scheduled_end = new_end
    appointment.provider_id = new_provider_id
    appointment.chair_id = new_chair_id
    appointment.status = "rescheduled"
    appointment.source_metadata = {
        "rescheduled_from_start": original_start.isoformat(),
        "rescheduled_from_provider_id": str(original_provider_id),
    }

    reschedule_time = datetime.now(timezone.utc)
    session.add(
        AppointmentHistory(
            appointment_id=appointment_id,
            changed_field="scheduled_start",
            old_value=original_start.isoformat(),
            new_value=new_start.isoformat(),
            changed_at=reschedule_time,
        )
    )
    session.add(
        AppointmentHistory(
            appointment_id=appointment_id,
            changed_field="provider_id",
            old_value=str(original_provider_id),
            new_value=str(new_provider_id),
            changed_at=reschedule_time,
        )
    )
    await session.commit()

    # exactly one appointments row exists for this appointment -- the id is
    # unchanged and no duplicate row was inserted
    all_appointments = (await session.execute(select(Appointment))).scalars().all()
    assert len(all_appointments) == 1
    reloaded = all_appointments[0]
    assert reloaded.id == appointment_id
    assert reloaded.status == "rescheduled"
    assert reloaded.scheduled_start == new_start
    assert reloaded.provider_id == new_provider_id
    assert reloaded.source_metadata["rescheduled_from_provider_id"] == str(original_provider_id)

    history_rows = (
        (
            await session.execute(
                select(AppointmentHistory).where(
                    AppointmentHistory.appointment_id == appointment_id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(history_rows) == 2
    by_field = {row.changed_field: row for row in history_rows}
    assert by_field["scheduled_start"].old_value == original_start.isoformat()
    assert by_field["scheduled_start"].new_value == new_start.isoformat()
    assert by_field["provider_id"].old_value == str(original_provider_id)
    assert by_field["provider_id"].new_value == str(new_provider_id)


# --------------------------------------------------------------------------
# appointment_import_batches
# --------------------------------------------------------------------------


async def test_batch_uploaded_by_staff_id_required(session):
    batch = AppointmentImportBatch(filename="import.csv", status="validating")
    session.add(batch)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_batch_filename_required(session):
    batch = AppointmentImportBatch(uploaded_by_staff_id=uuid.uuid4(), status="validating")
    session.add(batch)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_batch_status_required(session):
    batch = AppointmentImportBatch(uploaded_by_staff_id=uuid.uuid4(), filename="import.csv")
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


@pytest.mark.parametrize("status_value", ["validating", "rejected", "committed"])
async def test_batch_status_accepts_all_documented_values(session, status_value):
    batch = make_batch(status=status_value)
    session.add(batch)
    await session.commit()
    await session.refresh(batch)
    assert batch.status == status_value


# --------------------------------------------------------------------------
# appointment_import_errors
# --------------------------------------------------------------------------


async def test_error_batch_id_required(session):
    error = AppointmentImportError(
        row_number=3, column_name="patient_code", error_message="Unknown patient_code"
    )
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_row_number_required(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = AppointmentImportError(
        batch_id=batch.id, column_name="patient_code", error_message="Unknown patient_code"
    )
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_column_name_required(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = AppointmentImportError(
        batch_id=batch.id, row_number=3, error_message="Unknown patient_code"
    )
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_message_required(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = AppointmentImportError(batch_id=batch.id, row_number=3, column_name="patient_code")
    session.add(error)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_error_unmatched_patient_defaults_to_false(session):
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = AppointmentImportError(
        batch_id=batch.id,
        row_number=5,
        column_name="scheduled_start",
        error_message="Unparseable date",
    )
    session.add(error)
    await session.commit()
    await session.refresh(error)
    assert error.unmatched_patient is False


async def test_error_unmatched_patient_can_be_flagged_true(session):
    # US12 AC: rows referencing an unknown patient_code are flagged
    # unmatched_patient=True.
    batch = make_batch()
    session.add(batch)
    await session.commit()

    error = AppointmentImportError(
        batch_id=batch.id,
        row_number=8,
        column_name="patient_code",
        error_message="No patient found for patient_code PT-000999",
        unmatched_patient=True,
    )
    session.add(error)
    await session.commit()
    await session.refresh(error)
    assert error.unmatched_patient is True
    assert error.batch_id == batch.id
    assert error.row_number == 8
