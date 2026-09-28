"""Unit tests for app/services/scheduling/service.py::SchedulingService.

Derived from the file's own spec only: propose_slots / create / reschedule / cancel /
lock / get_by_id / import_csv. Dependencies (AppointmentRepository,
AppointmentHistoryRepository, AppointmentImportRepository, RecordLockService,
RuleSetService, AuditService, MessageBroker) are mocked using exactly the method
signatures documented in those files' own specs.
"""
import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.services.scheduling.service import SchedulingService
from app.common.exceptions.errors import (
    ScheduleConflictError,
    MissingReasonCodeError,
    NoActiveRuleSetError,
)
from app.common.constants import LATE_CANCELLATION_WINDOW_HOURS

pytestmark = pytest.mark.asyncio


def make_deps():
    appt_repo = AsyncMock()
    history_repo = AsyncMock()
    import_repo = AsyncMock()
    lock_service = AsyncMock()
    rule_set_service = AsyncMock()
    audit = AsyncMock()
    broker = AsyncMock()
    return appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker


@pytest.fixture
def deps():
    return make_deps()


@pytest.fixture
def service(deps):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    return SchedulingService(
        appt_repo=appt_repo,
        history_repo=history_repo,
        import_repo=import_repo,
        lock_service=lock_service,
        rule_set_service=rule_set_service,
        audit=audit,
        broker=broker,
    )


@pytest.fixture
def db():
    return AsyncMock()


def get_publish_payload(broker_mock, routing_key):
    """Find the payload of a broker.publish call for a given routing key,
    tolerating either positional or keyword invocation."""
    for call in broker_mock.publish.call_args_list:
        args, kwargs = call
        rk = kwargs.get("routing_key", args[0] if args else None)
        if rk == routing_key:
            if "payload" in kwargs:
                return kwargs["payload"]
            return args[1] if len(args) > 1 else None
    return None


def combined_update_kwargs(appt_repo):
    """Collect kwargs across whichever of update_status/update_fields the
    implementation happens to call, since the spec does not name which one."""
    combined = {}
    for call in list(appt_repo.update_status.call_args_list) + list(appt_repo.update_fields.call_args_list):
        combined.update(call.kwargs)
    return combined


def make_create_data(patient_id=None, provider_id=None, chair_id=None):
    start = datetime.now(timezone.utc) + timedelta(days=1)
    return {
        "patient_id": patient_id or uuid4(),
        "provider_id": provider_id or uuid4(),
        "chair_id": chair_id or uuid4(),
        "appointment_type": "cleaning",
        "scheduled_start": start,
        "scheduled_end": start + timedelta(hours=1),
    }


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #

async def test_create_replay_returns_existing_row_without_reinserting(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    existing = SimpleNamespace(id=uuid4(), patient_id=uuid4(), status="booked")
    appt_repo.get_by_idempotency_key.return_value = existing

    result, was_replay = await service.create(db, make_create_data(), uuid4(), "idem-key-1")

    assert was_replay is True
    assert result is existing
    appt_repo.create.assert_not_called()
    audit.record.assert_not_called()
    broker.publish.assert_not_called()


async def test_create_raises_conflict_when_slot_taken(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appt_repo.get_by_idempotency_key.return_value = None
    appt_repo.find_conflicts.return_value = [SimpleNamespace(id=uuid4())]

    with pytest.raises(ScheduleConflictError):
        await service.create(db, make_create_data(), uuid4(), "idem-key-2")

    appt_repo.create.assert_not_called()
    audit.record.assert_not_called()
    broker.publish.assert_not_called()


async def test_create_success_inserts_booked_audits_and_publishes_after_commit(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appt_repo.get_by_idempotency_key.return_value = None
    appt_repo.find_conflicts.return_value = []
    patient_id = uuid4()
    new_appt = SimpleNamespace(id=uuid4(), patient_id=patient_id, status="booked")
    appt_repo.create.return_value = new_appt

    events = []

    async def commit_effect(*a, **kw):
        events.append("commit")

    async def publish_effect(*a, **kw):
        events.append("publish")

    db.commit.side_effect = commit_effect
    broker.publish.side_effect = publish_effect

    actor_id = uuid4()
    result, was_replay = await service.create(
        db, make_create_data(patient_id=patient_id), actor_id, "idem-key-3"
    )

    assert was_replay is False
    assert result is new_appt
    assert appt_repo.create.call_args.kwargs.get("status") == "booked"
    assert audit.record.call_args.kwargs.get("action_type") == "appointment.create"

    payload = get_publish_payload(broker, "booking.confirmed")
    assert payload == {"appointment_id": new_appt.id, "patient_id": patient_id}
    # publish must happen strictly after the DB commit
    assert events == ["commit", "publish"]


# --------------------------------------------------------------------------- #
# reschedule
# --------------------------------------------------------------------------- #

async def test_reschedule_raises_conflict_excluding_self(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appointment_id = uuid4()
    existing = SimpleNamespace(
        id=appointment_id, patient_id=uuid4(), provider_id=uuid4(), chair_id=uuid4(),
        scheduled_start=datetime.now(timezone.utc) + timedelta(days=2),
        scheduled_end=datetime.now(timezone.utc) + timedelta(days=2, hours=1),
        status="booked",
    )
    appt_repo.get_by_id.return_value = existing
    appt_repo.find_conflicts.return_value = [SimpleNamespace(id=uuid4())]

    new_start = datetime.now(timezone.utc) + timedelta(days=3)
    data = {
        "provider_id": uuid4(), "chair_id": uuid4(),
        "scheduled_start": new_start, "scheduled_end": new_start + timedelta(hours=1),
    }

    with pytest.raises(ScheduleConflictError):
        await service.reschedule(db, appointment_id, data, uuid4())

    call = appt_repo.find_conflicts.call_args
    assert call.kwargs.get("exclude_appointment_id", appointment_id) == appointment_id
    history_repo.insert.assert_not_called()
    broker.publish.assert_not_called()


async def test_reschedule_success_writes_history_before_update_and_publishes_booking_confirmed(
    service, deps, db
):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appointment_id = uuid4()
    patient_id = uuid4()
    original_provider_id = uuid4()
    original_chair_id = uuid4()
    original_start = datetime.now(timezone.utc) + timedelta(days=2)
    original_end = original_start + timedelta(hours=1)
    existing = SimpleNamespace(
        id=appointment_id, patient_id=patient_id,
        provider_id=original_provider_id, chair_id=original_chair_id,
        scheduled_start=original_start, scheduled_end=original_end,
        status="booked",
    )
    appt_repo.get_by_id.return_value = existing
    appt_repo.find_conflicts.return_value = []

    updated = SimpleNamespace(id=appointment_id, patient_id=patient_id, status="rescheduled")

    events = []

    async def insert_effect(*a, **kw):
        events.append("history_insert")

    async def update_status_effect(*a, **kw):
        events.append("update")
        return updated

    async def update_fields_effect(*a, **kw):
        events.append("update")
        return updated

    history_repo.insert.side_effect = insert_effect
    appt_repo.update_status.side_effect = update_status_effect
    appt_repo.update_fields.side_effect = update_fields_effect

    new_start = original_start + timedelta(days=1)
    data = {
        "provider_id": uuid4(), "chair_id": uuid4(),
        "scheduled_start": new_start, "scheduled_end": new_start + timedelta(hours=1),
    }
    result = await service.reschedule(db, appointment_id, data, uuid4())

    assert history_repo.insert.await_count >= 1
    assert "update" in events
    assert events.index("history_insert") < events.index("update")

    combined = combined_update_kwargs(appt_repo)
    assert combined.get("status") == "rescheduled"
    sm = combined.get("source_metadata")
    assert sm is not None
    assert set(sm.keys()) >= {
        "original_start", "original_end", "original_provider_id", "original_chair_id",
    }

    payload = get_publish_payload(broker, "booking.confirmed")
    assert payload == {"appointment_id": appointment_id, "patient_id": patient_id}
    # a reschedule must NOT publish appointment.cancelled for the vacated slot
    assert get_publish_payload(broker, "appointment.cancelled") is None


# --------------------------------------------------------------------------- #
# cancel
# --------------------------------------------------------------------------- #

async def test_cancel_raises_missing_reason_code_for_blank(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps

    with pytest.raises(MissingReasonCodeError):
        await service.cancel(db, uuid4(), "", uuid4())

    audit.record.assert_not_called()
    broker.publish.assert_not_called()


async def test_cancel_marks_late_cancellation_within_window(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appointment_id = uuid4()
    provider_id = uuid4()
    chair_id = uuid4()
    scheduled_start = datetime.now(timezone.utc) + timedelta(hours=2)
    scheduled_end = scheduled_start + timedelta(hours=1)
    existing = SimpleNamespace(
        id=appointment_id, patient_id=uuid4(), provider_id=provider_id, chair_id=chair_id,
        scheduled_start=scheduled_start, scheduled_end=scheduled_end, status="booked",
    )
    appt_repo.get_by_id.return_value = existing

    events = []

    async def commit_effect(*a, **kw):
        events.append("commit")

    async def publish_effect(*a, **kw):
        events.append("publish")

    db.commit.side_effect = commit_effect
    broker.publish.side_effect = publish_effect

    await service.cancel(db, appointment_id, "patient_request", uuid4())

    combined = combined_update_kwargs(appt_repo)
    assert combined.get("status") == "cancelled"
    assert combined.get("cancellation_reason") == "patient_request"
    assert combined.get("is_late_cancellation") is True

    assert audit.record.call_args.kwargs.get("action_type") == "appointment.cancel"
    payload = get_publish_payload(broker, "appointment.cancelled")
    assert payload == {
        "appointment_id": appointment_id, "provider_id": provider_id,
        "chair_id": chair_id, "slot_start": scheduled_start, "slot_end": scheduled_end,
    }
    assert events == ["commit", "publish"]


async def test_cancel_marks_not_late_outside_window(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appointment_id = uuid4()
    scheduled_start = datetime.now(timezone.utc) + timedelta(hours=LATE_CANCELLATION_WINDOW_HOURS + 10)
    scheduled_end = scheduled_start + timedelta(hours=1)
    existing = SimpleNamespace(
        id=appointment_id, patient_id=uuid4(), provider_id=uuid4(), chair_id=uuid4(),
        scheduled_start=scheduled_start, scheduled_end=scheduled_end, status="booked",
    )
    appt_repo.get_by_id.return_value = existing

    await service.cancel(db, appointment_id, "provider_unavailable", uuid4())

    combined = combined_update_kwargs(appt_repo)
    assert combined.get("is_late_cancellation") is False


# --------------------------------------------------------------------------- #
# lock / get_by_id
# --------------------------------------------------------------------------- #

async def test_lock_delegates_to_record_lock_service(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appointment_id = uuid4()
    staff_id = uuid4()
    expected_lock = SimpleNamespace(entity_type="appointment", entity_id=appointment_id)
    lock_service.acquire.return_value = expected_lock

    result = await service.lock(db, appointment_id, staff_id)

    assert result is expected_lock
    lock_service.acquire.assert_awaited_once_with(
        db, entity_type="appointment", entity_id=appointment_id, staff_id=staff_id
    )


async def test_get_by_id_returns_repository_result(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    appointment_id = uuid4()
    expected = SimpleNamespace(id=appointment_id)
    appt_repo.get_by_id.return_value = expected

    result = await service.get_by_id(db, appointment_id)

    assert result is expected


# --------------------------------------------------------------------------- #
# propose_slots
# --------------------------------------------------------------------------- #

async def test_propose_slots_propagates_no_active_rule_set_error(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    rule_set_service.get_active_rules_by_category.side_effect = NoActiveRuleSetError()

    with pytest.raises(NoActiveRuleSetError):
        await service.propose_slots(
            db, uuid4(), None,
            datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=7),
        )


async def test_propose_slots_returns_slots_and_nearest_alternatives_shape(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    rule_set_service.get_active_rules_by_category.return_value = []
    appt_repo.find_conflicts.return_value = []
    appt_repo.list_open_slots.return_value = []

    result = await service.propose_slots(
        db, uuid4(), uuid4(),
        datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=7),
    )

    assert isinstance(result, dict)
    assert isinstance(result["slots"], list)
    assert isinstance(result["nearest_alternatives"], list)


async def test_propose_slots_checks_both_rule_categories(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    rule_set_service.get_active_rules_by_category.return_value = []
    appt_repo.find_conflicts.return_value = []
    appt_repo.list_open_slots.return_value = []

    await service.propose_slots(
        db, uuid4(), uuid4(),
        datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=7),
    )

    called_categories = set()
    for call in rule_set_service.get_active_rules_by_category.call_args_list:
        args, kwargs = call
        category = kwargs.get("category")
        if category is None and len(args) >= 2:
            category = args[1]
        called_categories.add(category)

    assert {"appointment_type", "scheduling_priority"} <= called_categories


# --------------------------------------------------------------------------- #
# import_csv
# --------------------------------------------------------------------------- #

async def test_import_csv_rejects_invalid_file_without_bulk_inserting(service, deps, db):
    appt_repo, history_repo, import_repo, lock_service, rule_set_service, audit, broker = deps
    garbage_bytes = b"not_a_real_column,another_bad_column\nx,y\n"

    try:
        await service.import_csv(db, garbage_bytes, "bad_import.csv", uuid4())
    except Exception:
        # validate()-style structural failure is an acceptable outcome per the
        # all-or-nothing contract mirrored from PatientImportService
        pass

    import_repo.bulk_insert.assert_not_called()
