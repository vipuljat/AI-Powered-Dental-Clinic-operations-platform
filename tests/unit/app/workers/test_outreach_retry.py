"""Unit tests for app/workers/outreach_retry.py.

These tests exercise the module's public surface directly:
    - run(broker, session_factory)
    - handle_booking_confirmed(payload, headers)
    - handle_waitlist_offer_created(payload, headers)

All DB access, repository/service calls, and the message broker are mocked per the
project's test-substitution rules (no live infra); dependencies are patched exactly as
named in the spec ("WHAT THIS CODE CALLS"): AppointmentRepository, PatientRepository,
OutreachMessageRepository (app.repositories.scheduling/patients/outreach.repository) and
OutreachService (app.services.outreach.service), accessed via app.workers.outreach_retry's
own module namespace.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import app.workers.outreach_retry as outreach_retry

pytestmark = pytest.mark.asyncio


class _FakeSessionCM:
    """A minimal async-context-manager standing in for `session_factory()`."""

    def __init__(self, db):
        self._db = db

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_db():
    return MagicMock(name="AsyncSession")


@pytest.fixture
def patch_get_db(monkeypatch, fake_db):
    """Patch the module-level get_db() dependency used by the event handlers."""

    async def _fake_get_db():
        yield fake_db

    monkeypatch.setattr(outreach_retry, "get_db", _fake_get_db)
    return fake_db


# ---------------------------------------------------------------------------
# handle_booking_confirmed
# ---------------------------------------------------------------------------


async def test_handle_booking_confirmed_loads_appointment_and_patient(
    monkeypatch, patch_get_db
):
    appointment = SimpleNamespace(id="appt-1")
    patient = SimpleNamespace(id="pat-1", language="es")
    message = SimpleNamespace(status="sent")

    appt_repo = MagicMock()
    appt_repo.get_by_id = AsyncMock(return_value=appointment)
    appt_repo.update_fields = AsyncMock()
    patient_repo = MagicMock()
    patient_repo.get_by_id = AsyncMock(return_value=patient)
    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock(return_value=message)

    monkeypatch.setattr(outreach_retry, "AppointmentRepository", appt_repo)
    monkeypatch.setattr(outreach_retry, "PatientRepository", patient_repo)
    monkeypatch.setattr(outreach_retry, "OutreachService", outreach_service)

    payload = {"appointment_id": "appt-1", "patient_id": "pat-1"}
    await outreach_retry.handle_booking_confirmed(payload, {})

    assert appt_repo.get_by_id.await_count == 1
    assert "appt-1" in appt_repo.get_by_id.await_args.args
    assert patient_repo.get_by_id.await_count == 1
    assert "pat-1" in patient_repo.get_by_id.await_args.args


async def test_handle_booking_confirmed_dispatches_confirmation_campaign(
    monkeypatch, patch_get_db
):
    appointment = SimpleNamespace(id="appt-1")
    patient = SimpleNamespace(id="pat-1", language="fr")
    message = SimpleNamespace(status="sent")

    appt_repo = MagicMock()
    appt_repo.get_by_id = AsyncMock(return_value=appointment)
    appt_repo.update_fields = AsyncMock()
    patient_repo = MagicMock()
    patient_repo.get_by_id = AsyncMock(return_value=patient)
    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock(return_value=message)

    monkeypatch.setattr(outreach_retry, "AppointmentRepository", appt_repo)
    monkeypatch.setattr(outreach_retry, "PatientRepository", patient_repo)
    monkeypatch.setattr(outreach_retry, "OutreachService", outreach_service)

    payload = {"appointment_id": "appt-1", "patient_id": "pat-1"}
    await outreach_retry.handle_booking_confirmed(payload, {})

    outreach_service.dispatch.assert_awaited_once()
    kwargs = outreach_service.dispatch.await_args.kwargs
    assert kwargs["campaign_type"] == "confirmation"
    assert kwargs["patient_id"] == "pat-1"
    assert kwargs["language"] == "fr"
    assert kwargs["related_entity_type"] == "appointment"
    assert kwargs["related_entity_id"] == "appt-1"


@pytest.mark.parametrize(
    "dispatch_status,expected_confirmation_status",
    [("sent", "sent"), ("failed", "failed"), ("suppressed", "suppressed")],
)
async def test_handle_booking_confirmed_writes_back_confirmation_status(
    monkeypatch, patch_get_db, dispatch_status, expected_confirmation_status
):
    appointment = SimpleNamespace(id="appt-9")
    patient = SimpleNamespace(id="pat-9", language="en")
    message = SimpleNamespace(status=dispatch_status)

    appt_repo = MagicMock()
    appt_repo.get_by_id = AsyncMock(return_value=appointment)
    appt_repo.update_fields = AsyncMock()
    patient_repo = MagicMock()
    patient_repo.get_by_id = AsyncMock(return_value=patient)
    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock(return_value=message)

    monkeypatch.setattr(outreach_retry, "AppointmentRepository", appt_repo)
    monkeypatch.setattr(outreach_retry, "PatientRepository", patient_repo)
    monkeypatch.setattr(outreach_retry, "OutreachService", outreach_service)

    payload = {"appointment_id": "appt-9", "patient_id": "pat-9"}
    await outreach_retry.handle_booking_confirmed(payload, {})

    appt_repo.update_fields.assert_awaited_once()
    args, kwargs = appt_repo.update_fields.await_args
    assert "appt-9" in args
    assert kwargs.get("confirmation_status") == expected_confirmation_status


# ---------------------------------------------------------------------------
# handle_waitlist_offer_created
# ---------------------------------------------------------------------------


async def test_handle_waitlist_offer_created_dispatches_waitlist_offer_campaign(
    monkeypatch, patch_get_db
):
    entry = SimpleNamespace(id="wl-1", patient_id="pat-5", language="en")
    message = SimpleNamespace(status="sent")

    waitlist_repo = MagicMock()
    waitlist_repo.get_by_id = AsyncMock(return_value=entry)
    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock(return_value=message)

    monkeypatch.setattr(outreach_retry, "WaitlistEntryRepository", waitlist_repo)
    monkeypatch.setattr(outreach_retry, "OutreachService", outreach_service)

    payload = {"waitlist_entry_id": "wl-1", "idle_chair_alert_id": "alert-1"}
    await outreach_retry.handle_waitlist_offer_created(payload, {})

    outreach_service.dispatch.assert_awaited_once()
    kwargs = outreach_service.dispatch.await_args.kwargs
    assert kwargs["campaign_type"] == "waitlist_offer"
    assert kwargs["related_entity_type"] == "waitlist_entry"
    assert kwargs["related_entity_id"] == "wl-1"


async def test_handle_waitlist_offer_created_loads_entry_by_id_from_payload(
    monkeypatch, patch_get_db
):
    entry = SimpleNamespace(id="wl-2", patient_id="pat-6", language="en")
    message = SimpleNamespace(status="sent")

    waitlist_repo = MagicMock()
    waitlist_repo.get_by_id = AsyncMock(return_value=entry)
    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock(return_value=message)

    monkeypatch.setattr(outreach_retry, "WaitlistEntryRepository", waitlist_repo)
    monkeypatch.setattr(outreach_retry, "OutreachService", outreach_service)

    payload = {"waitlist_entry_id": "wl-2", "idle_chair_alert_id": "alert-9"}
    await outreach_retry.handle_waitlist_offer_created(payload, {})

    waitlist_repo.get_by_id.assert_awaited_once()
    assert "wl-2" in waitlist_repo.get_by_id.await_args.args


async def test_handle_waitlist_offer_created_does_not_touch_appointment_status(
    monkeypatch, patch_get_db
):
    """This worker only sends the offer; it never processes the patient's channel
    response nor touches appointment confirmation_status (that belongs to
    WaitlistFillService.respond via POST /waitlist/offers/{id}/respond)."""
    entry = SimpleNamespace(id="wl-3", patient_id="pat-7", language="en")
    message = SimpleNamespace(status="sent")

    waitlist_repo = MagicMock()
    waitlist_repo.get_by_id = AsyncMock(return_value=entry)
    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock(return_value=message)
    appt_repo = MagicMock()
    appt_repo.update_fields = AsyncMock()

    monkeypatch.setattr(outreach_retry, "WaitlistEntryRepository", waitlist_repo)
    monkeypatch.setattr(outreach_retry, "OutreachService", outreach_service)
    monkeypatch.setattr(outreach_retry, "AppointmentRepository", appt_repo)

    payload = {"waitlist_entry_id": "wl-3", "idle_chair_alert_id": "alert-3"}
    await outreach_retry.handle_waitlist_offer_created(payload, {})

    appt_repo.update_fields.assert_not_awaited()


# ---------------------------------------------------------------------------
# run() wiring: consumer registration
# ---------------------------------------------------------------------------


async def test_run_registers_booking_confirmed_consumer(monkeypatch):
    async def _never_returns(*args, **kwargs):
        await asyncio.sleep(3600)

    broker = MagicMock()
    broker.consume = AsyncMock(side_effect=_never_returns)

    retry_repo = MagicMock()
    retry_repo.list_retry_due = AsyncMock(return_value=[])
    monkeypatch.setattr(outreach_retry, "OutreachMessageRepository", retry_repo)

    fake_db = MagicMock()
    session_factory = MagicMock(side_effect=lambda: _FakeSessionCM(fake_db))

    try:
        await asyncio.wait_for(outreach_retry.run(broker, session_factory), timeout=0.2)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    assert (
        call(
            "outreach_retry.booking_confirmed",
            "booking.confirmed",
            outreach_retry.handle_booking_confirmed,
        )
        in broker.consume.await_args_list
    )


async def test_run_registers_waitlist_offer_created_consumer(monkeypatch):
    async def _never_returns(*args, **kwargs):
        await asyncio.sleep(3600)

    broker = MagicMock()
    broker.consume = AsyncMock(side_effect=_never_returns)

    retry_repo = MagicMock()
    retry_repo.list_retry_due = AsyncMock(return_value=[])
    monkeypatch.setattr(outreach_retry, "OutreachMessageRepository", retry_repo)

    fake_db = MagicMock()
    session_factory = MagicMock(side_effect=lambda: _FakeSessionCM(fake_db))

    try:
        await asyncio.wait_for(outreach_retry.run(broker, session_factory), timeout=0.2)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    assert (
        call(
            "outreach_retry.waitlist_offer_created",
            "waitlist_offer.created",
            outreach_retry.handle_waitlist_offer_created,
        )
        in broker.consume.await_args_list
    )
