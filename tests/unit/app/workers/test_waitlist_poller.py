"""Unit tests for app/workers/waitlist_poller.py, derived from its spec only.

Public surface: ``run(broker, session_factory)`` and
``handle_cancellation(payload, headers)``. ``handle_cancellation`` does not
receive session_factory/get_db as an argument directly, so -- per the
verified real signature of app.core.database.get_db given for this task --
these tests patch ``get_db`` inside the worker module to hand back a fake
AsyncSession, consistent with app.core.database.configure_session_factory
existing precisely so ad-hoc worker sessions can be sourced from the
run()-provided session_factory via get_db().
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import app.workers.waitlist_poller as waitlist_poller

pytestmark = pytest.mark.asyncio


def _bind(sig_func, call):
    sig = inspect.signature(sig_func)
    bound = sig.bind(*call.args, **call.kwargs)
    bound.apply_defaults()
    return bound.arguments


def _patch_get_db(monkeypatch, fake_db):
    async def _fake_get_db():
        yield fake_db

    monkeypatch.setattr(waitlist_poller, "get_db", _fake_get_db, raising=False)


def _patch_idle_chair_service(monkeypatch, detect_result=None):
    fake_service = MagicMock()
    fake_service.detect_from_slot = AsyncMock(return_value=detect_result or (MagicMock(), True))
    monkeypatch.setattr(waitlist_poller, "IdleChairPollingService", lambda *a, **kw: fake_service)
    return fake_service


def _cancellation_payload():
    now = datetime.now(timezone.utc)
    return {
        "appointment_id": uuid4(),
        "provider_id": uuid4(),
        "chair_id": uuid4(),
        "slot_start": now,
        "slot_end": now + timedelta(minutes=30),
    }


# ---------------------------------------------------------------------------
# handle_cancellation
# ---------------------------------------------------------------------------


async def test_handle_cancellation_calls_detect_from_slot_with_payload_fields_and_auto_detected_source(monkeypatch):
    fake_db = MagicMock()
    fake_db.commit = AsyncMock()
    _patch_get_db(monkeypatch, fake_db)
    fake_service = _patch_idle_chair_service(monkeypatch)

    payload = _cancellation_payload()
    await waitlist_poller.handle_cancellation(payload, {})

    fake_service.detect_from_slot.assert_awaited_once()
    bound = _bind(
        lambda db, provider_id, chair_id, slot_start, slot_end, source, appointment_id=None: None,
        fake_service.detect_from_slot.await_args,
    )
    assert bound["db"] is fake_db
    assert bound["provider_id"] == payload["provider_id"]
    assert bound["chair_id"] == payload["chair_id"]
    assert bound["slot_start"] == payload["slot_start"]
    assert bound["slot_end"] == payload["slot_end"]
    assert bound["appointment_id"] == payload["appointment_id"]


async def test_handle_cancellation_uses_auto_detected_source(monkeypatch):
    """FR-E5.1: handle_cancellation is the event-driven detection path, so
    it must be distinguishable from the scheduled-scan path via
    source='auto_detected' -- dedup (FR-E5.2) then applies uniformly."""
    fake_db = MagicMock()
    fake_db.commit = AsyncMock()
    _patch_get_db(monkeypatch, fake_db)
    fake_service = _patch_idle_chair_service(monkeypatch)

    await waitlist_poller.handle_cancellation(_cancellation_payload(), {})

    bound = _bind(
        lambda db, provider_id, chair_id, slot_start, slot_end, source, appointment_id=None: None,
        fake_service.detect_from_slot.await_args,
    )
    assert bound["source"] == "auto_detected"


async def test_handle_cancellation_commits_the_session(monkeypatch):
    fake_db = MagicMock()
    fake_db.commit = AsyncMock()
    _patch_get_db(monkeypatch, fake_db)
    _patch_idle_chair_service(monkeypatch)

    await waitlist_poller.handle_cancellation(_cancellation_payload(), {})

    fake_db.commit.assert_awaited()


async def test_handle_cancellation_opens_a_fresh_session_each_call(monkeypatch):
    """Interaction contract: each iteration opens and closes its own
    AsyncSession -- never a long-lived, reused session across calls."""
    call_count = {"n": 0}

    async def _fake_get_db():
        call_count["n"] += 1
        db = MagicMock()
        db.commit = AsyncMock()
        yield db

    monkeypatch.setattr(waitlist_poller, "get_db", _fake_get_db, raising=False)
    _patch_idle_chair_service(monkeypatch)

    await waitlist_poller.handle_cancellation(_cancellation_payload(), {})
    await waitlist_poller.handle_cancellation(_cancellation_payload(), {})

    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class _FakeBroker:
    def __init__(self):
        self.consume = AsyncMock()


class _FakeSessionCM:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _fake_session_factory():
    return _FakeSessionCM()


class _FakeAnyService:
    def __init__(self, *a, **kw):
        pass

    async def run_cycle(self, *a, **kw):
        return 0

    async def sweep_expired_offers(self, *a, **kw):
        return 0

    def __getattr__(self, name):
        async def _m(*a, **kw):
            return 0
        return _m


async def test_run_registers_cancellation_consumer_with_documented_queue_routing_key_and_handler(monkeypatch):
    monkeypatch.setattr(waitlist_poller, "IdleChairPollingService", _FakeAnyService, raising=False)
    monkeypatch.setattr(waitlist_poller, "WaitlistFillService", _FakeAnyService, raising=False)
    fake_broker = _FakeBroker()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            waitlist_poller.run(fake_broker, _fake_session_factory), timeout=0.3
        )

    fake_broker.consume.assert_awaited_once()
    bound = _bind(
        lambda queue_name, routing_key, handler: None, fake_broker.consume.await_args
    )
    assert bound["queue_name"] == "waitlist_poller.appointment_cancelled"
    assert bound["routing_key"] == "appointment.cancelled"
    assert bound["handler"] is waitlist_poller.handle_cancellation


async def test_run_keeps_polling_and_sweep_loops_alive_rather_than_returning(monkeypatch):
    """FR-E5.1/FR-E5.5: run() supervises three concurrent loops (scheduled
    poll, event-driven cancellation consumer, offer-timeout sweep) -- it
    must not complete on its own."""
    monkeypatch.setattr(waitlist_poller, "IdleChairPollingService", _FakeAnyService, raising=False)
    monkeypatch.setattr(waitlist_poller, "WaitlistFillService", _FakeAnyService, raising=False)
    fake_broker = _FakeBroker()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            waitlist_poller.run(fake_broker, _fake_session_factory), timeout=0.3
        )
