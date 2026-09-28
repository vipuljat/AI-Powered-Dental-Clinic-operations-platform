"""Unit tests for app/workers/recommendation_engine.py, derived from its
spec only.

Public surface: ``run(broker, session_factory)`` and
``handle_booking_confirmed(payload, headers)``. ``handle_booking_confirmed``
does not receive a session_factory/get_db argument directly, so -- per the
verified real signature of app.core.database.get_db given for this task --
these tests patch ``get_db`` inside the worker module to hand back a fake
AsyncSession, consistent with app.core.database.configure_session_factory
existing precisely so ad-hoc worker sessions can be sourced from the
run()-provided session_factory via get_db().
"""
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import app.workers.recommendation_engine as recommendation_engine

pytestmark = pytest.mark.asyncio


def _bind(sig_func, call):
    sig = inspect.signature(sig_func)
    bound = sig.bind(*call.args, **call.kwargs)
    bound.apply_defaults()
    return bound.arguments


def _patch_get_db(monkeypatch, fake_db):
    async def _fake_get_db():
        yield fake_db

    monkeypatch.setattr(recommendation_engine, "get_db", _fake_get_db, raising=False)


def _patch_risk_scoring(monkeypatch, score_rules_result=None, score_ml_result=None):
    fake_service = MagicMock()
    fake_service.score_rules_based = AsyncMock(return_value=score_rules_result or MagicMock())
    fake_service.score_ml = AsyncMock(return_value=score_ml_result)
    monkeypatch.setattr(recommendation_engine, "RiskScoringService", lambda *a, **kw: fake_service)
    return fake_service


# ---------------------------------------------------------------------------
# handle_booking_confirmed
# ---------------------------------------------------------------------------


async def test_handle_booking_confirmed_calls_score_rules_based_with_appointment_id_from_payload(monkeypatch):
    fake_db = MagicMock()
    _patch_get_db(monkeypatch, fake_db)
    fake_service = _patch_risk_scoring(monkeypatch)

    appointment_id = uuid4()
    patient_id = uuid4()
    await recommendation_engine.handle_booking_confirmed(
        {"appointment_id": appointment_id, "patient_id": patient_id}, {}
    )

    fake_service.score_rules_based.assert_awaited_once()
    bound = _bind(lambda db, appointment_id: None, fake_service.score_rules_based.await_args)
    assert bound["appointment_id"] == appointment_id


async def test_handle_booking_confirmed_also_calls_score_ml_immediately_after(monkeypatch):
    """FR-E8.1/US40 + FR-E8.2: a single consumer covers both rules-based and
    ML scoring without needing two separate event handlers."""
    fake_db = MagicMock()
    _patch_get_db(monkeypatch, fake_db)
    fake_service = _patch_risk_scoring(monkeypatch)

    appointment_id = uuid4()
    await recommendation_engine.handle_booking_confirmed(
        {"appointment_id": appointment_id, "patient_id": uuid4()}, {}
    )

    fake_service.score_ml.assert_awaited_once()
    bound = _bind(lambda db, appointment_id: None, fake_service.score_ml.await_args)
    assert bound["appointment_id"] == appointment_id


async def test_handle_booking_confirmed_calls_score_ml_after_score_rules_based(monkeypatch):
    order = []
    fake_db = MagicMock()
    _patch_get_db(monkeypatch, fake_db)

    fake_service = MagicMock()

    async def _score_rules_based(*a, **kw):
        order.append("score_rules_based")
        return MagicMock()

    async def _score_ml(*a, **kw):
        order.append("score_ml")
        return None

    fake_service.score_rules_based = AsyncMock(side_effect=_score_rules_based)
    fake_service.score_ml = AsyncMock(side_effect=_score_ml)
    monkeypatch.setattr(recommendation_engine, "RiskScoringService", lambda *a, **kw: fake_service)

    await recommendation_engine.handle_booking_confirmed(
        {"appointment_id": uuid4(), "patient_id": uuid4()}, {}
    )

    assert order == ["score_rules_based", "score_ml"]


async def test_handle_booking_confirmed_does_not_raise_when_score_ml_is_a_no_op(monkeypatch):
    """score_ml is documented as a no-op unless the ML data-sufficiency gate
    has already passed -- i.e. it may legitimately return None."""
    fake_db = MagicMock()
    _patch_get_db(monkeypatch, fake_db)
    _patch_risk_scoring(monkeypatch, score_ml_result=None)

    # Should complete without raising.
    result = await recommendation_engine.handle_booking_confirmed(
        {"appointment_id": uuid4(), "patient_id": uuid4()}, {}
    )
    assert result is None


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


class _FakeUtilisationEngine:
    def __init__(self, *a, **kw):
        pass

    async def generate(self, *a, **kw):
        return []


async def test_run_registers_booking_confirmed_consumer_with_documented_queue_routing_key_and_handler(monkeypatch):
    monkeypatch.setattr(recommendation_engine, "UtilisationEngine", _FakeUtilisationEngine, raising=False)
    fake_broker = _FakeBroker()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            recommendation_engine.run(fake_broker, _fake_session_factory), timeout=0.3
        )

    fake_broker.consume.assert_awaited_once()
    bound = _bind(
        lambda queue_name, routing_key, handler: None, fake_broker.consume.await_args
    )
    assert bound["queue_name"] == "recommendation_engine.booking_confirmed"
    assert bound["routing_key"] == "booking.confirmed"
    assert bound["handler"] is recommendation_engine.handle_booking_confirmed


async def test_run_does_not_return_early(monkeypatch):
    """FR-E8.6/US45: utilisation_loop's periodic UtilisationEngine.generate
    is a scheduled job -- run() must keep supervising it, not return."""
    monkeypatch.setattr(recommendation_engine, "UtilisationEngine", _FakeUtilisationEngine, raising=False)
    fake_broker = _FakeBroker()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            recommendation_engine.run(fake_broker, _fake_session_factory), timeout=0.3
        )
