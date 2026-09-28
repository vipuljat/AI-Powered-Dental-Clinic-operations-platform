"""Unit tests for app/core/messaging.py.

Covers `InMemoryBroker` (the `project_rules.testing` in-process fake every
composition root wires in when `settings.environment == "test"`),
`get_broker`'s environment-driven dispatch between it and `RabbitMQBroker`,
and the correlation-id propagation contract `app.core.logging` and every
worker rely on to keep a single request/message chain traceable across
process boundaries.

`RabbitMQBroker` is the real `aio-pika`-backed implementation; there is no
running RabbitMQ broker available to a unit test (per `project_rules.testing`
that substitution is exactly what `InMemoryBroker` exists to avoid), so this
file only exercises the parts of `RabbitMQBroker` observable without a live
connection: that `get_broker` selects it for any non-"test" environment, and
that constructing it does not itself attempt any network I/O (only
`connect()` would). Its `connect`/`close`/`publish`/`consume` behaviour
against a real broker, and the retry-then-dead-letter-queue behaviour the
spec describes for `MessageBroker.consume` in general, are not observable
through this file's own public surface without a live `aio-pika` connection,
so they are not asserted here.
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from app.core.logging import _correlation_id_var, get_correlation_id, set_correlation_id
from app.core.messaging import InMemoryBroker, RabbitMQBroker, get_broker


@pytest.fixture(autouse=True)
def _reset_correlation_id():
    token = _correlation_id_var.set(None)
    yield
    _correlation_id_var.reset(token)


async def _start_consumer(broker, queue_name, routing_key):
    """Bind `queue_name` to `routing_key` on `broker` and hand back the
    running (otherwise-infinite) `consume()` task plus a list that
    accumulates every `(payload, headers)` pair the handler receives, and an
    `asyncio.Event` set each time a message arrives."""
    calls = []
    event = asyncio.Event()

    async def handler(payload, headers):
        calls.append((payload, headers))
        event.set()

    task = asyncio.create_task(broker.consume(queue_name, routing_key, handler))
    # Yield control long enough for consume() to run its (synchronous, no
    # real network) binding step and reach its first suspension point before
    # the test publishes anything.
    await asyncio.sleep(0.01)
    return task, calls, event


async def _stop_consumer(task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# get_broker: environment-driven dispatch (project_rules.testing)
# ---------------------------------------------------------------------------

def test_get_broker_returns_in_memory_broker_when_environment_is_test():
    settings = SimpleNamespace(environment="test", rabbitmq_url="amqp://guest:guest@localhost:5672/")

    broker = get_broker(settings)

    assert isinstance(broker, InMemoryBroker)


def test_get_broker_returns_rabbitmq_broker_when_environment_is_development():
    settings = SimpleNamespace(environment="development", rabbitmq_url="amqp://guest:guest@localhost:5672/")

    broker = get_broker(settings)

    assert isinstance(broker, RabbitMQBroker)


def test_get_broker_returns_rabbitmq_broker_when_environment_is_production():
    settings = SimpleNamespace(environment="production", rabbitmq_url="amqp://guest:guest@prod-broker:5672/")

    broker = get_broker(settings)

    assert isinstance(broker, RabbitMQBroker)


def test_rabbitmq_broker_construction_does_not_touch_the_network():
    # Constructing a RabbitMQBroker must not itself dial out -- only an
    # explicit connect() call would. If __init__ tried to connect, this
    # would hang/raise against the amqp url below (nothing is listening).
    broker = RabbitMQBroker("amqp://guest:guest@127.0.0.1:1/")

    assert isinstance(broker, RabbitMQBroker)
    assert hasattr(broker, "connect")
    assert hasattr(broker, "close")


# ---------------------------------------------------------------------------
# InMemoryBroker: publish() / consume() round trip, exact routing-key match
# ---------------------------------------------------------------------------

async def test_consume_receives_a_published_message_matching_its_routing_key():
    broker = InMemoryBroker()
    task, calls, event = await _start_consumer(broker, "outreach_retry", "booking.confirmed")

    await broker.publish("booking.confirmed", {"appointment_id": "a1", "patient_id": "p1"})
    await asyncio.wait_for(event.wait(), timeout=1)
    await _stop_consumer(task)

    assert len(calls) == 1
    payload, headers = calls[0]
    assert payload == {"appointment_id": "a1", "patient_id": "p1"}
    assert "correlation_id" in headers


async def test_publish_fans_out_to_every_queue_bound_to_the_same_routing_key():
    broker = InMemoryBroker()
    task1, calls1, event1 = await _start_consumer(broker, "outreach_retry", "waitlist_offer.created")
    task2, calls2, event2 = await _start_consumer(broker, "audit_listener", "waitlist_offer.created")

    await broker.publish(
        "waitlist_offer.created",
        {"waitlist_entry_id": "w1", "idle_chair_alert_id": "i1"},
    )
    await asyncio.wait_for(event1.wait(), timeout=1)
    await asyncio.wait_for(event2.wait(), timeout=1)
    await _stop_consumer(task1)
    await _stop_consumer(task2)

    assert calls1[0][0] == {"waitlist_entry_id": "w1", "idle_chair_alert_id": "i1"}
    assert calls2[0][0] == {"waitlist_entry_id": "w1", "idle_chair_alert_id": "i1"}


async def test_publish_does_not_deliver_to_a_differently_named_routing_key():
    # Exact string match only -- no wildcard support -- per this file's spec.
    broker = InMemoryBroker()
    task_a, calls_a, event_a = await _start_consumer(broker, "qa", "escalation.triggered")
    task_b, calls_b, event_b = await _start_consumer(broker, "qb", "escalation.triggered.extra")

    await broker.publish(
        "escalation.triggered",
        {"triage_session_id_or_call_interaction_id": "t1", "trigger_reason": "red_flag"},
    )
    await asyncio.wait_for(event_a.wait(), timeout=1)
    # Give the differently-bound consumer a chance to have (wrongly) matched.
    await asyncio.sleep(0.05)
    await _stop_consumer(task_a)
    await _stop_consumer(task_b)

    assert len(calls_a) == 1
    assert calls_b == []


async def test_publish_with_no_bound_consumer_does_not_raise():
    broker = InMemoryBroker()

    await broker.publish("nobody.listening", {"x": 1})


async def test_consume_handler_exception_propagates_to_the_caller_for_in_memory_broker():
    # InMemoryBroker's consume() needs no retry/dead-letter path in tests --
    # handler exceptions simply propagate (raised, not swallowed).
    broker = InMemoryBroker()

    async def failing_handler(payload, headers):
        raise ValueError("handler exploded")

    task = asyncio.create_task(broker.consume("q1", "booking.confirmed", failing_handler))
    await asyncio.sleep(0.01)

    await broker.publish("booking.confirmed", {"appointment_id": "a1"})

    with pytest.raises(ValueError, match="handler exploded"):
        await asyncio.wait_for(task, timeout=1)


# ---------------------------------------------------------------------------
# Correlation-id propagation (project_rules.logging / app.core.logging)
# ---------------------------------------------------------------------------

async def test_publish_sets_correlation_id_header_from_ambient_context_when_not_provided():
    broker = InMemoryBroker()
    set_correlation_id("corr-ambient-1")
    task, calls, event = await _start_consumer(broker, "q1", "booking.confirmed")

    await broker.publish("booking.confirmed", {"appointment_id": "a1"})
    await asyncio.wait_for(event.wait(), timeout=1)
    await _stop_consumer(task)

    _, headers = calls[0]
    assert headers["correlation_id"] == "corr-ambient-1"


async def test_publish_with_no_ambient_correlation_id_sets_header_to_none():
    broker = InMemoryBroker()
    assert get_correlation_id() is None
    task, calls, event = await _start_consumer(broker, "q1", "booking.confirmed")

    await broker.publish("booking.confirmed", {"appointment_id": "a1"})
    await asyncio.wait_for(event.wait(), timeout=1)
    await _stop_consumer(task)

    _, headers = calls[0]
    assert headers["correlation_id"] is None


async def test_publish_preserves_an_explicitly_provided_correlation_id_header():
    broker = InMemoryBroker()
    set_correlation_id("corr-ambient-should-not-be-used")
    task, calls, event = await _start_consumer(broker, "q1", "appointment.cancelled")

    await broker.publish(
        "appointment.cancelled",
        {"appointment_id": "a1"},
        headers={"correlation_id": "corr-explicit-2"},
    )
    await asyncio.wait_for(event.wait(), timeout=1)
    await _stop_consumer(task)

    _, headers = calls[0]
    assert headers["correlation_id"] == "corr-explicit-2"


async def test_publish_fills_in_correlation_id_when_headers_given_without_one():
    broker = InMemoryBroker()
    set_correlation_id("corr-ambient-3")
    task, calls, event = await _start_consumer(broker, "q1", "escalation.triggered")

    await broker.publish(
        "escalation.triggered",
        {"trigger_reason": "red_flag"},
        headers={"source": "triage_service"},
    )
    await asyncio.wait_for(event.wait(), timeout=1)
    await _stop_consumer(task)

    _, headers = calls[0]
    assert headers["correlation_id"] == "corr-ambient-3"
    assert headers["source"] == "triage_service"


async def test_consume_sets_correlation_id_before_invoking_handler():
    broker = InMemoryBroker()
    observed = {}
    event = asyncio.Event()

    async def handler(payload, headers):
        observed["seen"] = get_correlation_id()
        event.set()

    task = asyncio.create_task(broker.consume("q1", "booking.confirmed", handler))
    await asyncio.sleep(0.01)

    await broker.publish(
        "booking.confirmed",
        {"appointment_id": "a1"},
        headers={"correlation_id": "corr-from-publisher"},
    )
    await asyncio.wait_for(event.wait(), timeout=1)
    await _stop_consumer(task)

    assert observed["seen"] == "corr-from-publisher"
