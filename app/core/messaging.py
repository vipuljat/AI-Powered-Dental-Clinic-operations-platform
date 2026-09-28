"""The single message-broker abstraction every service that publishes a
domain event, and every worker that consumes one, is built against —
``publish``/``consume`` with an identical signature whether the real
RabbitMQ connection (`RabbitMQBroker`) or the in-process test fake
(`InMemoryBroker`) is wired in underneath.

Owns correlation-id propagation into/out of message headers
(project_rules.logging). Does not know about any specific queue/event name
(`booking.confirmed` etc.) — those are defined and used by the
services/workers that call this file, per project_rules.other's event
contract; this file enforces none of that shape itself.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, runtime_checkable

from app.common.constants import AUDIT_WRITE_MAX_RETRIES
from app.core.logging import get_correlation_id, get_logger, set_correlation_id

if TYPE_CHECKING:
    from app.core.config import Settings

logger = get_logger(__name__)

# The single topic exchange every publish/consume call binds against when a
# real RabbitMQ connection is used (RabbitMQBroker).
_EXCHANGE_NAME = "dental_platform_events"

# project_rules.errors: "after N retries a failure routes to a dead-letter
# queue rather than crashing the consumer loop" — no separate BRD-specified
# value exists for worker-message retries, so AUDIT_WRITE_MAX_RETRIES's
# sibling default of 3 is reused here (see spec's open question).
_MAX_HANDLER_RETRIES = AUDIT_WRITE_MAX_RETRIES

MessageHandler = Callable[[dict, dict], Awaitable[None]]


@runtime_checkable
class MessageBroker(Protocol):
    """The transport-agnostic publish/consume contract every caller depends on."""

    async def publish(
        self, routing_key: str, payload: dict, headers: dict | None = None
    ) -> None:
        """Publish `payload` under `routing_key`, propagating a correlation id."""
        ...

    async def consume(
        self, queue_name: str, routing_key: str, handler: MessageHandler
    ) -> None:
        """Bind `queue_name` to `routing_key` and run forever, awaiting
        `handler(payload, headers)` per message; on handler exception, retry
        up to a bounded count then route the message to a dead-letter queue
        named ``f"{queue_name}.dlq"`` rather than crashing the consume loop.
        """
        ...


def _with_correlation_header(headers: dict | None) -> dict:
    """Return a headers dict guaranteed to carry a `correlation_id`.

    project_rules.logging: `publish` always sets
    `headers["correlation_id"] = get_correlation_id()` when the caller does
    not explicitly pass one.
    """

    merged = dict(headers) if headers else {}
    if "correlation_id" not in merged:
        merged["correlation_id"] = get_correlation_id()
    return merged


class RabbitMQBroker:
    """Real aio-pika-backed implementation of `MessageBroker`.

    A topic exchange named "dental_platform_events" is declared once; every
    publish/consume binds against it by routing_key.
    """

    def __init__(self, amqp_url: str) -> None:
        self._amqp_url = amqp_url
        self._connection = None
        self._channel = None
        self._exchange = None

    async def connect(self) -> None:
        """Open the AMQP connection/channel and declare the shared exchange.

        Constructed lazily — never at import time — so this module can be
        imported without a live broker; the actual TCP connection is only
        opened when a composition root calls `connect()` during startup.
        """

        import aio_pika

        self._connection = await aio_pika.connect_robust(self._amqp_url)
        self._channel = await self._connection.channel()
        self._exchange = await self._channel.declare_exchange(
            _EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )

    async def close(self) -> None:
        """Close the AMQP connection, if one is open."""

        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._channel = None
            self._exchange = None

    async def publish(
        self, routing_key: str, payload: dict, headers: dict | None = None
    ) -> None:
        import json

        import aio_pika

        if self._exchange is None:
            await self.connect()

        merged_headers = _with_correlation_header(headers)
        message = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            headers=merged_headers,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
        )
        await self._exchange.publish(message, routing_key=routing_key)

    async def consume(
        self, queue_name: str, routing_key: str, handler: MessageHandler
    ) -> None:
        import json

        import aio_pika

        if self._channel is None or self._exchange is None:
            await self.connect()

        dlq_name = f"{queue_name}.dlq"
        dead_letter_queue = await self._channel.declare_queue(dlq_name, durable=True)
        await dead_letter_queue.bind(self._exchange, routing_key=dlq_name)

        queue = await self._channel.declare_queue(queue_name, durable=True)
        await queue.bind(self._exchange, routing_key=routing_key)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process(requeue=False, ignore_processed=True):
                    payload = json.loads(message.body.decode("utf-8"))
                    headers = dict(message.headers or {})
                    set_correlation_id(headers.get("correlation_id"))

                    attempt = 0
                    while True:
                        try:
                            await handler(payload, headers)
                            break
                        except Exception:  # noqa: BLE001 - broad by design, retried then dead-lettered
                            attempt += 1
                            logger.error(
                                "message_handler_failed",
                                extra={
                                    "queue_name": queue_name,
                                    "routing_key": routing_key,
                                    "attempt": attempt,
                                },
                                exc_info=True,
                            )
                            if attempt >= _MAX_HANDLER_RETRIES:
                                await self._exchange.publish(
                                    aio_pika.Message(
                                        body=json.dumps(payload).encode("utf-8"),
                                        headers=headers,
                                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                                        content_type="application/json",
                                    ),
                                    routing_key=dlq_name,
                                )
                                logger.error(
                                    "message_dead_lettered",
                                    extra={
                                        "queue_name": queue_name,
                                        "dlq_name": dlq_name,
                                        "routing_key": routing_key,
                                    },
                                )
                                break


class InMemoryBroker:
    """`asyncio.Queue`-per-routing_key fake used only when
    settings.environment == "test" (project_rules.testing).

    `publish()` appends to every queue whose bound routing_key matches
    (exact string match — no wildcard support needed since every event name
    in this tree is a literal string). `consume()` loops `await queue.get()`
    and calls `handler`; no real network, no retries needed for tests since
    handler exceptions simply propagate to the caller (raised, not
    swallowed).
    """

    def __init__(self) -> None:
        # routing_key -> list of (queue_name, asyncio.Queue) bound to it.
        self._bindings: dict[str, list[tuple[str, "asyncio.Queue[tuple[dict, dict]]"]]] = {}

    def _queues_for(self, routing_key: str) -> "list[tuple[str, asyncio.Queue]]":
        return self._bindings.setdefault(routing_key, [])

    async def publish(
        self, routing_key: str, payload: dict, headers: dict | None = None
    ) -> None:
        merged_headers = _with_correlation_header(headers)
        for _queue_name, queue in self._queues_for(routing_key):
            await queue.put((dict(payload), dict(merged_headers)))

    async def consume(
        self, queue_name: str, routing_key: str, handler: MessageHandler
    ) -> None:
        queue: "asyncio.Queue[tuple[dict, dict]]" = asyncio.Queue()
        self._queues_for(routing_key).append((queue_name, queue))

        while True:
            payload, headers = await queue.get()
            set_correlation_id(headers.get("correlation_id"))
            await handler(payload, headers)


def get_broker(settings: "Settings") -> MessageBroker:
    """Factory: `InMemoryBroker()` if `settings.environment == "test"`, else
    `RabbitMQBroker(settings.rabbitmq_url)` (project_rules.testing).
    """

    if settings.environment == "test":
        return InMemoryBroker()
    return RabbitMQBroker(settings.rabbitmq_url)
