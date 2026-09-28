"""The one request-scoped ASGI middleware registered on `app/main.py`'s
`FastAPI()` instance.

Generates/propagates a correlation id per inbound request and attaches it to
the response headers and to every log line emitted while handling that
request (project_rules.logging). Also owns a small in-process rate limiter
used by the one endpoint the BRD documents a `429` for
(`POST /auth/password-reset/request`) -- public-facing rate limiting at the
Ingress/ALB layer (architecture.md §10) is out of this codebase's scope, but
the documented `429` response itself must still be produced by the app for
the acceptance suite to observe it with no real Ingress present.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware

from app.common.exceptions.errors import RateLimitedError
from app.core.logging import set_correlation_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI
    from starlette.requests import Request
    from starlette.responses import Response

CORRELATION_ID_HEADER = "X-Request-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Reads/generates a correlation id and propagates it end to end.

    project_rules.logging: "a request_id/correlation_id is generated in
    app.common.middleware for every inbound request and propagated into
    RabbitMQ message headers by app.core.messaging so a worker processing
    that message logs the same correlation id." This class is that one place
    the id is generated/read.
    """

    async def dispatch(
        self,
        request: "Request",
        call_next: Callable[["Request"], Awaitable["Response"]],
    ) -> "Response":
        correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)

        response.headers[CORRELATION_ID_HEADER] = correlation_id
        return response


class InMemoryRateLimiter:
    """A single-process, in-memory sliding-window rate limiter.

    architecture.md §5.2, `POST /auth/password-reset/request`, Errors:
    `429 too many requests` -- this class (constructed once at
    composition-root startup with
    `max_calls=PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR, per_seconds=3600`)
    is what `AuthService.request_reset` calls with `key=email` to produce
    that `429`.

    Interaction contract (services/console/service.py `AuthService`):
    `AuthService` depends on an `InMemoryRateLimiter` instance passed into
    its constructor (built once by the composition root and shared across
    requests, never constructed per-request) -- `AuthService.request_reset
    (email)` calls `limiter.check(email)` before generating a reset token,
    and lets `RateLimitedError` propagate to the global handler for the
    documented `429`.

    In-process `dict[str, list[float]]` -- single-process assumption is
    acceptable at Phase-1's 50-concurrent-user, single-clinic scale
    (architecture.md §10).
    """

    def __init__(self, max_calls: int, per_seconds: int) -> None:
        self._max_calls = max_calls
        self._per_seconds = per_seconds
        self._calls: dict[str, list[float]] = {}

    def check(self, key: str) -> None:
        """Raise `RateLimitedError` if `key` exceeded its call budget.

        Otherwise records this call (for future window checks) and returns
        `None`.
        """

        now = time.monotonic()
        window_start = now - self._per_seconds
        history = [t for t in self._calls.get(key, []) if t > window_start]

        if len(history) >= self._max_calls:
            self._calls[key] = history
            raise RateLimitedError()

        history.append(now)
        self._calls[key] = history


def register_middleware(app: "FastAPI") -> None:
    """Attach `CorrelationIdMiddleware` to `app`."""

    app.add_middleware(CorrelationIdMiddleware)
