"""Structured JSON logging configuration and the correlation-id contextvar
every request/worker-message log line carries.

Owns log-level conventions (DEBUG/INFO/WARN/ERROR) project-wide. Does not
itself generate the correlation id on inbound HTTP requests (that is
``app/common/middleware.py``) — it only stores/exposes the current value and
formats log records.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.core.config import Settings

# project_rules.logging: a request_id/correlation_id is generated in
# app.common.middleware for every inbound request and propagated into
# RabbitMQ message headers by app.core.messaging so a worker processing that
# message logs the same correlation id. This module only stores/reads the
# current value via this contextvar (module-private, per the Public surface).
_correlation_id_var: "ContextVar[str | None]" = ContextVar("correlation_id", default=None)

# Standard `logging.LogRecord` attributes we must not re-serialize verbatim
# as "extra" fields (they are already surfaced explicitly below).
_RESERVED_LOG_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)


class _JSONFormatter(logging.Formatter):
    """Renders every log record as a single JSON line.

    Injects `correlation_id` (from `get_correlation_id()`), `level`,
    `logger`, `message`, and any extra fields (`error_code`, `entity_id`,
    etc.) passed via `logger.error(..., extra={...})` — never a raw stack
    trace to the client (this formats for server-side log sinks only; the
    HTTP response body is built separately by
    app.common.exceptions.handlers).
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            if key == "message":
                continue
            payload[key] = value

        if record.exc_info:
            # Server-side sink only: the formatted traceback text is included
            # here for operators, never returned to an HTTP client (that
            # mapping is owned by app.common.exceptions.handlers, which never
            # forwards a raw stack trace in the response envelope).
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(settings: "Settings") -> None:
    """Install the JSON log formatter on the root logger.

    Called once at each composition root startup (`app/main.py`,
    `app/worker.py`). DEBUG/INFO/WARN/ERROR are used consistently by callers
    across the tree; this function only wires the formatter/handler and the
    root level (DEBUG in "test"/"development", INFO otherwise).
    """

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root_logger.addHandler(handler)

    root_logger.setLevel(
        logging.DEBUG if settings.environment in ("test", "development") else logging.INFO
    )


def get_logger(name: str) -> logging.Logger:
    """Return a standard-library logger; formatting is applied by the root handler."""

    return logging.getLogger(name)


def set_correlation_id(value: str) -> None:
    """Set the current correlation id for this request/message-processing context."""

    _correlation_id_var.set(value)


def get_correlation_id() -> str | None:
    """Return the current correlation id, or `None` if none has been set."""

    return _correlation_id_var.get()
