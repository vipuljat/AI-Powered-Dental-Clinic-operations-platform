"""The single global FastAPI exception-handling registration point.

`register_exception_handlers(app)` is called once, by every composition root
that builds a `FastAPI()` instance -- `app/main.py` and, for verification
purposes, any throwaway app a route file's own `verify` command constructs --
so a domain exception raised anywhere below the route layer becomes the
documented status + envelope (architecture.md §8) without every route
re-implementing a `try/except`.

This file performs no independent status-code decision-making: each handler
only reads `exc.status_code` / `exc.code` / `exc.message` / `exc.details` off
the exception instance (for `AppError` subclasses), or produces the two other
documented, fixed envelopes (422 validation failure, 500 catch-all).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.common.exceptions.errors import AppError
from app.common.responses import error_envelope
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI, Request

logger = get_logger(__name__)


def register_exception_handlers(app: "FastAPI") -> None:
    """Register the three global handlers on ``app``.

    errors.handler_setup (project rules `Interaction contract`): any file
    that builds a throwaway ``FastAPI()`` app in its own verify command MUST
    call this before asserting any status code produced by a raised
    ``AppError`` -- otherwise the exception has nothing to convert it to a
    response.
    """

    @app.exception_handler(AppError)
    async def _handle_app_error(request: "Request", exc: AppError) -> JSONResponse:
        # Every AppError subclass fixes its own status_code/code/default_message
        # (see app/common/exceptions/errors.py) -- this handler only reads them.
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: "Request", exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic body/query validation failures -> 422, same envelope shape,
        # code = "VALIDATION_FAILED", details = Pydantic's own error list.
        return JSONResponse(
            status_code=422,
            content=error_envelope(
                "VALIDATION_FAILED",
                "Validation failed.",
                exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: "Request", exc: Exception) -> JSONResponse:
        # FR-E1 (cross-cutting): ERROR log entries carry error_code + context;
        # never a raw stack trace (or the exception message) is returned to
        # the client -- log the full exception server-side only, and return a
        # generic envelope.
        logger.error(
            "Unhandled exception while processing request",
            exc_info=exc,
            extra={"error_code": "INTERNAL_ERROR", "path": str(request.url.path)},
        )
        return JSONResponse(
            status_code=500,
            content=error_envelope(
                "INTERNAL_ERROR",
                "An unexpected error occurred.",
                None,
            ),
        )
