"""Shared response-shaping helpers: the error envelope builder used by
app/common/exceptions/handlers.py, and the pagination envelope used by every
list endpoint (architecture.md §8: ``?page=&page_size=`` offset pagination
with ``total`` in the response).

Contains no route or business logic.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.common.constants import DEFAULT_PAGE_SIZE


def error_envelope(code: str, message: str, details: list[dict] | None = None) -> dict:
    """Build the single global error envelope, verbatim architecture.md §8:

    ``{ "error": { "code": "...", "message": "...", "details": [...] } }``
    """
    return {"error": {"code": code, "message": message, "details": details or []}}


class PageParams(BaseModel):
    """Offset pagination query parameters.

    Defaults ``page=1``, ``page_size=DEFAULT_PAGE_SIZE`` (50) when the caller
    omits either query parameter (architecture.md §8).
    """

    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, ge=1)


def paginated_response(items: list, page: int, page_size: int, total: int) -> dict:
    """Build the pagination envelope, verbatim architecture.md §8 and the
    ``GET /console/audit`` example response:

    ``{ "items": [...], "page": 1, "page_size": 50, "total": 214 }``

    ``page``/``page_size`` echo the caller's requested (or defaulted) values;
    ``total`` is the full unpaginated match count, not the count of the
    current page.
    """
    return {"items": items, "page": page, "page_size": page_size, "total": total}
