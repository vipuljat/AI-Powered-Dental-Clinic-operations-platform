"""Unit tests for app.common.responses — error envelope and pagination helpers."""

from app.common import constants
from app.common.responses import PageParams, error_envelope, paginated_response


# ---- error_envelope ---------------------------------------------------------


def test_error_envelope_basic_shape():
    result = error_envelope("NOT_FOUND", "Patient not found")
    assert result == {
        "error": {
            "code": "NOT_FOUND",
            "message": "Patient not found",
            "details": [],
        }
    }


def test_error_envelope_with_details():
    details = [{"field": "email", "issue": "invalid"}]
    result = error_envelope("VALIDATION_FAILED", "Invalid input", details=details)
    assert result["error"]["details"] == details


def test_error_envelope_defaults_details_to_empty_list_when_none():
    result = error_envelope("CONFLICT", "conflict", details=None)
    assert result["error"]["details"] == []


def test_error_envelope_preserves_code_and_message_verbatim():
    result = error_envelope("RATE_LIMITED", "Too many requests")
    assert result["error"]["code"] == "RATE_LIMITED"
    assert result["error"]["message"] == "Too many requests"


# ---- PageParams --------------------------------------------------------------


def test_page_params_defaults():
    params = PageParams()
    assert params.page == 1
    assert params.page_size == constants.DEFAULT_PAGE_SIZE
    assert params.page_size == 50


def test_page_params_accepts_explicit_values():
    params = PageParams(page=3, page_size=20)
    assert params.page == 3
    assert params.page_size == 20


# ---- paginated_response -------------------------------------------------------


def test_paginated_response_shape():
    items = [{"id": 1}, {"id": 2}]
    result = paginated_response(items, page=1, page_size=50, total=214)
    assert result == {
        "items": items,
        "page": 1,
        "page_size": 50,
        "total": 214,
    }


def test_paginated_response_echoes_requested_page_and_page_size():
    result = paginated_response([], page=4, page_size=10, total=35)
    assert result["page"] == 4
    assert result["page_size"] == 10


def test_paginated_response_total_is_full_match_count_not_page_count():
    items = [{"id": 1}]  # only one item on this page
    result = paginated_response(items, page=1, page_size=50, total=214)
    assert result["total"] == 214
    assert len(result["items"]) == 1


def test_paginated_response_with_empty_items():
    result = paginated_response([], page=1, page_size=50, total=0)
    assert result["items"] == []
    assert result["total"] == 0
