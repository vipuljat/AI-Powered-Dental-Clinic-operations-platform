"""Unit tests for app/common/exceptions/handlers.py

register_exception_handlers(app) is the single global FastAPI exception
registration point (architecture.md §8). These tests build a throwaway
FastAPI() app, register the handlers under test, mount routes that raise the
documented exception types, and assert on the resulting HTTP response shape
-- never on any implementation-internal helper.
"""
import logging
import os

os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.common.exceptions.errors import AppError
from app.common.exceptions.handlers import register_exception_handlers


class _StubAppError(AppError):
    """A concrete AppError subclass that sets its public attributes directly,
    so the test controls status_code/code/message/details without depending
    on AppError's own constructor signature (not part of this file's spec)."""

    def __init__(self, status_code, code, message, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details if details is not None else []
        Exception.__init__(self, message)


class _NestedStubAppError(_StubAppError):
    """A second-generation subclass, to prove the handler catches
    AppError *and every subclass*, not just direct children."""


class _Payload(BaseModel):
    name: str


def _build_app():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/app-error")
    def raise_app_error():
        raise _StubAppError(409, "CONFLICT_TEST", "a conflict occurred", details=[{"field": "x"}])

    @app.get("/nested-app-error")
    def raise_nested_app_error():
        raise _NestedStubAppError(404, "NOT_FOUND_TEST", "not found", details=[])

    @app.get("/app-error-no-details")
    def raise_app_error_no_details():
        raise _StubAppError(400, "BAD_REQUEST_TEST", "bad request")

    @app.post("/validate")
    def validate(payload: _Payload):
        return {"name": payload.name}

    @app.get("/boom")
    def boom():
        raise ValueError("db password=hunter2 leaked in a stack frame")

    return app


@pytest.fixture
def app():
    return _build_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def test_register_exception_handlers_returns_none():
    app = FastAPI()
    assert register_exception_handlers(app) is None


def test_app_error_handler_returns_its_own_status_code_and_envelope(client):
    resp = client.get("/app-error")

    assert resp.status_code == 409
    body = resp.json()
    assert body == {
        "error": {
            "code": "CONFLICT_TEST",
            "message": "a conflict occurred",
            "details": [{"field": "x"}],
        }
    }


def test_app_error_handler_reads_details_default_when_absent(client):
    resp = client.get("/app-error-no-details")

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "BAD_REQUEST_TEST"
    assert body["error"]["message"] == "bad request"
    assert body["error"]["details"] == []


def test_app_error_handler_catches_subclasses_too(client):
    resp = client.get("/nested-app-error")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "NOT_FOUND_TEST"
    assert body["error"]["message"] == "not found"


def test_app_error_handler_performs_no_independent_status_decision(client):
    # Two different AppError subclasses with different fixed status codes
    # must come back exactly as attached to the exception instance -- the
    # handler is not allowed to normalise/override them.
    conflict = client.get("/app-error")
    not_found = client.get("/nested-app-error")

    assert conflict.status_code == 409
    assert not_found.status_code == 404
    assert conflict.status_code != not_found.status_code


def test_validation_error_handler_returns_422_with_validation_failed_code(client):
    resp = client.post("/validate", json={})

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert isinstance(body["error"]["details"], list)
    assert len(body["error"]["details"]) >= 1
    # Pydantic's own error list identifies the missing field.
    locs = [tuple(item.get("loc", [])) for item in body["error"]["details"]]
    assert any("name" in loc for loc in locs)


def test_validation_error_handler_uses_same_envelope_shape(client):
    resp = client.post("/validate", json={})

    body = resp.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "details"}


def test_catch_all_handler_returns_generic_500_without_leaking_exception_details(client):
    resp = client.get("/boom")

    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    # The raw exception message (which here simulates a leaked secret) must
    # never reach the client.
    raw_text = resp.text
    assert "hunter2" not in raw_text
    assert "db password=hunter2 leaked in a stack frame" not in raw_text
    assert "Traceback" not in raw_text
    assert body["error"]["message"] != "db password=hunter2 leaked in a stack frame"


def test_catch_all_handler_response_matches_global_envelope_shape(client):
    resp = client.get("/boom")

    body = resp.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "details"}
    assert isinstance(body["error"]["code"], str)
    assert isinstance(body["error"]["message"], str)


def test_catch_all_handler_logs_the_exception_as_error_with_error_code(client, caplog):
    with caplog.at_level(logging.ERROR):
        resp = client.get("/boom")

    assert resp.status_code == 500
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "the catch-all handler must log the exception at ERROR"

    def record_mentions_internal_error(record):
        if getattr(record, "error_code", None) == "INTERNAL_ERROR":
            return True
        return "INTERNAL_ERROR" in record.getMessage()

    assert any(record_mentions_internal_error(r) for r in error_records)
