"""Unit tests for app/common/middleware.py

Covers CorrelationIdMiddleware (per-request correlation id generation /
propagation into the response headers and into app.core.logging), the
in-process InMemoryRateLimiter used for POST /auth/password-reset/request's
documented 429, and register_middleware(app)'s wiring of the former onto a
FastAPI app.
"""
import importlib
import os
import sys
import time
import uuid

os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.common.exceptions.errors import RateLimitedError
from app.common.middleware import (
    CorrelationIdMiddleware,
    InMemoryRateLimiter,
    register_middleware,
)


# ---------------------------------------------------------------------------
# CorrelationIdMiddleware
# ---------------------------------------------------------------------------


def _build_app_with_middleware():
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


@pytest.fixture
def client():
    return TestClient(_build_app_with_middleware())


def test_dispatch_echoes_inbound_x_request_id_header(client):
    resp = client.get("/ping", headers={"X-Request-ID": "existing-correlation-id"})

    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"] == "existing-correlation-id"


def test_dispatch_generates_a_uuid4_when_no_header_present(client):
    resp = client.get("/ping")

    assert resp.status_code == 200
    generated = resp.headers["X-Request-ID"]
    parsed = uuid.UUID(generated)
    assert parsed.version == 4


def test_dispatch_generates_a_fresh_id_per_request_without_header(client):
    first = client.get("/ping").headers["X-Request-ID"]
    second = client.get("/ping").headers["X-Request-ID"]

    assert first != second


def test_dispatch_calls_set_correlation_id_before_call_next(monkeypatch):
    # Force a fresh import of app.common.middleware after patching
    # app.core.logging.set_correlation_id, so this works regardless of
    # whether middleware.py did `import app.core.logging as x` (attribute
    # lookup at call time) or `from app.core.logging import
    # set_correlation_id` (name bound at import time) -- either way the
    # freshly (re)imported module observes the patched function.
    sys.modules.pop("app.common.middleware", None)

    calls = []

    def fake_set_correlation_id(value):
        calls.append(value)

    monkeypatch.setattr("app.core.logging.set_correlation_id", fake_set_correlation_id)

    mw = importlib.import_module("app.common.middleware")

    app = FastAPI()
    app.add_middleware(mw.CorrelationIdMiddleware)

    seen_during_request = []

    @app.get("/ping")
    def ping():
        # set_correlation_id must already have been called by the time the
        # route handler (i.e. call_next's downstream) runs.
        seen_during_request.append(list(calls))
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/ping", headers={"X-Request-ID": "corr-id-42"})

    assert resp.status_code == 200
    assert calls == ["corr-id-42"]
    assert seen_during_request == [["corr-id-42"]]

    # Leave module state pristine for any other test in this session.
    sys.modules.pop("app.common.middleware", None)
    importlib.import_module("app.common.middleware")


# ---------------------------------------------------------------------------
# InMemoryRateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_calls_up_to_max_within_window():
    limiter = InMemoryRateLimiter(max_calls=3, per_seconds=3600)

    assert limiter.check("alice@example.com") is None
    assert limiter.check("alice@example.com") is None
    assert limiter.check("alice@example.com") is None


def test_rate_limiter_raises_rate_limited_error_once_max_calls_exceeded():
    limiter = InMemoryRateLimiter(max_calls=2, per_seconds=3600)
    limiter.check("bob@example.com")
    limiter.check("bob@example.com")

    with pytest.raises(RateLimitedError):
        limiter.check("bob@example.com")


def test_rate_limiter_tracks_each_key_independently():
    limiter = InMemoryRateLimiter(max_calls=1, per_seconds=3600)
    limiter.check("carol@example.com")

    with pytest.raises(RateLimitedError):
        limiter.check("carol@example.com")

    # A different key must be unaffected by carol's exhausted quota.
    assert limiter.check("dave@example.com") is None


def test_rate_limiter_allows_again_once_the_trailing_window_elapses():
    limiter = InMemoryRateLimiter(max_calls=1, per_seconds=1)
    limiter.check("erin@example.com")

    with pytest.raises(RateLimitedError):
        limiter.check("erin@example.com")

    time.sleep(1.2)

    assert limiter.check("erin@example.com") is None


# ---------------------------------------------------------------------------
# register_middleware
# ---------------------------------------------------------------------------


def test_register_middleware_installs_correlation_id_middleware_on_the_app():
    app = FastAPI()
    register_middleware(app)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    client = TestClient(app)
    resp = client.get("/ping")

    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    # Must be a well-formed id, proving CorrelationIdMiddleware (not some
    # no-op) is what got installed.
    uuid.UUID(resp.headers["X-Request-ID"])
