"""Unit tests for app/core/logging.py.

Covers `configure_logging` installing a JSON formatter that injects
`correlation_id`/`level`/`logger`/`message` plus any `extra=` fields into
every emitted log line, and the `set_correlation_id`/`get_correlation_id`
contextvar pair `app.core.messaging` and workers rely on to keep a single
request's log lines traceable across process boundaries.

`configure_logging`'s own signature takes a `Settings` instance, but this
file's spec does not document which settings fields (if any) it reads, so a
`unittest.mock.MagicMock` stands in: any attribute access it doesn't care
about simply resolves to another Mock rather than raising `AttributeError`,
which keeps these tests decoupled from an undocumented internal detail.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from app.core.logging import (
    _correlation_id_var,
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def _reset_correlation_id():
    token = _correlation_id_var.set(None)
    yield
    _correlation_id_var.reset(token)


@pytest.fixture(autouse=True)
def _restore_root_logging():
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.setLevel(original_level)


def _json_lines(text_blob: str) -> list[dict]:
    parsed = []
    for line in text_blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return parsed


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

def test_get_logger_returns_a_logger_with_the_given_name():
    logger = get_logger("app.core.test_logging_probe")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "app.core.test_logging_probe"


# ---------------------------------------------------------------------------
# correlation id contextvar
# ---------------------------------------------------------------------------

def test_get_correlation_id_defaults_to_none():
    assert get_correlation_id() is None


def test_set_and_get_correlation_id_round_trips():
    set_correlation_id("corr-abc-123")
    assert get_correlation_id() == "corr-abc-123"


def test_set_correlation_id_overwrites_previous_value():
    set_correlation_id("first-id")
    set_correlation_id("second-id")
    assert get_correlation_id() == "second-id"


# ---------------------------------------------------------------------------
# configure_logging: JSON formatting + correlation id + extra fields
# ---------------------------------------------------------------------------

def test_configure_logging_emits_valid_json_with_correlation_id_for_info(capsys):
    configure_logging(MagicMock())
    set_correlation_id("req-info-1")
    logger = get_logger("app.core.test_logging_probe.info")

    logger.info("hello world")

    captured = capsys.readouterr()
    records = _json_lines(captured.out + captured.err)
    match = next(r for r in records if r.get("message") == "hello world")
    assert match["level"] == "INFO"
    assert match["logger"] == "app.core.test_logging_probe.info"
    assert match["correlation_id"] == "req-info-1"


def test_configure_logging_emits_error_code_and_entity_id_from_extra(capsys):
    configure_logging(MagicMock())
    set_correlation_id("req-err-2")
    logger = get_logger("app.core.test_logging_probe.error")

    logger.error(
        "something failed",
        extra={"error_code": "SOME_ERROR_CODE", "entity_id": "entity-42"},
    )

    captured = capsys.readouterr()
    records = _json_lines(captured.out + captured.err)
    match = next(r for r in records if r.get("message") == "something failed")
    assert match["level"] == "ERROR"
    assert match["correlation_id"] == "req-err-2"
    assert match["error_code"] == "SOME_ERROR_CODE"
    assert match["entity_id"] == "entity-42"


def test_configure_logging_json_line_has_no_correlation_id_when_unset(capsys):
    configure_logging(MagicMock())
    logger = get_logger("app.core.test_logging_probe.no_corr")

    logger.info("no correlation id set for this line")

    captured = capsys.readouterr()
    records = _json_lines(captured.out + captured.err)
    match = next(r for r in records if r.get("message") == "no correlation id set for this line")
    assert match.get("correlation_id") is None
