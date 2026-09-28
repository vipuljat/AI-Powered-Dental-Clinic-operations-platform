"""Unit tests for app/workers/escalation_dispatcher.py.

Scope notes (per the unit-test-authoring role): this worker's dependencies
(app/core/messaging.py's MessageBroker, app/repositories/intelligence/repository.py's
escalation repository, app/services/console/service.py's AuditService) could not be
opened to confirm their exact method signatures/class names, so tests that would
require guessing those internals are kept loose (identity/kwargs-subset checks,
permissive fakes) rather than asserting an exact private interface. The one fact
confirmed directly by project rules text is that console's audit dependency class
is named ``AuditService``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.workers import escalation_dispatcher


class _FakeSession(AsyncMock):
    """A permissive stand-in for AsyncSession supporting `async with ... as session`."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


def _session_factory():
    """Mimics async_sessionmaker: calling it returns a fresh async-context-manager session."""
    return _FakeSession()


class _FakeBroker:
    def __init__(self):
        self.consume = AsyncMock()


@pytest.fixture
def fake_broker():
    return _FakeBroker()


@pytest.fixture
def mocked_dependencies(monkeypatch):
    """Best-effort isolation of handle_escalation_triggered's collaborators.

    raising=False is used throughout because the exact attribute names inside
    app.workers.escalation_dispatcher's own namespace were not confirmed (the source
    files that would confirm them were off-limits to read); if a guessed name is
    wrong this simply becomes a no-op patch rather than a hard failure.
    """
    audit_instance = AsyncMock()
    audit_cls = MagicMock(return_value=audit_instance)
    monkeypatch.setattr(escalation_dispatcher, "AuditService", audit_cls, raising=False)

    repo_instance = AsyncMock()
    repo_cls = MagicMock(return_value=repo_instance)
    monkeypatch.setattr(escalation_dispatcher, "EscalationRepository", repo_cls, raising=False)

    fake_logger = MagicMock()
    monkeypatch.setattr(escalation_dispatcher, "logger", fake_logger, raising=False)

    return {
        "audit_cls": audit_cls,
        "audit_instance": audit_instance,
        "repo_cls": repo_cls,
        "repo_instance": repo_instance,
        "logger": fake_logger,
    }


# ---------------------------------------------------------------------------
# run() -- consumer registration contract (given verbatim in the spec's public
# surface pseudocode, so these are the highest-confidence tests in this file).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_registers_consumer_with_documented_queue_and_routing_key(fake_broker):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    assert fake_broker.consume.call_count == 1
    args, kwargs = fake_broker.consume.call_args
    all_args = list(args) + list(kwargs.values())
    assert "escalation_dispatcher.escalation_triggered" in all_args
    assert "escalation.triggered" in all_args


@pytest.mark.asyncio
async def test_run_registers_the_module_level_handler_function(fake_broker):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    args, kwargs = fake_broker.consume.call_args
    all_args = list(args) + list(kwargs.values())
    # the handler passed to consume() must be handle_escalation_triggered itself,
    # not a wrapper/closure with a different identity, per the documented pseudocode.
    assert escalation_dispatcher.handle_escalation_triggered in all_args


@pytest.mark.asyncio
async def test_run_calls_consume_exactly_once(fake_broker):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    assert fake_broker.consume.call_count == 1


@pytest.mark.asyncio
async def test_run_returns_none(fake_broker):
    result = await escalation_dispatcher.run(fake_broker, _session_factory)

    assert result is None


# ---------------------------------------------------------------------------
# handle_escalation_triggered -- notification/audit side-effect contract.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_escalation_triggered_records_audit_with_documented_kwargs(
    fake_broker, mocked_dependencies
):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    payload = {
        "triage_session_id_or_call_interaction_id": "11111111-1111-1111-1111-111111111111",
        "trigger_reason": "critical_keyword_detected",
    }
    headers = {"correlation_id": "corr-1"}

    await escalation_dispatcher.handle_escalation_triggered(payload, headers)

    audit_instance = mocked_dependencies["audit_instance"]
    assert audit_instance.record.await_count >= 1
    _, call_kwargs = audit_instance.record.call_args
    assert call_kwargs.get("action_type") == "escalation.notify"
    assert call_kwargs.get("actor_type") == "system"


@pytest.mark.asyncio
async def test_handle_escalation_triggered_logs_at_error_level_with_trigger_reason(
    fake_broker, mocked_dependencies
):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    payload = {
        "triage_session_id_or_call_interaction_id": "22222222-2222-2222-2222-222222222222",
        "trigger_reason": "self_harm_language",
    }

    await escalation_dispatcher.handle_escalation_triggered(payload, {})

    fake_logger = mocked_dependencies["logger"]
    assert fake_logger.error.called
    logged_text = " ".join(str(c) for c in fake_logger.error.call_args_list)
    assert "self_harm_language" in logged_text


@pytest.mark.asyncio
async def test_handle_escalation_triggered_does_not_log_raw_traceback_text(
    fake_broker, mocked_dependencies
):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    payload = {
        "triage_session_id_or_call_interaction_id": "33333333-3333-3333-3333-333333333333",
        "trigger_reason": "escalation_keyword",
    }

    await escalation_dispatcher.handle_escalation_triggered(payload, {})

    fake_logger = mocked_dependencies["logger"]
    logged_text = " ".join(str(c) for c in fake_logger.error.call_args_list)
    assert "Traceback (most recent call last)" not in logged_text


@pytest.mark.asyncio
async def test_handle_escalation_triggered_raises_for_missing_required_payload_fields(
    fake_broker, mocked_dependencies
):
    await escalation_dispatcher.run(fake_broker, _session_factory)

    with pytest.raises(Exception):
        await escalation_dispatcher.handle_escalation_triggered({}, {})


@pytest.mark.asyncio
async def test_handle_escalation_triggered_uses_trigger_reference_from_payload(
    fake_broker, mocked_dependencies
):
    """It must key its audit/log entry off the payload's own entity reference,
    not perform any per-staff routing/fan-out decision (US44 Alternate Flow)."""
    await escalation_dispatcher.run(fake_broker, _session_factory)

    entity_ref = "44444444-4444-4444-4444-444444444444"
    payload = {
        "triage_session_id_or_call_interaction_id": entity_ref,
        "trigger_reason": "high_risk_answer",
    }

    await escalation_dispatcher.handle_escalation_triggered(payload, {})

    audit_instance = mocked_dependencies["audit_instance"]
    _, call_kwargs = audit_instance.record.call_args
    # no staff/role targeting kwarg should be present on the audit call -- this
    # worker performs no per-staff fan-out routing.
    assert "assigned_to" not in call_kwargs
    assert "target_staff_id" not in call_kwargs
