"""Unit tests for app/workers/recall_scanner.py, derived from its spec only.

recall_scanner.py owns every fixed-interval batch job that is not idle-chair
polling or outreach retry: the recall scan, the record-lock expiry sweep, the
appointment-completion sweep (which drives post-appointment education), the
pre-appointment education trigger scan, and the monthly ML precision
evaluation. Only three names are declared on its Public surface:
``run``, ``complete_and_educate`` and ``trigger_pre_appointment_education``.
The five interval loops named in ``run``'s doc-comment (recall_scan_loop,
lock_sweep_loop, completion_sweep_loop, pre_appointment_loop,
ml_evaluation_loop) are not independently callable/importable per the spec,
so they are only exercised indirectly through the ``run`` smoke test below.
"""
import asyncio
import inspect
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import app.workers.recall_scanner as recall_scanner

pytestmark = pytest.mark.asyncio


def _bind(sig_func, call):
    """Bind a mock's recorded call.args/call.kwargs against a reference
    signature so tests don't care whether the implementation passed an
    argument positionally or by keyword."""
    sig = inspect.signature(sig_func)
    bound = sig.bind(*call.args, **call.kwargs)
    bound.apply_defaults()
    return bound.arguments


class _Appointment:
    def __init__(self):
        self.id = uuid4()


# ---------------------------------------------------------------------------
# complete_and_educate
# ---------------------------------------------------------------------------


def _patch_completion_deps(monkeypatch, completable, post_appointment_result=None):
    fake_appt_repo = MagicMock()
    fake_appt_repo.list_completable = AsyncMock(return_value=completable)
    fake_appt_repo.update_status = AsyncMock(return_value=None)
    monkeypatch.setattr(recall_scanner, "AppointmentRepository", lambda *a, **kw: fake_appt_repo)

    fake_edu = MagicMock()
    fake_edu.post_appointment = AsyncMock(return_value=post_appointment_result or MagicMock())
    monkeypatch.setattr(recall_scanner, "EducationTriggerService", lambda *a, **kw: fake_edu)

    return fake_appt_repo, fake_edu


async def test_complete_and_educate_returns_zero_and_skips_education_when_nothing_completable(monkeypatch):
    fake_appt_repo, fake_edu = _patch_completion_deps(monkeypatch, completable=[])
    fake_db = MagicMock()

    count = await recall_scanner.complete_and_educate(fake_db)

    assert count == 0
    fake_edu.post_appointment.assert_not_awaited()
    fake_appt_repo.update_status.assert_not_awaited()


async def test_complete_and_educate_queries_list_completable_with_tz_aware_as_of(monkeypatch):
    fake_appt_repo, _ = _patch_completion_deps(monkeypatch, completable=[])
    fake_db = MagicMock()

    before = datetime.now(timezone.utc)
    await recall_scanner.complete_and_educate(fake_db)
    after = datetime.now(timezone.utc)

    fake_appt_repo.list_completable.assert_awaited_once()
    args = _bind(lambda db, as_of: None, fake_appt_repo.list_completable.await_args)
    assert args["db"] is fake_db
    assert args["as_of"].tzinfo is not None
    assert before <= args["as_of"] <= after


async def test_complete_and_educate_updates_status_to_completed_for_each_completable_appointment(monkeypatch):
    appt1, appt2 = _Appointment(), _Appointment()
    fake_appt_repo, _ = _patch_completion_deps(monkeypatch, completable=[appt1, appt2])
    fake_db = MagicMock()

    await recall_scanner.complete_and_educate(fake_db)

    assert fake_appt_repo.update_status.await_count == 2
    seen_ids = set()
    for call in fake_appt_repo.update_status.await_args_list:
        bound = _bind(lambda db, id, status, **extra: None, call)
        assert bound["status"] == "completed"
        assert bound["db"] is fake_db
        seen_ids.add(bound["id"])
    assert seen_ids == {appt1.id, appt2.id}


async def test_complete_and_educate_calls_post_appointment_for_each_completed_appointment(monkeypatch):
    appt1, appt2 = _Appointment(), _Appointment()
    fake_appt_repo, fake_edu = _patch_completion_deps(monkeypatch, completable=[appt1, appt2])
    fake_db = MagicMock()

    await recall_scanner.complete_and_educate(fake_db)

    assert fake_edu.post_appointment.await_count == 2
    seen_ids = set()
    for call in fake_edu.post_appointment.await_args_list:
        bound = _bind(lambda db, appointment_id: None, call)
        assert bound["db"] is fake_db
        seen_ids.add(bound["appointment_id"])
    assert seen_ids == {appt1.id, appt2.id}


async def test_complete_and_educate_returns_count_of_completed_appointments(monkeypatch):
    completable = [_Appointment(), _Appointment(), _Appointment()]
    _patch_completion_deps(monkeypatch, completable=completable)
    fake_db = MagicMock()

    count = await recall_scanner.complete_and_educate(fake_db)

    assert count == 3


async def test_complete_and_educate_updates_status_before_dispatching_education_for_each_appointment(monkeypatch):
    """open_questions[Q-69814105]/[Q-867306e9]: post-appointment education is
    triggered by the completion state-transition itself, in the same worker
    iteration -- so for a given appointment, update_status must be observed
    before post_appointment is called."""
    appt = _Appointment()
    order = []

    fake_appt_repo = MagicMock()
    fake_appt_repo.list_completable = AsyncMock(return_value=[appt])

    async def _update_status(*a, **kw):
        order.append("update_status")

    fake_appt_repo.update_status = AsyncMock(side_effect=_update_status)
    monkeypatch.setattr(recall_scanner, "AppointmentRepository", lambda *a, **kw: fake_appt_repo)

    fake_edu = MagicMock()

    async def _post_appointment(*a, **kw):
        order.append("post_appointment")
        return MagicMock()

    fake_edu.post_appointment = AsyncMock(side_effect=_post_appointment)
    monkeypatch.setattr(recall_scanner, "EducationTriggerService", lambda *a, **kw: fake_edu)

    fake_db = MagicMock()
    await recall_scanner.complete_and_educate(fake_db)

    assert order == ["update_status", "post_appointment"]


# ---------------------------------------------------------------------------
# trigger_pre_appointment_education
# ---------------------------------------------------------------------------


class _FakeApptRepoGeneric:
    """Answers any lookup method with the same fixed candidate list, since
    the spec does not name the exact repository method used to find
    booked appointments within the pre-appointment education window."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = []

    def __getattr__(self, name):
        async def _method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self._candidates
        return _method


def _patch_pre_appointment_deps(monkeypatch, candidates):
    fake_repo = _FakeApptRepoGeneric(candidates)
    monkeypatch.setattr(recall_scanner, "AppointmentRepository", lambda *a, **kw: fake_repo)

    fake_edu = MagicMock()
    fake_edu.pre_appointment = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(recall_scanner, "EducationTriggerService", lambda *a, **kw: fake_edu)
    return fake_repo, fake_edu


async def test_trigger_pre_appointment_education_returns_zero_when_no_qualifying_appointments(monkeypatch):
    _, fake_edu = _patch_pre_appointment_deps(monkeypatch, candidates=[])
    fake_db = MagicMock()

    count = await recall_scanner.trigger_pre_appointment_education(fake_db)

    assert count == 0
    fake_edu.pre_appointment.assert_not_awaited()


async def test_trigger_pre_appointment_education_calls_pre_appointment_for_each_candidate(monkeypatch):
    appt1, appt2 = _Appointment(), _Appointment()
    _, fake_edu = _patch_pre_appointment_deps(monkeypatch, candidates=[appt1, appt2])
    fake_db = MagicMock()

    count = await recall_scanner.trigger_pre_appointment_education(fake_db)

    assert count == 2
    assert fake_edu.pre_appointment.await_count == 2
    seen_ids = set()
    for call in fake_edu.pre_appointment.await_args_list:
        bound = _bind(lambda db, appointment_id: None, call)
        assert bound["db"] is fake_db
        seen_ids.add(bound["appointment_id"])
    assert seen_ids == {appt1.id, appt2.id}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class _FakeAnyService:
    """Generic stand-in for whichever of the five services a loop needs
    (RecallScanService, RecordLockService, AppointmentRepository,
    EducationTriggerService, RiskScoringService); every method any of
    those classes exposes per the spec is stubbed so run() never blows up
    on a real DB call while its five interval loops are starting up."""

    def __init__(self, *a, **kw):
        pass

    async def run(self, *a, **kw):
        return 0

    async def sweep_expired(self, *a, **kw):
        return 0

    async def list_completable(self, *a, **kw):
        return []

    async def update_status(self, *a, **kw):
        return None

    async def post_appointment(self, *a, **kw):
        return None

    async def pre_appointment(self, *a, **kw):
        return None

    async def evaluate_precision(self, *a, **kw):
        return None

    def __getattr__(self, name):
        async def _m(*a, **kw):
            return None
        return _m


class _FakeSessionCM:
    async def __aenter__(self):
        return MagicMock()

    async def __aexit__(self, *exc):
        return False


def _fake_session_factory():
    return _FakeSessionCM()


async def test_run_keeps_all_five_interval_loops_alive_rather_than_returning(monkeypatch):
    """Responsibility: 'every one is a fixed-interval loop' -- run() must not
    complete on its own; it supervises five concurrently-gathered loops."""
    for name in (
        "RecallScanService",
        "RecordLockService",
        "AppointmentRepository",
        "EducationTriggerService",
        "RiskScoringService",
    ):
        monkeypatch.setattr(recall_scanner, name, lambda *a, **kw: _FakeAnyService(), raising=False)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(recall_scanner.run(_fake_session_factory), timeout=0.3)


def test_run_is_a_coroutine_function():
    assert inspect.iscoroutinefunction(recall_scanner.run)
