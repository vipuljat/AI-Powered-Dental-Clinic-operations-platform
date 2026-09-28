"""Unit tests for app/worker.py — the RabbitMQ/scheduled-job consumer entrypoint.

These tests exercise app.worker.main() as a pure composition root: every
infrastructure singleton factory it is documented to call (configure_logging,
build_engine, build_session_factory, create_all_tables, get_broker,
get_storage_client) and every one of the five app/workers/*.py `run(...)`
coroutines it is documented to start are patched out, so the test asserts only
what main() wires together and how — never the internals of those
collaborators (which are covered by their own unit tests).
"""

import asyncio
import runpy
import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.worker as worker_module

pytestmark = pytest.mark.asyncio


@pytest.fixture
def patched_worker():
    """Patch every collaborator app.worker.main() is documented to call.

    Yields a dict of the mocks plus the sentinel values they resolve to, so
    each test can assert main() propagated the right objects to the right
    calls.
    """
    settings_sentinel = MagicMock(name="settings")
    settings_sentinel.environment = "test"
    engine_sentinel = MagicMock(name="engine")
    session_factory_sentinel = MagicMock(name="session_factory")
    broker_sentinel = MagicMock(name="broker")
    storage_sentinel = MagicMock(name="storage")

    with ExitStack() as stack:
        m_configure_logging = stack.enter_context(
            patch("app.worker.configure_logging")
        )
        m_get_settings = stack.enter_context(patch("app.worker.get_settings"))
        m_build_engine = stack.enter_context(patch("app.worker.build_engine"))
        m_build_session_factory = stack.enter_context(
            patch("app.worker.build_session_factory")
        )
        m_create_all_tables = stack.enter_context(
            patch("app.worker.create_all_tables", new_callable=AsyncMock)
        )
        m_get_broker = stack.enter_context(patch("app.worker.get_broker"))
        m_get_storage_client = stack.enter_context(
            patch("app.worker.get_storage_client")
        )
        m_wp_run = stack.enter_context(
            patch("app.workers.waitlist_poller.run", new_callable=AsyncMock)
        )
        m_or_run = stack.enter_context(
            patch("app.workers.outreach_retry.run", new_callable=AsyncMock)
        )
        m_rs_run = stack.enter_context(
            patch("app.workers.recall_scanner.run", new_callable=AsyncMock)
        )
        m_ed_run = stack.enter_context(
            patch("app.workers.escalation_dispatcher.run", new_callable=AsyncMock)
        )
        m_re_run = stack.enter_context(
            patch("app.workers.recommendation_engine.run", new_callable=AsyncMock)
        )

        m_get_settings.return_value = settings_sentinel
        m_build_engine.return_value = engine_sentinel
        m_build_session_factory.return_value = session_factory_sentinel
        m_get_broker.return_value = broker_sentinel
        m_get_storage_client.return_value = storage_sentinel

        yield {
            "configure_logging": m_configure_logging,
            "get_settings": m_get_settings,
            "build_engine": m_build_engine,
            "build_session_factory": m_build_session_factory,
            "create_all_tables": m_create_all_tables,
            "get_broker": m_get_broker,
            "get_storage_client": m_get_storage_client,
            "waitlist_poller_run": m_wp_run,
            "outreach_retry_run": m_or_run,
            "recall_scanner_run": m_rs_run,
            "escalation_dispatcher_run": m_ed_run,
            "recommendation_engine_run": m_re_run,
            "settings": settings_sentinel,
            "engine": engine_sentinel,
            "session_factory": session_factory_sentinel,
            "broker": broker_sentinel,
            "storage": storage_sentinel,
        }


async def test_main_configures_logging_with_settings(patched_worker):
    await worker_module.main()

    patched_worker["configure_logging"].assert_called_once_with(
        patched_worker["settings"]
    )


async def test_main_builds_engine_from_settings(patched_worker):
    await worker_module.main()

    patched_worker["build_engine"].assert_called_once_with(patched_worker["settings"])


async def test_main_creates_all_tables_on_the_built_engine(patched_worker):
    await worker_module.main()

    patched_worker["create_all_tables"].assert_awaited_once_with(
        patched_worker["engine"]
    )


async def test_main_gets_broker_from_settings(patched_worker):
    await worker_module.main()

    patched_worker["get_broker"].assert_called_once_with(patched_worker["settings"])


async def test_main_gets_storage_client_from_settings(patched_worker):
    await worker_module.main()

    patched_worker["get_storage_client"].assert_called_once_with(
        patched_worker["settings"]
    )


async def test_main_starts_waitlist_poller_with_broker_and_session_factory(
    patched_worker,
):
    await worker_module.main()

    patched_worker["waitlist_poller_run"].assert_awaited_once_with(
        patched_worker["broker"], patched_worker["session_factory"]
    )


async def test_main_starts_outreach_retry_with_broker_and_session_factory(
    patched_worker,
):
    await worker_module.main()

    patched_worker["outreach_retry_run"].assert_awaited_once_with(
        patched_worker["broker"], patched_worker["session_factory"]
    )


async def test_main_starts_recall_scanner_with_session_factory_only(patched_worker):
    await worker_module.main()

    patched_worker["recall_scanner_run"].assert_awaited_once_with(
        patched_worker["session_factory"]
    )


async def test_main_starts_escalation_dispatcher_with_broker_and_session_factory(
    patched_worker,
):
    await worker_module.main()

    patched_worker["escalation_dispatcher_run"].assert_awaited_once_with(
        patched_worker["broker"], patched_worker["session_factory"]
    )


async def test_main_starts_recommendation_engine_with_broker_and_session_factory(
    patched_worker,
):
    await worker_module.main()

    patched_worker["recommendation_engine_run"].assert_awaited_once_with(
        patched_worker["broker"], patched_worker["session_factory"]
    )


async def test_main_returns_none(patched_worker):
    result = await worker_module.main()

    assert result is None


async def test_main_runs_all_five_workers_concurrently_not_sequentially(
    patched_worker,
):
    """architecture.md §5.1 / the public-surface asyncio.gather(...) call:
    all five workers' run() coroutines must be started together, not awaited
    one after another. If main() awaited them sequentially, five 0.05s
    sleeps would take ~0.25s; run concurrently they complete in ~0.05s.
    """
    start_times = []

    async def _slow_run(*args, **kwargs):
        start_times.append(time.monotonic())
        await asyncio.sleep(0.05)

    for key in (
        "waitlist_poller_run",
        "outreach_retry_run",
        "recall_scanner_run",
        "escalation_dispatcher_run",
        "recommendation_engine_run",
    ):
        patched_worker[key].side_effect = _slow_run

    started = time.monotonic()
    await worker_module.main()
    elapsed = time.monotonic() - started

    # Sequential execution of 5 x 0.05s sleeps would take ~0.25s; concurrent
    # execution completes in about one sleep's duration.
    assert elapsed < 0.15
    # All five coroutines must have been scheduled before the first one
    # finished sleeping.
    assert (max(start_times) - min(start_times)) < 0.05


def test_dunder_main_guard_invokes_asyncio_run():
    """`if __name__ == "__main__": asyncio.run(main())` — running the module
    as __main__ must hand a coroutine off to asyncio.run rather than block
    the test process on a real event loop.
    """
    with patch("asyncio.run") as mock_run:
        runpy.run_module("app.worker", run_name="__main__")

    mock_run.assert_called_once()
    # The single positional argument handed to asyncio.run must be an
    # awaitable (main()'s coroutine object); close it to avoid a dangling
    # "coroutine was never awaited" resource warning since asyncio.run was
    # mocked out and never actually drove it.
    (scheduled_coro,) = mock_run.call_args.args
    assert asyncio.iscoroutine(scheduled_coro)
    scheduled_coro.close()
