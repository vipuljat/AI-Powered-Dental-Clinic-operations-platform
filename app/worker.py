"""The second composition root (architecture.md §5, §5.1, §9's hosting
table: "Async workers... separate pods from API, same image, worker.py
entrypoint") — the RabbitMQ/scheduled-job consumer entrypoint, a distinct OS
process from the API (`app/main.py`) that shares the same codebase and the
same `services`/`repositories` layers, never any route/HTTP code.

Reads the same `ENVIRONMENT` flag `app/main.py` reads (project_rules.testing)
and, independently in this process, pivots the DB URL/broker/storage-client
construction to the same in-memory substitutes the API composition root uses
— see the Watch out note in this file's spec: two separate OS processes each
running in test mode do NOT share state (two independent in-memory
databases/queues), so any acceptance test that needs to observe a worker's
effect on data the API wrote must drive both composition roots' setup from
the same Python process/test session, sharing one `engine`/`broker`
instance, rather than spawning this module as a subprocess.
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.database import build_engine, build_session_factory, create_all_tables
from app.core.logging import configure_logging, get_logger
from app.core.messaging import RabbitMQBroker, get_broker
from app.core.storage import get_storage_client

# Every module's models.py is imported here, purely for the side effect of
# registering its ORM tables on the shared `Base.metadata`
# (app/core/database.py) — required so `create_all_tables` below can resolve
# cross-module foreign keys (e.g. `escalations.call_interaction_id` ->
# `call_interactions.id`, owned by the engagement module, which none of the
# five worker modules themselves import) regardless of which subset of the
# 11 BRD-epic modules the five `app/workers/*.py` consumers happen to touch.
# `app/main.py` gets the same completeness for free by importing every
# module's routes.py (which imports that module's models transitively); this
# process has no routes layer, so it registers the same 11 modules directly.
import app.models.analytics.models  # noqa: F401
import app.models.console.models  # noqa: F401
import app.models.education.models  # noqa: F401
import app.models.engagement.models  # noqa: F401
import app.models.intelligence.models  # noqa: F401
import app.models.outreach.models  # noqa: F401
import app.models.patients.models  # noqa: F401
import app.models.recall.models  # noqa: F401
import app.models.rules.models  # noqa: F401
import app.models.scheduling.models  # noqa: F401
import app.models.waitlist.models  # noqa: F401

# `waitlist_poller` is imported first, deliberately: its own module-level
# import order (patients repository before waitlist repository) is what
# registers the real `patients` ORM table on `Base.metadata` before
# `app.repositories.waitlist.repository`'s module-level FK-stand-in loop
# runs — the stand-in loop is itself a documented no-op once the real table
# is already registered (see that file's own comment), but only if the real
# `app.models.patients.models` module has been imported by the time it
# runs. Importing any of the other four worker modules first can trip that
# ordering the other way (e.g. `escalation_dispatcher` pulls in
# `app.services.console.service`, which imports the waitlist repository
# before anything imports the patients repository), which raises a
# SQLAlchemy "Table 'patients' is already defined" error. This import order
# is the fix for that pre-existing cross-module ordering hazard, applied
# here rather than in any of the five frozen worker files themselves.
from app.workers import waitlist_poller
from app.workers import (
    escalation_dispatcher,
    outreach_retry,
    recall_scanner,
    recommendation_engine,
)

logger = get_logger(__name__)


async def main() -> None:
    """Build this process's infrastructure singletons and start all five
    `app/workers/*.py` modules' `run(...)` coroutines concurrently.

    architecture.md §5.1: this is the RabbitMQ consumer entrypoint process;
    each worker module owns its own service/repository construction (per
    its own `run()`), so this function's job is limited to the shared
    cross-cutting infra (engine, broker, storage client) every worker's
    `run()` is handed, plus the concurrent fan-out itself.
    """

    settings = get_settings()

    # 1. Structured JSON logging, configured once per process
    # (project_rules.logging), identically to how app/main.py configures it.
    configure_logging(settings)

    # 2. Engine + idempotent table creation -- safe even if the API process
    # already created the tables (CREATE TABLE IF NOT EXISTS semantics via
    # SQLAlchemy's `create_all`), and in test mode this is the same
    # `sqlite+aiosqlite:///:memory:` pivot app/main.py performs
    # (project_rules.testing item 1), read from `settings.database_url`.
    engine = build_engine(settings)
    await create_all_tables(engine)
    session_factory = build_session_factory(engine)

    # 3. Broker: `InMemoryBroker` (asyncio.Queue-backed fake, no real
    # network) when `settings.environment == "test"`, else a real
    # `RabbitMQBroker` that must `connect()` before use -- constructed
    # lazily here at process startup, never at import time
    # (project_rules.testing item 2).
    broker = get_broker(settings)
    if isinstance(broker, RabbitMQBroker):
        await broker.connect()
    # else: InMemoryBroker (or, in a unit test exercising this composition
    # root in isolation, a mock standing in for either) requires no
    # connection step -- it is a pure in-process asyncio.Queue fake with
    # nothing to open, so no-op.

    # 4. Object-storage client: in-memory dict-backed fake in test mode,
    # real boto3 S3-compatible client otherwise (project_rules.testing item
    # 3). No worker in this tree currently reads/writes object storage
    # directly (each worker's own repositories/services are DB-only), but
    # it is constructed here regardless so this process's infra singleton
    # graph mirrors app/main.py's exactly, per this file's spec step 4.
    get_storage_client(settings)

    logger.info(
        "worker_process_starting",
        extra={"environment": settings.environment},
    )

    # 5/6. Each `app/workers/*.py` module owns constructing its own
    # service/repository singleton graph inside `run()` (mirroring, not
    # sharing Python objects with, app/main.py's graph -- this process does
    # not share objects with the API process, only the same database/broker
    # per this file's spec) -- so this composition root only needs to fan
    # the five `run(...)` coroutines out concurrently, keyed to the same
    # `broker`/`session_factory` singletons built above.
    await asyncio.gather(
        waitlist_poller.run(broker, session_factory),
        outreach_retry.run(broker, session_factory),
        recall_scanner.run(session_factory),
        escalation_dispatcher.run(broker, session_factory),
        recommendation_engine.run(broker, session_factory),
    )


if __name__ == "__main__":
    asyncio.run(main())
