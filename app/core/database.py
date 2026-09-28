"""Async SQLAlchemy engine, session factory, declarative ``Base``, and the
``get_db`` FastAPI dependency every repository is constructed with.

Owns the environment pivot (``test`` -> SQLite in-memory, else the configured
Postgres URL) described in project_rules.testing. Does not define any table/
model itself — those live in ``models/*/models.py``, all of which import
``Base`` from here so ``create_all_tables`` creates every table across all 11
modules in one call.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

if TYPE_CHECKING:
    from app.core.config import Settings


class Base(DeclarativeBase):
    """The single declarative base every ORM model in every module imports.

    A model that declared its own local ``Base`` would silently never be
    created by ``create_all_tables`` — every ``models/*/models.py`` file must
    import this exact class.
    """


def build_engine(settings: "Settings") -> AsyncEngine:
    """Build the process-wide async engine, applying the test-mode pivot.

    project_rules.testing: ``ENVIRONMENT=test`` switches the DB URL to
    ``sqlite+aiosqlite:///:memory:``, read once at each composition root —
    this is the one function that performs that switch. ``app/main.py`` and
    ``app/worker.py`` call this at startup with a one-line comment explaining
    why a real Postgres connection is not constructed there.
    """

    if settings.environment == "test":
        effective_url = "sqlite+aiosqlite:///:memory:"
        # Watch out: the SQLite in-memory database is process-local and
        # connection-scoped. StaticPool reuses the single physical connection
        # across every AsyncSession; without it each new session would see a
        # freshly created, empty :memory: database and prior writes would
        # appear to have vanished.
        return create_async_engine(
            effective_url,
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    return create_async_engine(settings.database_url, echo=False)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the process-wide session factory bound to ``engine``."""

    return async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def create_all_tables(engine: AsyncEngine) -> None:
    """Create every table registered on ``Base.metadata``.

    Called once at startup by every composition root, in every environment —
    no Alembic migration files exist in this tree (see project-level open
    question). Must run to completion before the first request/worker
    message is processed.
    """

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# Process-wide engine/session-factory singletons. `get_db` (a bare, no-arg
# FastAPI dependency) reads these rather than constructing its own engine —
# each composition root calls `configure_session_factory` once at startup,
# right after `build_engine`/`build_session_factory`/`create_all_tables`, so
# every request's session comes from the exact engine the schema was created
# against (critical for the SQLite-in-memory/StaticPool test pivot, where a
# second, distinct engine would see an empty database).
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def configure_session_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Wire the process-wide session factory `get_db` reads from.

    Called once by each composition root (`app/main.py`'s lifespan,
    `app/worker.py`'s `main()`) right after `build_session_factory`, so
    `get_db` always yields sessions from the same engine `create_all_tables`
    ran against.
    """

    global _session_factory
    _session_factory = session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding one ``AsyncSession`` per request.

    Commits on clean exit, rolls back on exception, always closes the
    session. Every ``repositories/*/repository.py`` class receives its
    ``AsyncSession`` through this dependency (via ``Depends(get_db)``,
    re-exported/composed by ``app/core/dependencies.py``) — repositories
    never construct their own session or engine.
    """

    if _session_factory is None:
        # Fallback for callers that never ran the composition-root startup
        # path (e.g. an ad-hoc script) — builds a default engine/session
        # factory from process settings so `get_db` never raises on a
        # missing singleton.
        from app.core.config import get_settings

        settings = get_settings()
        engine = build_engine(settings)
        session_factory = build_session_factory(engine)
        global _engine
        _engine = engine
        configure_session_factory(session_factory)

    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    else:
        await session.commit()
    finally:
        await session.close()
