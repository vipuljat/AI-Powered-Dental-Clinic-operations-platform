"""Unit tests for app/core/database.py.

Covers the environment pivot performed by `build_engine` (project_rules.testing:
`ENVIRONMENT=test` -> `sqlite+aiosqlite:///:memory:` with a shared `StaticPool`
connection, otherwise `settings.database_url`), `build_session_factory` +
`create_all_tables` producing working sessions/tables against the shared
`Base`, and the `get_db` FastAPI dependency's commit-on-success /
rollback-on-exception contract.

`Settings` (app/core/config.py) is not this file's own spec, so rather than
importing the real class we build a minimal stand-in exposing exactly the two
attributes `build_engine` is documented to read (`environment`,
`database_url`) -- the field names themselves are drawn from
specs/app/core/config.py.py.md's own public-surface listing, which is the
documented contract `database.py` depends on.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from sqlalchemy import Column, Integer, String, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

# The real composition roots set this before ever importing app.core.database;
# do the same here so importing the module (and its own `get_db` process-wide
# singleton, if built at import/first-use time) never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from app.core.database import (  # noqa: E402  (must follow the env var default above)
    Base,
    build_engine,
    build_session_factory,
    create_all_tables,
    get_db,
)


@dataclass
class _FakeSettings:
    environment: str
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/dental_test"


# ---------------------------------------------------------------------------
# build_engine: the ENVIRONMENT pivot
# ---------------------------------------------------------------------------

def test_build_engine_returns_async_engine():
    engine = build_engine(_FakeSettings(environment="test"))
    assert isinstance(engine, AsyncEngine)


def test_build_engine_uses_sqlite_memory_url_in_test_environment():
    engine = build_engine(_FakeSettings(environment="test"))
    # SQLAlchemy percent-encodes the ":memory:" path when rendering a URL to
    # a string, so compare the parsed components rather than str(engine.url).
    assert engine.url.drivername == "sqlite+aiosqlite"
    assert engine.url.database == ":memory:"


def test_build_engine_uses_static_pool_in_test_environment():
    engine = build_engine(_FakeSettings(environment="test"))
    assert isinstance(engine.pool, StaticPool)


def test_build_engine_echo_is_false():
    engine = build_engine(_FakeSettings(environment="test"))
    assert engine.sync_engine.echo is False


def test_build_engine_uses_configured_database_url_outside_test_environment():
    settings = _FakeSettings(
        environment="production",
        database_url="postgresql+asyncpg://user:pass@dbhost:5432/dental_prod",
    )
    engine = build_engine(settings)
    assert engine.url.drivername == "postgresql+asyncpg"
    assert engine.url.host == "dbhost"
    assert engine.url.database == "dental_prod"


def test_build_engine_ignores_database_url_when_environment_is_test():
    # Watch out: the test-mode substitution must win regardless of whatever
    # database_url happens to be configured.
    settings = _FakeSettings(
        environment="test",
        database_url="postgresql+asyncpg://should-not-be-used@dbhost:5432/x",
    )
    engine = build_engine(settings)
    assert engine.url.drivername == "sqlite+aiosqlite"
    assert engine.url.database == ":memory:"


# ---------------------------------------------------------------------------
# build_session_factory / create_all_tables
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_engine():
    engine = build_engine(_FakeSettings(environment="test"))
    yield engine


async def test_build_session_factory_returns_working_async_sessionmaker(test_engine):
    factory = build_session_factory(test_engine)
    assert isinstance(factory, async_sessionmaker)
    async with factory() as session:
        assert isinstance(session, AsyncSession)
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


async def test_static_pool_shares_one_connection_across_sessions(test_engine):
    # Watch out: StaticPool is required so a write from one session is still
    # visible to the next session against the same in-memory database.
    factory = build_session_factory(test_engine)
    async with factory() as first_session:
        await first_session.execute(text("CREATE TABLE probe (id INTEGER)"))
        await first_session.execute(text("INSERT INTO probe (id) VALUES (1)"))
        await first_session.commit()

    async with factory() as second_session:
        result = await second_session.execute(text("SELECT id FROM probe"))
        assert result.scalar() == 1


async def test_create_all_tables_creates_every_base_subclass_table(test_engine):
    class _ScratchModel(Base):
        __tablename__ = "database_py_scratch_model"
        id = Column(Integer, primary_key=True)
        name = Column(String)

    await create_all_tables(test_engine)

    factory = build_session_factory(test_engine)
    async with factory() as session:
        # Would raise "no such table" if create_all_tables had not run
        # Base.metadata.create_all against this engine.
        result = await session.execute(text("SELECT * FROM database_py_scratch_model"))
        assert result.fetchall() == []


# ---------------------------------------------------------------------------
# get_db: the FastAPI dependency
# ---------------------------------------------------------------------------

async def test_get_db_yields_an_async_session():
    agen = get_db()
    session = await agen.__anext__()
    try:
        assert isinstance(session, AsyncSession)
    finally:
        # Drain the generator to run its cleanup path.
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()


async def test_get_db_commits_on_clean_exit_and_is_visible_to_a_later_session():
    agen = get_db()
    session = await agen.__anext__()
    await session.execute(
        text("CREATE TABLE IF NOT EXISTS get_db_probe (id INTEGER PRIMARY KEY, val TEXT)")
    )
    await session.execute(text("INSERT INTO get_db_probe (id, val) VALUES (101, 'committed')"))
    # Clean exit: advancing past the yield with no exception in flight.
    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

    other_agen = get_db()
    other_session = await other_agen.__anext__()
    result = await other_session.execute(
        text("SELECT val FROM get_db_probe WHERE id = 101")
    )
    assert result.scalar() == "committed"
    with pytest.raises(StopAsyncIteration):
        await other_agen.__anext__()


async def test_get_db_rolls_back_on_exception():
    agen = get_db()
    session = await agen.__anext__()
    await session.execute(
        text("CREATE TABLE IF NOT EXISTS get_db_probe (id INTEGER PRIMARY KEY, val TEXT)")
    )
    await session.execute(text("INSERT INTO get_db_probe (id, val) VALUES (202, 'should-roll-back')"))

    # Mirrors how FastAPI drives a generator dependency when the route
    # handler raises: it throws the exception back into the generator at
    # the yield point rather than just abandoning it.
    with pytest.raises(RuntimeError):
        await agen.athrow(RuntimeError("boom"))

    other_agen = get_db()
    other_session = await other_agen.__anext__()
    result = await other_session.execute(
        text("SELECT val FROM get_db_probe WHERE id = 202")
    )
    assert result.first() is None
    with pytest.raises(StopAsyncIteration):
        await other_agen.__anext__()
