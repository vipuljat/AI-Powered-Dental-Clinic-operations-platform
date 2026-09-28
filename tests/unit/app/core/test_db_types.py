"""Unit tests for app/core/db_types.py.

`JSONBType`/`VectorType` are the two SQLAlchemy `TypeDecorator`s that let
every `models/*/models.py` file declare one column type and have it render
correctly on both real Postgres (JSONB / pgvector `VECTOR`) and the SQLite
in-memory engine `project_rules.testing` substitutes during every test run
(resolves `open_questions[Q-dad9c2c8]`).

The `pgvector` python package is *not* a declared project dependency (see
specs/pyproject.toml.md's runtime dependency list -- it lists `asyncpg` but
not `pgvector`), so it is not installed in this test environment. Per the
file's own spec, `VectorType.load_dialect_impl` only needs
`pgvector.sqlalchemy.Vector` when `dialect.name == "postgresql"`, which the
project's `ENVIRONMENT=test` SQLite substitution never actually exercises at
runtime. To still verify that documented postgres-branch delegation here
(without requiring the real third-party package), a minimal stand-in module
is registered under `sys.modules` before importing the module under test, so
"import pgvector.sqlalchemy" resolves successfully however/whenever the
implementation performs that import.
"""
from __future__ import annotations

import json
import sys
import types as py_types

import pytest
import sqlalchemy
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, select
from sqlalchemy import types as sa_types
from sqlalchemy.dialects import postgresql, sqlite

# ---------------------------------------------------------------------------
# Make `pgvector.sqlalchemy.Vector` importable even though the real
# third-party package is not installed in this environment (see module
# docstring). Falls back to the real package transparently if it is ever
# added to the project's dependencies.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only if pgvector is actually installed
    import pgvector.sqlalchemy as _real_pgvector_sqlalchemy

    PGVectorVectorClass = _real_pgvector_sqlalchemy.Vector
except ModuleNotFoundError:

    class _FakeVector(sqlalchemy.types.UserDefinedType):
        """Minimal stand-in for pgvector.sqlalchemy.Vector: stores the
        requested dimensionality on `.dim`, exactly like the real class,
        and is DDL-compilable so tests can drive it through the ordinary
        SQLAlchemy dialect machinery."""

        cache_ok = True

        def __init__(self, dim=None, *args, **kwargs):
            self.dim = dim

        def get_col_spec(self, **kw):
            return f"VECTOR({self.dim})"

    _fake_pgvector_pkg = py_types.ModuleType("pgvector")
    _fake_pgvector_sqlalchemy_mod = py_types.ModuleType("pgvector.sqlalchemy")
    _fake_pgvector_sqlalchemy_mod.Vector = _FakeVector
    _fake_pgvector_pkg.sqlalchemy = _fake_pgvector_sqlalchemy_mod
    sys.modules.setdefault("pgvector", _fake_pgvector_pkg)
    sys.modules.setdefault("pgvector.sqlalchemy", _fake_pgvector_sqlalchemy_mod)

    PGVectorVectorClass = _FakeVector

from app.core.db_types import JSONBType, VectorType  # noqa: E402  (import must follow the sys.modules shim above)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pg_dialect():
    return postgresql.dialect()


@pytest.fixture(scope="module")
def sqlite_dialect():
    return sqlite.dialect()


# ---------------------------------------------------------------------------
# Class shape / public surface
# ---------------------------------------------------------------------------

def test_jsonb_type_is_a_type_decorator():
    assert issubclass(JSONBType, sqlalchemy.types.TypeDecorator)


def test_jsonb_type_impl_is_json():
    assert JSONBType.impl is sa_types.JSON


def test_jsonb_type_cache_ok_is_true():
    assert JSONBType.cache_ok is True


def test_vector_type_is_a_type_decorator():
    assert issubclass(VectorType, sqlalchemy.types.TypeDecorator)


def test_vector_type_impl_is_text():
    assert VectorType.impl is sa_types.Text


def test_vector_type_cache_ok_is_true():
    assert VectorType.cache_ok is True


def test_vector_type_default_dimensions_is_1536():
    assert VectorType().dimensions == 1536


def test_vector_type_accepts_custom_dimensions_positionally():
    assert VectorType(768).dimensions == 768


def test_vector_type_accepts_custom_dimensions_as_keyword():
    assert VectorType(dimensions=384).dimensions == 384


# ---------------------------------------------------------------------------
# JSONBType.load_dialect_impl
# ---------------------------------------------------------------------------

def test_jsonb_load_dialect_impl_returns_postgres_jsonb_on_postgres(pg_dialect):
    result = JSONBType().load_dialect_impl(pg_dialect)
    assert isinstance(result, postgresql.JSONB)


def test_jsonb_load_dialect_impl_returns_generic_json_on_sqlite(sqlite_dialect):
    result = JSONBType().load_dialect_impl(sqlite_dialect)
    assert isinstance(result, sa_types.JSON)
    assert not isinstance(result, postgresql.JSONB)


# ---------------------------------------------------------------------------
# JSONBType.process_bind_param / process_result_value: no-op passthrough
# ---------------------------------------------------------------------------

def test_jsonb_process_bind_param_passes_through_dict(pg_dialect, sqlite_dialect):
    payload = {"a": 1, "b": [1, 2, 3]}
    t = JSONBType()
    assert t.process_bind_param(payload, sqlite_dialect) == payload
    assert t.process_bind_param(payload, pg_dialect) == payload


def test_jsonb_process_bind_param_passes_through_none(sqlite_dialect):
    assert JSONBType().process_bind_param(None, sqlite_dialect) is None


def test_jsonb_process_result_value_passes_through_dict(pg_dialect, sqlite_dialect):
    payload = {"a": 1, "b": [1, 2, 3]}
    t = JSONBType()
    assert t.process_result_value(payload, sqlite_dialect) == payload
    assert t.process_result_value(payload, pg_dialect) == payload


def test_jsonb_process_result_value_passes_through_none(sqlite_dialect):
    assert JSONBType().process_result_value(None, sqlite_dialect) is None


# ---------------------------------------------------------------------------
# VectorType.load_dialect_impl
# ---------------------------------------------------------------------------

def test_vector_load_dialect_impl_returns_pgvector_vector_on_postgres(pg_dialect):
    result = VectorType(dimensions=42).load_dialect_impl(pg_dialect)
    assert isinstance(result, PGVectorVectorClass)
    assert result.dim == 42


def test_vector_load_dialect_impl_returns_generic_text_on_sqlite(sqlite_dialect):
    result = VectorType().load_dialect_impl(sqlite_dialect)
    assert isinstance(result, sa_types.Text)
    assert not isinstance(result, PGVectorVectorClass)


# ---------------------------------------------------------------------------
# VectorType.process_bind_param
# ---------------------------------------------------------------------------

def test_vector_process_bind_param_json_encodes_on_sqlite(sqlite_dialect):
    value = [0.1, 0.2, 0.3]
    result = VectorType(dimensions=3).process_bind_param(value, sqlite_dialect)
    assert result == json.dumps(value)


def test_vector_process_bind_param_passes_through_on_postgres(pg_dialect):
    value = [0.1, 0.2, 0.3]
    result = VectorType(dimensions=3).process_bind_param(value, pg_dialect)
    assert result == value


# ---------------------------------------------------------------------------
# VectorType.process_result_value
# ---------------------------------------------------------------------------

def test_vector_process_result_value_json_decodes_on_sqlite(sqlite_dialect):
    value = [0.1, 0.2, 0.3]
    encoded = json.dumps(value)
    result = VectorType(dimensions=3).process_result_value(encoded, sqlite_dialect)
    assert result == value


def test_vector_process_result_value_none_is_passthrough_on_sqlite(sqlite_dialect):
    assert VectorType(dimensions=3).process_result_value(None, sqlite_dialect) is None


def test_vector_process_result_value_passes_through_on_postgres(pg_dialect):
    value = [0.1, 0.2, 0.3]
    result = VectorType(dimensions=3).process_result_value(value, pg_dialect)
    assert result == value


# ---------------------------------------------------------------------------
# End-to-end round trip against the real SQLite fallback engine, mirroring
# how a models.py column declared `Column(JSONBType())` /
# `Column(VectorType(dimensions=N))` actually behaves under the
# `sqlite+aiosqlite:///:memory:` substitution described in PROJECT.md's
# Testing section (a plain synchronous sqlite engine is used here purely as
# the lightest-weight way to drive the real TypeDecorator bind/result cycle
# end to end -- no application code, network, or the async driver is needed
# to prove the column round-trips correctly).
# ---------------------------------------------------------------------------

@pytest.fixture()
def sqlite_engine():
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


def test_jsonb_and_vector_columns_round_trip_through_sqlite_engine(sqlite_engine):
    metadata = MetaData()
    table = Table(
        "widgets",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("payload", JSONBType()),
        Column("embedding", VectorType(dimensions=4)),
    )
    metadata.create_all(sqlite_engine)

    payload = {"label": "molar", "tags": ["urgent", "recall"]}
    embedding = [0.1, 0.2, 0.3, 0.4]

    with sqlite_engine.begin() as conn:
        conn.execute(table.insert().values(id=1, payload=payload, embedding=embedding))

    with sqlite_engine.connect() as conn:
        row = conn.execute(select(table).where(table.c.id == 1)).one()

    assert row.payload == payload
    assert row.embedding == embedding


def test_jsonb_and_vector_columns_round_trip_none_through_sqlite_engine(sqlite_engine):
    metadata = MetaData()
    table = Table(
        "widgets_nullable",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("payload", JSONBType(), nullable=True),
        Column("embedding", VectorType(dimensions=4), nullable=True),
    )
    metadata.create_all(sqlite_engine)

    with sqlite_engine.begin() as conn:
        conn.execute(table.insert().values(id=1, payload=None, embedding=None))

    with sqlite_engine.connect() as conn:
        row = conn.execute(select(table).where(table.c.id == 1)).one()

    assert row.payload is None
    assert row.embedding is None
