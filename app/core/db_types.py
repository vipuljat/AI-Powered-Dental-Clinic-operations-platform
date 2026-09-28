"""Portable column types shared by every `models/*/models.py` file.

Resolves `open_questions[Q-dad9c2c8]`: pgvector's `VECTOR(1536)` and Postgres
`JSONB` have no SQLite equivalent, but `project_rules.testing` substitutes a
SQLite in-memory engine for every test run, so the ORM models must still load
and round-trip data against that engine unmodified. `JSONBType` and
`VectorType` are the single place that gap is closed: native `JSONB`/`VECTOR`
on the `postgresql` dialect, a JSON-encoded fallback on every other dialect
(in practice, SQLite).

This module never constructs an engine or session — see `database.py` for
that.
"""

from __future__ import annotations

import json

from sqlalchemy import types
from sqlalchemy.engine.interfaces import Dialect


class JSONBType(types.TypeDecorator):
    """Postgres `JSONB` in prod, plain `JSON` (SQLite fallback) in tests.

    Used by every `models/*/models.py` column documented in architecture.md
    §4.2 as `JSONB` (e.g. `patients.demographics`, `rule_definitions.rule_value`,
    `audit_log.original_payload`/`override_payload`,
    `triage_sessions.responses`, `utilisation_recommendations.recommended_change`).
    """

    impl = types.JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> types.TypeEngine:
        if dialect.name == "postgresql":
            from sqlalchemy.dialects import postgresql

            return dialect.type_descriptor(postgresql.JSONB())
        return dialect.type_descriptor(types.JSON())

    def process_bind_param(self, value, dialect):
        # JSON handles encoding on both dialects — no-op passthrough.
        return value

    def process_result_value(self, value, dialect):
        # JSON handles decoding on both dialects — no-op passthrough.
        return value


class VectorType(types.TypeDecorator):
    """Postgres/pgvector `VECTOR(n)` in prod, JSON-encoded `TEXT` in tests.

    Used by `call_transcript_segments.embedding` (architecture.md §4.2) and any
    other embedding column. Any semantic/embedding-similarity behaviour built
    on this type is a documented no-op under the SQLite fallback: the column
    round-trips a JSON-encoded list correctly, but no vector-similarity query
    (`<->`, `<=>`) is available outside Postgres. A service that needs
    similarity search must feature-detect the dialect (or catch the resulting
    `NotImplementedError`) rather than assume it always works.
    """

    impl = types.Text
    cache_ok = True

    def __init__(self, dimensions: int = 1536, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dimensions = dimensions

    def load_dialect_impl(self, dialect: Dialect) -> types.TypeEngine:
        if dialect.name == "postgresql":
            # Imported lazily so the pgvector package is only required when
            # actually running against Postgres — the SQLite test engine
            # never needs it importable.
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dimensions))
        return dialect.type_descriptor(types.Text())

    def process_bind_param(self, value: list[float] | None, dialect):
        if dialect.name == "postgresql":
            return value
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect) -> list[float] | None:
        if dialect.name == "postgresql":
            return value
        if value is None:
            return None
        return json.loads(value)
