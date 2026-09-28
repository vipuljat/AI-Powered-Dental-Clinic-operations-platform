"""Unit tests for app/models/rules/models.py.

Exercises the three rules/config ORM tables (rule_sets, rule_definitions,
configuration_parameters) directly against an in-memory SQLite database,
mirroring the ENVIRONMENT=test substitution described in project_rules.testing
(app/core/database.py switches to sqlite+aiosqlite:///:memory: with a
StaticPool at the composition root, so the same ORM models run unmodified
against both SQLite and Postgres).

This test file exercises only app/models/rules/models.py in isolation (per
the unit-test-authoring contract: no other module's spec/code is available
here). rule_sets.created_by_staff_id / rule_sets.approved_by_staff_id /
configuration_parameters.created_by_staff_id are plain FK columns pointing at
staff_users.id, a table owned by the console module's own models.py (per the
Interaction contract, "no cross-module ORM relationship()"). Since every ORM
model shares one Base.metadata, that FK target must be resolvable for
Base.metadata.create_all to compile this file's own tables' DDL. Register a
minimal stand-in table (id column only) here -- rather than importing
another module's ORM classes -- to keep this test scoped to this file's own
spec; the guard is a no-op if the real console models already registered
"staff_users" on the shared metadata (e.g. when this file runs alongside the
full suite).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.rules.models import ConfigurationParameter, RuleDefinition, RuleSet

if "staff_users" not in Base.metadata.tables:
    Table("staff_users", Base.metadata, Column("id", Uuid(), primary_key=True))

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# factories for minimal-valid rows (overridable per-test)
# --------------------------------------------------------------------------


def make_rule_set(**overrides):
    values = dict(
        id=uuid.uuid4(),
        version_number=1,
        status="draft",
        created_by_staff_id=uuid.uuid4(),
        approved_by_staff_id=None,
        approved_at=None,
        created_at=_now(),
    )
    values.update(overrides)
    return RuleSet(**values)


def make_rule_definition(**overrides):
    values = dict(
        id=uuid.uuid4(),
        rule_set_id=uuid.uuid4(),
        category="recall_interval",
        rule_key="cleaning.default_interval_months",
        rule_value={"months": 6, "unit": "month"},
    )
    values.update(overrides)
    return RuleDefinition(**values)


def make_configuration_parameter(**overrides):
    values = dict(
        id=uuid.uuid4(),
        name="outreach.retry_backoff_minutes",
        value={"steps": [5, 15, 60]},
        version_number=1,
        effective_at=_now(),
        created_by_staff_id=uuid.uuid4(),
        rolled_back_from_id=None,
    )
    values.update(overrides)
    return ConfigurationParameter(**values)


# --------------------------------------------------------------------------
# table names / public surface
# --------------------------------------------------------------------------


def test_tablenames():
    assert RuleSet.__tablename__ == "rule_sets"
    assert RuleDefinition.__tablename__ == "rule_definitions"
    assert ConfigurationParameter.__tablename__ == "configuration_parameters"


# --------------------------------------------------------------------------
# declared column shape (Data shapes tables)
# --------------------------------------------------------------------------


def test_rule_set_declared_columns_and_nullability():
    cols = {c.name: c for c in RuleSet.__table__.columns}
    assert set(cols) == {
        "id",
        "version_number",
        "status",
        "created_by_staff_id",
        "approved_by_staff_id",
        "approved_at",
        "created_at",
    }
    assert cols["version_number"].nullable is False
    assert cols["status"].nullable is False
    assert cols["created_by_staff_id"].nullable is False
    assert cols["approved_by_staff_id"].nullable is True
    assert cols["approved_at"].nullable is True
    assert cols["created_at"].nullable is False


def test_rule_definition_declared_columns_and_nullability():
    cols = {c.name: c for c in RuleDefinition.__table__.columns}
    assert set(cols) == {"id", "rule_set_id", "category", "rule_key", "rule_value"}
    assert cols["rule_set_id"].nullable is False
    assert cols["category"].nullable is False
    assert cols["rule_key"].nullable is False
    assert cols["rule_value"].nullable is False


def test_configuration_parameter_declared_columns_and_nullability():
    cols = {c.name: c for c in ConfigurationParameter.__table__.columns}
    assert set(cols) == {
        "id",
        "name",
        "value",
        "version_number",
        "effective_at",
        "created_by_staff_id",
        "rolled_back_from_id",
    }
    assert cols["name"].nullable is False
    assert cols["value"].nullable is False
    assert cols["version_number"].nullable is False
    assert cols["effective_at"].nullable is False
    assert cols["created_by_staff_id"].nullable is False
    assert cols["rolled_back_from_id"].nullable is True


# --------------------------------------------------------------------------
# foreign keys (Data shapes tables)
# --------------------------------------------------------------------------


def test_rule_set_created_by_staff_id_fk_targets_staff_users():
    fks = RuleSet.__table__.c.created_by_staff_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "staff_users.id"


def test_rule_set_approved_by_staff_id_fk_targets_staff_users():
    fks = RuleSet.__table__.c.approved_by_staff_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "staff_users.id"


def test_rule_definition_rule_set_id_fk_targets_rule_sets():
    fks = RuleDefinition.__table__.c.rule_set_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "rule_sets.id"


def test_configuration_parameter_created_by_staff_id_fk_targets_staff_users():
    fks = ConfigurationParameter.__table__.c.created_by_staff_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "staff_users.id"


def test_configuration_parameter_rolled_back_from_id_is_self_referential_fk():
    fks = ConfigurationParameter.__table__.c.rolled_back_from_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "configuration_parameters.id"


# --------------------------------------------------------------------------
# no relationship() attributes at all (Interaction contract)
# --------------------------------------------------------------------------


def test_rule_set_declares_no_orm_relationships():
    assert inspect(RuleSet).relationships.keys() == []


def test_rule_definition_declares_no_orm_relationships():
    assert inspect(RuleDefinition).relationships.keys() == []


def test_configuration_parameter_declares_no_orm_relationships():
    assert inspect(ConfigurationParameter).relationships.keys() == []


# --------------------------------------------------------------------------
# rule_sets behaviour
# --------------------------------------------------------------------------


async def test_rule_set_id_is_a_generated_uuid(session):
    rule_set = make_rule_set()
    session.add(rule_set)
    await session.flush()
    assert isinstance(rule_set.id, uuid.UUID)


async def test_rule_set_approved_fields_nullable_while_in_draft(session):
    rule_set = make_rule_set(status="draft", approved_by_staff_id=None, approved_at=None)
    session.add(rule_set)
    await session.commit()  # must not raise: no approval yet for a draft rule set
    session.expunge_all()

    fetched = (
        await session.execute(select(RuleSet).where(RuleSet.id == rule_set.id))
    ).scalar_one()
    assert fetched.approved_by_staff_id is None
    assert fetched.approved_at is None


async def test_rule_set_approved_fields_roundtrip_once_clinic_management_approves(session):
    approver_id = uuid.uuid4()
    approved_at = _now()
    rule_set = make_rule_set(
        status="active", approved_by_staff_id=approver_id, approved_at=approved_at
    )
    session.add(rule_set)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(select(RuleSet).where(RuleSet.id == rule_set.id))
    ).scalar_one()
    assert fetched.approved_by_staff_id == approver_id
    assert fetched.approved_at is not None


async def test_rule_set_status_supports_the_full_lifecycle(session):
    for status_value in ("draft", "in_review", "active", "superseded"):
        session.add(make_rule_set(id=uuid.uuid4(), status=status_value))
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(RuleSet))).scalars().all()
    persisted = {getattr(r.status, "value", r.status) for r in fetched}
    assert persisted == {"draft", "in_review", "active", "superseded"}


async def test_rule_set_superseded_rows_remain_queryable_after_supersession(session):
    # FR-E3.5: superseded rows must remain queryable for audit/diff history,
    # since the supersede-then-activate transition never deletes a row.
    superseded = make_rule_set(status="superseded", version_number=1)
    active = make_rule_set(status="active", version_number=2)
    session.add_all([superseded, active])
    await session.commit()
    session.expunge_all()

    fetched_ids = {
        r.id for r in (await session.execute(select(RuleSet))).scalars().all()
    }
    assert superseded.id in fetched_ids
    assert active.id in fetched_ids


async def test_rule_set_multiple_active_rows_not_rejected_at_db_level(session):
    # FR-E3.5's "only one active rule set" invariant is enforced by
    # RuleSetRepository.activate's supersede-then-activate transaction, not by
    # a DB constraint declared on this model -- so the ORM/DB layer alone must
    # allow two simultaneously-active rows (a broken caller could otherwise
    # violate the invariant undetected at this layer, which is exactly the
    # gap the repository is documented to close).
    first_active = make_rule_set(status="active", version_number=1)
    second_active = make_rule_set(status="active", version_number=2)
    session.add_all([first_active, second_active])
    await session.commit()  # must not raise
    session.expunge_all()

    active_rows = (
        await session.execute(select(RuleSet).where(RuleSet.status == "active"))
    ).scalars().all()
    assert len(active_rows) == 2


async def test_rule_set_missing_version_number_raises_integrity_error(session):
    rule_set = make_rule_set(version_number=None)
    session.add(rule_set)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_rule_set_missing_status_raises_integrity_error(session):
    rule_set = make_rule_set(status=None)
    session.add(rule_set)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_rule_set_missing_created_by_staff_id_raises_integrity_error(session):
    rule_set = make_rule_set(created_by_staff_id=None)
    session.add(rule_set)
    with pytest.raises(IntegrityError):
        await session.commit()


# --------------------------------------------------------------------------
# rule_definitions behaviour
# --------------------------------------------------------------------------


async def test_rule_definition_rule_value_is_structured_json_not_free_text(session):
    # FR-E3.1/RMR004: rule_value is JSONB-structured data, not a free-text string.
    structured = {"category_rules": [{"min_age": 0, "max_age": 12, "label": "pediatric"}]}
    row = make_rule_definition(rule_value=structured)
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(select(RuleDefinition).where(RuleDefinition.id == row.id))
    ).scalar_one()
    assert fetched.rule_value == structured
    assert isinstance(fetched.rule_value, dict)


async def test_rule_definition_category_supports_every_declared_category(session):
    categories = [
        "patient_category",
        "risk_classification",
        "recall_interval",
        "appointment_type",
        "scheduling_priority",
        "triage_question",
        "escalation_trigger",
        "family_scheduling",
    ]
    for category in categories:
        session.add(make_rule_definition(id=uuid.uuid4(), category=category))
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(RuleDefinition))).scalars().all()
    persisted = {getattr(r.category, "value", r.category) for r in fetched}
    assert persisted == set(categories)


async def test_rule_definition_missing_rule_set_id_raises_integrity_error(session):
    row = make_rule_definition(rule_set_id=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_rule_definition_missing_category_raises_integrity_error(session):
    row = make_rule_definition(category=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_rule_definition_missing_rule_key_raises_integrity_error(session):
    row = make_rule_definition(rule_key=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


# --------------------------------------------------------------------------
# configuration_parameters behaviour
# --------------------------------------------------------------------------


async def test_configuration_parameter_value_is_structured_json_roundtrip(session):
    structured = {"cadence_days": [1, 3, 7], "channel_priority": ["whatsapp", "sms", "email"]}
    row = make_configuration_parameter(value=structured)
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(
            select(ConfigurationParameter).where(ConfigurationParameter.id == row.id)
        )
    ).scalar_one()
    assert fetched.value == structured
    assert isinstance(fetched.value, dict)


async def test_configuration_parameter_rollback_lineage_roundtrips(session, session_factory):
    original = make_configuration_parameter(version_number=1)
    session.add(original)
    await session.commit()
    await session.refresh(original)

    rollback_row = make_configuration_parameter(
        version_number=2, rolled_back_from_id=original.id
    )
    session.add(rollback_row)
    await session.commit()
    rollback_id = rollback_row.id

    # re-query from a fresh session to force a real round trip rather than
    # relying on the identity map
    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(ConfigurationParameter, rollback_id)
        assert reloaded.rolled_back_from_id == original.id


async def test_configuration_parameter_rolled_back_from_id_nullable_when_not_a_rollback(session):
    row = make_configuration_parameter(rolled_back_from_id=None)
    session.add(row)
    await session.commit()  # must not raise: most rows are not rollbacks
    session.expunge_all()

    fetched = (
        await session.execute(
            select(ConfigurationParameter).where(ConfigurationParameter.id == row.id)
        )
    ).scalar_one()
    assert fetched.rolled_back_from_id is None


async def test_configuration_parameter_missing_name_raises_integrity_error(session):
    row = make_configuration_parameter(name=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


# NOTE: a "missing value raises IntegrityError" case was intentionally not
# written here (mirroring rule_value above): `value` is stored through a
# JSONB TypeDecorator, and whether a Python `None` is encoded as a JSON
# `null` literal (satisfying the NOT NULL column constraint) or passed
# through as a real SQL NULL (violating it) is an internal encoding choice
# the spec does not state, so it is not reliably observable through this
# file's declared public surface alone.


async def test_configuration_parameter_missing_version_number_raises_integrity_error(session):
    row = make_configuration_parameter(version_number=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_configuration_parameter_missing_effective_at_raises_integrity_error(session):
    row = make_configuration_parameter(effective_at=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_configuration_parameter_missing_created_by_staff_id_raises_integrity_error(
    session,
):
    row = make_configuration_parameter(created_by_staff_id=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


# --------------------------------------------------------------------------
# two separate tables / two separate write paths (Watch out)
# --------------------------------------------------------------------------


def test_rule_set_has_an_approval_gate_but_configuration_parameter_does_not():
    # FR-E3.2 (clinical rules, Clinic-Management-approved) vs FR-E3.4
    # (outreach config, Delivery-Team-only, no approval step): the
    # approval-gate columns exist only on rule_sets.
    rule_set_columns = {c.name for c in RuleSet.__table__.columns}
    config_columns = {c.name for c in ConfigurationParameter.__table__.columns}
    assert {"approved_by_staff_id", "approved_at"} <= rule_set_columns
    assert "approved_by_staff_id" not in config_columns
    assert "approved_at" not in config_columns


def test_rules_and_configuration_are_declared_as_separate_tables():
    assert RuleSet.__tablename__ != ConfigurationParameter.__tablename__
    assert RuleDefinition.__tablename__ != ConfigurationParameter.__tablename__
