"""Unit tests for app/repositories/rules/repository.py.

Exercises the three data-access classes (RuleSetRepository,
RuleDefinitionRepository, ConfigurationParameterRepository) directly against
an in-memory SQLite database, mirroring the ENVIRONMENT=test substitution
described in project_rules.testing. Nothing here mocks the AsyncSession --
these are real ORM round-trips against app.models.rules.models's three
tables, so a repository method that builds the wrong query is caught the
same way it would be against Postgres.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

# The real composition roots set this before ever importing app.core.database;
# do the same here defensively so importing it never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from sqlalchemy import Column, Table, Uuid  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.common.enums import RuleCategory, RuleSetStatus  # noqa: E402
from app.core.database import Base  # noqa: E402
from app.models.rules.models import ConfigurationParameter, RuleDefinition, RuleSet  # noqa: E402
from app.repositories.rules.repository import (  # noqa: E402
    ConfigurationParameterRepository,
    RuleDefinitionRepository,
    RuleSetRepository,
)

# rule_sets.created_by_staff_id / rule_sets.approved_by_staff_id /
# configuration_parameters.created_by_staff_id declare an FK to
# staff_users.id, a table owned by the console module that this task has no
# spec for and must not import for its own sake. Register a minimal
# stand-in table with just the `id` primary key FK resolution needs, purely
# so this in-memory engine can create *this* spec's own three tables. This
# is a no-op if the real console models already registered "staff_users" on
# the shared metadata (e.g. when this file runs alongside the full suite).
try:  # pragma: no cover - exercised implicitly by the fixtures below
    import app.models.console.models  # noqa: F401
except ModuleNotFoundError:
    pass

if "staff_users" not in Base.metadata.tables:
    Table("staff_users", Base.metadata, Column("id", Uuid(), primary_key=True))

_RULES_TABLES = [
    RuleSet.__table__,
    RuleDefinition.__table__,
    ConfigurationParameter.__table__,
]

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
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=_RULES_TABLES)
        )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def db(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


@pytest.fixture
def rule_set_repo():
    return RuleSetRepository()


@pytest.fixture
def rule_def_repo():
    return RuleDefinitionRepository()


@pytest.fixture
def config_repo():
    return ConfigurationParameterRepository()


def _rule_dict(category, rule_key: str, rule_value: dict) -> dict:
    return {"category": category, "rule_key": rule_key, "rule_value": rule_value}


def _assert_close_to_now(value: datetime, *, tolerance_seconds: int = 10) -> None:
    # Compared with tzinfo normalised away: the sqlite+aiosqlite fallback this
    # test suite runs against (per project_rules.testing) is not guaranteed to
    # round-trip a TIMESTAMPTZ column's UTC offset bit-for-bit, only its
    # wall-clock value -- app/models/recall/repository tests make the same
    # accommodation for this column type.
    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    assert value is not None
    value_naive = value.replace(tzinfo=None) if value.tzinfo is not None else value
    assert abs((now_naive - value_naive).total_seconds()) < tolerance_seconds


# --------------------------------------------------------------------------
# RuleSetRepository.create_draft
# --------------------------------------------------------------------------


async def test_create_draft_first_ever_row_is_version_1(db, rule_set_repo):
    staff_id = uuid.uuid4()

    draft = await rule_set_repo.create_draft(db, staff_id)

    assert isinstance(draft, RuleSet)
    assert draft.version_number == 1
    assert draft.created_by_staff_id == staff_id


async def test_create_draft_sets_draft_status(db, rule_set_repo):
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())

    assert draft.status == RuleSetStatus.draft
    assert draft.status == "draft"


async def test_create_draft_version_increments_from_current_max_regardless_of_status(
    db, rule_set_repo
):
    staff_id = uuid.uuid4()
    d1 = await rule_set_repo.create_draft(db, staff_id)  # v1, left as draft
    d2 = await rule_set_repo.create_draft(db, staff_id)  # v2
    await rule_set_repo.set_status(db, d2.id, "in_review")
    d3 = await rule_set_repo.create_draft(db, staff_id)  # v3
    await rule_set_repo.activate(db, d3.id, uuid.uuid4())  # v3 -> active

    d4 = await rule_set_repo.create_draft(db, staff_id)

    assert [d1.version_number, d2.version_number, d3.version_number, d4.version_number] == [
        1,
        2,
        3,
        4,
    ]


# --------------------------------------------------------------------------
# RuleSetRepository.get_by_id
# --------------------------------------------------------------------------


async def test_get_by_id_returns_the_matching_rule_set(db, rule_set_repo):
    created = await rule_set_repo.create_draft(db, uuid.uuid4())

    fetched = await rule_set_repo.get_by_id(db, created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.version_number == created.version_number


async def test_get_by_id_returns_none_for_unknown_id(db, rule_set_repo):
    assert await rule_set_repo.get_by_id(db, uuid.uuid4()) is None


# --------------------------------------------------------------------------
# RuleSetRepository.get_active
# --------------------------------------------------------------------------


async def test_get_active_returns_none_when_nothing_is_active(db, rule_set_repo):
    await rule_set_repo.create_draft(db, uuid.uuid4())

    assert await rule_set_repo.get_active(db) is None


async def test_get_active_returns_the_single_active_rule_set(db, rule_set_repo):
    staff_id = uuid.uuid4()
    draft = await rule_set_repo.create_draft(db, staff_id)
    other_draft = await rule_set_repo.create_draft(db, staff_id)
    await rule_set_repo.activate(db, draft.id, uuid.uuid4())

    active = await rule_set_repo.get_active(db)

    assert active is not None
    assert active.id == draft.id
    assert active.id != other_draft.id


# --------------------------------------------------------------------------
# RuleSetRepository.set_status
# --------------------------------------------------------------------------


async def test_set_status_updates_the_status_field(db, rule_set_repo):
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())

    updated = await rule_set_repo.set_status(db, draft.id, "in_review")

    assert updated.status == "in_review"
    reloaded = await rule_set_repo.get_by_id(db, draft.id)
    assert reloaded.status == "in_review"


async def test_set_status_optionally_records_the_approver(db, rule_set_repo):
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())
    approver = uuid.uuid4()

    updated = await rule_set_repo.set_status(db, draft.id, "active", approved_by_staff_id=approver)

    assert updated.approved_by_staff_id == approver


async def test_set_status_leaves_approver_untouched_when_not_supplied(db, rule_set_repo):
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())

    updated = await rule_set_repo.set_status(db, draft.id, "in_review")

    assert updated.approved_by_staff_id is None


# --------------------------------------------------------------------------
# RuleSetRepository.activate -- FR-E3.5/US15/T39 atomic supersede+activate
# --------------------------------------------------------------------------


async def test_activate_marks_the_target_active_and_records_the_approval(db, rule_set_repo):
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())
    approver = uuid.uuid4()

    activated = await rule_set_repo.activate(db, draft.id, approver)

    assert activated.status == "active"
    assert activated.approved_by_staff_id == approver
    _assert_close_to_now(activated.approved_at)


async def test_activate_supersedes_the_previously_active_rule_set_in_the_same_call(
    db, rule_set_repo
):
    staff_id = uuid.uuid4()
    first_draft = await rule_set_repo.create_draft(db, staff_id)
    await rule_set_repo.activate(db, first_draft.id, uuid.uuid4())

    second_draft = await rule_set_repo.create_draft(db, staff_id)
    await rule_set_repo.activate(db, second_draft.id, uuid.uuid4())

    reloaded_first = await rule_set_repo.get_by_id(db, first_draft.id)
    reloaded_second = await rule_set_repo.get_by_id(db, second_draft.id)

    assert reloaded_first.status == "superseded"
    assert reloaded_second.status == "active"


async def test_activate_never_leaves_two_rule_sets_simultaneously_active(db, rule_set_repo):
    # T39 AC: "the approved version becomes the active rule set atomically;
    # prior version remains in force until approved" -- after a single
    # `activate` call, exactly one row is active, never zero and never two.
    staff_id = uuid.uuid4()
    first_draft = await rule_set_repo.create_draft(db, staff_id)
    await rule_set_repo.activate(db, first_draft.id, uuid.uuid4())
    second_draft = await rule_set_repo.create_draft(db, staff_id)

    await rule_set_repo.activate(db, second_draft.id, uuid.uuid4())

    reloaded_first = await rule_set_repo.get_by_id(db, first_draft.id)
    reloaded_second = await rule_set_repo.get_by_id(db, second_draft.id)
    active_rows = [row for row in (reloaded_first, reloaded_second) if row.status == "active"]

    assert len(active_rows) == 1
    assert active_rows[0].id == second_draft.id


# --------------------------------------------------------------------------
# RuleDefinitionRepository.bulk_insert / list_by_set / list_by_set_and_category
# --------------------------------------------------------------------------


async def test_bulk_insert_persists_all_given_rules_under_the_rule_set(
    db, rule_set_repo, rule_def_repo
):
    rule_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    rules = [
        _rule_dict(RuleCategory.recall_interval, "cleaning.default_interval_months", {"months": 6}),
        _rule_dict(RuleCategory.risk_classification, "risk.high_threshold", {"score": 80}),
    ]

    inserted = await rule_def_repo.bulk_insert(db, rule_set.id, rules)

    assert len(inserted) == 2
    assert all(isinstance(r, RuleDefinition) for r in inserted)
    assert all(r.rule_set_id == rule_set.id for r in inserted)
    assert {r.rule_key for r in inserted} == {
        "cleaning.default_interval_months",
        "risk.high_threshold",
    }


async def test_list_by_set_returns_only_rules_belonging_to_that_set(
    db, rule_set_repo, rule_def_repo
):
    set_a = await rule_set_repo.create_draft(db, uuid.uuid4())
    set_b = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, set_a.id, [_rule_dict(RuleCategory.recall_interval, "a.key", {"v": 1})]
    )
    await rule_def_repo.bulk_insert(
        db, set_b.id, [_rule_dict(RuleCategory.recall_interval, "b.key", {"v": 2})]
    )

    results = await rule_def_repo.list_by_set(db, set_a.id)

    assert len(results) == 1
    assert results[0].rule_key == "a.key"


async def test_list_by_set_and_category_filters_by_both_set_and_category(
    db, rule_set_repo, rule_def_repo
):
    rule_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db,
        rule_set.id,
        [
            _rule_dict(RuleCategory.recall_interval, "recall.key", {"v": 1}),
            _rule_dict(RuleCategory.risk_classification, "risk.key", {"v": 2}),
        ],
    )

    results = await rule_def_repo.list_by_set_and_category(db, rule_set.id, "risk_classification")

    assert len(results) == 1
    assert results[0].rule_key == "risk.key"
    assert results[0].category == RuleCategory.risk_classification


# --------------------------------------------------------------------------
# RuleSetRepository.get_diff
# --------------------------------------------------------------------------


async def test_get_diff_returns_a_dict(db, rule_set_repo, rule_def_repo):
    active_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, active_set.id, [_rule_dict(RuleCategory.recall_interval, "shared.key", {"months": 6})]
    )
    await rule_set_repo.activate(db, active_set.id, uuid.uuid4())
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, draft.id, [_rule_dict(RuleCategory.recall_interval, "shared.key", {"months": 6})]
    )

    diff = await rule_set_repo.get_diff(db, draft.id)

    assert isinstance(diff, dict)


async def test_get_diff_surfaces_a_rule_added_only_in_the_draft(db, rule_set_repo, rule_def_repo):
    active_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_set_repo.activate(db, active_set.id, uuid.uuid4())
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, draft.id, [_rule_dict(RuleCategory.recall_interval, "new_only_in_draft.key", {"months": 9})]
    )

    diff = await rule_set_repo.get_diff(db, draft.id)

    assert "new_only_in_draft.key" in json.dumps(diff, default=str)


async def test_get_diff_surfaces_a_rule_removed_relative_to_the_active_set(
    db, rule_set_repo, rule_def_repo
):
    active_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, active_set.id, [_rule_dict(RuleCategory.recall_interval, "only_in_active.key", {"months": 3})]
    )
    await rule_set_repo.activate(db, active_set.id, uuid.uuid4())
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())

    diff = await rule_set_repo.get_diff(db, draft.id)

    assert "only_in_active.key" in json.dumps(diff, default=str)


async def test_get_diff_surfaces_a_changed_rule_value_for_the_same_category_and_key(
    db, rule_set_repo, rule_def_repo
):
    active_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, active_set.id, [_rule_dict(RuleCategory.recall_interval, "shared.key", {"months": 6})]
    )
    await rule_set_repo.activate(db, active_set.id, uuid.uuid4())
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db, draft.id, [_rule_dict(RuleCategory.recall_interval, "shared.key", {"months": 12})]
    )

    diff = await rule_set_repo.get_diff(db, draft.id)
    diff_text = json.dumps(diff, default=str)

    assert "shared.key" in diff_text
    assert "12" in diff_text


async def test_get_diff_compares_by_the_category_and_rule_key_pair_not_rule_key_alone(
    db, rule_set_repo, rule_def_repo
):
    # "compares draft's rule_definitions against the current active set's, by
    # (category, rule_key)": the same rule_key text under two different
    # categories are two distinct rules, not one changed rule.
    active_set = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db,
        active_set.id,
        [_rule_dict(RuleCategory.appointment_type, "shared_key_text", {"marker": "ACTIVE_ONE"})],
    )
    await rule_set_repo.activate(db, active_set.id, uuid.uuid4())
    draft = await rule_set_repo.create_draft(db, uuid.uuid4())
    await rule_def_repo.bulk_insert(
        db,
        draft.id,
        [_rule_dict(RuleCategory.recall_interval, "shared_key_text", {"marker": "DRAFT_ONE"})],
    )

    diff = await rule_set_repo.get_diff(db, draft.id)
    diff_text = json.dumps(diff, default=str)

    # Both the active-only appointment_type rule and the draft-only
    # recall_interval rule must be visible -- neither is silently dropped by
    # treating identical rule_key text alone as "the same rule".
    assert "ACTIVE_ONE" in diff_text
    assert "DRAFT_ONE" in diff_text


# --------------------------------------------------------------------------
# ConfigurationParameterRepository.save_version
# --------------------------------------------------------------------------


async def test_save_version_first_ever_row_for_a_name_is_version_1(db, config_repo):
    staff_id = uuid.uuid4()

    param = await config_repo.save_version(
        db, "outreach.retry_backoff_minutes", {"steps": [5, 15, 60]}, staff_id
    )

    assert isinstance(param, ConfigurationParameter)
    assert param.version_number == 1
    assert param.name == "outreach.retry_backoff_minutes"
    assert param.value == {"steps": [5, 15, 60]}
    assert param.created_by_staff_id == staff_id


async def test_save_version_increments_version_number_for_the_same_name(db, config_repo):
    staff_id = uuid.uuid4()
    await config_repo.save_version(db, "outreach.retry_backoff_minutes", {"steps": [5]}, staff_id)

    second = await config_repo.save_version(
        db, "outreach.retry_backoff_minutes", {"steps": [5, 15]}, staff_id
    )

    assert second.version_number == 2


async def test_save_version_numbering_is_independent_per_name(db, config_repo):
    staff_id = uuid.uuid4()
    await config_repo.save_version(db, "name.a", {"v": 1}, staff_id)
    await config_repo.save_version(db, "name.a", {"v": 2}, staff_id)

    first_for_b = await config_repo.save_version(db, "name.b", {"v": 1}, staff_id)

    assert first_for_b.version_number == 1


async def test_save_version_sets_effective_at_to_a_timezone_aware_utc_timestamp(db, config_repo):
    param = await config_repo.save_version(db, "name.c", {"v": 1}, uuid.uuid4())

    _assert_close_to_now(param.effective_at)


# --------------------------------------------------------------------------
# ConfigurationParameterRepository.get_latest
# --------------------------------------------------------------------------


async def test_get_latest_returns_none_when_no_versions_exist(db, config_repo):
    assert await config_repo.get_latest(db, "name.never_saved") is None


async def test_get_latest_returns_the_highest_version_number_for_that_name(db, config_repo):
    staff_id = uuid.uuid4()
    await config_repo.save_version(db, "name.d", {"v": 1}, staff_id)
    await config_repo.save_version(db, "name.d", {"v": 2}, staff_id)
    third = await config_repo.save_version(db, "name.d", {"v": 3}, staff_id)

    latest = await config_repo.get_latest(db, "name.d")

    assert latest is not None
    assert latest.id == third.id
    assert latest.version_number == 3


async def test_get_latest_is_scoped_to_the_given_name(db, config_repo):
    staff_id = uuid.uuid4()
    await config_repo.save_version(db, "name.e", {"v": 1}, staff_id)
    await config_repo.save_version(db, "name.e", {"v": 2}, staff_id)

    only_one = await config_repo.save_version(db, "name.f", {"v": 99}, staff_id)

    latest_f = await config_repo.get_latest(db, "name.f")
    assert latest_f.id == only_one.id
    assert latest_f.version_number == 1


# --------------------------------------------------------------------------
# ConfigurationParameterRepository.rollback
# --------------------------------------------------------------------------


async def test_rollback_creates_a_new_row_copying_name_and_value_from_the_target_version(
    db, config_repo
):
    creator = uuid.uuid4()
    v1 = await config_repo.save_version(db, "name.rollback_a", {"steps": [5]}, creator)
    await config_repo.save_version(db, "name.rollback_a", {"steps": [5, 15]}, creator)

    rollback_actor = uuid.uuid4()
    restored = await config_repo.rollback(
        db, version_number=v1.version_number, created_by_staff_id=rollback_actor
    )

    assert restored.name == "name.rollback_a"
    assert restored.value == {"steps": [5]}
    assert restored.created_by_staff_id == rollback_actor


async def test_rollback_version_number_is_current_max_plus_one_not_target_plus_one(
    db, config_repo
):
    creator = uuid.uuid4()
    v1 = await config_repo.save_version(db, "name.rollback_b", {"v": "one"}, creator)
    await config_repo.save_version(db, "name.rollback_b", {"v": "two"}, creator)
    await config_repo.save_version(db, "name.rollback_b", {"v": "three"}, creator)  # latest is 3

    restored = await config_repo.rollback(
        db, version_number=v1.version_number, created_by_staff_id=creator
    )

    assert restored.version_number == 4


async def test_rollback_rolled_back_from_id_names_the_version_restored_from(db, config_repo):
    # Watch out: `rolled_back_from_id` names the version being restored FROM
    # (the target of the rollback, matching architecture.md's `{"version_number":
    # 8, "rolled_back_from": 7}` example) -- never "the version being
    # replaced". Rolling back to version 1 while version 2 is latest must
    # point at version 1's row id, never version 2's.
    creator = uuid.uuid4()
    v1 = await config_repo.save_version(db, "name.rollback_c", {"v": "original"}, creator)
    v2 = await config_repo.save_version(db, "name.rollback_c", {"v": "changed"}, creator)

    restored = await config_repo.rollback(db, version_number=1, created_by_staff_id=creator)

    assert restored.rolled_back_from_id == v1.id
    assert restored.rolled_back_from_id != v2.id


async def test_rollback_restored_row_becomes_the_new_latest(db, config_repo):
    await config_repo.save_version(db, "name.rollback_d", {"v": "one"}, uuid.uuid4())
    await config_repo.save_version(db, "name.rollback_d", {"v": "two"}, uuid.uuid4())

    restored = await config_repo.rollback(
        db, version_number=1, created_by_staff_id=uuid.uuid4()
    )

    latest = await config_repo.get_latest(db, "name.rollback_d")
    assert latest.id == restored.id
    assert latest.value == {"v": "one"}
