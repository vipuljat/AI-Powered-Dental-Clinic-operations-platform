"""Unit tests for app/repositories/intelligence/repository.py.

Exercises the five data-access classes declared over `risk_scores`,
`triage_sessions`, `escalations`, `utilisation_recommendations` and
`ml_model_evaluations` directly against an in-memory SQLite database - the
project's test-substitution mechanism per PROJECT.md's Testing section
(ENVIRONMENT=test swaps the DB URL to sqlite+aiosqlite:///:memory: while the
same ORM models run unmodified thanks to app.core.db_types's portable
JSONB/VECTOR TypeDecorators and SQLAlchemy 2.x's cross-dialect Uuid type).
No bespoke mocking of the database is used.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
import pytest_asyncio

# The real composition roots read ENVIRONMENT before ever importing
# app.core.database; set it defensively here too so importing it never tries
# to dial a real, unreachable Postgres server using the class default
# `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from sqlalchemy import Column, Table  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.database import Base  # noqa: E402
from app.models.intelligence.models import (  # noqa: E402
    Escalation,
    MlModelEvaluation,
    RiskScore,
    TriageSession,
    UtilisationRecommendation,
)
from app.repositories.intelligence.repository import (  # noqa: E402
    EscalationRepository,
    MlModelEvaluationRepository,
    RiskScoreRepository,
    TriageSessionRepository,
    UtilisationRecommendationRepository,
)

# risk_scores.appointment_id, triage_sessions.patient_id,
# escalations.call_interaction_id/routed_to_staff_id/resolved_by_staff_id and
# utilisation_recommendations.decided_by_staff_id declare FKs to tables
# (appointments, patients, call_interactions, staff_users) owned by other
# modules' own models.py files that this task has no spec for and must not
# import for its own sake. If those modules happen to already be importable
# (and so already registered on the shared Base.metadata) reuse them as-is;
# otherwise register minimal stand-in tables with just the `id` primary key
# FK resolution needs, purely so this in-memory engine can create *this*
# spec's own five tables.
for _mod in (
    "app.models.scheduling.models",  # appointments
    "app.models.patients.models",  # patients
    "app.models.engagement.models",  # call_interactions
    "app.models.console.models",  # staff_users
):
    try:  # pragma: no cover - exercised implicitly by the fixtures below
        __import__(_mod)
    except ModuleNotFoundError:
        pass

_uuid_type = RiskScore.__table__.c.id.type

for _table_name, _fk_column in (
    ("appointments", RiskScore.__table__.c.appointment_id),
    ("patients", TriageSession.__table__.c.patient_id),
    ("call_interactions", Escalation.__table__.c.call_interaction_id),
    ("staff_users", Escalation.__table__.c.routed_to_staff_id),
):
    if _table_name not in Base.metadata.tables:
        Table(_table_name, Base.metadata, Column("id", _uuid_type, primary_key=True))

_INTELLIGENCE_TABLES = [
    RiskScore.__table__,
    TriageSession.__table__,
    Escalation.__table__,
    UtilisationRecommendation.__table__,
    MlModelEvaluation.__table__,
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
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=_INTELLIGENCE_TABLES)
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
def risk_repo():
    return RiskScoreRepository()


@pytest.fixture
def triage_repo():
    return TriageSessionRepository()


@pytest.fixture
def escalation_repo():
    return EscalationRepository()


@pytest.fixture
def util_repo():
    return UtilisationRecommendationRepository()


@pytest.fixture
def ml_repo():
    return MlModelEvaluationRepository()


def make_risk_fields(**overrides):
    defaults = dict(
        appointment_id=uuid4(),
        score_value=Decimal("0.8500"),
        risk_level="high",
        source="rules",
        model_version=None,
        computed_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return defaults


def make_escalation_fields(**overrides):
    defaults = dict(
        trigger_reason="high_risk_no_show_score",
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return defaults


def make_utilisation_fields(**overrides):
    defaults = dict(
        generated_at=datetime.now(timezone.utc),
        rationale="Chair 3 sits idle every Tuesday afternoon.",
        recommended_change={"chair_id": str(uuid4()), "action": "reassign_provider"},
    )
    defaults.update(overrides)
    return defaults


def make_ml_eval_fields(**overrides):
    defaults = dict(
        model_version="v1.2.0",
        precision_value=Decimal("0.9100"),
        evaluated_at=datetime.now(timezone.utc),
        data_volume_months=6,
        status="passing",
    )
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------
# RiskScoreRepository
# --------------------------------------------------------------------------


class TestRiskScoreRepository:
    async def test_create_persists_and_returns_given_fields(self, db, risk_repo):
        appointment_id = uuid4()
        computed_at = datetime.now(timezone.utc)

        result = await risk_repo.create(
            db,
            **make_risk_fields(
                appointment_id=appointment_id,
                score_value=Decimal("0.7000"),
                risk_level="medium",
                source="ml",
                model_version="v1.0.0",
                computed_at=computed_at,
            ),
        )

        assert isinstance(result, RiskScore)
        assert result.id is not None
        assert result.appointment_id == appointment_id
        assert result.risk_level == "medium"
        assert result.source == "ml"
        assert result.model_version == "v1.0.0"

    async def test_create_allows_model_version_to_be_none_for_rules_source(self, db, risk_repo):
        result = await risk_repo.create(
            db, **make_risk_fields(source="rules", model_version=None)
        )
        assert result.model_version is None
        assert result.source == "rules"

    async def test_get_latest_for_appointment_returns_none_when_no_scores_exist(
        self, db, risk_repo
    ):
        assert await risk_repo.get_latest_for_appointment(db, uuid4()) is None

    async def test_get_latest_for_appointment_returns_most_recently_computed_score(
        self, db, risk_repo
    ):
        appointment_id = uuid4()
        older = datetime(2026, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 3, 1, tzinfo=timezone.utc)

        await risk_repo.create(
            db, **make_risk_fields(appointment_id=appointment_id, risk_level="low", computed_at=older)
        )
        await risk_repo.create(
            db, **make_risk_fields(appointment_id=appointment_id, risk_level="high", computed_at=newer)
        )

        latest = await risk_repo.get_latest_for_appointment(db, appointment_id)

        assert latest is not None
        assert latest.risk_level == "high"
        assert latest.computed_at == newer

    async def test_get_latest_for_appointment_ignores_other_appointments(self, db, risk_repo):
        target = uuid4()
        other = uuid4()
        await risk_repo.create(db, **make_risk_fields(appointment_id=target, risk_level="low"))
        await risk_repo.create(db, **make_risk_fields(appointment_id=other, risk_level="high"))

        latest = await risk_repo.get_latest_for_appointment(db, target)

        assert latest is not None
        assert latest.risk_level == "low"

    async def test_get_distribution_counts_only_latest_score_per_appointment_in_range(
        self, db, risk_repo
    ):
        window_start = datetime(2026, 2, 1, tzinfo=timezone.utc)
        window_end = datetime(2026, 2, 28, tzinfo=timezone.utc)

        appt_a = uuid4()
        appt_b = uuid4()
        appt_outside = uuid4()

        # Appointment A was re-scored: an older "low" score, superseded by a
        # newer "high" score that falls inside the window -- only the
        # latter should count.
        await risk_repo.create(
            db,
            **make_risk_fields(
                appointment_id=appt_a,
                risk_level="low",
                computed_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            ),
        )
        await risk_repo.create(
            db,
            **make_risk_fields(
                appointment_id=appt_a,
                risk_level="high",
                computed_at=datetime(2026, 2, 10, tzinfo=timezone.utc),
            ),
        )
        # Appointment B: a single "medium" score inside the window.
        await risk_repo.create(
            db,
            **make_risk_fields(
                appointment_id=appt_b,
                risk_level="medium",
                computed_at=datetime(2026, 2, 15, tzinfo=timezone.utc),
            ),
        )
        # Appointment outside the window entirely.
        await risk_repo.create(
            db,
            **make_risk_fields(
                appointment_id=appt_outside,
                risk_level="high",
                computed_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        )

        distribution = await risk_repo.get_distribution(db, window_start, window_end)

        assert distribution.get("high", 0) == 1
        assert distribution.get("medium", 0) == 1
        assert distribution.get("low", 0) == 0
        assert sum(distribution.values()) == 2

    async def test_get_distribution_without_date_bounds_includes_every_appointment(
        self, db, risk_repo
    ):
        await risk_repo.create(
            db,
            **make_risk_fields(
                risk_level="low", computed_at=datetime(2020, 1, 1, tzinfo=timezone.utc)
            ),
        )
        await risk_repo.create(
            db,
            **make_risk_fields(
                risk_level="high", computed_at=datetime(2030, 1, 1, tzinfo=timezone.utc)
            ),
        )

        distribution = await risk_repo.get_distribution(db, None, None)

        assert distribution.get("low", 0) == 1
        assert distribution.get("high", 0) == 1


# --------------------------------------------------------------------------
# TriageSessionRepository
# --------------------------------------------------------------------------


class TestTriageSessionRepository:
    async def test_create_persists_channel_and_patient_id(self, db, triage_repo):
        patient_id = uuid4()
        session = await triage_repo.create(db, channel="webchat", patient_id=patient_id)

        assert isinstance(session, TriageSession)
        assert session.id is not None
        assert session.channel == "webchat"
        assert session.patient_id == patient_id

    async def test_create_allows_patient_id_none_for_unverified_visitor(self, db, triage_repo):
        session = await triage_repo.create(db, channel="voice", patient_id=None)
        assert session.patient_id is None
        assert session.channel == "voice"

    async def test_create_starts_with_empty_responses_and_no_outcome(self, db, triage_repo):
        session = await triage_repo.create(db, channel="webchat", patient_id=None)

        assert session.responses == {}
        assert session.urgency_classification is None
        assert session.recommended_block is None
        assert session.escalated is False

    async def test_get_by_id_returns_none_for_unknown_id(self, db, triage_repo):
        assert await triage_repo.get_by_id(db, uuid4()) is None

    async def test_get_by_id_returns_created_session(self, db, triage_repo):
        created = await triage_repo.create(db, channel="webchat", patient_id=None)
        fetched = await triage_repo.get_by_id(db, created.id)

        assert fetched is not None
        assert fetched.id == created.id

    async def test_append_response_merges_new_key_into_responses(self, db, triage_repo):
        created = await triage_repo.create(db, channel="webchat", patient_id=None)

        updated = await triage_repo.append_response(
            db, created.id, "pain_level", "severe"
        )

        assert updated.responses == {"pain_level": "severe"}

    async def test_append_response_preserves_previously_stored_answers(self, db, triage_repo):
        created = await triage_repo.create(db, channel="webchat", patient_id=None)

        await triage_repo.append_response(db, created.id, "pain_level", "severe")
        second = await triage_repo.append_response(db, created.id, "swelling", "yes")

        assert second.responses == {"pain_level": "severe", "swelling": "yes"}

        reloaded = await triage_repo.get_by_id(db, created.id)
        assert reloaded.responses == {"pain_level": "severe", "swelling": "yes"}

    async def test_append_response_overwrites_same_question_key(self, db, triage_repo):
        created = await triage_repo.create(db, channel="webchat", patient_id=None)

        await triage_repo.append_response(db, created.id, "pain_level", "mild")
        updated = await triage_repo.append_response(db, created.id, "pain_level", "severe")

        assert updated.responses == {"pain_level": "severe"}

    async def test_set_outcome_persists_classification_and_escalated_flag(self, db, triage_repo):
        created = await triage_repo.create(db, channel="voice", patient_id=None)

        updated = await triage_repo.set_outcome(
            db,
            created.id,
            urgency_classification="emergency",
            recommended_block="urgent_care",
            escalated=True,
        )

        assert updated.urgency_classification == "emergency"
        assert updated.recommended_block == "urgent_care"
        assert updated.escalated is True

        reloaded = await triage_repo.get_by_id(db, created.id)
        assert reloaded.escalated is True
        assert reloaded.urgency_classification == "emergency"

    async def test_set_outcome_allows_non_escalated_outcome_with_null_block(self, db, triage_repo):
        created = await triage_repo.create(db, channel="webchat", patient_id=None)

        updated = await triage_repo.set_outcome(
            db, created.id, urgency_classification="routine", recommended_block=None, escalated=False
        )

        assert updated.urgency_classification == "routine"
        assert updated.recommended_block is None
        assert updated.escalated is False

    async def test_update_fields_applies_arbitrary_staff_correction(self, db, triage_repo):
        # Interaction contract: services/console/service.py's OverrideService
        # depends directly on this method to correct an AI-originated triage
        # outcome.
        created = await triage_repo.create(db, channel="webchat", patient_id=None)
        await triage_repo.set_outcome(
            db, created.id, urgency_classification="emergency", recommended_block="urgent_care", escalated=True
        )

        corrected = await triage_repo.update_fields(
            db, created.id, urgency_classification="routine", escalated=False
        )

        assert corrected.urgency_classification == "routine"
        assert corrected.escalated is False

        reloaded = await triage_repo.get_by_id(db, created.id)
        assert reloaded.urgency_classification == "routine"
        assert reloaded.escalated is False


# --------------------------------------------------------------------------
# EscalationRepository
# --------------------------------------------------------------------------


class TestEscalationRepository:
    async def test_create_persists_given_fields(self, db, escalation_repo):
        triage_session_id = uuid4()
        result = await escalation_repo.create(
            db, **make_escalation_fields(triage_session_id=triage_session_id, trigger_reason="ai_low_confidence")
        )

        assert isinstance(result, Escalation)
        assert result.id is not None
        assert result.triage_session_id == triage_session_id
        assert result.trigger_reason == "ai_low_confidence"

    async def test_create_defaults_status_to_unacknowledged_when_omitted(self, db, escalation_repo):
        result = await escalation_repo.create(db, **make_escalation_fields())
        assert result.status == "unacknowledged"

    async def test_get_by_id_returns_none_for_unknown_id(self, db, escalation_repo):
        assert await escalation_repo.get_by_id(db, uuid4()) is None

    async def test_get_by_id_returns_created_escalation(self, db, escalation_repo):
        created = await escalation_repo.create(db, **make_escalation_fields())
        fetched = await escalation_repo.get_by_id(db, created.id)

        assert fetched is not None
        assert fetched.id == created.id

    async def test_acknowledge_sets_status_staff_and_timestamp(self, db, escalation_repo):
        created = await escalation_repo.create(
            db, **make_escalation_fields(status="unacknowledged")
        )
        staff_id = uuid4()

        acknowledged = await escalation_repo.acknowledge(db, created.id, staff_id)

        assert acknowledged.status == "acknowledged"
        assert acknowledged.routed_to_staff_id == staff_id
        assert acknowledged.acknowledged_at is not None

        reloaded = await escalation_repo.get_by_id(db, created.id)
        assert reloaded.status == "acknowledged"
        assert reloaded.routed_to_staff_id == staff_id

    async def test_resolve_sets_status_notes_and_timestamp(self, db, escalation_repo):
        created = await escalation_repo.create(
            db, **make_escalation_fields(status="acknowledged")
        )
        staff_id = uuid4()

        resolved = await escalation_repo.resolve(
            db, created.id, staff_id, "Called patient back, rescheduled for tomorrow."
        )

        assert resolved.status == "resolved"
        assert resolved.resolution_notes == "Called patient back, rescheduled for tomorrow."

        reloaded = await escalation_repo.get_by_id(db, created.id)
        assert reloaded.status == "resolved"
        assert reloaded.resolution_notes == "Called patient back, rescheduled for tomorrow."

    async def test_acknowledge_performs_write_unconditionally_even_out_of_order(
        self, db, escalation_repo
    ):
        # FR-E8.5/SFR001: the repository itself does not guard the
        # unacknowledged -> acknowledged -> resolved ordering -- that is the
        # caller's (EscalationService's) responsibility. Calling acknowledge
        # on an already-resolved row still performs the write.
        created = await escalation_repo.create(db, **make_escalation_fields(status="resolved"))
        staff_id = uuid4()

        result = await escalation_repo.acknowledge(db, created.id, staff_id)

        assert result.status == "acknowledged"
        assert result.routed_to_staff_id == staff_id

    async def test_resolve_performs_write_unconditionally_even_out_of_order(
        self, db, escalation_repo
    ):
        # Same guarantee for resolve(): calling it directly from
        # "unacknowledged" (skipping "acknowledged") still writes the row.
        created = await escalation_repo.create(db, **make_escalation_fields(status="unacknowledged"))
        staff_id = uuid4()

        result = await escalation_repo.resolve(db, created.id, staff_id, "Handled directly.")

        assert result.status == "resolved"
        assert result.resolution_notes == "Handled directly."

    async def test_list_unacknowledged_returns_only_unacknowledged_rows(self, db, escalation_repo):
        await escalation_repo.create(db, **make_escalation_fields(status="unacknowledged"))
        acked = await escalation_repo.create(db, **make_escalation_fields(status="unacknowledged"))
        await escalation_repo.acknowledge(db, acked.id, uuid4())
        await escalation_repo.create(db, **make_escalation_fields(status="resolved"))

        unacked = await escalation_repo.list_unacknowledged(db)

        assert len(unacked) == 1
        assert unacked[0].status == "unacknowledged"

    async def test_list_unacknowledged_returns_empty_list_when_none_pending(self, db, escalation_repo):
        await escalation_repo.create(db, **make_escalation_fields(status="resolved"))
        assert await escalation_repo.list_unacknowledged(db) == []


# --------------------------------------------------------------------------
# UtilisationRecommendationRepository
# --------------------------------------------------------------------------


class TestUtilisationRecommendationRepository:
    async def test_create_persists_given_fields(self, db, util_repo):
        change = {"chair_id": str(uuid4()), "action": "reassign_provider"}
        result = await util_repo.create(
            db, **make_utilisation_fields(rationale="Reassign chair 2 on Fridays.", recommended_change=change)
        )

        assert isinstance(result, UtilisationRecommendation)
        assert result.id is not None
        assert result.rationale == "Reassign chair 2 on Fridays."
        assert result.recommended_change == change

    async def test_create_defaults_status_to_pending_when_omitted(self, db, util_repo):
        result = await util_repo.create(db, **make_utilisation_fields())
        assert result.status == "pending"

    async def test_get_by_id_returns_none_for_unknown_id(self, db, util_repo):
        assert await util_repo.get_by_id(db, uuid4()) is None

    async def test_get_by_id_returns_created_recommendation(self, db, util_repo):
        created = await util_repo.create(db, **make_utilisation_fields())
        fetched = await util_repo.get_by_id(db, created.id)

        assert fetched is not None
        assert fetched.id == created.id

    async def test_list_by_status_filters_correctly(self, db, util_repo):
        await util_repo.create(db, **make_utilisation_fields(status="pending"))
        applied = await util_repo.create(db, **make_utilisation_fields(status="applied"))

        pending_only = await util_repo.list_by_status(db, "pending")
        applied_only = await util_repo.list_by_status(db, "applied")

        assert len(pending_only) == 1
        assert pending_only[0].status == "pending"
        assert len(applied_only) == 1
        assert applied_only[0].id == applied.id

    async def test_update_status_persists_status_and_decided_by_staff(self, db, util_repo):
        created = await util_repo.create(db, **make_utilisation_fields(status="pending"))
        staff_id = uuid4()

        updated = await util_repo.update_status(db, created.id, "applied", staff_id)

        assert updated.status == "applied"
        assert updated.decided_by_staff_id == staff_id

        reloaded = await util_repo.get_by_id(db, created.id)
        assert reloaded.status == "applied"
        assert reloaded.decided_by_staff_id == staff_id

    async def test_get_adoption_rate_divides_applied_by_applied_plus_dismissed(
        self, db, util_repo
    ):
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        after_since = since + timedelta(days=5)

        await util_repo.create(
            db, **make_utilisation_fields(status="applied", generated_at=after_since)
        )
        await util_repo.create(
            db, **make_utilisation_fields(status="applied", generated_at=after_since)
        )
        await util_repo.create(
            db, **make_utilisation_fields(status="dismissed", generated_at=after_since)
        )

        rate = await util_repo.get_adoption_rate(db, since)

        assert rate == pytest.approx(2 / 3)

    async def test_get_adoption_rate_excludes_pending_recommendations_from_denominator(
        self, db, util_repo
    ):
        # FR-E8.7/US45: an unreviewed backlog of "pending" rows must not
        # depress the reported adoption rate -- they are excluded from both
        # numerator and denominator.
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)
        after_since = since + timedelta(days=1)

        await util_repo.create(
            db, **make_utilisation_fields(status="applied", generated_at=after_since)
        )
        for _ in range(5):
            await util_repo.create(
                db, **make_utilisation_fields(status="pending", generated_at=after_since)
            )

        rate = await util_repo.get_adoption_rate(db, since)

        assert rate == pytest.approx(1.0)

    async def test_get_adoption_rate_excludes_rows_generated_before_since(self, db, util_repo):
        since = datetime(2026, 6, 1, tzinfo=timezone.utc)
        before_since = since - timedelta(days=30)
        after_since = since + timedelta(days=1)

        await util_repo.create(
            db, **make_utilisation_fields(status="dismissed", generated_at=before_since)
        )
        await util_repo.create(
            db, **make_utilisation_fields(status="applied", generated_at=after_since)
        )

        rate = await util_repo.get_adoption_rate(db, since)

        assert rate == pytest.approx(1.0)


# --------------------------------------------------------------------------
# MlModelEvaluationRepository
# --------------------------------------------------------------------------


class TestMlModelEvaluationRepository:
    async def test_create_persists_given_fields(self, db, ml_repo):
        result = await ml_repo.create(
            db, **make_ml_eval_fields(model_version="v2.0.0", status="underperforming")
        )

        assert isinstance(result, MlModelEvaluation)
        assert result.id is not None
        assert result.model_version == "v2.0.0"
        assert result.status == "underperforming"

    async def test_get_latest_returns_none_when_no_evaluations_exist(self, db, ml_repo):
        assert await ml_repo.get_latest(db) is None

    async def test_get_latest_returns_most_recently_evaluated_row(self, db, ml_repo):
        older = datetime(2026, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 5, 1, tzinfo=timezone.utc)

        await ml_repo.create(
            db, **make_ml_eval_fields(model_version="v1.0.0", evaluated_at=older, status="passing")
        )
        await ml_repo.create(
            db, **make_ml_eval_fields(model_version="v2.0.0", evaluated_at=newer, status="underperforming")
        )

        latest = await ml_repo.get_latest(db)

        assert latest is not None
        assert latest.model_version == "v2.0.0"
        assert latest.evaluated_at == newer
        assert latest.status == "underperforming"
