"""Unit tests for app/models/intelligence/models.py.

These tests exercise the SQLAlchemy ORM declarations directly (no HTTP, no
routes/services). They spin up an isolated in-memory SQLite database per test
via the same async-SQLAlchemy stack the app uses in ENVIRONMENT=test, create
only the tables declared by this module's ``Base.metadata`` (since this test
file imports nothing else), and assert on the declared column shape, defaults,
NOT NULL constraints and JSONB round-tripping described in the spec.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.intelligence.models import (
    Escalation,
    MlModelEvaluation,
    RiskScore,
    TriageSession,
    UtilisationRecommendation,
)

# This test file exercises only app/models/intelligence/models.py in isolation
# (per the unit-test-authoring contract, no other module's spec/code is
# available here). Several of its columns are plain FK columns pointing at
# other modules' tables (appointments, patients, staff_users,
# call_interactions) per the Interaction contract ("no cross-module ORM
# relationship()"). Those tables are not registered on Base.metadata unless
# their own models.py has also been imported, which DDL creation needs in
# order to resolve the FK's target table. Register minimal stand-in tables
# (id column only) so `create_all` can order/create *this* module's tables
# without requiring any other module's implementation to exist.
for _name in ("appointments", "patients", "staff_users", "call_interactions"):
    if _name not in Base.metadata.tables:
        Table(_name, Base.metadata, Column("id", Uuid(), primary_key=True))


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with maker() as s:
        yield s

    await engine.dispose()


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# factories for minimal-valid rows (overridable per-test)
# --------------------------------------------------------------------------


def make_risk_score(**overrides):
    values = dict(
        id=uuid.uuid4(),
        appointment_id=uuid.uuid4(),
        score_value=Decimal("0.8765"),
        risk_level="high",
        source="rules",
        model_version=None,
        computed_at=_now(),
    )
    values.update(overrides)
    return RiskScore(**values)


def make_triage_session(**overrides):
    values = dict(
        id=uuid.uuid4(),
        patient_id=uuid.uuid4(),
        channel="webchat",
        responses={"q1": "yes", "q2": "no"},
        urgency_classification=None,
        recommended_block=None,
        created_at=_now(),
    )
    values.update(overrides)
    return TriageSession(**values)


def make_escalation(**overrides):
    values = dict(
        id=uuid.uuid4(),
        triage_session_id=uuid.uuid4(),
        call_interaction_id=None,
        trigger_reason="high_risk_score",
        routed_to_staff_id=None,
        resolved_by_staff_id=None,
        acknowledged_at=None,
        resolved_at=None,
        resolution_notes=None,
        created_at=_now(),
    )
    values.update(overrides)
    return Escalation(**values)


def make_utilisation_recommendation(**overrides):
    values = dict(
        id=uuid.uuid4(),
        generated_at=_now(),
        rationale="Chair 2 idle 3 afternoons/week; move hygiene block earlier.",
        recommended_change={"chair_id": str(uuid.uuid4()), "action": "shift_block"},
        decided_by_staff_id=None,
        decided_at=None,
    )
    values.update(overrides)
    return UtilisationRecommendation(**values)


def make_ml_model_evaluation(**overrides):
    values = dict(
        id=uuid.uuid4(),
        model_version="v1.2.0",
        precision_value=Decimal("0.9123"),
        evaluated_at=_now(),
        data_volume_months=6,
        status="passing",
    )
    values.update(overrides)
    return MlModelEvaluation(**values)


# --------------------------------------------------------------------------
# table names / public surface
# --------------------------------------------------------------------------


def test_tablenames():
    assert RiskScore.__tablename__ == "risk_scores"
    assert TriageSession.__tablename__ == "triage_sessions"
    assert Escalation.__tablename__ == "escalations"
    assert UtilisationRecommendation.__tablename__ == "utilisation_recommendations"
    assert MlModelEvaluation.__tablename__ == "ml_model_evaluations"


# --------------------------------------------------------------------------
# declared column shape (Data shapes tables)
# --------------------------------------------------------------------------


def test_risk_score_declared_columns_and_nullability():
    cols = {c.name: c for c in RiskScore.__table__.columns}
    assert set(cols) == {
        "id",
        "appointment_id",
        "score_value",
        "risk_level",
        "source",
        "model_version",
        "computed_at",
    }
    assert cols["appointment_id"].nullable is False
    assert cols["score_value"].nullable is False
    assert cols["risk_level"].nullable is False
    assert cols["source"].nullable is False
    assert cols["model_version"].nullable is True
    assert cols["computed_at"].nullable is False


def test_triage_session_declared_columns_and_nullability():
    cols = {c.name: c for c in TriageSession.__table__.columns}
    assert set(cols) == {
        "id",
        "patient_id",
        "channel",
        "responses",
        "urgency_classification",
        "recommended_block",
        "escalated",
        "created_at",
    }
    assert cols["patient_id"].nullable is True
    assert cols["channel"].nullable is False
    assert cols["responses"].nullable is False
    assert cols["urgency_classification"].nullable is True
    assert cols["recommended_block"].nullable is True
    assert cols["escalated"].nullable is False
    assert cols["created_at"].nullable is False


def test_escalation_declared_columns_and_nullability():
    cols = {c.name: c for c in Escalation.__table__.columns}
    assert set(cols) == {
        "id",
        "triage_session_id",
        "call_interaction_id",
        "trigger_reason",
        "status",
        "routed_to_staff_id",
        "resolved_by_staff_id",
        "acknowledged_at",
        "resolved_at",
        "resolution_notes",
        "created_at",
    }
    assert cols["triage_session_id"].nullable is True
    assert cols["call_interaction_id"].nullable is True
    assert cols["trigger_reason"].nullable is False
    assert cols["status"].nullable is False
    assert cols["routed_to_staff_id"].nullable is True
    assert cols["resolved_by_staff_id"].nullable is True
    assert cols["acknowledged_at"].nullable is True
    assert cols["resolved_at"].nullable is True
    assert cols["resolution_notes"].nullable is True
    assert cols["created_at"].nullable is False


def test_utilisation_recommendation_declared_columns_and_nullability():
    cols = {c.name: c for c in UtilisationRecommendation.__table__.columns}
    assert set(cols) == {
        "id",
        "generated_at",
        "rationale",
        "recommended_change",
        "status",
        "decided_by_staff_id",
        "decided_at",
    }
    assert cols["generated_at"].nullable is False
    assert cols["rationale"].nullable is False
    assert cols["recommended_change"].nullable is False
    assert cols["status"].nullable is False
    assert cols["decided_by_staff_id"].nullable is True
    assert cols["decided_at"].nullable is True


def test_ml_model_evaluation_declared_columns_and_nullability():
    cols = {c.name: c for c in MlModelEvaluation.__table__.columns}
    assert set(cols) == {
        "id",
        "model_version",
        "precision_value",
        "evaluated_at",
        "data_volume_months",
        "status",
    }
    assert cols["model_version"].nullable is False
    assert cols["precision_value"].nullable is False
    assert cols["evaluated_at"].nullable is False
    assert cols["data_volume_months"].nullable is False
    assert cols["status"].nullable is False


# --------------------------------------------------------------------------
# no cross-module relationship() attributes (Interaction contract)
# --------------------------------------------------------------------------


def test_risk_score_has_no_appointment_relationship():
    # only the plain FK column may exist, no relationship() to Appointment
    assert "appointment_id" in RiskScore.__table__.columns
    assert not hasattr(RiskScore, "appointment")


def test_triage_session_has_no_patient_relationship():
    assert "patient_id" in TriageSession.__table__.columns
    assert not hasattr(TriageSession, "patient")


def test_escalation_has_no_cross_module_relationships():
    assert not hasattr(Escalation, "triage_session")
    assert not hasattr(Escalation, "call_interaction")
    assert not hasattr(Escalation, "routed_to_staff")
    assert not hasattr(Escalation, "resolved_by_staff")


def test_utilisation_recommendation_has_no_staff_relationship():
    assert not hasattr(UtilisationRecommendation, "decided_by_staff")


# --------------------------------------------------------------------------
# risk_scores behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_score_persists_and_roundtrips_fields(session):
    row = make_risk_score(risk_level="medium", source="ml", model_version="v3")
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(RiskScore).where(RiskScore.id == row.id))).scalar_one()
    assert fetched.risk_level == "medium" or getattr(fetched.risk_level, "value", None) == "medium"
    assert fetched.source == "ml" or getattr(fetched.source, "value", None) == "ml"
    assert fetched.model_version == "v3"
    assert Decimal(str(fetched.score_value)) == Decimal("0.8765")


@pytest.mark.asyncio
async def test_risk_score_missing_computed_at_raises_integrity_error(session):
    row = make_risk_score(computed_at=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.asyncio
async def test_risk_score_missing_appointment_id_raises_integrity_error(session):
    row = make_risk_score(appointment_id=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


# --------------------------------------------------------------------------
# triage_sessions behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_session_responses_is_structured_json_roundtrip(session):
    structured = {"q1": {"choice": "yes"}, "q2": {"choice": "no"}}
    row = make_triage_session(responses=structured)
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(select(TriageSession).where(TriageSession.id == row.id))
    ).scalar_one()
    assert fetched.responses == structured
    assert isinstance(fetched.responses, dict)


@pytest.mark.asyncio
async def test_triage_session_escalated_defaults_false(session):
    row = make_triage_session()
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(select(TriageSession).where(TriageSession.id == row.id))
    ).scalar_one()
    assert fetched.escalated is False


@pytest.mark.asyncio
async def test_triage_session_patient_id_nullable_for_unverified_visitor(session):
    row = make_triage_session(patient_id=None, channel="voice")
    session.add(row)
    # must not raise: patient_id is nullable for an unverified visitor
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(select(TriageSession).where(TriageSession.id == row.id))
    ).scalar_one()
    assert fetched.patient_id is None


# NOTE: a "missing responses raises IntegrityError" case was intentionally not
# written here: `responses` is stored through a JSONB TypeDecorator, and
# whether a Python `None` is encoded as a JSON `null` literal (satisfying the
# NOT NULL column constraint) or passed through as a real SQL NULL (violating
# it) is an internal encoding choice the spec does not state, so it is not
# reliably observable through this file's declared public surface alone.


# --------------------------------------------------------------------------
# escalations behaviour (FR-E8.5 / SFR001)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_status_defaults_to_unacknowledged(session):
    row = make_escalation()
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(Escalation).where(Escalation.id == row.id))).scalar_one()
    status = getattr(fetched.status, "value", fetched.status)
    assert status == "unacknowledged"


@pytest.mark.asyncio
async def test_escalation_status_supports_the_forward_states(session):
    for status_value in ("unacknowledged", "acknowledged", "resolved"):
        row = make_escalation(id=uuid.uuid4(), status=status_value)
        session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(Escalation))).scalars().all()
    persisted = {getattr(r.status, "value", r.status) for r in fetched}
    assert persisted == {"unacknowledged", "acknowledged", "resolved"}


@pytest.mark.asyncio
async def test_escalation_missing_trigger_reason_raises_integrity_error(session):
    row = make_escalation(trigger_reason=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.asyncio
async def test_escalation_allows_both_triage_session_and_call_interaction_to_be_null(session):
    # both FKs are nullable individually (one escalation source, not both required)
    row = make_escalation(triage_session_id=None, call_interaction_id=uuid.uuid4())
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(Escalation).where(Escalation.id == row.id))).scalar_one()
    assert fetched.triage_session_id is None
    assert fetched.call_interaction_id is not None


# --------------------------------------------------------------------------
# utilisation_recommendations behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_utilisation_recommendation_status_defaults_to_pending(session):
    row = make_utilisation_recommendation()
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(
            select(UtilisationRecommendation).where(UtilisationRecommendation.id == row.id)
        )
    ).scalar_one()
    status = getattr(fetched.status, "value", fetched.status)
    assert status == "pending"


@pytest.mark.asyncio
async def test_utilisation_recommendation_recommended_change_json_roundtrip(session):
    change = {"chair_id": "chair-9", "action": "add_hygiene_block", "day": "Wednesday"}
    row = make_utilisation_recommendation(recommended_change=change)
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(
            select(UtilisationRecommendation).where(UtilisationRecommendation.id == row.id)
        )
    ).scalar_one()
    assert fetched.recommended_change == change


@pytest.mark.asyncio
async def test_utilisation_recommendation_missing_rationale_raises_integrity_error(session):
    row = make_utilisation_recommendation(rationale=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


# --------------------------------------------------------------------------
# ml_model_evaluations behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ml_model_evaluation_persists_and_roundtrips_fields(session):
    row = make_ml_model_evaluation(status="underperforming", data_volume_months=3)
    session.add(row)
    await session.commit()
    session.expunge_all()

    fetched = (
        await session.execute(select(MlModelEvaluation).where(MlModelEvaluation.id == row.id))
    ).scalar_one()
    status = getattr(fetched.status, "value", fetched.status)
    assert status == "underperforming"
    assert fetched.data_volume_months == 3
    assert Decimal(str(fetched.precision_value)) == Decimal("0.9123")


@pytest.mark.asyncio
async def test_ml_model_evaluation_missing_status_raises_integrity_error(session):
    row = make_ml_model_evaluation(status=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()


@pytest.mark.asyncio
async def test_ml_model_evaluation_missing_model_version_raises_integrity_error(session):
    row = make_ml_model_evaluation(model_version=None)
    session.add(row)
    with pytest.raises(IntegrityError):
        await session.commit()
