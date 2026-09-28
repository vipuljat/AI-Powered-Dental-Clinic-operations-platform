"""SQLAlchemy ORM declarations for the intelligence module's five tables
(architecture.md §4.2): ``risk_scores``, ``triage_sessions``, ``escalations``,
``utilisation_recommendations``, ``ml_model_evaluations``.

Pure schema — no query methods, no business logic.

Interaction contract: ``appointment_id``/``triage_session_id``/
``call_interaction_id``/``routed_to_staff_id``/``resolved_by_staff_id``/
``decided_by_staff_id`` are plain ``UUID`` foreign-key columns only — no
cross-module ORM ``relationship()`` is declared here. Only
``repositories/intelligence/repository.py`` imports these classes directly;
``services/console/service.py``'s ``OverrideService`` depends on
``TriageSessionRepository`` (never these ORM classes) per the console
layering rule, and ``services/analytics/service.py`` depends on
``RiskScoreRepository``/``UtilisationRecommendationRepository`` for
read-only aggregation, again never these ORM classes directly.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import (
    Channel,
    EscalationStatus,
    MlEvaluationStatus,
    RiskLevel,
    ScoreSource,
    UtilisationRecommendationStatus,
)
from app.core.database import Base
from app.core.db_types import JSONBType


class RiskScore(Base):
    """A single computed no-show/risk score for one appointment, produced
    either by the rules engine or the ML model (architecture.md §4.2)."""

    __tablename__ = "risk_scores"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    appointment_id: Mapped[UUID] = mapped_column(ForeignKey("appointments.id"), nullable=False)
    score_value: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    risk_level: Mapped[RiskLevel] = mapped_column(SAEnum(RiskLevel, native_enum=False), nullable=False)
    source: Mapped[ScoreSource] = mapped_column(SAEnum(ScoreSource, native_enum=False), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TriageSession(Base):
    """A patient/visitor symptom-triage session captured over webchat or
    voice.

    RR002-safe: ``responses`` is ``JSONB`` holding only structured
    multiple-choice answers — never a free-text field for patient-entered
    symptoms, so no free-text diagnosis is ever stored here.
    """

    __tablename__ = "triage_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Null for an unverified visitor.
    patient_id: Mapped[UUID | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    channel: Mapped[Channel] = mapped_column(SAEnum(Channel, native_enum=False), nullable=False)
    # RR002-safe: structured multiple-choice answers only, never free-text.
    responses: Mapped[dict] = mapped_column(JSONBType, nullable=False)
    urgency_classification: Mapped[str | None] = mapped_column(String(50), nullable=True)
    recommended_block: Mapped[str | None] = mapped_column(String(100), nullable=True)
    escalated: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Escalation(Base):
    """FR-E8.5/SFR001: an in-console alert raised for a triage session or
    call interaction that requires staff attention.

    ``status`` defaults to ``unacknowledged`` and only ever transitions
    forward (``unacknowledged`` -> ``acknowledged`` -> ``resolved``) — no
    route/service in this tree exposes a way to move it backward; that
    forward-only invariant is enforced by the service layer, this file only
    declares the default.
    """

    __tablename__ = "escalations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    triage_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("triage_sessions.id"), nullable=True
    )
    call_interaction_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("call_interactions.id"), nullable=True
    )
    trigger_reason: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[EscalationStatus] = mapped_column(
        SAEnum(EscalationStatus, native_enum=False),
        nullable=False,
        default=EscalationStatus.unacknowledged,
        server_default=EscalationStatus.unacknowledged.value,
    )
    routed_to_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    resolved_by_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_notes: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class UtilisationRecommendation(Base):
    """A system-generated scheduling/utilisation recommendation awaiting a
    clinic_management decision (apply/dismiss)."""

    __tablename__ = "utilisation_recommendations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rationale: Mapped[str] = mapped_column(String(1000), nullable=False)
    recommended_change: Mapped[dict] = mapped_column(JSONBType, nullable=False)
    status: Mapped[UtilisationRecommendationStatus] = mapped_column(
        SAEnum(UtilisationRecommendationStatus, native_enum=False),
        nullable=False,
        default=UtilisationRecommendationStatus.pending,
        server_default=UtilisationRecommendationStatus.pending.value,
    )
    decided_by_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MlModelEvaluation(Base):
    """A monthly (per SM4) precision evaluation of the ML risk-scoring
    model, driving the passing/underperforming rollback decision."""

    __tablename__ = "ml_model_evaluations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    precision_value: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    # Monthly per SM4.
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data_volume_months: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[MlEvaluationStatus] = mapped_column(
        SAEnum(MlEvaluationStatus, native_enum=False), nullable=False
    )
