"""Data-access classes over ``risk_scores``, ``triage_sessions``,
``escalations``, ``utilisation_recommendations`` and ``ml_model_evaluations``
(architecture.md §4.2).

Interaction contract: ``TriageSessionRepository.update_fields`` is the method
``services/console/service.py``'s ``OverrideService`` depends on directly (a
cross-module repository dependency permitted by the console layering rule) to
apply a staff correction to an AI-originated triage classification —
``services/console/service.py`` never imports the ``TriageSession`` ORM class
or this module's other repositories directly for that purpose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import EscalationStatus, UtilisationRecommendationStatus
from app.common.utils import now_utc
from app.models.intelligence.models import (
    Escalation,
    MlModelEvaluation,
    RiskScore,
    TriageSession,
    UtilisationRecommendation,
)


def _to_utc(dt: datetime | None) -> datetime | None:
    """Restore an explicit UTC offset on a naive ``datetime``.

    The `sqlite+aiosqlite:///:memory:` DB used when ``ENVIRONMENT=test``
    (app/core/config.py) has no native timezone-aware storage, so a
    ``DateTime(timezone=True)`` column round-trips as a naive value even
    though every timestamp in this system is UTC by convention
    (architecture.md §8's "all timestamps are timezone-aware UTC"). Postgres
    preserves the offset natively in prod, so this is a no-op there; on
    SQLite it re-attaches the UTC offset so callers never see a naive
    "most recent" timestamp diverge from the aware one that was written.
    """
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class RiskScoreRepository:
    """Data-access for ``risk_scores``."""

    async def create(self, db: AsyncSession, **fields) -> RiskScore:
        score = RiskScore(**fields)
        db.add(score)
        await db.flush()
        await db.refresh(score)
        score.computed_at = _to_utc(score.computed_at)
        return score

    async def get_latest_for_appointment(
        self, db: AsyncSession, appointment_id: UUID
    ) -> RiskScore | None:
        """Return the most recently computed score for ``appointment_id``."""
        stmt = (
            select(RiskScore)
            .where(RiskScore.appointment_id == appointment_id)
            .order_by(RiskScore.computed_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        score = result.scalar_one_or_none()
        if score is not None:
            score.computed_at = _to_utc(score.computed_at)
        return score

    async def get_distribution(
        self,
        db: AsyncSession,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> dict[str, int]:
        """Counts grouped by ``risk_level`` over the latest score per
        appointment within ``[date_from, date_to]`` (either bound optional).

        Only the single most-recent ``RiskScore`` row per ``appointment_id``
        is counted — an appointment re-scored multiple times contributes to
        the distribution exactly once, under its current risk level.
        """
        row_number = (
            func.row_number()
            .over(
                partition_by=RiskScore.appointment_id,
                order_by=RiskScore.computed_at.desc(),
            )
            .label("rn")
        )
        filters = []
        if date_from is not None:
            filters.append(RiskScore.computed_at >= date_from)
        if date_to is not None:
            filters.append(RiskScore.computed_at <= date_to)

        latest_subq = (
            select(RiskScore.risk_level.label("risk_level"), row_number)
            .where(*filters)
            .subquery()
        )
        stmt = (
            select(latest_subq.c.risk_level, func.count())
            .where(latest_subq.c.rn == 1)
            .group_by(latest_subq.c.risk_level)
        )
        result = await db.execute(stmt)
        distribution: dict[str, int] = {}
        for risk_level, count in result.all():
            key = risk_level.value if hasattr(risk_level, "value") else risk_level
            distribution[key] = count
        return distribution


class TriageSessionRepository:
    """Data-access for ``triage_sessions``."""

    async def create(
        self, db: AsyncSession, channel: str, patient_id: UUID | None
    ) -> TriageSession:
        session = TriageSession(
            channel=channel,
            patient_id=patient_id,
            responses={},
            created_at=now_utc(),
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        return session

    async def get_by_id(self, db: AsyncSession, id: UUID) -> TriageSession | None:
        result = await db.execute(select(TriageSession).where(TriageSession.id == id))
        return result.scalar_one_or_none()

    async def append_response(
        self, db: AsyncSession, id: UUID, question_key: str, answer: str
    ) -> TriageSession:
        """RR002-safe: merges ``{question_key: answer}`` into the existing
        ``responses`` JSONB dict without overwriting previously recorded
        structured multiple-choice answers."""
        result = await db.execute(select(TriageSession).where(TriageSession.id == id))
        session = result.scalar_one()
        merged = dict(session.responses or {})
        merged[question_key] = answer
        # Reassigning the whole dict (rather than mutating in place) ensures
        # SQLAlchemy's change tracking sees the new value even though
        # `JSONBType` is a plain (non-`Mutable`) TypeDecorator.
        session.responses = merged
        await db.flush()
        await db.refresh(session)
        return session

    async def set_outcome(
        self,
        db: AsyncSession,
        id: UUID,
        urgency_classification: str | None,
        recommended_block: str | None,
        escalated: bool,
    ) -> TriageSession:
        result = await db.execute(select(TriageSession).where(TriageSession.id == id))
        session = result.scalar_one()
        session.urgency_classification = urgency_classification
        session.recommended_block = recommended_block
        session.escalated = escalated
        await db.flush()
        await db.refresh(session)
        return session

    async def update_fields(self, db: AsyncSession, id: UUID, **fields) -> TriageSession:
        """Used by ``services/console/service.py``'s ``OverrideService`` to
        apply a staff correction to an AI-originated triage outcome."""
        result = await db.execute(select(TriageSession).where(TriageSession.id == id))
        session = result.scalar_one()
        for key, value in fields.items():
            setattr(session, key, value)
        await db.flush()
        await db.refresh(session)
        return session


class EscalationRepository:
    """Data-access for ``escalations``.

    FR-E8.5/SFR001: ``acknowledge``/``resolve`` perform their write
    unconditionally once called — they do not themselves guard against an
    out-of-order status transition (e.g. resolving an already-resolved or
    still-unacknowledged escalation). Enforcing the forward-only
    ``unacknowledged`` -> ``acknowledged`` -> ``resolved`` ordering is the
    responsibility of ``services/intelligence/service.py``'s
    ``EscalationService``.
    """

    async def create(self, db: AsyncSession, **fields) -> Escalation:
        fields.setdefault("created_at", now_utc())
        escalation = Escalation(**fields)
        db.add(escalation)
        await db.flush()
        await db.refresh(escalation)
        return escalation

    async def get_by_id(self, db: AsyncSession, id: UUID) -> Escalation | None:
        result = await db.execute(select(Escalation).where(Escalation.id == id))
        return result.scalar_one_or_none()

    async def acknowledge(self, db: AsyncSession, id: UUID, staff_id: UUID) -> Escalation:
        """Sets ``status='acknowledged'``, ``routed_to_staff_id=staff_id``,
        ``acknowledged_at=now_utc()`` — only valid from
        ``status='unacknowledged'``; the caller enforces that transition."""
        result = await db.execute(select(Escalation).where(Escalation.id == id))
        escalation = result.scalar_one()
        escalation.status = EscalationStatus.acknowledged
        escalation.routed_to_staff_id = staff_id
        escalation.acknowledged_at = now_utc()
        await db.flush()
        await db.refresh(escalation)
        return escalation

    async def resolve(
        self, db: AsyncSession, id: UUID, staff_id: UUID, resolution_notes: str
    ) -> Escalation:
        """Sets ``status='resolved'``, ``resolved_by_staff_id=staff_id``,
        ``resolved_at=now_utc()``, ``resolution_notes`` — only valid from
        ``status='acknowledged'``; the caller enforces that transition."""
        result = await db.execute(select(Escalation).where(Escalation.id == id))
        escalation = result.scalar_one()
        escalation.status = EscalationStatus.resolved
        escalation.resolved_by_staff_id = staff_id
        escalation.resolved_at = now_utc()
        escalation.resolution_notes = resolution_notes
        await db.flush()
        await db.refresh(escalation)
        return escalation

    async def list_unacknowledged(self, db: AsyncSession) -> list[Escalation]:
        result = await db.execute(
            select(Escalation)
            .where(Escalation.status == EscalationStatus.unacknowledged)
            .order_by(Escalation.created_at)
        )
        return list(result.scalars().all())


class UtilisationRecommendationRepository:
    """Data-access for ``utilisation_recommendations``."""

    async def create(self, db: AsyncSession, **fields) -> UtilisationRecommendation:
        fields.setdefault("generated_at", now_utc())
        recommendation = UtilisationRecommendation(**fields)
        db.add(recommendation)
        await db.flush()
        await db.refresh(recommendation)
        return recommendation

    async def get_by_id(
        self, db: AsyncSession, id: UUID
    ) -> UtilisationRecommendation | None:
        result = await db.execute(
            select(UtilisationRecommendation).where(UtilisationRecommendation.id == id)
        )
        return result.scalar_one_or_none()

    async def list_by_status(
        self, db: AsyncSession, status: str
    ) -> list[UtilisationRecommendation]:
        result = await db.execute(
            select(UtilisationRecommendation)
            .where(UtilisationRecommendation.status == UtilisationRecommendationStatus(status))
            .order_by(UtilisationRecommendation.generated_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self, db: AsyncSession, id: UUID, status: str, decided_by_staff_id: UUID
    ) -> UtilisationRecommendation:
        result = await db.execute(
            select(UtilisationRecommendation).where(UtilisationRecommendation.id == id)
        )
        recommendation = result.scalar_one()
        recommendation.status = UtilisationRecommendationStatus(status)
        recommendation.decided_by_staff_id = decided_by_staff_id
        recommendation.decided_at = now_utc()
        await db.flush()
        await db.refresh(recommendation)
        return recommendation

    async def get_adoption_rate(self, db: AsyncSession, since: datetime) -> float:
        """FR-E8.7/US45: divides ``applied`` by ``applied + dismissed`` since
        ``since``, deliberately excluding still-``pending`` recommendations
        from the denominator so an unreviewed backlog does not depress the
        reported adoption rate. Returns ``0.0`` when the denominator is
        zero (no decided recommendations in range yet)."""
        applied_stmt = select(func.count()).where(
            UtilisationRecommendation.generated_at >= since,
            UtilisationRecommendation.status == UtilisationRecommendationStatus.applied,
        )
        dismissed_stmt = select(func.count()).where(
            UtilisationRecommendation.generated_at >= since,
            UtilisationRecommendation.status == UtilisationRecommendationStatus.dismissed,
        )
        applied_count = (await db.execute(applied_stmt)).scalar_one()
        dismissed_count = (await db.execute(dismissed_stmt)).scalar_one()
        denominator = applied_count + dismissed_count
        if denominator == 0:
            return 0.0
        return applied_count / denominator


class MlModelEvaluationRepository:
    """Data-access for ``ml_model_evaluations`` (SM4 monthly precision
    evaluation driving the ML model's passing/underperforming decision)."""

    async def create(self, db: AsyncSession, **fields) -> MlModelEvaluation:
        fields.setdefault("evaluated_at", now_utc())
        evaluation = MlModelEvaluation(**fields)
        db.add(evaluation)
        await db.flush()
        await db.refresh(evaluation)
        evaluation.evaluated_at = _to_utc(evaluation.evaluated_at)
        return evaluation

    async def get_latest(self, db: AsyncSession) -> MlModelEvaluation | None:
        stmt = select(MlModelEvaluation).order_by(MlModelEvaluation.evaluated_at.desc()).limit(1)
        result = await db.execute(stmt)
        evaluation = result.scalar_one_or_none()
        if evaluation is not None:
            evaluation.evaluated_at = _to_utc(evaluation.evaluated_at)
        return evaluation
