"""Data-access classes over ``recall_schedules``, ``unscheduled_treatments``,
``recall_compliance_baselines`` (architecture.md §4.2 recall module).

Interaction contract: every class here is used only from
``services/recall/service.py`` -- with one documented cross-module
exception per the analytics module's read-only-aggregator design:
``services/analytics/service.py`` depends directly on
``RecallScheduleRepository`` for read-only aggregation, never on
``services/recall/service.py``.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.utils import now_utc
from app.models.recall.models import (
    RecallComplianceBaseline,
    RecallSchedule,
    UnscheduledTreatment,
)


class RecallScheduleRepository:
    """Read/write access over ``recall_schedules``."""

    async def create(self, db: AsyncSession, **fields) -> RecallSchedule:
        schedule = RecallSchedule(**fields)
        db.add(schedule)
        await db.commit()
        await db.refresh(schedule)
        return schedule

    async def get_by_id(self, db: AsyncSession, id: UUID) -> RecallSchedule | None:
        result = await db.execute(select(RecallSchedule).where(RecallSchedule.id == id))
        return result.scalars().first()

    async def list_overdue(self, db: AsyncSession, as_of: date) -> list[RecallSchedule]:
        """``WHERE due_date < as_of AND status IN ('due', 'overdue')`` --
        reclassified overdue; the dormant threshold
        (``RECALL_DORMANT_THRESHOLD_MONTHS``) is applied by the
        caller/service, not here.
        """
        result = await db.execute(
            select(RecallSchedule).where(
                RecallSchedule.due_date < as_of,
                RecallSchedule.status.in_(["due", "overdue"]),
            )
        )
        return list(result.scalars().all())

    async def list_by_ids(self, db: AsyncSession, ids: list[UUID]) -> list[RecallSchedule]:
        result = await db.execute(select(RecallSchedule).where(RecallSchedule.id.in_(ids)))
        return list(result.scalars().all())

    async def update_status(
        self, db: AsyncSession, id: UUID, status: str, **extra_fields
    ) -> RecallSchedule:
        result = await db.execute(select(RecallSchedule).where(RecallSchedule.id == id))
        schedule = result.scalar_one()
        schedule.status = status
        for key, value in extra_fields.items():
            setattr(schedule, key, value)
        await db.commit()
        await db.refresh(schedule)
        return schedule

    async def link_appointment(
        self, db: AsyncSession, id: UUID, appointment_id: UUID
    ) -> RecallSchedule:
        result = await db.execute(select(RecallSchedule).where(RecallSchedule.id == id))
        schedule = result.scalar_one()
        schedule.appointment_id = appointment_id
        await db.commit()
        await db.refresh(schedule)
        return schedule

    async def count_by_status(
        self,
        db: AsyncSession,
        statuses: list[str],
        period_start: date,
        period_end: date,
    ) -> dict[str, int]:
        """FR-E7.4: backs the weekly compliance-rate calculation --
        ``RecallCampaignService`` divides these week-scoped counts to get a
        rate; the weekly refresh cadence itself is enforced by the caller
        (``app/workers/recall_scanner.py``), not here.
        """
        result = await db.execute(
            select(RecallSchedule).where(
                RecallSchedule.status.in_(statuses),
                RecallSchedule.due_date >= period_start,
                RecallSchedule.due_date <= period_end,
            )
        )
        schedules = result.scalars().all()
        counts: dict[str, int] = {status: 0 for status in statuses}
        for schedule in schedules:
            status_value = (
                schedule.status.value if hasattr(schedule.status, "value") else schedule.status
            )
            counts[status_value] = counts.get(status_value, 0) + 1
        return counts


class UnscheduledTreatmentRepository:
    """Read/write access over ``unscheduled_treatments``."""

    async def list_by_valuation(
        self, db: AsyncSession, status: str = "unscheduled"
    ) -> list[UnscheduledTreatment]:
        """``ORDER BY valuation_amount DESC NULLS LAST``, expressed
        portably (``NULLS LAST`` is Postgres-only syntax) via a
        ``valuation_amount IS NULL`` ascending sort key followed by the
        descending value sort, so nulls always sort last on both Postgres
        and the SQLite test engine.
        """
        result = await db.execute(
            select(UnscheduledTreatment)
            .where(UnscheduledTreatment.status == status)
            .order_by(
                UnscheduledTreatment.valuation_amount.is_(None),
                UnscheduledTreatment.valuation_amount.desc(),
            )
        )
        return list(result.scalars().all())

    async def get_by_id(self, db: AsyncSession, id: UUID) -> UnscheduledTreatment | None:
        result = await db.execute(
            select(UnscheduledTreatment).where(UnscheduledTreatment.id == id)
        )
        return result.scalars().first()

    async def update_valuation(
        self, db: AsyncSession, id: UUID, valuation_amount: float
    ) -> UnscheduledTreatment:
        result = await db.execute(
            select(UnscheduledTreatment).where(UnscheduledTreatment.id == id)
        )
        treatment = result.scalar_one()
        treatment.valuation_amount = valuation_amount
        await db.commit()
        await db.refresh(treatment)
        return treatment

    async def update_status(
        self, db: AsyncSession, id: UUID, status: str
    ) -> UnscheduledTreatment:
        result = await db.execute(
            select(UnscheduledTreatment).where(UnscheduledTreatment.id == id)
        )
        treatment = result.scalar_one()
        treatment.status = status
        await db.commit()
        await db.refresh(treatment)
        return treatment

    async def link_appointment(
        self, db: AsyncSession, id: UUID, appointment_id: UUID
    ) -> UnscheduledTreatment:
        result = await db.execute(
            select(UnscheduledTreatment).where(UnscheduledTreatment.id == id)
        )
        treatment = result.scalar_one()
        treatment.appointment_id = appointment_id
        await db.commit()
        await db.refresh(treatment)
        return treatment

    async def list_by_ids(self, db: AsyncSession, ids: list[UUID]) -> list[UnscheduledTreatment]:
        result = await db.execute(
            select(UnscheduledTreatment).where(UnscheduledTreatment.id.in_(ids))
        )
        return list(result.scalars().all())


class RecallComplianceBaselineRepository:
    """Read/write access over ``recall_compliance_baselines``.

    FR-E7.4: no ``update``/``delete`` method is exposed here -- a new
    baseline value is captured as a brand-new row so the history of
    captured baselines is preserved; ``get_latest`` reads the most recent
    one for a given metric.
    """

    async def get_latest(
        self, db: AsyncSession, metric_name: str
    ) -> RecallComplianceBaseline | None:
        result = await db.execute(
            select(RecallComplianceBaseline)
            .where(RecallComplianceBaseline.metric_name == metric_name)
            .order_by(RecallComplianceBaseline.captured_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    async def create(
        self, db: AsyncSession, metric_name: str, baseline_value: float
    ) -> RecallComplianceBaseline:
        baseline = RecallComplianceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            captured_at=now_utc(),
        )
        db.add(baseline)
        await db.commit()
        await db.refresh(baseline)
        return baseline
