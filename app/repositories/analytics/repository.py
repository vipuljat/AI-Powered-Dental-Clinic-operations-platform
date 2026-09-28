"""Data-access classes for the analytics module (architecture.md §5.4).

Owns reads/writes against ``operational_metrics_snapshots``,
``financial_roi_snapshots``, ``baseline_costs`` and ``dashboard_exports``.

``open_questions[Q-cf1e404b]``: ``DashboardExportRepository`` is declared here
even though architecture.md §5.4's table has no explicit row for it — the
``dashboard_exports`` table and ``DashboardExportService.generate``
(architecture.md §5.3) both exist, so a repository must own its writes; this
class is that owner.

Interaction contract: every method here is called only from
``services/analytics/service.py`` — no other module writes to or reads from
``operational_metrics_snapshots``/``financial_roi_snapshots``/
``baseline_costs``/``dashboard_exports``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analytics.models import (
    BaselineCost,
    DashboardExport,
    FinancialRoiSnapshot,
    OperationalMetricsSnapshot,
)


class OperationalMetricsRepository:
    """Data-access for ``operational_metrics_snapshots`` (RAR001 metrics)."""

    async def upsert_snapshot(
        self,
        db: AsyncSession,
        metric_name: str,
        metric_value: float,
        baseline_value: float | None,
        period_start: date,
        period_end: date,
    ) -> OperationalMetricsSnapshot:
        """Insert or update the snapshot for ``metric_name`` over
        ``[period_start, period_end]``.

        FR-E9.1: ``baseline_value`` may be ``None`` ("baseline pending") and
        is never coerced to a sentinel — it is written through verbatim.
        """
        result = await db.execute(
            select(OperationalMetricsSnapshot).where(
                OperationalMetricsSnapshot.metric_name == metric_name,
                OperationalMetricsSnapshot.period_start == period_start,
                OperationalMetricsSnapshot.period_end == period_end,
            )
        )
        snapshot = result.scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if snapshot is None:
            snapshot = OperationalMetricsSnapshot(
                metric_name=metric_name,
                metric_value=metric_value,
                baseline_value=baseline_value,
                period_start=period_start,
                period_end=period_end,
                computed_at=now,
            )
            db.add(snapshot)
        else:
            snapshot.metric_value = metric_value
            snapshot.baseline_value = baseline_value
            snapshot.computed_at = now
        await db.flush()
        await db.refresh(snapshot)
        return snapshot

    async def get_range(
        self,
        db: AsyncSession,
        metric_name: str | None,
        date_from: date,
        date_to: date,
    ) -> list[OperationalMetricsSnapshot]:
        """Return all snapshots overlapping ``[date_from, date_to]``,
        optionally filtered to a single ``metric_name``, ordered by
        ``period_start``."""
        stmt = select(OperationalMetricsSnapshot).where(
            OperationalMetricsSnapshot.period_start <= date_to,
            OperationalMetricsSnapshot.period_end >= date_from,
        )
        if metric_name is not None:
            stmt = stmt.where(OperationalMetricsSnapshot.metric_name == metric_name)
        stmt = stmt.order_by(OperationalMetricsSnapshot.period_start)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_latest(
        self, db: AsyncSession, metric_name: str
    ) -> OperationalMetricsSnapshot | None:
        """Return the most recently computed snapshot for ``metric_name``."""
        stmt = (
            select(OperationalMetricsSnapshot)
            .where(OperationalMetricsSnapshot.metric_name == metric_name)
            .order_by(OperationalMetricsSnapshot.computed_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


class FinancialRoiRepository:
    """Data-access for ``financial_roi_snapshots``."""

    async def upsert_snapshot(
        self,
        db: AsyncSession,
        period_month: date,
        recovered_appointments: int,
        recovered_revenue: float,
        baseline_cost: float | None,
        roi_percentage: float | None,
    ) -> FinancialRoiSnapshot:
        """Insert or update the ROI rollup for ``period_month``."""
        result = await db.execute(
            select(FinancialRoiSnapshot).where(
                FinancialRoiSnapshot.period_month == period_month,
            )
        )
        snapshot = result.scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if snapshot is None:
            snapshot = FinancialRoiSnapshot(
                period_month=period_month,
                recovered_appointments=recovered_appointments,
                recovered_revenue=recovered_revenue,
                baseline_cost=baseline_cost,
                roi_percentage=roi_percentage,
                computed_at=now,
            )
            db.add(snapshot)
        else:
            snapshot.recovered_appointments = recovered_appointments
            snapshot.recovered_revenue = recovered_revenue
            snapshot.baseline_cost = baseline_cost
            snapshot.roi_percentage = roi_percentage
            snapshot.computed_at = now
        await db.flush()
        await db.refresh(snapshot)
        return snapshot

    async def get_trend(self, db: AsyncSession, months: int) -> list[FinancialRoiSnapshot]:
        """Return the ``months`` most recent ROI snapshots, ordered oldest to
        newest."""
        stmt = (
            select(FinancialRoiSnapshot)
            .order_by(FinancialRoiSnapshot.period_month.desc())
            .limit(months)
        )
        result = await db.execute(stmt)
        rows = list(result.scalars().all())
        rows.reverse()
        return rows

    async def get_latest(self, db: AsyncSession) -> FinancialRoiSnapshot | None:
        """Return the most recently computed ROI snapshot."""
        stmt = select(FinancialRoiSnapshot).order_by(
            FinancialRoiSnapshot.period_month.desc()
        ).limit(1)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


class BaselineCostRepository:
    """Data-access for ``baseline_costs`` (T119 AC: audit-logged
    clinic_management-entered baseline cost figures)."""

    async def save_baseline(
        self,
        db: AsyncSession,
        entered_by_staff_id: UUID,
        amount: float,
        effective_period: date,
    ) -> BaselineCost:
        """Persist a new baseline cost figure for ``effective_period``."""
        baseline = BaselineCost(
            entered_by_staff_id=entered_by_staff_id,
            amount=amount,
            effective_period=effective_period,
            created_at=datetime.now(timezone.utc),
        )
        db.add(baseline)
        await db.flush()
        await db.refresh(baseline)
        return baseline

    async def get_for_period(self, db: AsyncSession, period: date) -> BaselineCost | None:
        """Return the most recently entered baseline cost row whose
        ``effective_period`` matches ``period``, or ``None`` if absent
        ("baseline pending")."""
        stmt = (
            select(BaselineCost)
            .where(BaselineCost.effective_period == period)
            .order_by(BaselineCost.created_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


class DashboardExportRepository:
    """Data-access for ``dashboard_exports``.

    ``open_questions[Q-cf1e404b]``: owns writes for
    ``DashboardExportService.generate`` (architecture.md §5.3) even though
    architecture.md §5.4's table has no explicit row naming this repository.
    """

    async def create(
        self,
        db: AsyncSession,
        requested_by_staff_id: UUID,
        dashboard_type: str,
        format: str,
        date_range_start: date,
        date_range_end: date,
        storage_uri: str,
    ) -> DashboardExport:
        """Persist a record of a generated dashboard export file.

        ``storage_uri`` is the object-storage location of the generated
        file — the file bytes themselves are never stored here.

        ``dashboard_type``/``format`` are written through verbatim as the
        plain strings the public surface declares (``str``, not
        ``DashboardType``/``ExportFormat``) — the analytics service is free
        to pass any dashboard-type/export-format label it names (e.g.
        ``"financial_roi"``, ``"operational_metrics"``) without this
        repository narrowing it to the small ``DashboardType``/
        ``ExportFormat`` vocabulary used elsewhere, and no server-generated
        column exists on this table that would require a post-insert
        ``refresh()`` round-trip.
        """
        export = DashboardExport(
            requested_by_staff_id=requested_by_staff_id,
            dashboard_type=dashboard_type,
            format=format,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            storage_uri=storage_uri,
            requested_at=datetime.now(timezone.utc),
        )
        db.add(export)
        await db.flush()
        return export

    async def get_by_id(self, db: AsyncSession, id: UUID) -> DashboardExport | None:
        """Return the export record with the given ``id``, or ``None``.

        Looked up via ``Session.get`` (primary-key identity-map lookup)
        rather than a fresh ``select`` — the idiomatic SQLAlchemy form for a
        by-id fetch, and one that returns the already-flushed in-session
        instance verbatim when present instead of forcing a superfluous
        round-trip.
        """
        return await db.get(DashboardExport, id)
