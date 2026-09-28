"""Unit tests for app/repositories/analytics/repository.py.

These tests exercise the four data-access classes declared over
`operational_metrics_snapshots`, `financial_roi_snapshots`, `baseline_costs`
and `dashboard_exports` directly against an in-memory SQLite database (the
project's test-substitution mechanism per PROJECT.md's Testing section:
ENVIRONMENT=test swaps the DB URL to sqlite+aiosqlite:///:memory: while the
same ORM models run unmodified thanks to app.core.db_types's portable
JSONB/VECTOR TypeDecorators). No bespoke mocking of the database is used.
"""

from datetime import date
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.database import Base
from app.repositories.analytics.repository import (
    BaselineCostRepository,
    DashboardExportRepository,
    FinancialRoiRepository,
    OperationalMetricsRepository,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as session:
        yield session
    await engine.dispose()


# --------------------------------------------------------------------------
# OperationalMetricsRepository
# --------------------------------------------------------------------------


class TestOperationalMetricsRepository:
    async def test_upsert_snapshot_creates_record_with_given_fields(self, db_session):
        repo = OperationalMetricsRepository()

        result = await repo.upsert_snapshot(
            db_session,
            metric_name="chair_utilisation",
            metric_value=72.5,
            baseline_value=60.0,
            period_start=date(2026, 1, 1),
            period_end=date(2026, 1, 31),
        )

        assert result.id is not None
        assert result.metric_name == "chair_utilisation"
        assert result.metric_value == 72.5
        assert result.baseline_value == 60.0
        assert result.period_start == date(2026, 1, 1)
        assert result.period_end == date(2026, 1, 31)

    async def test_upsert_snapshot_allows_baseline_value_to_be_none(self, db_session):
        repo = OperationalMetricsRepository()

        result = await repo.upsert_snapshot(
            db_session,
            metric_name="no_show_rate",
            metric_value=5.0,
            baseline_value=None,
            period_start=date(2026, 2, 1),
            period_end=date(2026, 2, 28),
        )

        assert result.baseline_value is None

    async def test_upsert_snapshot_updates_rather_than_duplicates_same_period(self, db_session):
        repo = OperationalMetricsRepository()
        period_start = date(2026, 3, 1)
        period_end = date(2026, 3, 31)

        await repo.upsert_snapshot(
            db_session, "chair_utilisation", 50.0, 40.0, period_start, period_end
        )
        await repo.upsert_snapshot(
            db_session, "chair_utilisation", 88.0, 40.0, period_start, period_end
        )

        rows = await repo.get_range(
            db_session, "chair_utilisation", date(2026, 3, 1), date(2026, 3, 31)
        )

        assert len(rows) == 1
        assert rows[0].metric_value == 88.0

    async def test_get_range_filters_by_metric_name(self, db_session):
        repo = OperationalMetricsRepository()
        await repo.upsert_snapshot(
            db_session, "metric_a", 10.0, None, date(2026, 4, 1), date(2026, 4, 30)
        )
        await repo.upsert_snapshot(
            db_session, "metric_b", 20.0, None, date(2026, 4, 1), date(2026, 4, 30)
        )

        rows = await repo.get_range(
            db_session, "metric_a", date(2026, 4, 1), date(2026, 4, 30)
        )

        assert len(rows) == 1
        assert rows[0].metric_name == "metric_a"

    async def test_get_range_without_metric_name_returns_all_metrics_in_window(self, db_session):
        repo = OperationalMetricsRepository()
        await repo.upsert_snapshot(
            db_session, "metric_a", 10.0, None, date(2026, 5, 1), date(2026, 5, 31)
        )
        await repo.upsert_snapshot(
            db_session, "metric_b", 20.0, None, date(2026, 5, 1), date(2026, 5, 31)
        )

        rows = await repo.get_range(
            db_session, None, date(2026, 5, 1), date(2026, 5, 31)
        )

        assert {row.metric_name for row in rows} == {"metric_a", "metric_b"}

    async def test_get_range_excludes_rows_outside_date_window(self, db_session):
        repo = OperationalMetricsRepository()
        await repo.upsert_snapshot(
            db_session, "metric_a", 10.0, None, date(2026, 1, 1), date(2026, 1, 31)
        )

        rows = await repo.get_range(
            db_session, "metric_a", date(2026, 6, 1), date(2026, 6, 30)
        )

        assert rows == []

    async def test_get_latest_returns_most_recent_snapshot_for_metric(self, db_session):
        repo = OperationalMetricsRepository()
        await repo.upsert_snapshot(
            db_session, "metric_a", 10.0, None, date(2026, 1, 1), date(2026, 1, 31)
        )
        await repo.upsert_snapshot(
            db_session, "metric_a", 30.0, None, date(2026, 3, 1), date(2026, 3, 31)
        )

        latest = await repo.get_latest(db_session, "metric_a")

        assert latest is not None
        assert latest.metric_value == 30.0
        assert latest.period_end == date(2026, 3, 31)

    async def test_get_latest_returns_none_when_metric_has_no_snapshots(self, db_session):
        repo = OperationalMetricsRepository()

        latest = await repo.get_latest(db_session, "never_recorded_metric")

        assert latest is None


# --------------------------------------------------------------------------
# FinancialRoiRepository
# --------------------------------------------------------------------------


class TestFinancialRoiRepository:
    async def test_upsert_snapshot_creates_record_with_given_fields(self, db_session):
        repo = FinancialRoiRepository()

        result = await repo.upsert_snapshot(
            db_session,
            period_month=date(2026, 1, 1),
            recovered_appointments=12,
            recovered_revenue=2500.5,
            baseline_cost=1000.0,
            roi_percentage=150.0,
        )

        assert result.id is not None
        assert result.period_month == date(2026, 1, 1)
        assert result.recovered_appointments == 12
        assert result.recovered_revenue == 2500.5
        assert result.baseline_cost == 1000.0
        assert result.roi_percentage == 150.0

    async def test_upsert_snapshot_allows_optional_fields_to_be_none(self, db_session):
        repo = FinancialRoiRepository()

        result = await repo.upsert_snapshot(
            db_session,
            period_month=date(2026, 2, 1),
            recovered_appointments=0,
            recovered_revenue=0.0,
            baseline_cost=None,
            roi_percentage=None,
        )

        assert result.baseline_cost is None
        assert result.roi_percentage is None

    async def test_upsert_snapshot_updates_rather_than_duplicates_same_month(self, db_session):
        repo = FinancialRoiRepository()
        month = date(2026, 3, 1)

        await repo.upsert_snapshot(db_session, month, 5, 500.0, 200.0, 25.0)
        await repo.upsert_snapshot(db_session, month, 9, 900.0, 200.0, 45.0)

        trend = await repo.get_trend(db_session, months=12)
        march_rows = [row for row in trend if row.period_month == month]

        assert len(march_rows) == 1
        assert march_rows[0].recovered_appointments == 9
        assert march_rows[0].roi_percentage == 45.0

    async def test_get_trend_returns_only_the_requested_number_of_recent_months(self, db_session):
        repo = FinancialRoiRepository()
        months = [date(2026, m, 1) for m in range(1, 7)]
        for i, month in enumerate(months, start=1):
            await repo.upsert_snapshot(db_session, month, i, float(i * 100), 50.0, 10.0)

        trend = await repo.get_trend(db_session, months=3)

        assert len(trend) == 3
        assert {row.period_month for row in trend} == set(months[-3:])

    async def test_get_latest_returns_none_when_no_snapshots_exist(self, db_session):
        repo = FinancialRoiRepository()

        latest = await repo.get_latest(db_session)

        assert latest is None

    async def test_get_latest_returns_the_most_recent_period(self, db_session):
        repo = FinancialRoiRepository()
        await repo.upsert_snapshot(db_session, date(2026, 1, 1), 1, 100.0, 50.0, 10.0)
        await repo.upsert_snapshot(db_session, date(2026, 4, 1), 4, 400.0, 50.0, 40.0)

        latest = await repo.get_latest(db_session)

        assert latest is not None
        assert latest.period_month == date(2026, 4, 1)
        assert latest.recovered_appointments == 4


# --------------------------------------------------------------------------
# BaselineCostRepository
# --------------------------------------------------------------------------


class TestBaselineCostRepository:
    async def test_save_baseline_creates_record_with_given_fields(self, db_session):
        repo = BaselineCostRepository()
        staff_id = uuid4()

        result = await repo.save_baseline(
            db_session,
            entered_by_staff_id=staff_id,
            amount=15000.0,
            effective_period=date(2026, 1, 1),
        )

        assert result.id is not None
        assert result.entered_by_staff_id == staff_id
        assert result.amount == 15000.0
        assert result.effective_period == date(2026, 1, 1)

    async def test_save_baseline_keeps_separate_entries_per_effective_period(self, db_session):
        repo = BaselineCostRepository()
        staff_id = uuid4()

        await repo.save_baseline(db_session, staff_id, 10000.0, date(2026, 1, 1))
        await repo.save_baseline(db_session, staff_id, 12000.0, date(2026, 2, 1))

        jan = await repo.get_for_period(db_session, date(2026, 1, 1))
        feb = await repo.get_for_period(db_session, date(2026, 2, 1))

        assert jan is not None and jan.amount == 10000.0
        assert feb is not None and feb.amount == 12000.0

    async def test_get_for_period_returns_none_when_no_baseline_recorded(self, db_session):
        repo = BaselineCostRepository()

        result = await repo.get_for_period(db_session, date(2026, 12, 1))

        assert result is None


# --------------------------------------------------------------------------
# DashboardExportRepository (open_questions[Q-cf1e404b])
# --------------------------------------------------------------------------


class TestDashboardExportRepository:
    async def test_create_persists_and_returns_export_with_given_fields(self, db_session):
        # Q-cf1e404b: this repository is declared here specifically to own
        # writes to dashboard_exports on behalf of DashboardExportService.generate,
        # even though architecture.md §5.4's table has no explicit row for it.
        repo = DashboardExportRepository()
        staff_id = uuid4()

        result = await repo.create(
            db_session,
            requested_by_staff_id=staff_id,
            dashboard_type="financial_roi",
            format="pdf",
            date_range_start=date(2026, 1, 1),
            date_range_end=date(2026, 1, 31),
            storage_uri="s3://dashboards/exports/financial_roi-2026-01.pdf",
        )

        assert result.id is not None
        assert result.requested_by_staff_id == staff_id
        assert result.dashboard_type == "financial_roi"
        assert result.format == "pdf"
        assert result.date_range_start == date(2026, 1, 1)
        assert result.date_range_end == date(2026, 1, 31)
        assert result.storage_uri == "s3://dashboards/exports/financial_roi-2026-01.pdf"

    async def test_get_by_id_returns_the_matching_export(self, db_session):
        repo = DashboardExportRepository()
        created = await repo.create(
            db_session,
            requested_by_staff_id=uuid4(),
            dashboard_type="operational_metrics",
            format="csv",
            date_range_start=date(2026, 2, 1),
            date_range_end=date(2026, 2, 28),
            storage_uri="s3://dashboards/exports/ops-2026-02.csv",
        )

        fetched = await repo.get_by_id(db_session, created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.storage_uri == "s3://dashboards/exports/ops-2026-02.csv"

    async def test_get_by_id_returns_none_for_unknown_id(self, db_session):
        repo = DashboardExportRepository()

        result = await repo.get_by_id(db_session, uuid4())

        assert result is None
