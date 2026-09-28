"""Unit tests for app/services/analytics/service.py.

Exercises `OperationalAnalyticsService`, `FinancialAnalyticsService`, and
`DashboardExportService` directly, with every collaborator (this module's own
repositories, the other modules' repositories it is allowed to read from per
the layering rule, `AuditService`, and `ObjectStorageClient`) replaced by
lightweight in-process fakes built from their documented public surfaces --
never a real DB/session/HTTP call.

Fakes intentionally ignore extra positional/keyword arguments they don't need
to inspect (e.g. `db`), so the tests exercise *outcomes* (the returned dict
shape, which exceptions are raised, which collaborator methods get invoked)
rather than the exact call signature the implementation happens to use
internally.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.common.exceptions.errors import NoDataForRangeError
from app.services.analytics.service import (
    DashboardExportService,
    FinancialAnalyticsService,
    OperationalAnalyticsService,
)

RAR001_METRICS = [
    "scheduling_time_minutes",
    "cancellation_recovery_rate",
    "no_show_rate",
    "recall_conversion",
    "utilisation_rate",
    "staff_time_saved",
]

DB = object()  # AsyncSession placeholder; every repository below is a fake, so no real session is needed


# ---------------------------------------------------------------------------
# fakes / builders
# ---------------------------------------------------------------------------

def _op_snapshot(metric_name, value=10.0, baseline=8.0, period_end=None, computed_at=None):
    return SimpleNamespace(
        metric_name=metric_name,
        metric_value=value,
        baseline_value=baseline,
        period_start=date(2026, 9, 1),
        period_end=period_end or date(2026, 9, 27),
        computed_at=computed_at or datetime(2026, 9, 27, tzinfo=timezone.utc),
    )


def _make_metrics_repo(snapshots):
    async def fake_get_range(db, metric_name, date_from, date_to):
        if metric_name is None:
            return list(snapshots)
        return [s for s in snapshots if s.metric_name == metric_name]

    async def fake_get_latest(db, metric_name):
        matches = [s for s in snapshots if s.metric_name == metric_name]
        return matches[-1] if matches else None

    return SimpleNamespace(
        upsert_snapshot=AsyncMock(),
        get_range=AsyncMock(side_effect=fake_get_range),
        get_latest=AsyncMock(side_effect=fake_get_latest),
    )


def _make_appt_repo():
    return SimpleNamespace(
        create=AsyncMock(),
        get_by_id=AsyncMock(return_value=None),
        get_by_idempotency_key=AsyncMock(return_value=None),
        list_open_slots=AsyncMock(return_value=[]),
        find_conflicts=AsyncMock(return_value=[]),
        update_status=AsyncMock(),
        update_fields=AsyncMock(),
        list_by_patient=AsyncMock(return_value=[]),
        list_completable=AsyncMock(return_value=[]),
        list_by_risk_source=AsyncMock(return_value=[]),
    )


def _make_alert_repo():
    return SimpleNamespace(
        create=AsyncMock(),
        find_matching=AsyncMock(return_value=None),
        get_by_id=AsyncMock(return_value=None),
        update_status=AsyncMock(),
        list_open=AsyncMock(return_value=[]),
    )


def _make_recall_repo():
    return SimpleNamespace(
        create=AsyncMock(),
        get_by_id=AsyncMock(return_value=None),
        list_overdue=AsyncMock(return_value=[]),
        list_by_ids=AsyncMock(return_value=[]),
        update_status=AsyncMock(),
        link_appointment=AsyncMock(),
        count_by_status=AsyncMock(return_value={}),
    )


def _make_operational_service(snapshots):
    return OperationalAnalyticsService(
        _make_metrics_repo(snapshots), _make_appt_repo(), _make_alert_repo(), _make_recall_repo()
    )


def _roi_snapshot(period_month, computed_at, roi_percentage=55.0, recovered_appointments=12,
                   recovered_revenue=3000.0, baseline_cost=1000.0):
    return SimpleNamespace(
        period_month=period_month,
        recovered_appointments=recovered_appointments,
        recovered_revenue=recovered_revenue,
        baseline_cost=baseline_cost,
        roi_percentage=roi_percentage,
        computed_at=computed_at,
    )


def _make_roi_repo(trend):
    return SimpleNamespace(
        upsert_snapshot=AsyncMock(),
        get_trend=AsyncMock(return_value=list(trend)),
        get_latest=AsyncMock(return_value=trend[-1] if trend else None),
    )


def _make_baseline(effective_period=date(2026, 9, 1)):
    return SimpleNamespace(
        id=uuid.uuid4(),
        entered_by_staff_id=uuid.uuid4(),
        amount=500.0,
        effective_period=effective_period,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


def _make_baseline_repo(baseline=None, save_return=None):
    return SimpleNamespace(
        save_baseline=AsyncMock(return_value=save_return or baseline or _make_baseline()),
        get_for_period=AsyncMock(return_value=baseline),
    )


def _make_audit():
    return SimpleNamespace(record=AsyncMock())


def _make_financial_service(trend, baseline=None):
    return FinancialAnalyticsService(_make_roi_repo(trend), _make_baseline_repo(baseline=baseline), _make_audit())


def _make_export_repo(created):
    return SimpleNamespace(
        create=AsyncMock(return_value=created),
        get_by_id=AsyncMock(return_value=created),
    )


def _make_storage():
    return SimpleNamespace(
        put_object=AsyncMock(return_value="memory://bucket/key"),
        get_object=AsyncMock(return_value=b""),
    )


def _await_values(mock_call):
    """Flatten an AsyncMock await_args' positional + keyword values for a loose membership check."""
    return list(mock_call.args) + list(mock_call.kwargs.values())


# ---------------------------------------------------------------------------
# OperationalAnalyticsService.get_metrics -- FR-E9.1/US46
# ---------------------------------------------------------------------------

async def test_get_metrics_returns_all_six_rar001_metrics_when_data_current():
    snapshots = [_op_snapshot(name) for name in RAR001_METRICS]
    service = _make_operational_service(snapshots)

    result = await service.get_metrics(DB, date(2026, 9, 1), date(2026, 9, 27), None)

    assert result["data_delayed"] is False
    for name in RAR001_METRICS:
        assert name in result


async def test_get_metrics_flags_stale_metric_as_delayed_not_a_stale_value():
    # 5 metrics are fresh (period_end == the requested date_to); no_show_rate's only
    # available snapshot is a week stale -- the documented "hasn't finished aggregating" case.
    snapshots = [_op_snapshot(name, period_end=date(2026, 9, 27)) for name in RAR001_METRICS if name != "no_show_rate"]
    snapshots.append(_op_snapshot("no_show_rate", period_end=date(2026, 9, 20)))
    service = _make_operational_service(snapshots)

    result = await service.get_metrics(DB, date(2026, 9, 1), date(2026, 9, 27), None)

    assert result["data_delayed"] is True
    assert isinstance(result["no_show_rate"], dict)
    assert result["no_show_rate"].get("delayed") is True


async def test_get_metrics_honours_metric_filter():
    snapshots = [_op_snapshot(name) for name in RAR001_METRICS]
    service = _make_operational_service(snapshots)

    result = await service.get_metrics(DB, None, date(2026, 9, 27), "recall_conversion")

    assert "recall_conversion" in result
    assert result["data_delayed"] is False


# ---------------------------------------------------------------------------
# FinancialAnalyticsService.get_roi -- FR-E9.2/US47
# ---------------------------------------------------------------------------

async def test_get_roi_requires_baseline_cost_before_computing_real_roi():
    trend = [_roi_snapshot(date(2026, 9, 1), datetime(2026, 9, 1, tzinfo=timezone.utc))]
    service = _make_financial_service(trend, baseline=None)

    result = await service.get_roi(DB, "monthly")

    assert result["roi_percentage"] is None
    assert result["baseline_cost_required"] is True
    assert isinstance(result["recovered_appointments"], int)
    assert isinstance(result["trend"], list)


async def test_get_roi_computes_real_percentage_once_baseline_captured():
    baseline = _make_baseline(effective_period=date(2026, 9, 1))
    trend = [_roi_snapshot(date(2026, 9, 1), datetime(2026, 9, 1, tzinfo=timezone.utc), roi_percentage=42.0)]
    service = _make_financial_service(trend, baseline=baseline)

    result = await service.get_roi(DB, "monthly")

    assert result.get("baseline_cost_required", False) is False
    assert result["roi_percentage"] is not None


async def test_set_baseline_cost_persists_and_audit_logs_the_action():
    baseline_repo = _make_baseline_repo()
    audit = _make_audit()
    service = FinancialAnalyticsService(_make_roi_repo([]), baseline_repo, audit)
    actor_id = uuid.uuid4()

    result = await service.set_baseline_cost(DB, 750.0, date(2026, 9, 1), actor_id)

    baseline_repo.save_baseline.assert_awaited_once()
    saved_call_values = _await_values(baseline_repo.save_baseline.await_args)
    assert 750.0 in saved_call_values
    assert actor_id in saved_call_values

    audit.record.assert_awaited_once()
    assert "baseline_cost.set" in _await_values(audit.record.await_args)

    assert result is baseline_repo.save_baseline.return_value


# ---------------------------------------------------------------------------
# DashboardExportService.generate -- FR-E9.3/US48
# ---------------------------------------------------------------------------

_EARLIEST = datetime(2026, 1, 10, tzinfo=timezone.utc)


def _build_export_service():
    op_snapshots = [
        _op_snapshot(name, period_end=date(2030, 1, 1), computed_at=_EARLIEST) for name in RAR001_METRICS
    ]
    operational_service = _make_operational_service(op_snapshots)

    roi_trend = [_roi_snapshot(date(2026, 3, 1), _EARLIEST)]
    baseline = _make_baseline(effective_period=date(2026, 3, 1))
    financial_service = _make_financial_service(roi_trend, baseline=baseline)

    created = SimpleNamespace(
        id=uuid.uuid4(),
        requested_by_staff_id=uuid.uuid4(),
        dashboard_type="operational",
        format="csv",
        date_range_start=date(2026, 3, 1),
        date_range_end=date(2026, 3, 31),
        storage_uri="memory://bucket/key",
        requested_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )
    export_repo = _make_export_repo(created)
    storage = _make_storage()

    service = DashboardExportService(export_repo, operational_service, financial_service, storage)
    return service, operational_service, financial_service, export_repo, storage, created


async def test_generate_raises_no_data_for_range_before_earliest_snapshot():
    service, *_ = _build_export_service()
    actor_id = uuid.uuid4()

    with pytest.raises(NoDataForRangeError):
        await service.generate(DB, "operational", "csv", date(2025, 1, 1), date(2025, 6, 1), actor_id)


async def test_generate_operational_uses_the_same_aggregation_call_as_the_live_dashboard():
    service, operational_service, _financial_service, export_repo, storage, created = _build_export_service()
    actor_id = uuid.uuid4()
    spy = AsyncMock(wraps=operational_service.get_metrics)
    operational_service.get_metrics = spy

    result = await service.generate(DB, "operational", "csv", date(2026, 3, 1), date(2026, 3, 31), actor_id)

    spy.assert_awaited()
    assert result["status"] == "processing"
    assert result["export_id"] == created.id
    storage.put_object.assert_awaited_once()
    export_repo.create.assert_awaited_once()


async def test_generate_financial_uses_the_same_aggregation_call_as_the_live_dashboard():
    service, _operational_service, financial_service, export_repo, storage, created = _build_export_service()
    actor_id = uuid.uuid4()
    spy = AsyncMock(wraps=financial_service.get_roi)
    financial_service.get_roi = spy

    result = await service.generate(DB, "financial", "pdf", date(2026, 3, 1), date(2026, 3, 31), actor_id)

    spy.assert_awaited()
    assert result["status"] == "processing"
    storage.put_object.assert_awaited_once()
