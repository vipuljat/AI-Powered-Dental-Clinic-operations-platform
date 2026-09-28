"""Unit tests for app/schemas/analytics/schemas.py.

These schemas are the whole request/response contract for
`routes/analytics/routes.py` (per the spec's Interaction Contract) -- field
names/types/aliases here are exactly what `services/analytics/service.py`'s
return dicts must match key-for-key. Tests exercise the public Pydantic v2
models directly (construction, serialization, validation errors) without any
DB/HTTP/service dependency.
"""
from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.schemas.analytics.schemas import (
    BaselineCostResponse,
    DateRange,
    ExportDashboardRequest,
    ExportDashboardResponse,
    FinancialDashboardResponse,
    MetricItem,
    OperationalDashboardResponse,
    SetBaselineCostRequest,
    TrendPoint,
)


# ---------------------------------------------------------------------------
# MetricItem
# ---------------------------------------------------------------------------

def test_metric_item_holds_name_value_and_optional_baseline():
    item = MetricItem(name="utilisation", value=87.5, baseline=80.0)
    assert item.name == "utilisation"
    assert item.value == 87.5
    assert item.baseline == 80.0


def test_metric_item_baseline_accepts_none():
    item = MetricItem(name="utilisation", value=87.5, baseline=None)
    assert item.baseline is None


def test_metric_item_requires_name_and_value():
    with pytest.raises(ValidationError):
        MetricItem(value=1.0, baseline=None)
    with pytest.raises(ValidationError):
        MetricItem(name="x", baseline=None)


# ---------------------------------------------------------------------------
# OperationalDashboardResponse
# ---------------------------------------------------------------------------

def test_operational_dashboard_response_shape():
    resp = OperationalDashboardResponse(
        metrics=[MetricItem(name="utilisation", value=87.5, baseline=80.0)],
        data_delayed=False,
    )
    assert resp.data_delayed is False
    assert len(resp.metrics) == 1
    assert resp.metrics[0].name == "utilisation"


def test_operational_dashboard_response_metrics_can_be_empty_list():
    resp = OperationalDashboardResponse(metrics=[], data_delayed=True)
    assert resp.metrics == []
    assert resp.data_delayed is True


def test_operational_dashboard_response_requires_data_delayed():
    with pytest.raises(ValidationError):
        OperationalDashboardResponse(metrics=[])


def test_operational_dashboard_response_serializes_nested_metric_items():
    resp = OperationalDashboardResponse(
        metrics=[MetricItem(name="no_show_rate", value=4.2, baseline=None)],
        data_delayed=False,
    )
    dumped = resp.model_dump()
    assert dumped == {
        "metrics": [{"name": "no_show_rate", "value": 4.2, "baseline": None}],
        "data_delayed": False,
    }


# ---------------------------------------------------------------------------
# TrendPoint
# ---------------------------------------------------------------------------

def test_trend_point_holds_month_and_roi_percentage():
    point = TrendPoint(month="2024-01", roi_percentage=12.5)
    assert point.month == "2024-01"
    assert point.roi_percentage == 12.5


def test_trend_point_roi_percentage_is_required_not_optional():
    with pytest.raises(ValidationError):
        TrendPoint(month="2024-01", roi_percentage=None)


# ---------------------------------------------------------------------------
# FinancialDashboardResponse -- US47 Alternate Flow (baseline-cost prompt)
# ---------------------------------------------------------------------------

def test_financial_dashboard_response_roi_percentage_can_be_null_when_baseline_required():
    """architecture.md Sec5.2 GET /analytics/financial: baseline_cost_required=true
    accompanies roi_percentage=null when no baseline_costs row exists yet."""
    resp = FinancialDashboardResponse(
        roi_percentage=None,
        recovered_appointments=0,
        recovered_revenue=0.0,
        trend=[],
        baseline_cost_required=True,
    )
    assert resp.roi_percentage is None
    assert resp.baseline_cost_required is True


def test_financial_dashboard_response_roi_percentage_populated_when_baseline_present():
    resp = FinancialDashboardResponse(
        roi_percentage=42.0,
        recovered_appointments=10,
        recovered_revenue=1500.0,
        trend=[TrendPoint(month="2024-01", roi_percentage=42.0)],
        baseline_cost_required=False,
    )
    assert resp.roi_percentage == 42.0
    assert resp.baseline_cost_required is False
    assert resp.trend[0].month == "2024-01"


def test_financial_dashboard_response_requires_all_non_optional_fields():
    with pytest.raises(ValidationError):
        FinancialDashboardResponse(
            roi_percentage=None,
            recovered_appointments=0,
            recovered_revenue=0.0,
            trend=[],
            # baseline_cost_required omitted
        )


def test_financial_dashboard_response_recovered_appointments_must_be_int():
    with pytest.raises(ValidationError):
        FinancialDashboardResponse(
            roi_percentage=None,
            recovered_appointments="not-an-int",
            recovered_revenue=0.0,
            trend=[],
            baseline_cost_required=True,
        )


# ---------------------------------------------------------------------------
# SetBaselineCostRequest / BaselineCostResponse
# ---------------------------------------------------------------------------

def test_set_baseline_cost_request_accepts_amount_and_effective_period():
    req = SetBaselineCostRequest(amount=12345.67, effective_period=date(2024, 1, 1))
    assert req.amount == 12345.67
    assert req.effective_period == date(2024, 1, 1)


def test_set_baseline_cost_request_parses_iso_date_string():
    req = SetBaselineCostRequest(amount=100.0, effective_period="2024-03-15")
    assert req.effective_period == date(2024, 3, 15)


def test_set_baseline_cost_request_rejects_malformed_date():
    with pytest.raises(ValidationError):
        SetBaselineCostRequest(amount=100.0, effective_period="not-a-date")


def test_set_baseline_cost_request_requires_amount():
    with pytest.raises(ValidationError):
        SetBaselineCostRequest(effective_period="2024-01-01")


def test_baseline_cost_response_holds_id_amount_and_effective_period():
    generated_id = uuid4()
    resp = BaselineCostResponse(id=generated_id, amount=5000.0, effective_period=date(2024, 6, 1))
    assert resp.id == generated_id
    assert isinstance(resp.id, UUID)
    assert resp.amount == 5000.0
    assert resp.effective_period == date(2024, 6, 1)


def test_baseline_cost_response_rejects_non_uuid_id():
    with pytest.raises(ValidationError):
        BaselineCostResponse(id="not-a-uuid", amount=5000.0, effective_period=date(2024, 6, 1))


# ---------------------------------------------------------------------------
# DateRange -- `from` alias
# ---------------------------------------------------------------------------

def test_date_range_accepts_from_alias_on_input():
    dr = DateRange(**{"from": "2024-01-01", "to": "2024-01-31"})
    assert dr.from_ == date(2024, 1, 1)
    assert dr.to == date(2024, 1, 31)


def test_date_range_serializes_back_to_from_alias():
    dr = DateRange(**{"from": "2024-01-01", "to": "2024-01-31"})
    dumped = dr.model_dump(by_alias=True)
    assert dumped == {"from": date(2024, 1, 1), "to": date(2024, 1, 31)}


def test_date_range_rejects_missing_from_field():
    with pytest.raises(ValidationError):
        DateRange(to="2024-01-31")


def test_date_range_accepts_any_well_formed_range_including_predating_go_live():
    """architecture.md Sec5.2 POST /analytics/export Errors: '422 selected range
    predates go-live (no data)' is a service-layer NoDataForRangeError, not a
    schema-level validation -- the schema itself accepts any well-formed date
    range, even one entirely in the past."""
    dr = DateRange(**{"from": "1999-01-01", "to": "1999-01-31"})
    assert dr.from_ == date(1999, 1, 1)
    assert dr.to == date(1999, 1, 31)


def test_date_range_accepts_from_before_or_after_to_without_schema_level_ordering_check():
    """The schema does not itself enforce from <= to; that is not a stated
    field-level contract, so an inverted range is still constructible."""
    dr = DateRange(**{"from": "2024-02-01", "to": "2024-01-01"})
    assert dr.from_ == date(2024, 2, 1)
    assert dr.to == date(2024, 1, 1)


# ---------------------------------------------------------------------------
# ExportDashboardRequest / ExportDashboardResponse
# ---------------------------------------------------------------------------

def test_export_dashboard_request_accepts_valid_enum_values_and_nested_date_range():
    req = ExportDashboardRequest(
        dashboard_type="operational",
        format="csv",
        date_range={"from": "2024-01-01", "to": "2024-01-31"},
    )
    assert req.dashboard_type == "operational"
    assert req.format == "csv"
    assert req.date_range.from_ == date(2024, 1, 1)
    assert req.date_range.to == date(2024, 1, 31)


def test_export_dashboard_request_accepts_financial_and_pdf():
    req = ExportDashboardRequest(
        dashboard_type="financial",
        format="pdf",
        date_range={"from": "2024-01-01", "to": "2024-01-31"},
    )
    assert req.dashboard_type == "financial"
    assert req.format == "pdf"


def test_export_dashboard_request_rejects_invalid_dashboard_type():
    with pytest.raises(ValidationError):
        ExportDashboardRequest(
            dashboard_type="not_a_real_type",
            format="csv",
            date_range={"from": "2024-01-01", "to": "2024-01-31"},
        )


def test_export_dashboard_request_rejects_invalid_format():
    with pytest.raises(ValidationError):
        ExportDashboardRequest(
            dashboard_type="operational",
            format="docx",
            date_range={"from": "2024-01-01", "to": "2024-01-31"},
        )


def test_export_dashboard_request_rejects_missing_date_range():
    with pytest.raises(ValidationError):
        ExportDashboardRequest(dashboard_type="operational", format="csv")


def test_export_dashboard_response_holds_export_id_and_status():
    generated_id = uuid4()
    resp = ExportDashboardResponse(export_id=generated_id, status="accepted")
    assert resp.export_id == generated_id
    assert isinstance(resp.export_id, UUID)
    assert resp.status == "accepted"


def test_export_dashboard_response_rejects_non_uuid_export_id():
    with pytest.raises(ValidationError):
        ExportDashboardResponse(export_id="not-a-uuid", status="accepted")
