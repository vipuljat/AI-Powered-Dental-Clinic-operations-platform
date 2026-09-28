"""Pydantic v2 request/response models for every `/analytics/*` endpoint
(architecture.md §5.2).

Field names/types/aliases here are exactly what `services/analytics/service.py`'s
return dicts must match key-for-key, and exactly what `routes/analytics/routes.py`
validates requests against. No additional out-of-band agreement exists beyond
what these signatures carry.
"""

from datetime import date
from uuid import UUID

from pydantic import BaseModel, Field

from app.common.enums import DashboardType, ExportFormat


class MetricItem(BaseModel):
    name: str
    value: float
    baseline: float | None = None


class OperationalDashboardResponse(BaseModel):
    metrics: list[MetricItem]
    data_delayed: bool


class TrendPoint(BaseModel):
    month: str
    roi_percentage: float


class FinancialDashboardResponse(BaseModel):
    # architecture.md §5.2 GET /analytics/financial: `roi_percentage` is
    # nullable because US47's Alternate Flow requires the system to prompt
    # Clinic Management for a baseline cost before ROI can be calculated —
    # when no `baseline_costs` row exists for the requested period,
    # `roi_percentage` is null and `baseline_cost_required` is true.
    roi_percentage: float | None = None
    recovered_appointments: int
    recovered_revenue: float
    trend: list[TrendPoint]
    baseline_cost_required: bool


class SetBaselineCostRequest(BaseModel):
    amount: float
    effective_period: date


class BaselineCostResponse(BaseModel):
    id: UUID
    amount: float
    effective_period: date


class DateRange(BaseModel):
    from_: date = Field(alias="from")
    to: date

    model_config = {"populate_by_name": True}


class ExportDashboardRequest(BaseModel):
    dashboard_type: DashboardType
    format: ExportFormat
    date_range: DateRange


class ExportDashboardResponse(BaseModel):
    export_id: UUID
    status: str
