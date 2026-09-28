"""HTTP routes for the analytics module (architecture.md §5.2).

Responsibility: `router` (prefix `/analytics`) is a thin adapter over
``OperationalAnalyticsService``/``FinancialAnalyticsService``/
``DashboardExportService`` — it validates the request, calls the service
layer, and shapes the response; it never touches a repository or the DB
session directly beyond handing it through to the service call.

Interaction contract: ``app/main.py`` mounts this ``router`` with the shared
``/api/v1`` prefix added on top of this file's own ``/analytics`` sub-prefix
(this file declares only its own sub-prefix, per the project's routing
convention).
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from app.core.config import get_settings
from app.core.database import get_db
from app.core.dependencies import require_role
from app.core.storage import get_storage_client
from app.repositories.analytics.repository import (
    BaselineCostRepository,
    DashboardExportRepository,
    FinancialRoiRepository,
    OperationalMetricsRepository,
)
from app.repositories.console.repository import AuditLogRepository
from app.repositories.recall.repository import RecallScheduleRepository
from app.repositories.scheduling.repository import AppointmentRepository
from app.repositories.waitlist.repository import IdleChairAlertRepository
from app.schemas.analytics.schemas import (
    BaselineCostResponse,
    ExportDashboardRequest,
    ExportDashboardResponse,
    FinancialDashboardResponse,
    OperationalDashboardResponse,
    SetBaselineCostRequest,
)
from app.services.analytics.service import (
    DashboardExportService,
    FinancialAnalyticsService,
    OperationalAnalyticsService,
)
from app.services.console.service import AuditService

router = APIRouter(prefix="/analytics")


def _operational_service() -> OperationalAnalyticsService:
    """Constructed per-call from stateless repositories (architecture.md
    §5.4 repositories carry no session state; the DB session itself is
    threaded through each method call, not the constructor)."""
    return OperationalAnalyticsService(
        OperationalMetricsRepository(),
        AppointmentRepository(),
        IdleChairAlertRepository(),
        RecallScheduleRepository(),
    )


def _financial_service() -> FinancialAnalyticsService:
    return FinancialAnalyticsService(
        FinancialRoiRepository(),
        BaselineCostRepository(),
        AuditService(AuditLogRepository()),
    )


def _export_service() -> DashboardExportService:
    # Object-storage client is resolved lazily here (per request), never at
    # import time, so ENVIRONMENT=test's in-memory fake substitution (wired
    # in app/core/storage.py) is honoured and no real client/socket is
    # opened merely by importing this module.
    storage = get_storage_client(get_settings())
    return DashboardExportService(
        DashboardExportRepository(),
        _operational_service(),
        _financial_service(),
        storage,
    )


@router.get("/operational", response_model=OperationalDashboardResponse)
async def get_operational_dashboard(
    from_: "date | None" = Query(None, alias="from"),
    to: "date | None" = None,
    metric: str | None = None,
    user=Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> OperationalDashboardResponse:
    """architecture.md §5.2 `GET /analytics/operational` — 200; FR-E9.1/US46
    the 6 RAR001 operational metrics, optionally narrowed to one ``metric``
    and/or one ``[from, to]`` window."""
    service = _operational_service()
    payload = await service.get_metrics(db, from_, to, metric)
    return OperationalDashboardResponse.model_validate(payload)


@router.get("/financial", response_model=FinancialDashboardResponse)
async def get_financial_dashboard(
    period: str = "monthly",
    user=Depends(require_role("clinic_management")),
    db=Depends(get_db),
) -> FinancialDashboardResponse:
    """architecture.md §5.2 `GET /analytics/financial` — 200, restricted to
    ``clinic_management``; FR-E9.2/US47 financial ROI dashboard."""
    service = _financial_service()
    payload = await service.get_roi(db, period)
    return FinancialDashboardResponse.model_validate(payload)


@router.post("/financial/baseline-cost", response_model=BaselineCostResponse, status_code=201)
async def set_financial_baseline_cost(
    body: SetBaselineCostRequest,
    user=Depends(require_role("clinic_management")),
    db=Depends(get_db),
) -> BaselineCostResponse:
    """architecture.md §5.2 `POST /analytics/financial/baseline-cost` — 201,
    restricted to ``clinic_management``; T119 AC captures a new baseline
    cost figure used to gate/derive ROI% (FR-E9.2/US47 Alternate Flow)."""
    service = _financial_service()
    baseline = await service.set_baseline_cost(db, body.amount, body.effective_period, user.id)
    return BaselineCostResponse(
        id=baseline.id, amount=baseline.amount, effective_period=baseline.effective_period
    )


@router.post("/export", response_model=ExportDashboardResponse, status_code=202)
async def export_dashboard(
    body: ExportDashboardRequest,
    user=Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> ExportDashboardResponse:
    """architecture.md §5.2 `POST /analytics/export` — 202 on success, 422
    when the requested range predates the platform's earliest data
    (FR-E9.3/US48, ``NoDataForRangeError`` -> 422 via the global handler)."""
    service = _export_service()
    payload = await service.generate(
        db,
        body.dashboard_type.value,
        body.format.value,
        body.date_range.from_,
        body.date_range.to,
        user.id,
    )
    return ExportDashboardResponse(export_id=payload["export_id"], status=payload["status"])
