"""Unit tests for app/routes/analytics/routes.py.

These exercise the router as a thin HTTP adapter: dependencies (get_current_user,
get_db) are overridden and the OperationalAnalyticsService / FinancialAnalyticsService /
DashboardExportService classes referenced from the routes module are monkeypatched with
fakes, so only this file's own contract (status codes, role gating, request/response
shape per architecture.md Sec 5.2) is under test.
"""
import os

os.environ.setdefault("ENVIRONMENT", "test")

import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.analytics import routes as routes_module
from app.core.dependencies import CurrentUser, get_current_user, get_db
from app.common.exceptions.handlers import register_exception_handlers
from app.common.enums import DashboardType, ExportFormat


def _fake_user(role: str) -> CurrentUser:
    # model_construct bypasses field validation so we never depend on the full,
    # undeclared field set of CurrentUser -- only the `role` attribute matters here.
    return CurrentUser.model_construct(id=uuid.uuid4(), role=role, email="user@example.com")


def _build_app(role: str) -> FastAPI:
    app = FastAPI()
    app.include_router(routes_module.router)
    register_exception_handlers(app)
    app.dependency_overrides[get_current_user] = lambda: _fake_user(role)
    app.dependency_overrides[get_db] = lambda: object()
    return app


def _make_operational_service(response, calls):
    class FakeOperationalService:
        def __init__(self, *args, **kwargs):
            pass

        async def get_metrics(self, db, date_from, date_to, metric):
            calls.append({"date_from": date_from, "date_to": date_to, "metric": metric})
            return response

    return FakeOperationalService


def _make_financial_service(get_roi_response, calls, baseline_result=None):
    class FakeFinancialService:
        def __init__(self, *args, **kwargs):
            pass

        async def get_roi(self, db, period):
            calls.append({"period": period})
            return get_roi_response

        async def set_baseline_cost(self, db, amount, effective_period, actor_id):
            calls.append(
                {"amount": amount, "effective_period": effective_period, "actor_id": actor_id}
            )
            return baseline_result

    return FakeFinancialService


def _make_export_service(response, calls):
    class FakeExportService:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, db, dashboard_type, format, date_from, date_to, actor_id):
            calls.append(
                {
                    "dashboard_type": dashboard_type,
                    "format": format,
                    "date_from": date_from,
                    "date_to": date_to,
                    "actor_id": actor_id,
                }
            )
            return response

    return FakeExportService


class _AttrDict(dict):
    """Dict that also supports attribute access, so it works whether the route
    builds the response via `**result`, `.model_validate(obj, from_attributes=True)`
    or direct attribute reads."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


# ---------------------------------------------------------------------------
# GET /analytics/operational
# ---------------------------------------------------------------------------


def test_get_operational_dashboard_200_for_front_office_staff(monkeypatch):
    calls = []
    fake = _make_operational_service(
        {"metrics": [{"name": "no_show_rate", "value": 12.5, "baseline": 15.0}], "data_delayed": False},
        calls,
    )
    monkeypatch.setattr(routes_module, "OperationalAnalyticsService", fake)

    app = _build_app("front_office_staff")
    client = TestClient(app)
    resp = client.get("/analytics/operational")

    assert resp.status_code == 200
    body = resp.json()
    assert body["metrics"][0]["name"] == "no_show_rate"
    assert body["data_delayed"] is False


def test_get_operational_dashboard_200_for_clinic_management(monkeypatch):
    calls = []
    fake = _make_operational_service(
        {"metrics": [{"name": "utilisation_rate", "value": 80.0, "baseline": None}], "data_delayed": True},
        calls,
    )
    monkeypatch.setattr(routes_module, "OperationalAnalyticsService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.get("/analytics/operational")

    assert resp.status_code == 200
    assert resp.json()["data_delayed"] is True


def test_get_operational_dashboard_403_for_delivery_team(monkeypatch):
    fake = _make_operational_service({"metrics": [], "data_delayed": False}, [])
    monkeypatch.setattr(routes_module, "OperationalAnalyticsService", fake)

    app = _build_app("delivery_team")
    client = TestClient(app)
    resp = client.get("/analytics/operational")

    assert resp.status_code == 403
    assert "error" in resp.json()


def test_get_operational_dashboard_forwards_query_params(monkeypatch):
    calls = []
    fake = _make_operational_service({"metrics": [], "data_delayed": False}, calls)
    monkeypatch.setattr(routes_module, "OperationalAnalyticsService", fake)

    app = _build_app("front_office_staff")
    client = TestClient(app)
    resp = client.get(
        "/analytics/operational",
        params={"from": "2026-01-01", "to": "2026-01-31", "metric": "no_show_rate"},
    )

    assert resp.status_code == 200
    assert len(calls) == 1
    assert str(calls[0]["date_from"]) == "2026-01-01"
    assert str(calls[0]["date_to"]) == "2026-01-31"
    assert calls[0]["metric"] == "no_show_rate"


# ---------------------------------------------------------------------------
# GET /analytics/financial
# ---------------------------------------------------------------------------


def test_get_financial_dashboard_200_for_clinic_management(monkeypatch):
    calls = []
    fake = _make_financial_service(
        {
            "roi_percentage": 8.2,
            "recovered_appointments": 10,
            "recovered_revenue": 5000.0,
            "trend": [{"month": "2026-08", "roi_percentage": 7.1}],
            "baseline_cost_required": False,
        },
        calls,
    )
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.get("/analytics/financial")

    assert resp.status_code == 200
    body = resp.json()
    assert body["roi_percentage"] == 8.2
    assert body["baseline_cost_required"] is False
    assert body["trend"][0]["month"] == "2026-08"


def test_get_financial_dashboard_403_for_front_office_staff(monkeypatch):
    fake = _make_financial_service(
        {
            "roi_percentage": None,
            "recovered_appointments": 0,
            "recovered_revenue": 0.0,
            "trend": [],
            "baseline_cost_required": True,
        },
        [],
    )
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("front_office_staff")
    client = TestClient(app)
    resp = client.get("/analytics/financial")

    assert resp.status_code == 403
    assert "error" in resp.json()


def test_get_financial_dashboard_baseline_cost_required_state(monkeypatch):
    """US47 Alternate Flow: roi_percentage null + baseline_cost_required true is a 200,
    not an error response."""
    calls = []
    fake = _make_financial_service(
        {
            "roi_percentage": None,
            "recovered_appointments": 3,
            "recovered_revenue": 1200.0,
            "trend": [],
            "baseline_cost_required": True,
        },
        calls,
    )
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.get("/analytics/financial")

    assert resp.status_code == 200
    body = resp.json()
    assert body["roi_percentage"] is None
    assert body["baseline_cost_required"] is True


def test_get_financial_dashboard_defaults_period_to_monthly(monkeypatch):
    calls = []
    fake = _make_financial_service(
        {
            "roi_percentage": 1.0,
            "recovered_appointments": 1,
            "recovered_revenue": 1.0,
            "trend": [],
            "baseline_cost_required": False,
        },
        calls,
    )
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.get("/analytics/financial")

    assert resp.status_code == 200
    assert calls[0]["period"] == "monthly"


# ---------------------------------------------------------------------------
# POST /analytics/financial/baseline-cost
# ---------------------------------------------------------------------------


def test_set_baseline_cost_201_for_clinic_management(monkeypatch):
    calls = []
    new_id = uuid.uuid4()
    result = _AttrDict(id=str(new_id), amount=500.0, effective_period="2026-01-01")
    fake = _make_financial_service({}, calls, baseline_result=result)
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.post(
        "/analytics/financial/baseline-cost",
        json={"amount": 500.0, "effective_period": "2026-01-01"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["amount"] == 500.0
    assert body["effective_period"] == "2026-01-01"
    assert calls[0]["amount"] == 500.0


def test_set_baseline_cost_403_for_front_office_staff(monkeypatch):
    result = _AttrDict(id=str(uuid.uuid4()), amount=500.0, effective_period="2026-01-01")
    fake = _make_financial_service({}, [], baseline_result=result)
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("front_office_staff")
    client = TestClient(app)
    resp = client.post(
        "/analytics/financial/baseline-cost",
        json={"amount": 500.0, "effective_period": "2026-01-01"},
    )

    assert resp.status_code == 403
    assert "error" in resp.json()


def test_set_baseline_cost_422_on_missing_amount(monkeypatch):
    result = _AttrDict(id=str(uuid.uuid4()), amount=500.0, effective_period="2026-01-01")
    fake = _make_financial_service({}, [], baseline_result=result)
    monkeypatch.setattr(routes_module, "FinancialAnalyticsService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.post(
        "/analytics/financial/baseline-cost",
        json={"effective_period": "2026-01-01"},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_FAILED"


# ---------------------------------------------------------------------------
# POST /analytics/export
# ---------------------------------------------------------------------------


def _valid_export_body():
    return {
        "dashboard_type": list(DashboardType)[0].value,
        "format": list(ExportFormat)[0].value,
        "date_range": {"from": "2026-01-01", "to": "2026-01-31"},
    }


def test_export_dashboard_202_for_front_office_staff(monkeypatch):
    calls = []
    export_id = uuid.uuid4()
    fake = _make_export_service({"export_id": str(export_id), "status": "processing"}, calls)
    monkeypatch.setattr(routes_module, "DashboardExportService", fake)

    app = _build_app("front_office_staff")
    client = TestClient(app)
    resp = client.post("/analytics/export", json=_valid_export_body())

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "processing"
    assert body["export_id"] == str(export_id)
    assert len(calls) == 1


def test_export_dashboard_202_for_clinic_management(monkeypatch):
    calls = []
    fake = _make_export_service({"export_id": str(uuid.uuid4()), "status": "processing"}, calls)
    monkeypatch.setattr(routes_module, "DashboardExportService", fake)

    app = _build_app("clinic_management")
    client = TestClient(app)
    resp = client.post("/analytics/export", json=_valid_export_body())

    assert resp.status_code == 202


def test_export_dashboard_403_for_delivery_team(monkeypatch):
    fake = _make_export_service({"export_id": str(uuid.uuid4()), "status": "processing"}, [])
    monkeypatch.setattr(routes_module, "DashboardExportService", fake)

    app = _build_app("delivery_team")
    client = TestClient(app)
    resp = client.post("/analytics/export", json=_valid_export_body())

    assert resp.status_code == 403
    assert "error" in resp.json()


def test_export_dashboard_422_on_missing_fields(monkeypatch):
    fake = _make_export_service({"export_id": str(uuid.uuid4()), "status": "processing"}, [])
    monkeypatch.setattr(routes_module, "DashboardExportService", fake)

    app = _build_app("front_office_staff")
    client = TestClient(app)
    resp = client.post("/analytics/export", json={})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_FAILED"
