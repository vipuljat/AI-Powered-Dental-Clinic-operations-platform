"""Unit tests for app/routes/recall/routes.py.

These tests exercise the `/recall` router in isolation as a thin HTTP adapter:
- role-based access control (require_role) for every endpoint,
- unauthenticated access is rejected,
- successful requests return the documented status codes and response shapes.

The three domain services this router sits on top of (RecallScanService,
RecallCampaignService, TreatmentRecoveryService) -- and SchedulingService, which
the convert endpoint's Interaction Contract says is consulted to validate the
appointment id -- are replaced with flexible in-process fakes: the exact private
method names the route bodies call to build inline services are not part of this
file's declared public surface, so the fakes respond to *any* awaited method call
with a pre-configured value rather than asserting a guessed method name. This
keeps the tests tied to the documented contract (status codes + schema shapes)
without inventing an internal interface.
"""
import uuid
from datetime import date, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.routes.recall import routes as recall_routes
from app.core.dependencies import get_db, get_current_user
from app.common.exceptions.handlers import register_exception_handlers


class AttrDict(dict):
    """A dict that also supports attribute access (mocks an ORM row or a plain
    dict payload equally well, since it is genuinely unknown which style the
    route bodies expect from their inline-constructed services)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc


def make_flexible_service(return_value):
    """Returns a class whose instances accept any constructor args and whose
    any awaited attribute access returns `return_value` when called."""

    class _FlexibleService:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            async def _method(*args, **kwargs):
                return return_value

            return _method

    return _FlexibleService


def build_app():
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(recall_routes.router)
    return app


def override_user(app, role):
    async def _fake_get_current_user():
        return SimpleNamespace(id=uuid.uuid4(), role=role)

    app.dependency_overrides[get_current_user] = _fake_get_current_user


def override_db(app):
    async def _fake_get_db():
        yield SimpleNamespace()

    app.dependency_overrides[get_db] = _fake_get_db


@pytest.fixture
def app():
    return build_app()


def test_router_prefix_is_recall():
    assert recall_routes.router.prefix == "/recall"


# --------------------------------------------------------------------------
# Role enforcement -- dependency resolution happens before the route body
# runs, so these do not require mocking the underlying services.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_queue_forbidden_for_delivery_team(app):
    override_user(app, "delivery_team")
    override_db(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/queue")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_post_campaigns_forbidden_for_delivery_team(app):
    override_user(app, "delivery_team")
    override_db(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/recall/campaigns", json={"recall_schedule_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_get_compliance_forbidden_for_delivery_team(app):
    override_user(app, "delivery_team")
    override_db(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/compliance")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_get_treatments_forbidden_for_delivery_team(app):
    override_user(app, "delivery_team")
    override_db(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/treatments")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_patch_treatment_valuation_forbidden_for_clinic_management(app):
    # Only front_office_staff is listed for this endpoint -- clinic_management,
    # which IS allowed on every other recall endpoint, must be rejected here.
    override_user(app, "clinic_management")
    override_db(app)
    treatment_id = uuid.uuid4()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            f"/recall/treatments/{treatment_id}", json={"valuation_amount": 500.0}
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_post_treatment_campaigns_forbidden_for_delivery_team(app):
    override_user(app, "delivery_team")
    override_db(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/recall/treatment-campaigns",
            json={"unscheduled_treatment_ids": [str(uuid.uuid4())]},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_convert_treatment_forbidden_for_clinic_management(app):
    # Only front_office_staff is listed for this endpoint.
    override_user(app, "clinic_management")
    override_db(app)
    treatment_id = uuid.uuid4()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/recall/treatments/{treatment_id}/convert",
            json={"appointment_id": str(uuid.uuid4()), "partial": False},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_request_rejected(app):
    # No auth override at all -- Auth section lists every recall endpoint as
    # requiring a valid session (none of them are in the unauthenticated list).
    override_db(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/queue")
    assert resp.status_code == 401


# --------------------------------------------------------------------------
# Success paths -- documented status codes and response shapes.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_recall_queue_returns_200_and_items(app, monkeypatch):
    override_user(app, "front_office_staff")
    override_db(app)
    patient_id = uuid.uuid4()
    item = AttrDict(
        patient_id=patient_id,
        risk_classification="high",
        due_date=date(2026, 9, 1),
        overdue_days=27,
        status="overdue",
    )
    fake_cls = make_flexible_service(AttrDict(items=[item]))
    monkeypatch.setattr(recall_routes, "RecallScanService", fake_cls, raising=False)
    monkeypatch.setattr(recall_routes, "RecallCampaignService", fake_cls, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/queue")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["items"], list)
    assert len(body["items"]) == 1
    assert body["items"][0]["status"] == "overdue"


@pytest.mark.asyncio
async def test_trigger_recall_campaign_returns_202(app, monkeypatch):
    override_user(app, "clinic_management")
    override_db(app)
    campaign_id = uuid.uuid4()
    fake_cls = make_flexible_service(
        {"campaign_id": campaign_id, "dispatched_count": 3, "status": "contacted"}
    )
    monkeypatch.setattr(recall_routes, "RecallCampaignService", fake_cls, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/recall/campaigns",
            json={"recall_schedule_ids": [str(uuid.uuid4()), str(uuid.uuid4())]},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["campaign_id"] == str(campaign_id)
    assert body["dispatched_count"] == 3
    assert body["status"] == "contacted"


@pytest.mark.asyncio
async def test_get_recall_compliance_returns_baseline_pending_state(app, monkeypatch):
    # US36 Alternate Flow: baseline_pending=true accompanies baseline_rate=null,
    # while compliance_rate is still populated.
    override_user(app, "front_office_staff")
    override_db(app)
    fake_cls = make_flexible_service(
        {
            "compliance_rate": 0.42,
            "baseline_rate": None,
            "baseline_pending": True,
            "period": "2026-W39",
        }
    )
    monkeypatch.setattr(recall_routes, "RecallCampaignService", fake_cls, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/compliance")

    assert resp.status_code == 200
    body = resp.json()
    assert body["compliance_rate"] == 0.42
    assert body["baseline_rate"] is None
    assert body["baseline_pending"] is True


@pytest.mark.asyncio
async def test_list_unscheduled_treatments_returns_200(app, monkeypatch):
    override_user(app, "front_office_staff")
    override_db(app)
    treatment_id = uuid.uuid4()
    patient_id = uuid.uuid4()
    row = AttrDict(
        id=treatment_id,
        patient_id=patient_id,
        treatment_code="D2740",
        valuation_amount=1200.0,
        days_unscheduled=45,
        status="unscheduled",
    )
    fake_cls = make_flexible_service([row])
    monkeypatch.setattr(recall_routes, "TreatmentRecoveryService", fake_cls, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/recall/treatments")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["items"], list)
    assert len(body["items"]) == 1
    assert body["items"][0]["treatment_code"] == "D2740"


@pytest.mark.asyncio
async def test_update_treatment_valuation_returns_200(app, monkeypatch):
    override_user(app, "front_office_staff")
    override_db(app)
    treatment_id = uuid.uuid4()
    patient_id = uuid.uuid4()
    fake_cls = make_flexible_service(
        AttrDict(
            id=treatment_id,
            patient_id=patient_id,
            treatment_code="D2740",
            valuation_amount=750.0,
            days_unscheduled=45,
            status="unscheduled",
        )
    )
    monkeypatch.setattr(recall_routes, "TreatmentRecoveryService", fake_cls, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            f"/recall/treatments/{treatment_id}", json={"valuation_amount": 750.0}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["valuation_amount"] == 750.0


@pytest.mark.asyncio
async def test_trigger_treatment_campaign_returns_202(app, monkeypatch):
    override_user(app, "clinic_management")
    override_db(app)
    campaign_id = uuid.uuid4()
    fake_cls = make_flexible_service(
        {"campaign_id": campaign_id, "dispatched_count": 2, "status": "contacted"}
    )
    monkeypatch.setattr(recall_routes, "TreatmentRecoveryService", fake_cls, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/recall/treatment-campaigns",
            json={"unscheduled_treatment_ids": [str(uuid.uuid4())]},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["campaign_id"] == str(campaign_id)
    assert body["dispatched_count"] == 2


@pytest.mark.asyncio
async def test_convert_treatment_returns_200_booked(app, monkeypatch):
    override_user(app, "front_office_staff")
    override_db(app)
    treatment_id = uuid.uuid4()
    appointment_id = uuid.uuid4()

    scheduling_fake = make_flexible_service(AttrDict(id=appointment_id))
    treatment_fake = make_flexible_service(
        AttrDict(id=treatment_id, status="booked", appointment_id=appointment_id)
    )
    monkeypatch.setattr(recall_routes, "SchedulingService", scheduling_fake, raising=False)
    monkeypatch.setattr(recall_routes, "TreatmentRecoveryService", treatment_fake, raising=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/recall/treatments/{treatment_id}/convert",
            json={"appointment_id": str(appointment_id), "partial": False},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "booked"
    assert body["appointment_id"] == str(appointment_id)
