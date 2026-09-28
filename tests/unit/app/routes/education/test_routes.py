"""Unit tests for app/routes/education/routes.py.

These tests exercise the FastAPI router in isolation: `EducationTriggerService`'s
two read-only wrapper methods (`get_tracking_funnel` / `list_deliveries_for_patient`)
are patched directly on the real service class (app/services/education/service.py),
so no repository/outreach construction is required beyond whatever the route itself
does internally, and no assumption is made about how routes.py imports the class.

Role checks are exercised via FastAPI's `dependency_overrides` on the shared
`get_current_user` dependency (the thing every `require_role(...)` closure itself
depends on), rather than by guessing the private `_dependency` closures that
`require_role(*roles)` returns fresh on every call.
"""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.common.enums import ContentDeliveryStatus, EducationTriggerType
from app.common.exceptions.handlers import register_exception_handlers
from app.core.dependencies import get_current_user
from app.routes.education.routes import router
import app.services.education.service as education_service_module


class _FakeUser:
    """Minimal stand-in for CurrentUser -- only the `role` attribute is used
    by require_role's dependency."""

    def __init__(self, role: str) -> None:
        self.role = role
        self.id = uuid4()


def _build_app(role: str | None):
    app = FastAPI()
    register_exception_handlers(app)
    # The router already carries its own "/education" sub-prefix per this
    # file's Responsibility/Interaction-contract; app/main.py is the one that
    # adds the shared "/api/v1" on top -- so we mount it bare here.
    app.include_router(router)

    if role is not None:
        async def _fake_get_current_user():
            return _FakeUser(role)

        app.dependency_overrides[get_current_user] = _fake_get_current_user

    return app


def _flat_call_values(call):
    return list(call.args) + list(call.kwargs.values())


@pytest.mark.asyncio
async def test_get_tracking_returns_200_for_front_office_staff(monkeypatch):
    funnel = {"delivered": 10, "opened": 4, "completed": 2, "channel_limited": ["sms"]}
    mock_method = AsyncMock(return_value=funnel)
    monkeypatch.setattr(
        education_service_module.EducationTriggerService, "get_tracking_funnel", mock_method
    )

    app = _build_app("front_office_staff")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/tracking")

    assert resp.status_code == 200
    body = resp.json()
    assert body["delivered"] == 10
    assert body["opened"] == 4
    assert body["completed"] == 2
    assert body["channel_limited"] == ["sms"]


@pytest.mark.asyncio
async def test_get_tracking_returns_200_for_clinic_management(monkeypatch):
    funnel = {"delivered": 1, "opened": 0, "completed": 0, "channel_limited": []}
    monkeypatch.setattr(
        education_service_module.EducationTriggerService,
        "get_tracking_funnel",
        AsyncMock(return_value=funnel),
    )

    app = _build_app("clinic_management")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/tracking")

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_get_tracking_passes_none_when_content_item_id_omitted(monkeypatch):
    mock_method = AsyncMock(
        return_value={"delivered": 0, "opened": 0, "completed": 0, "channel_limited": []}
    )
    monkeypatch.setattr(
        education_service_module.EducationTriggerService, "get_tracking_funnel", mock_method
    )

    app = _build_app("front_office_staff")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/tracking")

    assert resp.status_code == 200
    mock_method.assert_awaited_once()
    assert None in _flat_call_values(mock_method.call_args)


@pytest.mark.asyncio
async def test_get_tracking_passes_supplied_content_item_id(monkeypatch):
    mock_method = AsyncMock(
        return_value={"delivered": 0, "opened": 0, "completed": 0, "channel_limited": []}
    )
    monkeypatch.setattr(
        education_service_module.EducationTriggerService, "get_tracking_funnel", mock_method
    )
    content_item_id = uuid4()

    app = _build_app("front_office_staff")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/education/tracking?content_item_id={content_item_id}")

    assert resp.status_code == 200
    mock_method.assert_awaited_once()
    call_values = [str(v) for v in _flat_call_values(mock_method.call_args)]
    assert str(content_item_id) in call_values


@pytest.mark.asyncio
async def test_get_tracking_rejects_malformed_content_item_id(monkeypatch):
    monkeypatch.setattr(
        education_service_module.EducationTriggerService,
        "get_tracking_funnel",
        AsyncMock(return_value={"delivered": 0, "opened": 0, "completed": 0, "channel_limited": []}),
    )

    app = _build_app("front_office_staff")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/tracking?content_item_id=not-a-uuid")

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_get_tracking_forbidden_for_delivery_team(monkeypatch):
    monkeypatch.setattr(
        education_service_module.EducationTriggerService,
        "get_tracking_funnel",
        AsyncMock(return_value={"delivered": 0, "opened": 0, "completed": 0, "channel_limited": []}),
    )

    app = _build_app("delivery_team")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/tracking")

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_get_tracking_unauthenticated_returns_401():
    app = _build_app(None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/tracking")

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_list_deliveries_returns_200_for_front_office_staff(monkeypatch):
    trigger_type = next(iter(EducationTriggerType))
    status_value = next(iter(ContentDeliveryStatus))
    content_item_id = uuid4()
    deliveries = [
        {
            "content_item_id": content_item_id,
            "trigger_type": trigger_type,
            "status": status_value,
            "delivered_at": None,
        }
    ]
    mock_method = AsyncMock(return_value=deliveries)
    monkeypatch.setattr(
        education_service_module.EducationTriggerService,
        "list_deliveries_for_patient",
        mock_method,
    )

    patient_id = uuid4()
    app = _build_app("front_office_staff")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/education/deliveries?patient_id={patient_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["content_item_id"] == str(content_item_id)
    assert body["items"][0]["delivered_at"] is None
    mock_method.assert_awaited_once()
    call_values = [str(v) for v in _flat_call_values(mock_method.call_args)]
    assert str(patient_id) in call_values


@pytest.mark.asyncio
async def test_list_deliveries_requires_patient_id_query_param(monkeypatch):
    monkeypatch.setattr(
        education_service_module.EducationTriggerService,
        "list_deliveries_for_patient",
        AsyncMock(return_value=[]),
    )

    app = _build_app("front_office_staff")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/education/deliveries")

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_list_deliveries_forbidden_for_clinic_management(monkeypatch):
    # Per this route's declared require_role("front_office_staff") -- clinic_management
    # is NOT in the allowed-roles list for this specific read, unlike /education/tracking.
    monkeypatch.setattr(
        education_service_module.EducationTriggerService,
        "list_deliveries_for_patient",
        AsyncMock(return_value=[]),
    )

    patient_id = uuid4()
    app = _build_app("clinic_management")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/education/deliveries?patient_id={patient_id}")

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_deliveries_unauthenticated_returns_401():
    patient_id = uuid4()
    app = _build_app(None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/education/deliveries?patient_id={patient_id}")

    assert resp.status_code == 401
