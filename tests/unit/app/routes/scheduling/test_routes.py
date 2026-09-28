"""Unit tests for app/routes/scheduling/routes.py.

These tests exercise the router as a thin HTTP adapter over ``SchedulingService``:
the service layer is mocked out (patched at the module level where routes.py
would import it) and only the route's own behaviour -- status codes, request
validation, response shape, and error-envelope translation -- is asserted.
"""

import os

os.environ.setdefault("ENVIRONMENT", "test")

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.common.exceptions.errors import (
    ImportValidationError,
    MissingReasonCodeError,
    RecordLockedError,
    ScheduleConflictError,
)
from app.common.exceptions.handlers import register_exception_handlers
from app.core.dependencies import CurrentUser, get_current_user, get_db
from app.routes.scheduling import routes as scheduling_routes

pytestmark = pytest.mark.asyncio


def _build_app(role: str = "front_office_staff") -> FastAPI:
    app = FastAPI()
    app.include_router(scheduling_routes.router, prefix="/scheduling")
    register_exception_handlers(app)

    async def _current_user_override():
        return CurrentUser(id=uuid4(), role=role)

    async def _db_override():
        yield SimpleNamespace()

    app.dependency_overrides[get_current_user] = _current_user_override
    app.dependency_overrides[get_db] = _db_override
    return app


def _make_appointment(**overrides):
    base = dict(
        id=uuid4(),
        patient_id=uuid4(),
        provider_id=uuid4(),
        chair_id=uuid4(),
        status="booked",
        risk_flag=None,
        confirmation_status=None,
        risk_score_source=None,
        source_metadata={},
        is_late_cancellation=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _create_body():
    now = datetime.now(timezone.utc)
    return {
        "patient_id": str(uuid4()),
        "provider_id": str(uuid4()),
        "chair_id": str(uuid4()),
        "appointment_type": "checkup",
        "scheduled_start": now.isoformat(),
        "scheduled_end": (now + timedelta(minutes=30)).isoformat(),
    }


def _reschedule_body():
    now = datetime.now(timezone.utc)
    return {
        "provider_id": str(uuid4()),
        "chair_id": str(uuid4()),
        "scheduled_start": now.isoformat(),
        "scheduled_end": (now + timedelta(minutes=30)).isoformat(),
    }


@pytest.fixture
def mock_service():
    with patch("app.routes.scheduling.routes.SchedulingService") as service_cls:
        yield service_cls.return_value


# ---------------------------------------------------------------------------
# GET /scheduling/slots
# ---------------------------------------------------------------------------


async def test_propose_slots_returns_200_with_service_result(mock_service):
    slot = {
        "provider_id": str(uuid4()),
        "chair_id": str(uuid4()),
        "start": "2026-10-01T09:00:00+00:00",
        "end": "2026-10-01T09:30:00+00:00",
    }
    mock_service.propose_slots = AsyncMock(
        return_value={"slots": [slot], "nearest_alternatives": []}
    )
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/scheduling/slots",
            params={
                "patient_id": str(uuid4()),
                "from": "2026-10-01T00:00:00+00:00",
                "to": "2026-10-02T00:00:00+00:00",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["slots"]) == 1
    assert body["nearest_alternatives"] == []


async def test_propose_slots_missing_required_query_returns_422(mock_service):
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/scheduling/slots",
            params={"patient_id": str(uuid4()), "from": "2026-10-01T00:00:00+00:00"},
        )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /scheduling/appointments
# ---------------------------------------------------------------------------


async def test_create_appointment_returns_201(mock_service):
    appt = _make_appointment(status="booked")
    mock_service.create = AsyncMock(return_value=(appt, False))
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments",
            json=_create_body(),
            headers={"Idempotency-Key": "key-123"},
        )
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == str(appt.id)
    assert body["status"] == "booked"


async def test_create_appointment_missing_idempotency_key_returns_422(mock_service):
    mock_service.create = AsyncMock()
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/scheduling/appointments", json=_create_body())
    assert resp.status_code == 422
    mock_service.create.assert_not_called()


async def test_create_appointment_conflict_returns_409(mock_service):
    mock_service.create = AsyncMock(side_effect=ScheduleConflictError("slot already booked"))
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments",
            json=_create_body(),
            headers={"Idempotency-Key": "key-456"},
        )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "APPOINTMENT_CONFLICT"


async def test_create_appointment_replay_returns_original_result(mock_service):
    appt = _make_appointment(status="booked")
    mock_service.create = AsyncMock(return_value=(appt, True))
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments",
            json=_create_body(),
            headers={"Idempotency-Key": "replayed-key"},
        )
    assert resp.status_code == 201
    assert resp.json()["id"] == str(appt.id)


async def test_create_appointment_malformed_body_returns_422(mock_service):
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments",
            json={"patient_id": "not-a-uuid"},
            headers={"Idempotency-Key": "key-789"},
        )
    assert resp.status_code == 422


async def test_create_appointment_forbidden_for_wrong_role(mock_service):
    app = _build_app(role="delivery_team")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments",
            json=_create_body(),
            headers={"Idempotency-Key": "key-000"},
        )
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"
    mock_service.create.assert_not_called()


# ---------------------------------------------------------------------------
# PATCH /scheduling/appointments/{id}/reschedule
# ---------------------------------------------------------------------------


async def test_reschedule_appointment_returns_200(mock_service):
    appt = _make_appointment(
        status="rescheduled", source_metadata={"original_start": "2026-09-01T09:00:00+00:00"}
    )
    mock_service.reschedule = AsyncMock(return_value=appt)
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            f"/scheduling/appointments/{uuid4()}/reschedule", json=_reschedule_body()
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "rescheduled"
    assert body["source_metadata"] == {"original_start": "2026-09-01T09:00:00+00:00"}


async def test_reschedule_appointment_conflict_returns_409(mock_service):
    mock_service.reschedule = AsyncMock(side_effect=ScheduleConflictError("new slot conflicts"))
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.patch(
            f"/scheduling/appointments/{uuid4()}/reschedule", json=_reschedule_body()
        )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "APPOINTMENT_CONFLICT"


# ---------------------------------------------------------------------------
# POST /scheduling/appointments/{id}/cancel
# ---------------------------------------------------------------------------


async def test_cancel_appointment_returns_200(mock_service):
    appt = _make_appointment(status="cancelled", is_late_cancellation=True)
    mock_service.cancel = AsyncMock(return_value=appt)
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/scheduling/appointments/{uuid4()}/cancel", json={"reason_code": "patient_request"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cancelled"
    assert body["is_late_cancellation"] is True


async def test_cancel_appointment_missing_reason_code_returns_422(mock_service):
    mock_service.cancel = AsyncMock(side_effect=MissingReasonCodeError("reason_code required"))
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            f"/scheduling/appointments/{uuid4()}/cancel", json={"reason_code": ""}
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "MISSING_REASON_CODE"


# ---------------------------------------------------------------------------
# GET /scheduling/appointments/{id}
# ---------------------------------------------------------------------------


async def test_get_appointment_returns_200_for_front_office_staff(mock_service):
    appt = _make_appointment()
    mock_service.get_by_id = AsyncMock(return_value=appt)
    app = _build_app(role="front_office_staff")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/scheduling/appointments/{appt.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(appt.id)


async def test_get_appointment_returns_200_for_clinic_management(mock_service):
    appt = _make_appointment()
    mock_service.get_by_id = AsyncMock(return_value=appt)
    app = _build_app(role="clinic_management")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/scheduling/appointments/{appt.id}")
    assert resp.status_code == 200


async def test_get_appointment_forbidden_for_delivery_team(mock_service):
    app = _build_app(role="delivery_team")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/scheduling/appointments/{uuid4()}")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


# ---------------------------------------------------------------------------
# POST /scheduling/appointments/{id}/lock
# ---------------------------------------------------------------------------


async def test_lock_appointment_returns_200(mock_service):
    lock_row = SimpleNamespace(
        locked_by="staff-42", expires_at=datetime.now(timezone.utc) + timedelta(minutes=10)
    )
    mock_service.lock = AsyncMock(return_value=lock_row)
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(f"/scheduling/appointments/{uuid4()}/lock")
    assert resp.status_code == 200
    assert resp.json()["locked_by"] == "staff-42"


async def test_lock_appointment_conflict_returns_409(mock_service):
    mock_service.lock = AsyncMock(
        side_effect=RecordLockedError("already locked", details=[{"locked_by": "other-staff"}])
    )
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(f"/scheduling/appointments/{uuid4()}/lock")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "RECORD_LOCKED"


# ---------------------------------------------------------------------------
# POST /scheduling/appointments/import
# ---------------------------------------------------------------------------


async def test_import_appointments_returns_201(mock_service):
    mock_service.import_csv = AsyncMock(return_value={"imported": 3, "errors": []})
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments/import",
            files={"file": ("appointments.csv", b"a,b\n1,2\n", "text/csv")},
        )
    assert resp.status_code == 201
    assert resp.json()["imported"] == 3


async def test_import_appointments_validation_error_returns_422(mock_service):
    mock_service.import_csv = AsyncMock(
        side_effect=ImportValidationError("bad rows", details=[{"row": 1, "error": "bad"}])
    )
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/scheduling/appointments/import",
            files={"file": ("appointments.csv", b"bad,rows", "text/csv")},
        )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "IMPORT_VALIDATION_FAILED"
