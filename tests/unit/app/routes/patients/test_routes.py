"""Unit tests for app/routes/patients/routes.py.

`router` is documented as a thin HTTP adapter over PatientService /
PatientImportService / RecordLockService / ExportService: these tests mock
those service classes (by patching the async methods on the real classes, so
they work regardless of how routes.py names/imports them) and drive the
router directly through a minimal FastAPI app, asserting only the HTTP
contract from architecture.md §5.2 documented in the spec -- status codes,
response envelope shapes, and the specific "actor_id passed through" wiring
called out in the spec's Behaviour section.
"""
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.patients.routes import router
from app.core.dependencies import get_current_user, get_db, CurrentUser
from app.common.exceptions.handlers import register_exception_handlers
from app.common.exceptions.errors import (
    DuplicatePatientError,
    ImportValidationError,
    RecordLockedError,
    NotLockHolderError,
)

FRONT_OFFICE = "front_office_staff"
CLINIC_MGMT = "clinic_management"
DELIVERY_TEAM = "delivery_team"


def _build_app(role: str = FRONT_OFFICE, user_id: "uuid.UUID | None" = None):
    """Fresh app mounting only this module's router, auth/db dependencies faked."""
    user_id = user_id or uuid.uuid4()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)

    async def _fake_current_user():
        return CurrentUser(id=user_id, role=role)

    async def _fake_get_db():
        yield MagicMock()

    app.dependency_overrides[get_current_user] = _fake_current_user
    app.dependency_overrides[get_db] = _fake_get_db
    return app, user_id


def _flat(call):
    """All positional+keyword values from a mock call, for loose contains-checks."""
    if call is None:
        return []
    return list(call.args) + list(call.kwargs.values())


def _contains(call, expected) -> bool:
    values = _flat(call)
    return expected in values or str(expected) in [str(v) for v in values]


# ---------------------------------------------------------------------------
# POST /patients
# ---------------------------------------------------------------------------

def test_create_patient_returns_201_and_passes_actor_id():
    app, user_id = _build_app()
    patient = SimpleNamespace(
        id=uuid.uuid4(), patient_code="PT-00123",
        first_name="Ana", last_name="Silva", status="active",
    )
    with patch(
        "app.services.patients.service.PatientService.create",
        new=AsyncMock(return_value=(patient, None)),
    ) as mock_create:
        resp = TestClient(app).post(
            "/patients",
            json={"first_name": "Ana", "last_name": "Silva", "phone": "+351912345678"},
        )
    assert resp.status_code == 201
    body = resp.json()
    assert body["patient_code"] == "PT-00123"
    assert body["first_name"] == "Ana"
    assert _contains(mock_create.await_args, user_id)


def test_create_patient_duplicate_returns_409_with_candidate_envelope():
    app, _ = _build_app()
    with patch(
        "app.services.patients.service.PatientService.create",
        new=AsyncMock(
            side_effect=DuplicatePatientError(
                details=[{"duplicate_candidate": {"id": str(uuid.uuid4()), "patient_code": "PT-00118"}}]
            )
        ),
    ):
        resp = TestClient(app).post(
            "/patients",
            json={"first_name": "Ana", "last_name": "Silva", "phone": "+351912345678"},
        )
    assert resp.status_code == 409
    envelope = resp.json()["error"]
    assert envelope["code"] == "DUPLICATE_PATIENT"
    assert envelope["details"][0]["duplicate_candidate"]["patient_code"] == "PT-00118"


# ---------------------------------------------------------------------------
# POST /patients/import
# ---------------------------------------------------------------------------

def test_import_patients_committed_returns_201():
    app, _ = _build_app(role=FRONT_OFFICE)
    with patch(
        "app.services.patients.service.PatientImportService.commit",
        new=AsyncMock(return_value={"status": "committed", "imported": 3}),
    ) as mock_commit:
        resp = TestClient(app).post(
            "/patients/import",
            files={"file": ("patients.csv", b"first_name,last_name,phone\nAna,Silva,+351912345678\n", "text/csv")},
        )
    assert resp.status_code == 201
    assert resp.json()["status"] == "committed"
    mock_commit.assert_awaited_once()


def test_import_patients_rejected_returns_422_with_row_errors():
    app, _ = _build_app(role=FRONT_OFFICE)
    with patch(
        "app.services.patients.service.PatientImportService.commit",
        new=AsyncMock(
            side_effect=ImportValidationError(
                details=[{"row": 2, "column": "phone", "message": "invalid phone"}]
            )
        ),
    ):
        resp = TestClient(app).post(
            "/patients/import",
            files={"file": ("patients.csv", b"first_name,last_name,phone\nAna,Silva,bad\n", "text/csv")},
        )
    assert resp.status_code == 422
    envelope = resp.json()["error"]
    assert envelope["code"] == "IMPORT_VALIDATION_FAILED"
    assert envelope["details"][0]["row"] == 2


# ---------------------------------------------------------------------------
# GET /patients
# ---------------------------------------------------------------------------

def test_search_patients_returns_200_with_matching_items():
    app, _ = _build_app()
    item = SimpleNamespace(
        id=uuid.uuid4(), patient_code="PT-00001",
        first_name="Ana", last_name="Silva", phone="+351912345678",
    )
    with patch(
        "app.services.patients.service.PatientService.search",
        new=AsyncMock(return_value=[item]),
    ) as mock_search:
        resp = TestClient(app).get("/patients", params={"q": "Ana"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["patient_code"] == "PT-00001"
    assert _contains(mock_search.await_args, "Ana")


# ---------------------------------------------------------------------------
# GET /patients/{id}
# ---------------------------------------------------------------------------

def test_get_patient_profile_returns_200_with_aggregated_fields():
    app, _ = _build_app()
    patient_id = uuid.uuid4()
    profile = {
        "id": str(patient_id), "patient_code": "PT-00050",
        "first_name": "Ana", "last_name": "Silva", "status": "active",
        "appointments": [], "consent": [], "recall_status": None,
    }
    with patch(
        "app.services.patients.service.PatientService.get_profile",
        new=AsyncMock(return_value=profile),
    ):
        resp = TestClient(app).get(f"/patients/{patient_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_code"] == "PT-00050"
    assert body["appointments"] == []
    assert body["recall_status"] is None


# ---------------------------------------------------------------------------
# POST /patients/{id}/lock
# ---------------------------------------------------------------------------

def test_lock_patient_returns_200():
    app, _ = _build_app()
    patient_id = uuid.uuid4()
    lock = SimpleNamespace(locked_by="jane.doe", expires_at=datetime.now(timezone.utc))
    with patch(
        "app.services.patients.service.RecordLockService.acquire",
        new=AsyncMock(return_value=lock),
    ) as mock_acquire:
        resp = TestClient(app).post(f"/patients/{patient_id}/lock")
    assert resp.status_code == 200
    assert resp.json()["locked_by"] == "jane.doe"
    assert _contains(mock_acquire.await_args, "patient")
    assert _contains(mock_acquire.await_args, patient_id)


def test_lock_patient_returns_409_when_already_locked():
    app, _ = _build_app()
    patient_id = uuid.uuid4()
    with patch(
        "app.services.patients.service.RecordLockService.acquire",
        new=AsyncMock(side_effect=RecordLockedError(details=[{"locked_by": "jane.doe"}])),
    ):
        resp = TestClient(app).post(f"/patients/{patient_id}/lock")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "RECORD_LOCKED"


# ---------------------------------------------------------------------------
# PATCH /patients/{id}
# ---------------------------------------------------------------------------

def test_update_patient_returns_200_and_passes_actor_id():
    app, user_id = _build_app()
    patient_id = uuid.uuid4()
    updated = SimpleNamespace(
        id=patient_id, patient_code="PT-00050",
        first_name="Ana", last_name="Silva", status="active",
    )
    with patch(
        "app.services.patients.service.PatientService.update",
        new=AsyncMock(return_value=updated),
    ) as mock_update:
        resp = TestClient(app).patch(f"/patients/{patient_id}", json={"phone": "+351999999999"})
    assert resp.status_code == 200
    assert resp.json()["patient_code"] == "PT-00050"
    assert _contains(mock_update.await_args, user_id)


def test_update_patient_returns_423_when_caller_is_not_lock_holder():
    app, _ = _build_app()
    patient_id = uuid.uuid4()
    with patch(
        "app.services.patients.service.PatientService.update",
        new=AsyncMock(side_effect=NotLockHolderError("not the lock holder")),
    ):
        resp = TestClient(app).patch(f"/patients/{patient_id}", json={"phone": "+351999999999"})
    assert resp.status_code == 423
    assert resp.json()["error"]["code"] == "NOT_LOCK_HOLDER"


# ---------------------------------------------------------------------------
# POST /patients/{id}/archive
# ---------------------------------------------------------------------------

def test_archive_patient_returns_200():
    app, _ = _build_app()
    patient_id = uuid.uuid4()
    archived = SimpleNamespace(id=patient_id, status="archived", archived_reason="duplicate record")
    with patch(
        "app.services.patients.service.PatientService.archive",
        new=AsyncMock(return_value=archived),
    ) as mock_archive:
        resp = TestClient(app).post(f"/patients/{patient_id}/archive", json={"reason": "duplicate record"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "archived"
    assert body["archived_reason"] == "duplicate record"
    assert _contains(mock_archive.await_args, "duplicate record")


# ---------------------------------------------------------------------------
# POST /patients/export
# ---------------------------------------------------------------------------

def test_export_patients_returns_202_processing():
    app, _ = _build_app()
    with patch(
        "app.services.patients.service.ExportService.generate",
        new=AsyncMock(return_value={"export_id": uuid.uuid4(), "status": "processing", "download_url": None}),
    ) as mock_generate:
        resp = TestClient(app).post(
            "/patients/export",
            json={
                "entity": "patients",
                "date_range": {"from": "2026-01-01", "to": "2026-09-01"},
                "de_identified": False,
            },
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "processing"
    assert body["download_url"] is None
    mock_generate.assert_awaited_once()


# ---------------------------------------------------------------------------
# Role enforcement (public surface: every route declares Depends(require_role(...)))
# ---------------------------------------------------------------------------

def test_role_not_in_allowlist_gets_403():
    app, _ = _build_app(role=DELIVERY_TEAM)
    resp = TestClient(app).get("/patients", params={"q": "Ana"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Router prefix isolation (Interaction contract)
# ---------------------------------------------------------------------------

def test_router_carries_only_its_own_subprefix_not_api_v1():
    app, _ = _build_app()
    # app/main.py is responsible for adding /api/v1 on top; this router alone
    # must expose only its own /patients sub-prefix.
    resp = TestClient(app).get("/api/v1/patients", params={"q": "Ana"})
    assert resp.status_code == 404
