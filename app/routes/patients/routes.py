"""Thin HTTP adapter over ``PatientService``/``PatientImportService``/
``RecordLockService``/``ExportService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema, delegates to the service layer, and shapes the response. This
module's ``router`` carries only its own ``/patients`` sub-prefix; the shared
``/api/v1`` prefix is added on top by ``app/main.py`` when it calls
``app.include_router`` (Interaction contract).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.encoders import jsonable_encoder

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table
# is not already registered there — but `app.core.dependencies` transitively
# imports that waitlist repository (via `services.console.service`) before
# this module otherwise would import the real `app.models.patients.models`.
# Importing the real patients repository/model first, here, ensures the real
# declarative `Patient` table claims the "patients" name before that stub
# loop ever runs, so its `if _table_name not in Base.metadata.tables` check
# correctly no-ops instead of racing a second, conflicting declaration.
from app.repositories.patients.repository import (  # isort: skip
    PatientImportRepository,
    PatientRepository,
    RecordLockRepository,
)

from app.common.exceptions.errors import ImportValidationError
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.core.storage import get_storage_client
from app.repositories.console.repository import AuditLogRepository
from app.repositories.scheduling.repository import AppointmentRepository
from app.schemas.patients.schemas import (
    ArchivePatientRequest,
    ArchivePatientResponse,
    CreatePatientRequest,
    ExportPatientDataRequest,
    ExportResponse,
    LockResponse,
    PatientProfileResponse,
    PatientResponse,
    PatientSearchItem,
    PatientSearchResponse,
    UpdatePatientRequest,
)
from app.services.console.service import AuditService
from app.services.patients.service import (
    ExportService,
    PatientImportService,
    PatientService,
    RecordLockService,
)

router = APIRouter(prefix="/patients")


def _audit_service() -> AuditService:
    # Each request builds its own thin service/repository graph over the
    # request-scoped `AsyncSession` (`Depends(get_db)`) — no collaborator
    # here is a module-level singleton, matching the rest of this codebase's
    # `require_permission`/`AuthService` wiring (app/core/dependencies.py).
    return AuditService(AuditLogRepository())


def _patient_service() -> PatientService:
    return PatientService(PatientRepository(), RecordLockRepository(), _audit_service())


def _import_service() -> PatientImportService:
    return PatientImportService(PatientImportRepository(), PatientRepository(), _audit_service())


def _lock_service() -> RecordLockService:
    return RecordLockService(RecordLockRepository())


def _export_service() -> ExportService:
    settings = get_settings()
    return ExportService(PatientRepository(), AppointmentRepository(), get_storage_client(settings))


@router.post("", response_model=PatientResponse, status_code=201)
async def create_patient(
    body: CreatePatientRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> PatientResponse:
    """FR-E2.1/US8: 201 on success, 409 (``DuplicatePatientError``) if a
    phone-matching record already exists — the exception is left to
    propagate to the registered global handler, never caught here.
    """
    service = _patient_service()
    patient, _ = await service.create(db, body.model_dump(), user.id)
    return PatientResponse(
        id=patient.id,
        patient_code=patient.patient_code,
        first_name=patient.first_name,
        last_name=patient.last_name,
        status=patient.status,
    )


@router.post("/import", status_code=201)
async def import_patients(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> dict:
    """FR-E2.2/US9/A10: multipart/form-data upload; returns 201 when the
    whole batch committed, or 422 when it was rejected (architecture.md
    §5.2).

    Structural problems (missing columns, file/row-count limits exceeded)
    raise ``ImportValidationError`` (422) directly out of
    ``PatientImportService.commit`` -> ``validate``, and are left to
    propagate to the registered global handler. Row-level rejection instead
    comes back as a normal ``status="rejected"`` result (per that service's
    own docstring, so every bad row can be reported in one response) — that
    case is translated into the same 422 by raising
    ``ImportValidationError`` here, with the per-row errors as ``details``,
    matching this error class's own documented shape
    (``app/common/exceptions/errors.py``: "details = row-level errors[]").
    """
    service = _import_service()
    file_bytes = await file.read()
    result = await service.commit(db, file_bytes, file.filename or "upload.csv", user.id)

    if result["status"] == "rejected":
        raise ImportValidationError(
            message=f"Import batch rejected: {result['error_count']} row(s) failed validation.",
            details=result["errors"],
        )

    return jsonable_encoder(result)


@router.get("", response_model=PatientSearchResponse)
async def search_patients(
    q: str,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> PatientSearchResponse:
    """FR-E2.3/US10: matches name, phone, or PatientID."""
    service = _patient_service()
    patients = await service.search(db, q)
    return PatientSearchResponse(
        items=[
            PatientSearchItem(
                id=p.id,
                patient_code=p.patient_code,
                first_name=p.first_name,
                last_name=p.last_name,
                phone=p.phone,
            )
            for p in patients
        ]
    )


@router.get("/{id}", response_model=PatientProfileResponse)
async def get_patient(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> PatientProfileResponse:
    """FR-E2.3/US10: aggregated profile (patient + appointments + consent +
    recall_status) — 404 (``NotFoundError``) propagates unchanged if ``id``
    does not exist.
    """
    service = _patient_service()
    profile = await service.get_profile(db, id)
    return PatientProfileResponse(**profile)


@router.post("/{id}/lock", response_model=LockResponse)
async def lock_patient(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> LockResponse:
    """FR-E2.4/US10: 200 on success, 409 (``RecordLockedError``) if another
    staff member currently holds this record's lock.
    """
    service = _lock_service()
    lock = await service.acquire(db, "patient", id, user.id)
    return LockResponse(locked_by=str(lock.locked_by_staff_id), expires_at=lock.expires_at)


@router.patch("/{id}", response_model=PatientResponse)
async def update_patient(
    id: UUID,
    body: UpdatePatientRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> PatientResponse:
    """FR-E2.4/US10: requires an active lock held by the caller — ``user.id``
    is passed into ``PatientService.update``, which raises
    ``NotLockHolderError`` (423) if the caller is not the current lock
    holder (architecture.md §5.2).
    """
    service = _patient_service()
    data = body.model_dump(exclude_unset=True)
    patient = await service.update(db, id, data, user.id)
    return PatientResponse(
        id=patient.id,
        patient_code=patient.patient_code,
        first_name=patient.first_name,
        last_name=patient.last_name,
        status=patient.status,
    )


@router.post("/{id}/archive", response_model=ArchivePatientResponse)
async def archive_patient(
    id: UUID,
    body: ArchivePatientRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> ArchivePatientResponse:
    """FR-E2.5/US11: sets ``status="archived"``; future outreach/recall
    eligibility is halted at the query layer (``PatientRepository.list_active``),
    not here.
    """
    service = _patient_service()
    patient = await service.archive(db, id, body.reason, user.id)
    return ArchivePatientResponse(id=patient.id, status=patient.status, archived_reason=body.reason)


@router.post("/export", response_model=ExportResponse, status_code=202)
async def export_patients(
    body: ExportPatientDataRequest,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> ExportResponse:
    """FR-E2.6/US13: 202 — the export always starts in
    ``status="processing"`` with ``download_url=None`` (see
    ``ExportService.generate``'s own docstring for why this is
    fire-and-forget rather than awaited inline).
    """
    service = _export_service()
    result = await service.generate(
        db,
        body.entity,
        body.date_range.from_,
        body.date_range.to,
        body.de_identified,
        user.id,
    )
    return ExportResponse(**result)
