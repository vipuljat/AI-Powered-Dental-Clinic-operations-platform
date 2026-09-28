"""Data-access classes over ``patients``, ``patient_import_batches``,
``patient_import_errors``, ``record_locks``.

Every SQL/ORM statement the patients module issues lives here —
``services/patients/service.py`` never constructs a query itself. Each method
takes an ``AsyncSession`` explicitly; no repository owns/creates its own
session (that is the composition root's / ``get_db``'s job, per
app/core/database.py).

Interaction contract: ``RecordLockRepository`` is the one and only lock
repository in the codebase, over the one and only ``record_locks`` table
(declared in ``models/patients/models.py``). ``services/scheduling/service.py``
depends on this class directly (repo/service -> repo, acceptable per the
layering rule since it is not a service-to-service dependency) to acquire and
release appointment locks, passing ``entity_type="appointment"``.
"""

from __future__ import annotations

from datetime import timedelta, timezone
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import PatientStatus
from app.common.exceptions.errors import NotFoundError
from app.common.utils import new_patient_code, now_utc
from app.models.patients.models import (
    Patient,
    PatientImportBatch,
    PatientImportError,
    RecordLock,
)

__all__ = [
    "PatientImportRepository",
    "PatientRepository",
    "RecordLockRepository",
]


class PatientRepository:
    """Data access for ``patients``."""

    async def create(self, db: AsyncSession, **fields) -> Patient:
        patient = Patient(**fields)
        db.add(patient)
        await db.flush()
        await db.refresh(patient)
        return patient

    async def get_by_id(self, db: AsyncSession, id: UUID) -> Patient | None:
        result = await db.execute(select(Patient).where(Patient.id == id))
        return result.scalar_one_or_none()

    async def search(self, db: AsyncSession, q: str, include_archived: bool = False) -> list[Patient]:
        """FR-E2.3: matches name (first_name || ' ' || last_name), phone, or
        patient_code, case-insensitive substring.
        """
        pattern = f"%{q}%"
        full_name = Patient.first_name + " " + Patient.last_name
        stmt = select(Patient).where(
            or_(
                full_name.ilike(pattern),
                Patient.phone.ilike(pattern),
                Patient.patient_code.ilike(pattern),
            )
        )
        if not include_archived:
            stmt = stmt.where(Patient.status == PatientStatus.active)
        stmt = stmt.order_by(Patient.last_name, Patient.first_name)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update(self, db: AsyncSession, id: UUID, **fields) -> Patient:
        patient = await self.get_by_id(db, id)
        if patient is None:
            raise NotFoundError("Patient not found.")
        for key, value in fields.items():
            setattr(patient, key, value)
        await db.flush()
        await db.refresh(patient)
        return patient

    async def archive(self, db: AsyncSession, id: UUID, reason: str) -> Patient:
        patient = await self.get_by_id(db, id)
        if patient is None:
            raise NotFoundError("Patient not found.")
        patient.status = PatientStatus.archived
        patient.archived_reason = reason
        patient.archived_at = now_utc()
        await db.flush()
        await db.refresh(patient)
        return patient

    async def find_duplicate_by_phone(self, db: AsyncSession, phone: str) -> Patient | None:
        result = await db.execute(select(Patient).where(Patient.phone == phone))
        return result.scalars().first()

    async def next_patient_code(self, db: AsyncSession) -> str:
        result = await db.execute(select(func.count()).select_from(Patient))
        count = result.scalar_one()
        return new_patient_code(count + 1)

    async def list_active(self, db: AsyncSession, ids: list[UUID] | None = None) -> list[Patient]:
        """FR-E2.5: used by every other module's eligibility query to exclude
        archived patients.
        """
        stmt = select(Patient).where(Patient.status == PatientStatus.active)
        if ids is not None:
            stmt = stmt.where(Patient.id.in_(ids))
        result = await db.execute(stmt)
        return list(result.scalars().all())


class PatientImportRepository:
    """Data access for ``patient_import_batches`` / ``patient_import_errors``."""

    async def create_batch(
        self, db: AsyncSession, uploaded_by_staff_id: UUID, filename: str, total_rows: int
    ) -> PatientImportBatch:
        batch = PatientImportBatch(
            uploaded_by_staff_id=uploaded_by_staff_id,
            filename=filename,
            total_rows=total_rows,
            status="validating",
        )
        db.add(batch)
        await db.flush()
        await db.refresh(batch)
        return batch

    async def bulk_insert(self, db: AsyncSession, rows: list[dict]) -> list["Patient"]:
        """FR-E2.2/US9: performs all rows in one transaction — the caller
        (``PatientImportService.commit``) only invokes this after 100% of
        rows have independently validated, so a mid-batch failure here rolls
        the whole transaction back atomically (all-or-nothing, per A10).
        """
        patients = [Patient(**row) for row in rows]
        db.add_all(patients)
        await db.flush()
        for patient in patients:
            await db.refresh(patient)
        return patients

    async def log_errors(self, db: AsyncSession, batch_id: UUID, errors: list[dict]) -> None:
        for error in errors:
            db.add(
                PatientImportError(
                    batch_id=batch_id,
                    row_number=error["row_number"],
                    column_name=error["column_name"],
                    error_message=error["error_message"],
                )
            )
        await db.flush()

    async def update_batch_status(
        self, db: AsyncSession, batch_id: UUID, status: str, success_count: int, error_count: int
    ) -> PatientImportBatch:
        result = await db.execute(select(PatientImportBatch).where(PatientImportBatch.id == batch_id))
        batch = result.scalar_one_or_none()
        if batch is None:
            raise NotFoundError("Patient import batch not found.")
        batch.status = status
        batch.success_count = success_count
        batch.error_count = error_count
        await db.flush()
        await db.refresh(batch)
        return batch


class RecordLockRepository:
    """FR-E2.4: the single lock implementation shared by the patients module
    (patient edit locking, US10) and the scheduling module (appointment edit
    locking, US22).
    """

    async def acquire(
        self, db: AsyncSession, entity_type: str, entity_id: UUID, staff_id: UUID, ttl_minutes: int
    ) -> RecordLock:
        """Relies on the DB-level ``UNIQUE(entity_type, entity_id)`` constraint
        (rather than a SELECT-then-INSERT check) to keep lock acquisition
        correct under concurrent requests across multiple API processes, per
        ``project_rules.concurrency``. Raises ``sqlalchemy.exc.IntegrityError``
        on an already-locked row, which the service layer catches and
        converts to ``RecordLockedError`` (409).
        """
        locked_at = now_utc()
        lock = RecordLock(
            entity_type=entity_type,
            entity_id=entity_id,
            locked_by_staff_id=staff_id,
            locked_at=locked_at,
            expires_at=locked_at + timedelta(minutes=ttl_minutes),
        )
        db.add(lock)
        await db.flush()
        await db.refresh(lock)
        return lock

    async def release(self, db: AsyncSession, entity_type: str, entity_id: UUID, staff_id: UUID) -> None:
        """No-op if no matching row (e.g. it already expired and was swept,
        or the caller was never the holder — the service layer is
        responsible for the ``NotLockHolderError`` distinction, if any).
        """
        result = await db.execute(
            select(RecordLock).where(
                RecordLock.entity_type == entity_type,
                RecordLock.entity_id == entity_id,
                RecordLock.locked_by_staff_id == staff_id,
            )
        )
        lock = result.scalar_one_or_none()
        if lock is None:
            return
        await db.delete(lock)
        await db.flush()

    async def find_active(self, db: AsyncSession, entity_type: str, entity_id: UUID) -> RecordLock | None:
        result = await db.execute(
            select(RecordLock).where(
                RecordLock.entity_type == entity_type,
                RecordLock.entity_id == entity_id,
            )
        )
        return result.scalar_one_or_none()

    async def sweep_expired(self, db: AsyncSession) -> int:
        now = now_utc()
        result = await db.execute(select(RecordLock))
        locks = result.scalars().all()
        expired = [lock for lock in locks if _as_aware_utc(lock.expires_at) < now]
        for lock in expired:
            await db.delete(lock)
        if expired:
            await db.flush()
        return len(expired)


def _as_aware_utc(value):
    """Normalize a datetime read back from the DB to tz-aware UTC.

    SQLite (used per project_rules.testing's ENVIRONMENT=test pivot) does not
    retain a ``DateTime(timezone=True)`` column's UTC offset the way Postgres
    does — a value round-tripped through it comes back naive. Every timestamp
    in this tree is UTC (project_rules "all timestamps are timezone-aware
    UTC"), so a naive value read back is treated as already being UTC before
    it is compared against ``now_utc()``.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
