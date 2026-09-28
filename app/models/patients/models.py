"""SQLAlchemy ORM declarations for `patients`, `patient_import_batches`,
`patient_import_errors`, `record_locks`.

Pure schema — no query methods, no business logic. Query/mutation behaviour
lives in ``repositories/patients/repository.py``; this file only declares
columns and constraints.

Interaction contract: ``record_locks`` is declared exactly once, here.
Per ``project_rules.concurrency``, it is shared by both the patients module
(patient edit locking, US10) and the scheduling module (appointment edit
locking, US22) — ``models/scheduling/models.py`` does not redeclare this
table. Other modules that need to acquire/release a lock go through
``repositories/patients/repository.py``'s ``RecordLockRepository`` (used
directly by ``services/scheduling/service.py``), never a second ``RecordLock``
model.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import ImportBatchStatus, Language, LockEntityType, PatientStatus
from app.core.database import Base
from app.core.db_types import JSONBType


class Patient(Base):
    """A patient record.

    FR-E2.1: ``first_name``/``last_name``/``phone`` are ``NOT NULL`` — the one
    schema-level enforcement of "name and contact phone required as non-null
    fields". ``phone`` additionally serves as the duplicate-detection key used
    by ``services/patients/service.py``.
    """

    __tablename__ = "patients"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(150), nullable=False)
    last_name: Mapped[str] = mapped_column(String(150), nullable=False)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    phone: Mapped[str] = mapped_column(String(30), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    language: Mapped[Language] = mapped_column(
        SAEnum(Language, native_enum=False),
        nullable=False,
        default=Language.pt,
        server_default=Language.pt.value,
    )
    # [INFERRED] drives education content selection.
    developmental_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # [INFERRED] flexible demographic fields.
    demographics: Mapped[dict | None] = mapped_column(JSONBType, nullable=True)
    status: Mapped[PatientStatus] = mapped_column(
        SAEnum(PatientStatus, native_enum=False),
        nullable=False,
        default=PatientStatus.active,
        server_default=PatientStatus.active.value,
    )
    archived_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PatientImportBatch(Base):
    """A bulk patient-import upload (US-? bulk import) and its outcome."""

    __tablename__ = "patient_import_batches"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    uploaded_by_staff_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ImportBatchStatus] = mapped_column(
        SAEnum(ImportBatchStatus, native_enum=False), nullable=False
    )
    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PatientImportError(Base):
    """One row/column-level validation error raised while processing a
    ``PatientImportBatch``."""

    __tablename__ = "patient_import_errors"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    batch_id: Mapped[UUID] = mapped_column(ForeignKey("patient_import_batches.id"), nullable=False)
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    column_name: Mapped[str] = mapped_column(String(100), nullable=False)
    error_message: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RecordLock(Base):
    """FR-E2.4: locks a patient or appointment record for editing by only one
    staff member at a time.

    The ``UNIQUE(entity_type, entity_id)`` constraint is the DB-level
    enforcement ``RecordLockService.acquire``
    (``services/patients/service.py``) relies on — a second staff member's
    concurrent acquire attempt fails at the DB constraint rather than racing
    in application code. Shared, per ``project_rules.concurrency``, by the
    patients module (US10, patient edit locking) and the scheduling module
    (US22, appointment edit locking) via ``repositories/patients/repository.py``'s
    ``RecordLockRepository`` — never a second declaration of this table.
    """

    __tablename__ = "record_locks"
    __table_args__ = (UniqueConstraint("entity_type", "entity_id", name="uq_record_locks_entity"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    entity_type: Mapped[LockEntityType] = mapped_column(
        SAEnum(LockEntityType, native_enum=False), nullable=False
    )
    entity_id: Mapped[UUID] = mapped_column(nullable=False)
    locked_by_staff_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Timeout auto-release: expiry time past which the lock is treated as
    # stale and may be reclaimed by another staff member.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
