"""SQLAlchemy ORM declarations for `providers`, `chairs`, `appointments`,
`appointment_history`, `appointment_import_batches`, `appointment_import_errors`.

`appointments` is the platform's relational hub (architecture.md §4.1) — most
other modules FK against it.

Pure schema — no query methods, no business logic. Query/mutation behaviour
lives in ``repositories/scheduling/repository.py``; this file only declares
columns and constraints.

Interaction contract: ``patient_id``/``provider_id``/``chair_id``/
``created_by_staff_id`` are plain FK columns, with no cross-module ORM
``relationship()`` declared here. ``appointments.id`` is referenced by FK from
``idle_chair_alerts``, ``waitlist_entries``, ``outreach_messages``,
``recall_schedules``, ``unscheduled_treatments``, ``risk_scores``,
``content_deliveries``, all as plain UUID columns there, never as an imported
ORM class from this file. ``repositories/scheduling/repository.py`` is the
only file that imports these classes directly; every other module reaches
appointment data through ``AppointmentRepository`` (a cross-module repository
dependency, acceptable per the layering rule) or
``services/scheduling/service.py``'s ``SchedulingService``.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import (
    AppointmentStatus,
    ConfirmationStatus,
    ImportBatchStatus,
    RiskLevel,
    ScoreSource,
)
from app.core.database import Base
from app.core.db_types import JSONBType


class Provider(Base):
    """A clinical provider (dentist/hygienist) who can be assigned an appointment."""

    __tablename__ = "providers"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    specialty: Mapped[str | None] = mapped_column(String(100), nullable=True)


class Chair(Base):
    """A physical treatment chair/room that can be assigned an appointment."""

    __tablename__ = "chairs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    room: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(50), nullable=False)


class Appointment(Base):
    """The platform's relational hub (architecture.md §4.1).

    RMR001: ``idempotency_key`` is ``UNIQUE, NULLABLE`` — the mechanism
    ``SchedulingService.create`` relies on to make a retried
    ``POST /scheduling/appointments`` safe to replay rather than
    double-booking (a repeated key returns the original result instead of
    inserting a second row).

    FR-E4.4: on reschedule the SAME row's ``scheduled_start``/
    ``scheduled_end``/``provider_id``/``chair_id`` are updated in place and
    ``status`` set to ``"rescheduled"`` — a new ``appointments`` row is never
    created per reschedule, since that would break every FK pointing at the
    original appointment id (``outreach_messages``, ``risk_scores``,
    ``content_deliveries``, etc.). ``source_metadata`` records the reschedule
    lineage while ``AppointmentHistory`` preserves the original values that
    were overwritten.
    """

    __tablename__ = "appointments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), nullable=False)
    provider_id: Mapped[UUID] = mapped_column(ForeignKey("providers.id"), nullable=False)
    chair_id: Mapped[UUID] = mapped_column(ForeignKey("chairs.id"), nullable=False)
    # Null if AI-Agent-originated.
    created_by_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    appointment_type: Mapped[str] = mapped_column(String(100), nullable=False)
    scheduled_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scheduled_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[AppointmentStatus] = mapped_column(
        SAEnum(AppointmentStatus, native_enum=False),
        nullable=False,
        default=AppointmentStatus.booked,
        server_default=AppointmentStatus.booked.value,
    )
    # Preserved reschedule lineage (FR-E4.4).
    source_metadata: Mapped[dict | None] = mapped_column(JSONBType, nullable=True)
    cancellation_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_late_cancellation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    risk_flag: Mapped[RiskLevel | None] = mapped_column(SAEnum(RiskLevel, native_enum=False), nullable=True)
    # US41 transparency: which subsystem produced ``risk_flag``.
    risk_score_source: Mapped[ScoreSource | None] = mapped_column(
        SAEnum(ScoreSource, native_enum=False), nullable=True
    )
    confirmation_status: Mapped[ConfirmationStatus | None] = mapped_column(
        SAEnum(ConfirmationStatus, native_enum=False), nullable=True
    )
    # RMR001 replay safety for POST /scheduling/appointments.
    idempotency_key: Mapped[str | None] = mapped_column(String(100), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AppointmentHistory(Base):
    """FR-E4.4: preserves original values on reschedule (``changed_field``/
    ``old_value``/``new_value``) — the appointment row itself is updated in
    place, never deleted and re-inserted.
    """

    __tablename__ = "appointment_history"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    appointment_id: Mapped[UUID] = mapped_column(ForeignKey("appointments.id"), nullable=False)
    changed_field: Mapped[str] = mapped_column(String(100), nullable=False)
    old_value: Mapped[str | None] = mapped_column(String(255), nullable=True)
    new_value: Mapped[str | None] = mapped_column(String(255), nullable=True)
    changed_by_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AppointmentImportBatch(Base):
    """A bulk appointment-import upload and its outcome (mirrors
    ``patient_import_batches``)."""

    __tablename__ = "appointment_import_batches"

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


class AppointmentImportError(Base):
    """One row/column-level validation error raised while processing an
    ``AppointmentImportBatch`` (mirrors ``patient_import_errors``)."""

    __tablename__ = "appointment_import_errors"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    batch_id: Mapped[UUID] = mapped_column(ForeignKey("appointment_import_batches.id"), nullable=False)
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    column_name: Mapped[str] = mapped_column(String(100), nullable=False)
    error_message: Mapped[str] = mapped_column(String(500), nullable=False)
    # US12 AC: flags rows referencing an unknown patient_code.
    unmatched_patient: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
