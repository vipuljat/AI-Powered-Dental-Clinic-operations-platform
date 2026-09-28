"""Data-access classes over ``providers``, ``chairs``, ``appointments``,
``appointment_history``, ``appointment_import_batches``, ``appointment_import_errors``.

Every SQL/ORM statement the scheduling module issues lives here —
``services/scheduling/service.py`` never constructs a query itself. Each
method takes an ``AsyncSession`` explicitly; no repository owns/creates its
own session (that is the composition root's / ``get_db``'s job, per
app/core/database.py).

Interaction contract: this is the only file that imports the ORM classes
declared in ``models/scheduling/models.py`` directly (per that file's own
Interaction contract); every other module reaches appointment data through
``AppointmentRepository`` (a cross-module repository dependency, acceptable
per the layering rule) or ``services/scheduling/service.py``'s
``SchedulingService``. ``list_completable`` is polled by
``app/workers/recall_scanner.py``'s general periodic sweep (not a dedicated
scheduling worker — there is no ``scheduling`` worker file in this tree) to
transition ``booked`` -> ``completed`` appointments; this repository does not
perform that transition itself, it only surfaces candidates.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.constants import APPOINTMENT_COMPLETION_GRACE_MINUTES
from app.common.enums import AppointmentStatus
from app.common.exceptions.errors import NotFoundError
from app.common.utils import now_utc
from app.models.scheduling.models import (
    Appointment,
    AppointmentHistory,
    AppointmentImportBatch,
    AppointmentImportError,
    Chair,
    Provider,
)

__all__ = [
    "AppointmentHistoryRepository",
    "AppointmentImportRepository",
    "AppointmentRepository",
    "ChairRepository",
    "ProviderRepository",
]

_OPEN_STATUSES = (AppointmentStatus.booked, AppointmentStatus.rescheduled)


class ProviderRepository:
    """Data access for ``providers``."""

    async def list(self, db: AsyncSession) -> list[Provider]:
        result = await db.execute(select(Provider).order_by(Provider.name))
        return list(result.scalars().all())

    async def get_by_id(self, db: AsyncSession, id: UUID) -> Provider | None:
        result = await db.execute(select(Provider).where(Provider.id == id))
        return result.scalar_one_or_none()


class ChairRepository:
    """Data access for ``chairs``."""

    async def list(self, db: AsyncSession) -> list[Chair]:
        result = await db.execute(select(Chair).order_by(Chair.room, Chair.label))
        return list(result.scalars().all())

    async def get_by_id(self, db: AsyncSession, id: UUID) -> Chair | None:
        result = await db.execute(select(Chair).where(Chair.id == id))
        return result.scalar_one_or_none()


class AppointmentRepository:
    """Data access for ``appointments`` — the platform's relational hub
    (architecture.md §4.1).
    """

    async def create(self, db: AsyncSession, **fields) -> Appointment:
        appointment = Appointment(**fields)
        db.add(appointment)
        await db.flush()
        await db.refresh(appointment)
        return appointment

    async def get_by_id(self, db: AsyncSession, id: UUID) -> Appointment | None:
        result = await db.execute(select(Appointment).where(Appointment.id == id))
        return result.scalar_one_or_none()

    async def get_by_idempotency_key(self, db: AsyncSession, key: str) -> Appointment | None:
        """RMR001: checked by ``SchedulingService.create`` *before* calling
        ``create`` — a repeated ``Idempotency-Key`` returns the existing
        row's data rather than inserting a duplicate.
        """
        result = await db.execute(select(Appointment).where(Appointment.idempotency_key == key))
        return result.scalar_one_or_none()

    async def list_open_slots(
        self,
        db: AsyncSession,
        provider_id: UUID | None,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Appointment]:
        """Returns booked/rescheduled appointments in range, used by
        ``SchedulingService`` to derive free slots (the complement of the
        booked intervals returned here within the requested window).
        """
        stmt = select(Appointment).where(
            Appointment.status.in_(_OPEN_STATUSES),
            Appointment.scheduled_start < date_to,
            Appointment.scheduled_end > date_from,
        )
        if provider_id is not None:
            stmt = stmt.where(Appointment.provider_id == provider_id)
        stmt = stmt.order_by(Appointment.scheduled_start)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def find_conflicts(
        self,
        db: AsyncSession,
        provider_id: UUID,
        chair_id: UUID,
        start: datetime,
        end: datetime,
        exclude_appointment_id: UUID | None = None,
    ) -> list[Appointment]:
        """FR-E4.2/US18-20: the one query both ``SchedulingService.create``
        and ``.reschedule`` call to detect a double-booking — checks overlap
        against *either* the same ``provider_id`` *or* the same ``chair_id``,
        since either being double-booked is a conflict. Overlap is a
        standard interval test: ``start < other.end AND end > other.start``.
        """
        stmt = select(Appointment).where(
            Appointment.status.in_(_OPEN_STATUSES),
            or_(Appointment.provider_id == provider_id, Appointment.chair_id == chair_id),
            and_(Appointment.scheduled_start < end, Appointment.scheduled_end > start),
        )
        if exclude_appointment_id is not None:
            stmt = stmt.where(Appointment.id != exclude_appointment_id)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def update_status(self, db: AsyncSession, id: UUID, status: str, **extra_fields) -> Appointment:
        appointment = await self.get_by_id(db, id)
        if appointment is None:
            raise NotFoundError("Appointment not found.")
        appointment.status = status
        for key, value in extra_fields.items():
            setattr(appointment, key, value)
        await db.flush()
        await db.refresh(appointment)
        return appointment

    async def update_fields(self, db: AsyncSession, id: UUID, **fields) -> Appointment:
        appointment = await self.get_by_id(db, id)
        if appointment is None:
            raise NotFoundError("Appointment not found.")
        for key, value in fields.items():
            setattr(appointment, key, value)
        await db.flush()
        await db.refresh(appointment)
        return appointment

    async def list_by_patient(self, db: AsyncSession, patient_id: UUID) -> list[Appointment]:
        result = await db.execute(
            select(Appointment)
            .where(Appointment.patient_id == patient_id)
            .order_by(Appointment.scheduled_start.desc())
        )
        return list(result.scalars().all())

    async def list_completable(self, db: AsyncSession, as_of: datetime) -> list[Appointment]:
        """``status='booked' AND scheduled_end < as_of - APPOINTMENT_COMPLETION_GRACE_MINUTES``;
        feeds the ``recall_scanner.py`` sweep that transitions ``booked`` ->
        ``completed``.
        """
        cutoff = as_of - timedelta(minutes=APPOINTMENT_COMPLETION_GRACE_MINUTES)
        result = await db.execute(
            select(Appointment).where(
                Appointment.status == AppointmentStatus.booked,
                Appointment.scheduled_end < cutoff,
            )
        )
        return list(result.scalars().all())

    async def list_by_risk_source(self, db: AsyncSession, source: str) -> list[Appointment]:
        result = await db.execute(select(Appointment).where(Appointment.risk_score_source == source))
        return list(result.scalars().all())


class AppointmentHistoryRepository:
    """FR-E4.4: preserves original values overwritten on reschedule."""

    async def insert(
        self,
        db: AsyncSession,
        appointment_id: UUID,
        changed_field: str,
        old_value: str | None,
        new_value: str | None,
        changed_by_staff_id: UUID | None,
    ) -> AppointmentHistory:
        history = AppointmentHistory(
            appointment_id=appointment_id,
            changed_field=changed_field,
            old_value=old_value,
            new_value=new_value,
            changed_by_staff_id=changed_by_staff_id,
            changed_at=now_utc(),
        )
        db.add(history)
        await db.flush()
        await db.refresh(history)
        return history


class AppointmentImportRepository:
    """Data access for ``appointment_import_batches`` / ``appointment_import_errors``
    (mirrors ``PatientImportRepository``).
    """

    async def create_batch(
        self, db: AsyncSession, uploaded_by_staff_id: UUID, filename: str, total_rows: int
    ) -> AppointmentImportBatch:
        batch = AppointmentImportBatch(
            uploaded_by_staff_id=uploaded_by_staff_id,
            filename=filename,
            total_rows=total_rows,
            status="validating",
        )
        db.add(batch)
        await db.flush()
        await db.refresh(batch)
        return batch

    async def bulk_insert(self, db: AsyncSession, rows: list[dict]) -> list[Appointment]:
        """Performs all rows in one transaction — the caller only invokes
        this after 100% of rows have independently validated, so a
        mid-batch failure here rolls the whole transaction back atomically
        (all-or-nothing, mirroring the patients-module import contract).
        """
        appointments = [Appointment(**row) for row in rows]
        db.add_all(appointments)
        await db.flush()
        for appointment in appointments:
            await db.refresh(appointment)
        return appointments

    async def log_errors(self, db: AsyncSession, batch_id: UUID, errors: list[dict]) -> None:
        for error in errors:
            db.add(
                AppointmentImportError(
                    batch_id=batch_id,
                    row_number=error["row_number"],
                    column_name=error["column_name"],
                    error_message=error["error_message"],
                    unmatched_patient=error.get("unmatched_patient", False),
                )
            )
        await db.flush()

    async def update_batch_status(
        self, db: AsyncSession, batch_id: UUID, status: str, success_count: int, error_count: int
    ) -> AppointmentImportBatch:
        result = await db.execute(
            select(AppointmentImportBatch).where(AppointmentImportBatch.id == batch_id)
        )
        batch = result.scalar_one_or_none()
        if batch is None:
            raise NotFoundError("Appointment import batch not found.")
        batch.status = status
        batch.success_count = success_count
        batch.error_count = error_count
        await db.flush()
        await db.refresh(batch)
        return batch
