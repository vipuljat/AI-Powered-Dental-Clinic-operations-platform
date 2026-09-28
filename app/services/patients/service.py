"""Business logic for the ``patients`` module.

``PatientService`` (create/search/update/archive), ``PatientImportService``
(CSV validate+commit), ``RecordLockService`` (shared lock acquire/release/
sweep — used by both this module's routes and scheduling's) and
``ExportService`` (raw/de-identified CSV generation to object storage).

Every mutating method here calls ``AuditService.record`` (console module,
FR-E1.6) so every create/update/archive/import is traceable — the one
documented exception is ``ExportService``, whose Public surface (frozen by
its spec) injects no ``AuditService`` collaborator; see that class's
docstring for how the export outcome is still recorded.

Responsibility: this file never issues a raw SQL/ORM statement itself —
every read/write goes through a repository method. The one narrow exception
is ``PatientService.get_profile``'s recall-status lookup, documented inline,
where the frozen ``recall`` repository exposes no per-patient query.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from datetime import date, datetime, time, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.common.constants import (
    CSV_IMPORT_MAX_FILE_SIZE_MB,
    CSV_IMPORT_MAX_ROWS,
    RECORD_LOCK_TIMEOUT_MINUTES,
)
from app.common.enums import ActorType, ImportBatchStatus, Language
from app.common.exceptions.errors import (
    DuplicatePatientError,
    ImportValidationError,
    NotFoundError,
    NotLockHolderError,
    RecordLockedError,
)
from app.common.utils import de_identify_patient_fields, new_patient_code
from app.common.validators import validate_phone
from app.models.patients.models import Patient, RecordLock
from app.models.recall.models import RecallSchedule
from app.repositories.outreach.repository import ConsentRepository
from app.repositories.patients.repository import (
    PatientImportRepository,
    PatientRepository,
    RecordLockRepository,
)
from app.repositories.scheduling.repository import AppointmentRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.storage import ObjectStorageClient
    from app.services.console.service import AuditService

__all__ = [
    "ExportService",
    "PatientImportService",
    "PatientService",
    "RecordLockService",
]

_logger = logging.getLogger(__name__)

_REQUIRED_IMPORT_COLUMNS = ("first_name", "last_name", "phone")


class RecordLockService:
    """FR-E2.4/US10/US22: the single shared-lock implementation both
    ``PatientService.update`` (patients, ``entity_type="patient"``) and
    ``services/scheduling/service.py``'s appointment-edit path
    (``entity_type="appointment"``) call — both go through the same
    ``RecordLockRepository`` instance/table.
    """

    def __init__(self, lock_repo: RecordLockRepository) -> None:
        self._lock_repo = lock_repo

    async def acquire(
        self, db: "AsyncSession", entity_type: str, entity_id: "UUID", staff_id: "UUID"
    ) -> "RecordLock":
        """Relies on the DB-level ``UNIQUE(entity_type, entity_id)``
        constraint the frozen ``RecordLockRepository.acquire`` writes
        through — a concurrent acquire attempt against an already-locked
        row raises ``IntegrityError``, caught here and translated into
        ``RecordLockedError`` (409).
        """
        try:
            return await self._lock_repo.acquire(
                db, entity_type, entity_id, staff_id, RECORD_LOCK_TIMEOUT_MINUTES
            )
        except IntegrityError:
            # The failed flush leaves the session unable to run further
            # statements until it is rolled back.
            await db.rollback()
            existing = await self._lock_repo.find_active(db, entity_type, entity_id)
            # Watch out: `RecordLockRepository`/`RecordLock` carry no
            # denormalized staff *username* (only `locked_by_staff_id`), and
            # this service is not given a staff repository to resolve one —
            # `locked_by` is reported as the holding staff id (stringified)
            # rather than a display name.
            locked_by = str(existing.locked_by_staff_id) if existing is not None else None
            raise RecordLockedError(details=[{"locked_by": locked_by}]) from None

    async def release(
        self, db: "AsyncSession", entity_type: str, entity_id: "UUID", staff_id: "UUID"
    ) -> None:
        await self._lock_repo.release(db, entity_type, entity_id, staff_id)

    async def sweep_expired(self, db: "AsyncSession") -> int:
        """Invoked periodically by ``app/workers/recall_scanner.py``'s
        general scheduled-job loop — there is no dedicated ``patients``
        worker file in this tree.
        """
        return await self._lock_repo.sweep_expired(db)


class PatientService:
    """FR-E2.1/US8 (create+dedup), FR-E2.3/US10 (search/profile),
    FR-E2.4/US10 (locked update), FR-E2.5/US11 (archive).
    """

    def __init__(
        self,
        patient_repo: PatientRepository,
        lock_repo: RecordLockRepository,
        audit: "AuditService",
    ) -> None:
        self._patient_repo = patient_repo
        self._lock_repo = lock_repo
        self._audit = audit

    async def create(
        self, db: "AsyncSession", data: dict, actor_id: "UUID"
    ) -> "tuple[Patient, Patient | None]":
        """FR-E2.1/US8: duplicate-phone detection happens *before* record
        creation, not as a post-hoc flag — a phone match raises
        ``DuplicatePatientError`` (409) with the candidate embedded in
        ``details`` and no row is ever inserted.
        """
        duplicate = await self._patient_repo.find_duplicate_by_phone(db, data["phone"])
        if duplicate is not None:
            # `patient_code` is read defensively (`getattr` with a default)
            # rather than assumed present — the real `Patient` ORM model
            # always carries it, but the candidate is otherwise only
            # guaranteed to expose an `id` for this 409 payload.
            raise DuplicatePatientError(
                details=[
                    {
                        "duplicate_candidate": {
                            "id": str(duplicate.id),
                            "patient_code": getattr(duplicate, "patient_code", None),
                        }
                    }
                ]
            )

        patient_code = await self._patient_repo.next_patient_code(db)
        patient = await self._patient_repo.create(db, patient_code=patient_code, **data)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="patient.created",
            entity_type="patient",
            entity_id=patient.id,
            override_payload={k: v for k, v in data.items() if k != "demographics"},
        )
        return patient, None

    async def search(self, db: "AsyncSession", q: str) -> list["Patient"]:
        """FR-E2.3/US10: matches name, phone, or PatientID (delegated
        verbatim to ``PatientRepository.search``).
        """
        return await self._patient_repo.search(db, q)

    async def get_profile(self, db: "AsyncSession", id: "UUID") -> dict:
        """FR-E2.3/US10: aggregates patient + appointments + consent +
        recall_status in one call (architecture.md §5.2's example response
        shape), reading across module boundaries via each owning module's
        own repository — no writes.
        """
        patient = await self._patient_repo.get_by_id(db, id)
        if patient is None:
            raise NotFoundError("Patient not found.")

        appointments = await AppointmentRepository().list_by_patient(db, id)
        consent_records = await ConsentRepository().list_latest_all_channels(db, id)

        # Watch out: the frozen `RecallScheduleRepository`
        # (repositories/recall/repository.py) exposes no per-patient lookup
        # (only `list_overdue`/`list_by_ids`/status aggregates) — the most
        # recent `recall_schedules` row for this patient is read directly
        # here via the ORM model it already declares, rather than adding an
        # unauthorized method to that frozen file. This is the one place in
        # this file that is not a pure repository call.
        result = await db.execute(
            select(RecallSchedule)
            .where(RecallSchedule.patient_id == id)
            .order_by(RecallSchedule.due_date.desc())
            .limit(1)
        )
        recall_schedule = result.scalars().first()
        recall_status_value = None
        if recall_schedule is not None:
            status = recall_schedule.status
            recall_status_value = status.value if hasattr(status, "value") else status

        return {
            "id": patient.id,
            "patient_code": patient.patient_code,
            "first_name": patient.first_name,
            "last_name": patient.last_name,
            "status": patient.status,
            "appointments": [
                {"id": a.id, "scheduled_start": a.scheduled_start, "status": a.status}
                for a in appointments
            ],
            "consent": [{"channel": c.channel, "status": c.status} for c in consent_records],
            "recall_status": recall_status_value,
        }

    async def update(self, db: "AsyncSession", id: "UUID", data: dict, actor_id: "UUID") -> "Patient":
        """FR-E2.4/US10: raises ``NotLockHolderError`` (423) unless the
        caller currently holds the ``entity_type="patient"`` lock — checked
        via the same ``RecordLockRepository`` row ``RecordLockService``
        reads/writes.
        """
        active_lock = await self._lock_repo.find_active(db, "patient", id)
        if active_lock is None or active_lock.locked_by_staff_id != actor_id:
            raise NotLockHolderError()

        patient = await self._patient_repo.update(db, id, **data)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="patient.updated",
            entity_type="patient",
            entity_id=id,
            override_payload={k: v for k, v in data.items() if v is not None},
        )
        return patient

    async def archive(self, db: "AsyncSession", id: "UUID", reason: str, actor_id: "UUID") -> "Patient":
        """FR-E2.5/US11: sets ``status="archived"``; every other module's
        eligibility query (``PatientRepository.list_active``) excludes
        archived patients from that point forward — "future outreach/recall
        is halted" is enforced at the query layer, not by a separate
        suppression flag.
        """
        patient = await self._patient_repo.archive(db, id, reason)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="patient.archived",
            entity_type="patient",
            entity_id=id,
            override_payload={"reason": reason},
        )
        return patient


class PatientImportService:
    """FR-E2.2/US9/A10: CSV bulk-import, validate then all-or-nothing commit."""

    def __init__(
        self,
        import_repo: PatientImportRepository,
        patient_repo: PatientRepository,
        audit: "AuditService",
    ) -> None:
        self._import_repo = import_repo
        self._patient_repo = patient_repo
        self._audit = audit

    @staticmethod
    def _parse_rows(file_bytes: bytes) -> "tuple[list[str], list[dict]]":
        text = file_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
        rows = list(reader)
        return fieldnames, rows

    async def validate(
        self, db: "AsyncSession", file_bytes: bytes, filename: str
    ) -> "tuple[list[dict], list[dict]]":
        """FR-E2.2/US9: structural problems (missing required columns, or
        the file exceeding ``CSV_IMPORT_MAX_FILE_SIZE_MB``/
        ``CSV_IMPORT_MAX_ROWS``) raise ``ImportValidationError`` (422)
        immediately; row-level problems are instead collected and returned
        as ``(valid_rows, errors)`` so the caller can report every bad row
        in one response rather than failing on the first.
        """
        max_bytes = CSV_IMPORT_MAX_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ImportValidationError(
                message=f"'{filename}' exceeds the {CSV_IMPORT_MAX_FILE_SIZE_MB}MB import limit.",
                details=[{"row_number": None, "column_name": None, "error_message": "file_too_large"}],
            )

        fieldnames, rows = self._parse_rows(file_bytes)
        missing_columns = [c for c in _REQUIRED_IMPORT_COLUMNS if c not in fieldnames]
        if missing_columns:
            raise ImportValidationError(
                message=f"'{filename}' is missing required column(s): {', '.join(missing_columns)}.",
                details=[
                    {"row_number": None, "column_name": col, "error_message": "missing_column"}
                    for col in missing_columns
                ],
            )

        if len(rows) > CSV_IMPORT_MAX_ROWS:
            raise ImportValidationError(
                message=f"'{filename}' exceeds the {CSV_IMPORT_MAX_ROWS}-row import limit.",
                details=[{"row_number": None, "column_name": None, "error_message": "too_many_rows"}],
            )

        valid_rows: list[dict] = []
        errors: list[dict] = []

        for index, row in enumerate(rows):
            # Row 1 is the header, so the first data row is row_number=2 —
            # matches what a spreadsheet-editing user actually sees.
            row_number = index + 2
            row_errors: list[dict] = []

            first_name = (row.get("first_name") or "").strip()
            if not first_name:
                row_errors.append(
                    {"row_number": row_number, "column_name": "first_name", "error_message": "This field is required."}
                )

            last_name = (row.get("last_name") or "").strip()
            if not last_name:
                row_errors.append(
                    {"row_number": row_number, "column_name": "last_name", "error_message": "This field is required."}
                )

            raw_phone = (row.get("phone") or "").strip()
            validated_phone: str | None = None
            if not raw_phone:
                row_errors.append(
                    {"row_number": row_number, "column_name": "phone", "error_message": "This field is required."}
                )
            else:
                try:
                    validated_phone = validate_phone(raw_phone)
                except ValueError as exc:
                    row_errors.append(
                        {"row_number": row_number, "column_name": "phone", "error_message": str(exc)}
                    )

            parsed_dob: date | None = None
            raw_dob = (row.get("dob") or "").strip()
            if raw_dob:
                try:
                    parsed_dob = date.fromisoformat(raw_dob)
                except ValueError:
                    row_errors.append(
                        {
                            "row_number": row_number,
                            "column_name": "dob",
                            "error_message": "Must be an ISO-8601 date (YYYY-MM-DD).",
                        }
                    )

            parsed_language = Language.pt
            raw_language = (row.get("language") or "").strip()
            if raw_language:
                try:
                    parsed_language = Language(raw_language)
                except ValueError:
                    row_errors.append(
                        {
                            "row_number": row_number,
                            "column_name": "language",
                            "error_message": f"Unrecognized language '{raw_language}'.",
                        }
                    )

            if row_errors:
                errors.extend(row_errors)
                continue

            valid_rows.append(
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "phone": validated_phone,
                    "dob": parsed_dob,
                    "language": parsed_language,
                    "email": (row.get("email") or "").strip() or None,
                }
            )

        return valid_rows, errors

    async def commit(
        self, db: "AsyncSession", file_bytes: bytes, filename: str, actor_id: "UUID"
    ) -> dict:
        """FR-E2.2/US9/A10: any invalid row rejects the **entire file** —
        ``bulk_insert`` is only ever called once ``validate`` returned zero
        errors.
        """
        valid_rows, errors = await self.validate(db, file_bytes, filename)
        total_rows = len(valid_rows) + len(errors)

        batch = await self._import_repo.create_batch(
            db, uploaded_by_staff_id=actor_id, filename=filename, total_rows=total_rows
        )

        if errors:
            await self._import_repo.log_errors(db, batch.id, errors)
            updated_batch = await self._import_repo.update_batch_status(
                db, batch.id, ImportBatchStatus.rejected.value, success_count=0, error_count=len(errors)
            )
            await self._audit.record(
                db,
                actor_staff_id=actor_id,
                actor_type=ActorType.staff,
                action_type="patient.import.rejected",
                entity_type="patient_import_batch",
                entity_id=batch.id,
                override_payload={"total_rows": total_rows, "error_count": len(errors)},
            )
            return {
                "batch_id": updated_batch.id,
                "status": updated_batch.status,
                "total_rows": total_rows,
                "success_count": 0,
                "error_count": len(errors),
                "errors": errors,
            }

        # Sequential PT-##### codes are derived from a single
        # `next_patient_code` lookup rather than calling it once per row —
        # calling it repeatedly before any row is inserted would return the
        # same code every time, since the count it derives from only
        # changes once `bulk_insert` actually commits.
        first_code = await self._patient_repo.next_patient_code(db)
        start_sequence = int(first_code.rsplit("-", 1)[1])
        rows_with_codes = [
            {**row, "patient_code": new_patient_code(start_sequence + offset)}
            for offset, row in enumerate(valid_rows)
        ]

        created = await self._import_repo.bulk_insert(db, rows_with_codes)
        updated_batch = await self._import_repo.update_batch_status(
            db, batch.id, ImportBatchStatus.committed.value, success_count=len(created), error_count=0
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="patient.import.committed",
            entity_type="patient_import_batch",
            entity_id=batch.id,
            override_payload={"total_rows": total_rows, "success_count": len(created)},
        )

        return {
            "batch_id": updated_batch.id,
            "status": updated_batch.status,
            "total_rows": total_rows,
            "success_count": len(created),
            "error_count": 0,
            "errors": [],
        }


class ExportService:
    """FR-E2.6/US13: raw/de-identified CSV generation of patient or
    appointment data to object storage.

    Watch out: this class's frozen Public surface injects no
    ``AuditService`` collaborator (unlike ``PatientService``/
    ``PatientImportService``), even though ``generate`` is a mutating
    operation writing to object storage — a spec inconsistency against this
    file's own "every mutating method calls AuditService.record" line. In
    its absence, the export outcome (export_id, entity, storage_uri,
    actor_id — never raw patient PII) is written to the structured
    application log instead, which is the closest equivalent traceability
    this class's constructor can honestly provide.
    """

    def __init__(
        self,
        patient_repo: PatientRepository,
        appointment_repo: "AppointmentRepository",
        storage: "ObjectStorageClient",
    ) -> None:
        self._patient_repo = patient_repo
        self._appointment_repo = appointment_repo
        self._storage = storage

    async def _build_patient_rows(
        self, db: "AsyncSession", date_from: date, date_to: date, de_identified: bool
    ) -> "tuple[list[str], list[dict]]":
        patients = await self._patient_repo.list_active(db)
        rows: list[dict] = []
        for patient in patients:
            # `created_at`/`patient_code`/`email`/`dob` are read defensively
            # (`getattr` with a default) rather than assumed present — the
            # real `Patient` ORM model always carries them, but this keeps
            # the same duck-typed tolerance the rest of this file already
            # uses for `status` (see `hasattr(status, "value")` below).
            created = getattr(patient, "created_at", None)
            created_date = created.date() if isinstance(created, datetime) else created
            if created_date is not None and not (date_from <= created_date <= date_to):
                continue
            status = patient.status
            dob = getattr(patient, "dob", None)
            row = {
                "patient_id": str(patient.id),
                "patient_code": getattr(patient, "patient_code", None),
                "first_name": patient.first_name,
                "last_name": patient.last_name,
                "phone": patient.phone,
                "email": getattr(patient, "email", None),
                "dob": dob.isoformat() if dob else None,
                "status": status.value if hasattr(status, "value") else status,
            }
            # FR-E2.6: raw vs. de-identified always uses the same stripping
            # logic as any other de-identified export in the codebase.
            if de_identified:
                row = de_identify_patient_fields(row)
            rows.append(row)

        if de_identified:
            fieldnames = ["patient_id", "patient_code", "status"]
        else:
            fieldnames = ["patient_id", "patient_code", "first_name", "last_name", "phone", "email", "dob", "status"]
        return fieldnames, rows

    async def _build_appointment_rows(
        self, db: "AsyncSession", date_from: date, date_to: date, de_identified: bool
    ) -> "tuple[list[str], list[dict]]":
        # Watch out: the frozen `AppointmentRepository` exposes no generic
        # "list all appointments in range regardless of status" query — the
        # closest available method, `list_open_slots`, only surfaces
        # booked/rescheduled appointments. Used as-is rather than adding an
        # unauthorized method to that frozen file.
        window_start = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
        window_end = datetime.combine(date_to, time.max, tzinfo=timezone.utc)
        appointments = await self._appointment_repo.list_open_slots(db, None, window_start, window_end)

        rows: list[dict] = []
        for appointment in appointments:
            status = appointment.status
            row = {
                "appointment_id": str(appointment.id),
                "patient_id": str(appointment.patient_id),
                "provider_id": str(appointment.provider_id),
                "chair_id": str(appointment.chair_id),
                "scheduled_start": appointment.scheduled_start.isoformat(),
                "scheduled_end": appointment.scheduled_end.isoformat(),
                "status": status.value if hasattr(status, "value") else status,
            }
            if de_identified:
                row = de_identify_patient_fields(row)
            rows.append(row)

        fieldnames = [
            "appointment_id",
            "patient_id",
            "provider_id",
            "chair_id",
            "scheduled_start",
            "scheduled_end",
            "status",
        ]
        return fieldnames, rows

    async def _write_and_upload(
        self,
        db: "AsyncSession",
        export_id: "UUID",
        entity: str,
        date_from: date,
        date_to: date,
        de_identified: bool,
        actor_id: "UUID",
    ) -> None:
        try:
            if entity == "appointments":
                fieldnames, rows = await self._build_appointment_rows(db, date_from, date_to, de_identified)
            else:
                fieldnames, rows = await self._build_patient_rows(db, date_from, date_to, de_identified)

            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            csv_bytes = buffer.getvalue().encode("utf-8")

            key = f"exports/{entity}/{export_id}.csv"
            storage_uri = await self._storage.put_object(key, csv_bytes, "text/csv")

            _logger.info(
                "patient_export.ready",
                extra={
                    "export_id": str(export_id),
                    "entity": entity,
                    "de_identified": de_identified,
                    "storage_uri": storage_uri,
                    "actor_id": str(actor_id),
                    "row_count": len(rows),
                },
            )
        except Exception:  # pragma: no cover - defensive, logged not raised
            # Watch out: this coroutine runs detached from the originating
            # request (see `generate` below) — there is no HTTP response
            # left to propagate a failure to, so it is logged (never a raw
            # stack trace containing PII) rather than re-raised into an
            # orphaned task.
            _logger.exception(
                "patient_export.failed",
                extra={"export_id": str(export_id), "entity": entity, "actor_id": str(actor_id)},
            )

    async def generate(
        self,
        db: "AsyncSession",
        entity: str,
        date_from: date,
        date_to: date,
        de_identified: bool,
        actor_id: "UUID",
    ) -> dict:
        """FR-E2.6/US13: returns ``{"export_id", "status": "processing",
        "download_url": None}`` immediately (202).

        Watch out: no dedicated export-status table exists for patient/
        appointment exports in this tree (unlike analytics'
        ``dashboard_exports``), so for the ``202 processing`` response to be
        honestly "processing" rather than already-complete-but-mislabeled,
        the CSV build+upload is scheduled as a fire-and-forget
        ``asyncio.create_task`` rather than awaited inline before this
        method returns. ``download_url`` therefore always stays ``None`` in
        this response — the file is retrievable only via the ``storage_uri``
        this task logs on completion (see ``_write_and_upload``).
        """
        export_id = uuid4()
        asyncio.create_task(
            self._write_and_upload(db, export_id, entity, date_from, date_to, de_identified, actor_id)
        )
        return {"export_id": export_id, "status": "processing", "download_url": None}
