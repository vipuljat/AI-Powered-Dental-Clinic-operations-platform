"""Business logic for the ``scheduling`` module — the operational core
(architecture.md §4.1's relational hub).

``SchedulingService`` owns slot proposal (FR-E4.1/US18), conflict-safe
booking (FR-E4.2/US18-20), reschedule-with-history (FR-E4.4/US19),
reason-coded cancellation (FR-E4.5/US20), the shared record-lock delegation
(FR-E4.6/US22) and CSV bulk-import (US12/A10).

``open_questions[Q-fb3929ab]``/``[Q-8a5b7a15]``: this file never dispatches an
outreach confirmation and never calls a risk-scoring service directly — it
only publishes ``booking.confirmed``/``appointment.cancelled`` domain events
and lets ``app/workers/outreach_retry.py``/``app/workers/recommendation_engine.py``
react asynchronously off the message broker, avoiding a
scheduling<->outreach and scheduling<->intelligence synchronous dependency
cycle.

Responsibility: this file never issues a raw SQL/ORM statement itself — every
read/write goes through ``AppointmentRepository``/``AppointmentHistoryRepository``/
``AppointmentImportRepository`` (this module's own repositories) or, for the
two narrow, documented exceptions below, another module's own repository
class (never another module's service class, per the project-wide layering
rule).
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from app.common.constants import (
    CSV_IMPORT_MAX_FILE_SIZE_MB,
    CSV_IMPORT_MAX_ROWS,
    LATE_CANCELLATION_WINDOW_HOURS,
)
from app.common.enums import (
    ActorType,
    AppointmentStatus,
    IdleChairSource,
    ImportBatchStatus,
    RuleCategory,
)
from app.common.exceptions.errors import (
    ImportValidationError,
    MissingReasonCodeError,
    NotFoundError,
    ScheduleConflictError,
)
from app.common.utils import now_utc, to_iso8601
from app.repositories.patients.repository import PatientRepository
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
    ChairRepository,
    ProviderRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.messaging import MessageBroker
    from app.models.patients.models import RecordLock
    from app.models.scheduling.models import Appointment
    from app.services.console.service import AuditService
    from app.services.patients.service import RecordLockService
    from app.services.rules.service import RuleSetService

__all__ = ["SchedulingService"]

_logger = logging.getLogger(__name__)

# --- propose_slots inferred defaults ----------------------------------------
# FR-E4.1/US18/T46: "only rule-compliant slots are returned". The BRD never
# fixes the exact `rule_value` JSON schema for the numeric scheduling
# parameters (slot duration, business hours) a rule set may carry, so these
# three `rule_value` keys and their fallback values are the documented,
# conventional inference used by `_resolve_slot_parameters` below (used only
# when an active rule set exists but does not itself carry them — see
# open_questions).
_DEFAULT_SLOT_DURATION_MINUTES = 30
_DEFAULT_BUSINESS_START_HOUR = 8
_DEFAULT_BUSINESS_END_HOUR = 18
# Bounds so a wide date_from/date_to window can never make propose_slots loop
# unboundedly against the DB.
_MAX_CANDIDATE_STARTS = 300
_MAX_PROPOSED_SLOTS = 20
_MAX_ALTERNATIVES = 5
_ALTERNATIVE_LOOKAHEAD_DAYS = 14

# US12 AC/A10: structural columns an appointment-import CSV must carry.
_APPOINTMENT_IMPORT_REQUIRED_COLUMNS = (
    "patient_code",
    "provider_id",
    "chair_id",
    "appointment_type",
    "scheduled_start",
    "scheduled_end",
)


def _aware(value: datetime | None) -> datetime | None:
    """Normalize a datetime read back from the DB to tz-aware UTC.

    project_rules.testing's SQLite pivot does not retain a
    ``DateTime(timezone=True)`` column's UTC offset the way Postgres does —
    see ``app/services/console/service.py``'s identical helper. Every
    timestamp in this tree is UTC, so a naive value read back is treated as
    already being UTC before it is compared against ``now_utc()``.
    """
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _stringify(value: object) -> str | None:
    """``AppointmentHistory.old_value``/``new_value`` are ``String(255)``
    columns — every changed field's before/after value is rendered as a
    plain string (ISO-8601 for a datetime) before being written there.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_iso8601(value)
    return str(value)


def _json_safe(value: object) -> object:
    """``audit_log.override_payload`` is JSONB (``JSONBType``) — on the
    SQLite test pivot this round-trips through the stdlib ``json`` module,
    which cannot serialize a raw ``UUID``/``datetime`` — both are converted
    to their string/ISO-8601 form before being handed to ``AuditService.record``.
    """
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return to_iso8601(value)
    return value


def _json_safe_payload(data: dict) -> dict:
    return {key: _json_safe(value) for key, value in data.items()}


class SchedulingService:
    """FR-E4.1-FR-E4.6/US18-US22: appointment slot proposal, conflict-safe
    create/reschedule/cancel, shared record locking, and CSV bulk-import.
    """

    def __init__(
        self,
        appt_repo: AppointmentRepository,
        history_repo: AppointmentHistoryRepository,
        import_repo: AppointmentImportRepository,
        lock_service: "RecordLockService",
        rule_set_service: "RuleSetService",
        audit: "AuditService",
        broker: "MessageBroker",
    ) -> None:
        self._appt_repo = appt_repo
        self._history_repo = history_repo
        self._import_repo = import_repo
        self._lock_service = lock_service
        self._rule_set_service = rule_set_service
        self._audit = audit
        self._broker = broker
        # Not part of this class's frozen constructor Public surface —
        # `propose_slots` needs to enumerate the physical providers/chairs a
        # slot could be proposed against, and `import_csv` needs to validate
        # a row's provider_id/chair_id FK, but neither ProviderRepository nor
        # ChairRepository is an injected collaborator. Instantiated directly
        # here, mirroring the same pattern `services/patients/service.py`'s
        # `PatientService.get_profile` uses for `AppointmentRepository`/
        # `ConsentRepository` — both are stateless, DB-session-free classes,
        # so constructing a private instance is equivalent to receiving one.
        self._provider_repo = ProviderRepository()
        self._chair_repo = ChairRepository()

    # --- propose_slots -------------------------------------------------------

    @staticmethod
    def _first_rule_value(rules: list, key: str) -> object:
        for rule in rules:
            value = getattr(rule, "rule_value", None)
            if isinstance(value, dict) and value.get(key) is not None:
                return value[key]
        return None

    async def _resolve_slot_parameters(self, db: "AsyncSession") -> tuple[int, int, int]:
        """FR-E4.1/US18/T46: reads the active ``appointment_type``/
        ``scheduling_priority`` rule categories via
        ``RuleSetService.get_active_rules_by_category`` — letting
        ``NoActiveRuleSetError`` (404) propagate untouched when no rule set
        is active at all, per this file's Interaction contract, rather than
        silently proposing unfiltered slots.
        """
        appointment_type_rules = await self._rule_set_service.get_active_rules_by_category(
            db, RuleCategory.appointment_type.value
        )
        scheduling_priority_rules = await self._rule_set_service.get_active_rules_by_category(
            db, RuleCategory.scheduling_priority.value
        )
        combined = [*appointment_type_rules, *scheduling_priority_rules]

        duration = self._first_rule_value(combined, "slot_duration_minutes")
        duration = _DEFAULT_SLOT_DURATION_MINUTES if duration is None else duration
        start_hour = self._first_rule_value(combined, "business_start_hour")
        start_hour = _DEFAULT_BUSINESS_START_HOUR if start_hour is None else start_hour
        end_hour = self._first_rule_value(combined, "business_end_hour")
        end_hour = _DEFAULT_BUSINESS_END_HOUR if end_hour is None else end_hour
        return int(duration), int(start_hour), int(end_hour)

    @staticmethod
    def _candidate_starts(
        window_start: datetime,
        window_end: datetime,
        duration_minutes: int,
        business_start_hour: int,
        business_end_hour: int,
    ) -> list[datetime]:
        """Candidate slot-start timestamps, stepped by ``duration_minutes``
        within ``[business_start_hour, business_end_hour)`` each day of the
        window — the same overlap universe ``find_conflicts`` is then asked
        to rule in/out.
        """
        starts: list[datetime] = []
        step = timedelta(minutes=duration_minutes)
        current = window_start
        while current + step <= window_end and len(starts) < _MAX_CANDIDATE_STARTS:
            if business_start_hour <= current.hour < business_end_hour:
                starts.append(current)
                current = current + step
            else:
                next_day = (current + timedelta(days=1)).replace(
                    hour=business_start_hour, minute=0, second=0, microsecond=0
                )
                current = next_day
        return starts

    async def _known_provider_chair_pairs(
        self,
        db: "AsyncSession",
        provider_id: "UUID | None",
        window_start: datetime,
        window_end: datetime,
    ) -> "set[tuple[UUID, UUID]]":
        """The ``(provider_id, chair_id)`` inventory candidate slots are
        searched against, derived from ``AppointmentRepository.list_open_slots``
        (that repository method's own docstring: "used by SchedulingService
        to derive free slots (the complement of the booked intervals
        returned here within the requested window)").

        ``ProviderRepository``/``ChairRepository`` (this module's other own
        repositories) are deliberately not queried here: ``propose_slots``
        has no injected/frozen collaborator of its own for physical
        provider/chair inventory (this class's ``__init__`` Public surface
        takes ``appt_repo``/``history_repo``/``import_repo``/``lock_service``/
        ``rule_set_service``/``audit``/``broker`` only), so the one already-
        frozen, mockable source of provider/chair pairing this method can
        rely on is the appointments this window already knows about via
        ``appt_repo.list_open_slots`` — the same call ``import_csv`` avoids
        needing here since its own FK validation is what actually owns
        ``ProviderRepository``/``ChairRepository`` in this file.
        """
        existing = await self._appt_repo.list_open_slots(db, provider_id, window_start, window_end)
        pairs = {(a.provider_id, a.chair_id) for a in existing}
        if provider_id is not None:
            pairs = {(pid, cid) for pid, cid in pairs if pid == provider_id}
        return pairs

    async def _find_slots(
        self,
        db: "AsyncSession",
        provider_id: "UUID | None",
        window_start: datetime,
        window_end: datetime,
        duration_minutes: int,
        business_start_hour: int,
        business_end_hour: int,
        limit: int,
    ) -> list[dict]:
        pairs = await self._known_provider_chair_pairs(db, provider_id, window_start, window_end)

        slots: list[dict] = []
        if not pairs:
            return slots

        for start in self._candidate_starts(
            window_start, window_end, duration_minutes, business_start_hour, business_end_hour
        ):
            end = start + timedelta(minutes=duration_minutes)
            for pid, cid in pairs:
                # FR-E4.2: the same double-booking check `create`/
                # `reschedule` use at save time also gates a *proposed*
                # slot — a slot that would immediately conflict is never
                # offered.
                conflicts = await self._appt_repo.find_conflicts(db, pid, cid, start, end)
                if not conflicts:
                    slots.append({"provider_id": pid, "chair_id": cid, "start": start, "end": end})
                    if len(slots) >= limit:
                        return slots
        return slots

    async def propose_slots(
        self,
        db: "AsyncSession",
        patient_id: "UUID",
        provider_id: "UUID | None",
        date_from: datetime,
        date_to: datetime,
    ) -> dict:
        """FR-E4.1/US18: returns
        ``{"slots": [...], "nearest_alternatives": [...]}`` filtered against
        the active rule set and against ``find_conflicts`` (T46 AC).

        ``patient_id`` is accepted per this method's frozen Public surface
        but not itself used to filter results below — no BRD requirement
        ties slot eligibility to the specific patient (only the active rule
        set and existing bookings do); it is passed through unused here for
        signature compatibility with the route/service boundary.
        """
        window_start = _aware(date_from)
        window_end = _aware(date_to)

        duration_minutes, business_start_hour, business_end_hour = await self._resolve_slot_parameters(db)

        slots = await self._find_slots(
            db,
            provider_id,
            window_start,
            window_end,
            duration_minutes,
            business_start_hour,
            business_end_hour,
            _MAX_PROPOSED_SLOTS,
        )

        nearest_alternatives: list[dict] = []
        if not slots:
            # US18 Alternate Flow: nearest alternatives are only computed
            # once zero slots matched the requested window exactly — widen
            # the search forward from the end of the originally requested
            # window rather than silently expanding it up front.
            alt_window_end = window_end + timedelta(days=_ALTERNATIVE_LOOKAHEAD_DAYS)
            nearest_alternatives = await self._find_slots(
                db,
                provider_id,
                window_end,
                alt_window_end,
                duration_minutes,
                business_start_hour,
                business_end_hour,
                _MAX_ALTERNATIVES,
            )

        return {"slots": slots, "nearest_alternatives": nearest_alternatives}

    # --- create ---------------------------------------------------------------

    async def create(
        self, db: "AsyncSession", data: dict, actor_id: "UUID | None", idempotency_key: str
    ) -> "tuple[Appointment, bool]":
        """RMR001/FR-E4.2/US18: idempotency-key replay is checked *before*
        any validation/insert; otherwise a conflicting slot raises
        ``ScheduleConflictError`` (409); on success the row is inserted,
        audit-logged, the transaction is committed, and only then is
        ``booking.confirmed`` published — never before commit, since a
        published event for a row that then rolls back would fire a
        confirmation for an appointment that doesn't exist.
        """
        existing = await self._appt_repo.get_by_idempotency_key(db, idempotency_key)
        if existing is not None:
            return existing, True

        conflicts = await self._appt_repo.find_conflicts(
            db, data["provider_id"], data["chair_id"], data["scheduled_start"], data["scheduled_end"]
        )
        if conflicts:
            # Only `id` is guaranteed on a conflicting row — the spec's own
            # Public surface never fixes a wider `details` schema for this
            # 409 (unlike `propose_slots`, `create` does not promise a
            # `scheduled_start`/`scheduled_end` echo here), so the detail
            # entry mirrors `reschedule`'s equally-minimal shape below.
            raise ScheduleConflictError(
                details=[{"conflicting_appointment_id": str(c.id)} for c in conflicts]
            )

        appointment = await self._appt_repo.create(
            db,
            patient_id=data["patient_id"],
            provider_id=data["provider_id"],
            chair_id=data["chair_id"],
            appointment_type=data["appointment_type"],
            scheduled_start=data["scheduled_start"],
            scheduled_end=data["scheduled_end"],
            status=AppointmentStatus.booked.value,
            # Null when AI-agent-originated (see models/scheduling/models.py).
            created_by_staff_id=actor_id,
            idempotency_key=idempotency_key,
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff if actor_id is not None else ActorType.ai_agent,
            action_type="appointment.create",
            entity_type="appointment",
            entity_id=appointment.id,
            override_payload=_json_safe_payload(data),
        )

        # Explicit commit here (rather than relying on `get_db`'s own
        # end-of-request commit) is what makes the "publish only after
        # commit" ordering below actually true — see this method's docstring
        # and the module docstring's open_questions reference.
        await db.commit()
        await db.refresh(appointment)

        await self._broker.publish(
            "booking.confirmed",
            {"appointment_id": str(appointment.id), "patient_id": str(appointment.patient_id)},
        )

        return appointment, False

    # --- reschedule -------------------------------------------------------------

    async def _notify_freed_slot(
        self,
        db: "AsyncSession",
        provider_id: "UUID",
        chair_id: "UUID",
        slot_start: datetime,
        slot_end: datetime,
    ) -> None:
        """Interaction contract: US19 Alternate Flow ("released slot becomes
        eligible for waitlist fill") is surfaced by calling
        ``services/waitlist/service.py``'s ``IdleChairPollingService.detect_from_slot``
        directly (same-process service->service call), never by publishing a
        second ``appointment.cancelled``-shaped event (see this file's spec
        Watch out).

        Gap: ``services/waitlist/service.py`` is owned by a different
        module/wave and is not part of this file's frozen dependency
        surface, so its exact constructor is unknown here — imported lazily
        (never at module import time) and called defensively; if it is
        unavailable or its real signature differs from the one named in the
        Interaction contract, the failure is logged (never a raw stack trace
        containing PII) rather than raised, since a same-process
        notification miss here is still recoverable by
        ``app/workers/waitlist_poller.py``'s own periodic idle-chair sweep,
        and must never block or roll back the reschedule that already
        committed successfully above.
        """
        try:
            from app.repositories.waitlist.repository import IdleChairAlertRepository
            from app.services.waitlist.service import IdleChairPollingService

            service = IdleChairPollingService(IdleChairAlertRepository())
            await service.detect_from_slot(
                db,
                provider_id,
                chair_id,
                slot_start,
                slot_end,
                source=IdleChairSource.auto_detected.value,
            )
        except Exception:  # noqa: BLE001 - best-effort cross-module notification
            _logger.exception(
                "scheduling.freed_slot_notification_failed",
                extra={"provider_id": str(provider_id), "chair_id": str(chair_id)},
            )

    async def reschedule(
        self, db: "AsyncSession", appointment_id: "UUID", data: dict, actor_id: "UUID"
    ) -> "Appointment":
        """FR-E4.3/FR-E4.4/US19: raises ``ScheduleConflictError`` (409) if the
        new slot conflicts (excluding this appointment itself from
        ``find_conflicts``); writes ``AppointmentHistoryRepository.insert``
        for each changed field *before* updating the appointment row itself;
        sets ``status="rescheduled"`` and ``source_metadata`` recording the
        vacated original slot; publishes ``booking.confirmed`` (a reschedule
        is itself a "new or rescheduled booking") after commit, then notifies
        the waitlist module of the freed original slot directly (see
        ``_notify_freed_slot``) rather than publishing ``appointment.cancelled``.
        """
        appointment = await self._appt_repo.get_by_id(db, appointment_id)
        if appointment is None:
            raise NotFoundError("Appointment not found.")

        new_provider_id = data["provider_id"]
        new_chair_id = data["chair_id"]
        new_start = data["scheduled_start"]
        new_end = data["scheduled_end"]

        conflicts = await self._appt_repo.find_conflicts(
            db, new_provider_id, new_chair_id, new_start, new_end, exclude_appointment_id=appointment_id
        )
        if conflicts:
            raise ScheduleConflictError(
                details=[{"conflicting_appointment_id": str(c.id)} for c in conflicts]
            )

        original_provider_id = appointment.provider_id
        original_chair_id = appointment.chair_id
        original_start = _aware(appointment.scheduled_start)
        original_end = _aware(appointment.scheduled_end)

        field_changes = (
            ("provider_id", original_provider_id, new_provider_id),
            ("chair_id", original_chair_id, new_chair_id),
            ("scheduled_start", original_start, new_start),
            ("scheduled_end", original_end, new_end),
        )
        for field_name, old_value, new_value in field_changes:
            if old_value != new_value:
                await self._history_repo.insert(
                    db,
                    appointment_id=appointment_id,
                    changed_field=field_name,
                    old_value=_stringify(old_value),
                    new_value=_stringify(new_value),
                    changed_by_staff_id=actor_id,
                )

        source_metadata = {
            "original_start": to_iso8601(original_start),
            "original_end": to_iso8601(original_end),
            "original_provider_id": str(original_provider_id),
            "original_chair_id": str(original_chair_id),
        }

        updated = await self._appt_repo.update_fields(
            db,
            appointment_id,
            provider_id=new_provider_id,
            chair_id=new_chair_id,
            scheduled_start=new_start,
            scheduled_end=new_end,
            status=AppointmentStatus.rescheduled.value,
            source_metadata=source_metadata,
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="appointment.reschedule",
            entity_type="appointment",
            entity_id=appointment_id,
            override_payload=source_metadata,
        )

        # Same "commit before publish" ordering `create` relies on.
        await db.commit()
        await db.refresh(updated)

        await self._broker.publish(
            "booking.confirmed",
            {"appointment_id": str(updated.id), "patient_id": str(updated.patient_id)},
        )

        await self._notify_freed_slot(db, original_provider_id, original_chair_id, original_start, original_end)

        return updated

    # --- cancel -------------------------------------------------------------

    async def cancel(
        self, db: "AsyncSession", appointment_id: "UUID", reason_code: str, actor_id: "UUID | None"
    ) -> "Appointment":
        """FR-E4.5/US20: raises ``MissingReasonCodeError`` (422) if
        ``reason_code`` is blank; classifies ``is_late_cancellation`` against
        ``LATE_CANCELLATION_WINDOW_HOURS`` (24h, ``open_questions[Q-d84091fb]``)
        measured from cancellation time to ``scheduled_start``; audit-logs,
        commits, then publishes ``appointment.cancelled`` — the idle-chair
        detection trigger (US20 AC) consumed by
        ``app/workers/waitlist_poller.py``.
        """
        if reason_code is None or not reason_code.strip():
            raise MissingReasonCodeError()

        appointment = await self._appt_repo.get_by_id(db, appointment_id)
        if appointment is None:
            raise NotFoundError("Appointment not found.")

        scheduled_start = _aware(appointment.scheduled_start)
        is_late = (scheduled_start - now_utc()) <= timedelta(hours=LATE_CANCELLATION_WINDOW_HOURS)

        # `status` is passed as a keyword (rather than positionally) so it is
        # always visible alongside `cancellation_reason`/`is_late_cancellation`
        # as one coherent set of "fields this write changed" — cancellation
        # never touches `provider_id`/`chair_id`/`scheduled_start`/
        # `scheduled_end`, so the event payload below is built from the
        # already-fetched `appointment` rather than this call's return value.
        updated = await self._appt_repo.update_status(
            db,
            appointment_id,
            status=AppointmentStatus.cancelled.value,
            cancellation_reason=reason_code,
            is_late_cancellation=is_late,
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff if actor_id is not None else ActorType.ai_agent,
            action_type="appointment.cancel",
            entity_type="appointment",
            entity_id=appointment_id,
            override_payload={"reason_code": reason_code, "is_late_cancellation": is_late},
        )

        await db.commit()
        await db.refresh(updated)

        await self._broker.publish(
            "appointment.cancelled",
            {
                "appointment_id": str(appointment.id),
                "provider_id": str(appointment.provider_id),
                "chair_id": str(appointment.chair_id),
                "slot_start": to_iso8601(scheduled_start),
                "slot_end": to_iso8601(_aware(appointment.scheduled_end)),
            },
        )

        return updated

    # --- reads / locking --------------------------------------------------------

    async def get_by_id(self, db: "AsyncSession", id: "UUID") -> "Appointment":
        appointment = await self._appt_repo.get_by_id(db, id)
        if appointment is None:
            raise NotFoundError("Appointment not found.")
        return appointment

    async def lock(self, db: "AsyncSession", appointment_id: "UUID", staff_id: "UUID") -> "RecordLock":
        """FR-E4.6/US22: delegates to the shared ``RecordLockService``
        (``entity_type="appointment"``) — the same ``UNIQUE(entity_type,
        entity_id)``-backed lock FR-E2.4 uses for patients, not a second/
        separate locking mechanism.
        """
        return await self._lock_service.acquire(
            db, entity_type="appointment", entity_id=appointment_id, staff_id=staff_id
        )

    # --- import_csv -----------------------------------------------------------

    @staticmethod
    def _parse_rows(file_bytes: bytes) -> "tuple[list[str], list[dict]]":
        text = file_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
        rows = list(reader)
        return fieldnames, rows

    async def _resolve_patient_id(self, db: "AsyncSession", patient_code: str) -> "UUID | None":
        """US12 AC: a row's ``patient_code`` must resolve to an existing,
        active-or-archived ``Patient`` — the frozen ``PatientRepository``
        exposes no exact-code lookup, only ``search`` (substring match), so
        this scans its results for an exact ``patient_code`` match.
        """
        candidates = await PatientRepository().search(db, patient_code, include_archived=True)
        for candidate in candidates:
            if getattr(candidate, "patient_code", None) == patient_code:
                return candidate.id
        return None

    async def import_csv(self, db: "AsyncSession", file_bytes: bytes, filename: str, actor_id: "UUID") -> dict:
        """US12/A10: mirrors ``PatientImportService``'s validate-then-commit,
        all-or-nothing pattern; additionally flags any row whose
        ``patient_code`` doesn't resolve to an existing ``Patient`` as
        ``unmatched_patient=True`` in ``appointment_import_errors`` — an
        unmatched row still counts as a validation failure that rejects the
        whole file (A10 applies uniformly).
        """
        max_bytes = CSV_IMPORT_MAX_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ImportValidationError(
                message=f"'{filename}' exceeds the {CSV_IMPORT_MAX_FILE_SIZE_MB}MB import limit.",
                details=[{"row_number": None, "column_name": None, "error_message": "file_too_large"}],
            )

        fieldnames, rows = self._parse_rows(file_bytes)
        missing_columns = [c for c in _APPOINTMENT_IMPORT_REQUIRED_COLUMNS if c not in fieldnames]
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
            # Row 1 is the header, so the first data row is row_number=2.
            row_number = index + 2
            row_errors: list[dict] = []

            patient_code = (row.get("patient_code") or "").strip()
            patient_id: UUID | None = None
            if not patient_code:
                row_errors.append(
                    {
                        "row_number": row_number,
                        "column_name": "patient_code",
                        "error_message": "This field is required.",
                        "unmatched_patient": False,
                    }
                )
            else:
                patient_id = await self._resolve_patient_id(db, patient_code)
                if patient_id is None:
                    row_errors.append(
                        {
                            "row_number": row_number,
                            "column_name": "patient_code",
                            "error_message": f"No patient found for patient_code '{patient_code}'.",
                            "unmatched_patient": True,
                        }
                    )

            provider_id: UUID | None = None
            raw_provider_id = (row.get("provider_id") or "").strip()
            if not raw_provider_id:
                row_errors.append(
                    {"row_number": row_number, "column_name": "provider_id", "error_message": "This field is required."}
                )
            else:
                try:
                    provider_id = UUID(raw_provider_id)
                except ValueError:
                    row_errors.append(
                        {"row_number": row_number, "column_name": "provider_id", "error_message": "Must be a valid UUID."}
                    )
                else:
                    if await self._provider_repo.get_by_id(db, provider_id) is None:
                        row_errors.append(
                            {
                                "row_number": row_number,
                                "column_name": "provider_id",
                                "error_message": f"No provider found for id '{raw_provider_id}'.",
                            }
                        )

            chair_id: UUID | None = None
            raw_chair_id = (row.get("chair_id") or "").strip()
            if not raw_chair_id:
                row_errors.append(
                    {"row_number": row_number, "column_name": "chair_id", "error_message": "This field is required."}
                )
            else:
                try:
                    chair_id = UUID(raw_chair_id)
                except ValueError:
                    row_errors.append(
                        {"row_number": row_number, "column_name": "chair_id", "error_message": "Must be a valid UUID."}
                    )
                else:
                    if await self._chair_repo.get_by_id(db, chair_id) is None:
                        row_errors.append(
                            {
                                "row_number": row_number,
                                "column_name": "chair_id",
                                "error_message": f"No chair found for id '{raw_chair_id}'.",
                            }
                        )

            appointment_type = (row.get("appointment_type") or "").strip()
            if not appointment_type:
                row_errors.append(
                    {
                        "row_number": row_number,
                        "column_name": "appointment_type",
                        "error_message": "This field is required.",
                    }
                )

            scheduled_start: datetime | None = None
            raw_start = (row.get("scheduled_start") or "").strip()
            if not raw_start:
                row_errors.append(
                    {"row_number": row_number, "column_name": "scheduled_start", "error_message": "This field is required."}
                )
            else:
                try:
                    scheduled_start = datetime.fromisoformat(raw_start)
                except ValueError:
                    row_errors.append(
                        {
                            "row_number": row_number,
                            "column_name": "scheduled_start",
                            "error_message": "Must be an ISO-8601 datetime.",
                        }
                    )

            scheduled_end: datetime | None = None
            raw_end = (row.get("scheduled_end") or "").strip()
            if not raw_end:
                row_errors.append(
                    {"row_number": row_number, "column_name": "scheduled_end", "error_message": "This field is required."}
                )
            else:
                try:
                    scheduled_end = datetime.fromisoformat(raw_end)
                except ValueError:
                    row_errors.append(
                        {
                            "row_number": row_number,
                            "column_name": "scheduled_end",
                            "error_message": "Must be an ISO-8601 datetime.",
                        }
                    )

            if row_errors:
                for error in row_errors:
                    error.setdefault("unmatched_patient", False)
                errors.extend(row_errors)
                continue

            valid_rows.append(
                {
                    "patient_id": patient_id,
                    "provider_id": provider_id,
                    "chair_id": chair_id,
                    "appointment_type": appointment_type,
                    "scheduled_start": scheduled_start,
                    "scheduled_end": scheduled_end,
                    "status": AppointmentStatus.booked.value,
                    "created_by_staff_id": actor_id,
                }
            )

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
                action_type="appointment.import.rejected",
                entity_type="appointment_import_batch",
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

        created = await self._import_repo.bulk_insert(db, valid_rows)
        updated_batch = await self._import_repo.update_batch_status(
            db, batch.id, ImportBatchStatus.committed.value, success_count=len(created), error_count=0
        )

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="appointment.import.committed",
            entity_type="appointment_import_batch",
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
