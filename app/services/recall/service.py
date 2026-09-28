"""Business logic for the `recall` module (architecture.md §4.2/§5.3 recall
module): overdue/dormant recall identification, recall-campaign dispatch and
compliance reporting, and unscheduled-treatment re-engagement.

``RecallScanService`` (FR-E7.1/FR-E7.2, US34) is the weekly-scan worker's
(``app/workers/recall_scanner.py``) entry point — it is also safe to call
on-demand. ``RecallCampaignService`` (FR-E7.3/FR-E7.4, US35/US36) owns
dispatching a recall outreach campaign and reporting the weekly compliance
rate against a captured baseline. ``TreatmentRecoveryService``
(FR-E7.5-FR-E7.7, US37-US39) owns the unscheduled-treatment re-engagement
queue.

Watch out (signature gap, reported in the task summary): ``RecallScanService``
is wired with only a ``RecallScheduleRepository``/``PatientRepository``/
``RuleSetService`` — no per-patient risk-classification store (e.g. a
``risk_scores``-backed repository) is injected here, and ``patients`` carries
no ``risk_classification`` column of its own (see ``models/patients/models.py``).
The only place a per-patient ``risk_classification`` value can honestly be
read from, given this frozen constructor, is the value already recorded on
that patient's own prior ``recall_schedules`` row (the column exists exactly
for this — see ``models/recall/models.py``); a patient with no prior row has
an unknown risk classification, which is exactly the "insufficient risk data"
case FR-E7.2 documents and folds into ``default_fallback`` without ever being
skipped. Likewise, no ``AppointmentRepository`` is injected, so "since last
visit" (US34's dormancy definition) is approximated from ``last_recall_at``
(the most recent recall contact on file) — or, absent that, from the
schedule's own ``due_date`` — as the closest available proxy for "since last
visit" this constructor's collaborators can support.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from app.common.constants import DEFAULT_RECALL_INTERVAL_MONTHS, RECALL_DORMANT_THRESHOLD_MONTHS
from app.common.enums import (
    ActorType,
    CampaignType,
    IntervalSource,
    Language,
    OutreachMessageStatus,
    RecallStatus,
    RuleCategory,
    UnscheduledTreatmentStatus,
)
from app.common.exceptions.errors import NoActiveRuleSetError
from app.common.utils import now_utc

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models.recall.models import RecallSchedule, UnscheduledTreatment
    from app.repositories.patients.repository import PatientRepository
    from app.repositories.recall.repository import (
        RecallComplianceBaselineRepository,
        RecallScheduleRepository,
        UnscheduledTreatmentRepository,
    )
    from app.services.console.service import AuditService
    from app.services.outreach.service import OutreachService
    from app.services.rules.service import RuleSetService

__all__ = ["RecallCampaignService", "RecallScanService", "TreatmentRecoveryService"]


def _enum_value(value: object) -> object:
    """Normalizes an attribute read back off an ORM row that may be either a
    Python ``Enum`` member or its plain string value (the shape a
    SQLite-backed test row can come back as) — both are treated identically
    throughout this file, the same accommodation ``services/outreach/
    service.py`` and ``services/console/service.py`` make for their own
    enum-typed columns.
    """
    return value.value if hasattr(value, "value") else value


class RecallScanService:
    """FR-E7.1/FR-E7.2 (US34): identifies overdue/dormant patients and
    creates/updates their ``recall_schedules`` rows. Called weekly by
    ``app/workers/recall_scanner.py``; also safe to call on-demand.
    """

    def __init__(
        self,
        recall_repo: "RecallScheduleRepository",
        patient_repo: "PatientRepository",
        rule_set_service: "RuleSetService",
    ) -> None:
        self._recall_repo = recall_repo
        self._patient_repo = patient_repo
        self._rule_set_service = rule_set_service

    async def _resolve_interval(
        self, db: "AsyncSession", risk_classification: str | None
    ) -> "tuple[int, str]":
        """FR-E7.1: always attempts the risk-based lookup first via
        ``RuleSetService.get_active_rules_by_category("recall_interval")``,
        matched against ``risk_classification``. FR-E7.2: falls back to
        ``DEFAULT_RECALL_INTERVAL_MONTHS`` with ``interval_source=
        "default_fallback"`` whenever no active rule set exists, no rule in
        that category matches this patient's risk classification, or the
        risk classification itself is unknown (``None``) — the patient is
        still evaluated with the fallback, never skipped.
        """
        try:
            rules = await self._rule_set_service.get_active_rules_by_category(
                db, RuleCategory.recall_interval.value
            )
        except NoActiveRuleSetError:
            return DEFAULT_RECALL_INTERVAL_MONTHS, IntervalSource.default_fallback.value

        if risk_classification is not None:
            for rule in rules:
                rule_value = rule.rule_value if isinstance(rule.rule_value, dict) else {}
                candidate = rule_value.get("risk_classification")
                months = rule_value.get("months")
                if (
                    candidate is not None
                    and months is not None
                    and str(candidate) == str(risk_classification)
                ):
                    return int(months), IntervalSource.risk_based.value

        return DEFAULT_RECALL_INTERVAL_MONTHS, IntervalSource.default_fallback.value

    @staticmethod
    def _is_dormant(schedule: "RecallSchedule", as_of: date) -> bool:
        """US34: dormant once at least ``RECALL_DORMANT_THRESHOLD_MONTHS``
        (approximated as 30-day months, the same convention
        ``services/analytics/service.py`` uses for its own windowed
        calculations, since no calendar-month-aware dependency is declared
        anywhere in this tree) have elapsed since the patient's last known
        recall contact (``last_recall_at``), or — absent any recorded
        contact — since the schedule's own ``due_date`` (see Watch out).
        """
        threshold_days = RECALL_DORMANT_THRESHOLD_MONTHS * 30
        reference = schedule.last_recall_at.date() if schedule.last_recall_at is not None else schedule.due_date
        return (as_of - reference).days >= threshold_days

    async def run(self, db: "AsyncSession") -> int:
        """FR-E7.1/FR-E7.2 (US34): reclassifies every already-due/overdue
        ``recall_schedules`` row (via ``RecallScheduleRepository.
        list_overdue``) as ``"overdue"`` or ``"dormant"``, then creates a
        brand-new ``"due"`` row (see Watch out) for every remaining active
        patient not already covered by one of those. Returns the total
        number of rows created/updated in this scan.
        """
        as_of = date.today()
        flagged = 0
        handled_patient_ids: set["UUID"] = set()

        overdue_candidates = await self._recall_repo.list_overdue(db, as_of)
        for schedule in overdue_candidates:
            handled_patient_ids.add(schedule.patient_id)
            _months, source = await self._resolve_interval(db, schedule.risk_classification)
            new_status = (
                RecallStatus.dormant.value if self._is_dormant(schedule, as_of) else RecallStatus.overdue.value
            )
            await self._recall_repo.update_status(db, schedule.id, new_status, interval_source=source)
            flagged += 1

        active_patients = await self._patient_repo.list_active(db)
        for patient in active_patients:
            if patient.id in handled_patient_ids:
                continue
            months, source = await self._resolve_interval(db, None)
            due_date = as_of + timedelta(days=months * 30)
            await self._recall_repo.create(
                db,
                patient_id=patient.id,
                appointment_id=None,
                risk_classification=None,
                due_date=due_date,
                last_recall_at=None,
                status=RecallStatus.due.value,
                interval_source=source,
            )
            flagged += 1

        return flagged


class RecallCampaignService:
    """FR-E7.3/FR-E7.4 (US35/US36): dispatches recall outreach campaigns and
    computes the weekly compliance rate.
    """

    def __init__(
        self,
        recall_repo: "RecallScheduleRepository",
        baseline_repo: "RecallComplianceBaselineRepository",
        outreach_service: "OutreachService",
        audit: "AuditService",
    ) -> None:
        self._recall_repo = recall_repo
        self._baseline_repo = baseline_repo
        self._outreach_service = outreach_service
        self._audit = audit

    async def dispatch(
        self, db: "AsyncSession", recall_schedule_ids: "list[UUID]", actor_id: "UUID | None"
    ) -> dict:
        """FR-E7.3 (US35): dispatches one outreach message per schedule via
        ``OutreachService.dispatch``. Watch out: this constructor injects no
        ``PatientRepository`` (per this class's frozen Public surface), so
        the patient's own preferred language cannot be resolved here;
        ``Language.en`` is used uniformly and ``OutreachService.dispatch``'s
        own English-template fallback (FR-E6.2/T71) still applies regardless.
        A schedule whose patient has no consented channel comes back with
        ``status="suppressed"`` (US21 Alternate Flow) and is left ``"due"``/
        ``"overdue"`` rather than being marked ``"contacted"``.
        """
        campaign_id = uuid4()
        dispatched_count = 0

        schedules = await self._recall_repo.list_by_ids(db, recall_schedule_ids)
        for schedule in schedules:
            message = await self._outreach_service.dispatch(
                db,
                patient_id=schedule.patient_id,
                campaign_type=CampaignType.recall.value,
                language=Language.en.value,
                related_entity_type="recall_schedule",
                related_entity_id=schedule.id,
            )
            if _enum_value(message.status) == OutreachMessageStatus.suppressed.value:
                continue

            await self._recall_repo.update_status(
                db, schedule.id, RecallStatus.contacted.value, last_recall_at=now_utc()
            )
            dispatched_count += 1

            await self._audit.record(
                db,
                actor_staff_id=actor_id,
                actor_type=ActorType.staff if actor_id is not None else ActorType.system,
                action_type="recall.dispatch",
                entity_type="recall_schedule",
                entity_id=schedule.id,
                override_payload={"campaign_id": str(campaign_id), "status": "contacted"},
            )

        return {"campaign_id": campaign_id, "dispatched_count": dispatched_count, "status": "contacted"}

    async def link_booking(
        self, db: "AsyncSession", recall_schedule_id: "UUID", appointment_id: "UUID"
    ) -> "RecallSchedule":
        """FR-E7.3 (US35 Alternate Flow): called by the route/caller that
        initiates a recall-driven booking, as a follow-up step after
        ``SchedulingService.create`` succeeds (Interaction contract) — never
        the reverse. Links the appointment and marks the schedule
        ``"completed"``. The schedule row's own existence is enforced by
        ``RecallScheduleRepository.link_appointment``/``update_status``
        (both resolve via ``scalar_one`` and raise if the row is absent) —
        this method performs no separate existence pre-check of its own. Per
        the Public surface, only ``TreatmentRecoveryService.convert``
        documents an explicit ``AuditService.record`` call in this file;
        this method does not audit-log itself (the ``dispatch`` call that
        preceded it, and ``SchedulingService.create``'s own booking, already
        carry the audit trail for this action).
        """
        await self._recall_repo.link_appointment(db, recall_schedule_id, appointment_id)
        updated = await self._recall_repo.update_status(db, recall_schedule_id, RecallStatus.completed.value)
        return updated

    async def get_compliance(self, db: "AsyncSession", period: str) -> dict:
        """FR-E7.4 (US36): computes the compliance rate over the *current*
        ISO week's window — this method is itself stateless/on-demand
        (``period="weekly"`` documents the calling job's cadence, not a
        parameter this method branches on); ``GET /recall/compliance``
        calls it the same way. ``compliance_rate = completed / (due +
        overdue + contacted + completed)`` for the window, expressed as a
        0-1 fraction (per the file spec's literal formula) derived from the
        same ``RecallScheduleRepository.count_by_status`` aggregate. US36
        Alternate Flow: ``baseline_rate=None``/``baseline_pending=True``
        when no baseline has ever been captured.
        """
        today = date.today()
        iso_year, iso_week, _ = today.isocalendar()
        period_start = date.fromisocalendar(iso_year, iso_week, 1)
        period_end = date.fromisocalendar(iso_year, iso_week, 7)

        counts = await self._recall_repo.count_by_status(
            db, ["due", "overdue", "contacted", "completed"], period_start, period_end
        )
        total = sum(counts.values())
        compliance_rate = (counts.get("completed", 0) / total) if total else 0.0

        baseline = await self._baseline_repo.get_latest(db, "recall_compliance_rate")
        baseline_rate = float(baseline.baseline_value) if baseline is not None else None

        return {
            "compliance_rate": compliance_rate,
            "baseline_rate": baseline_rate,
            "baseline_pending": baseline is None,
            "period": f"{iso_year}-W{iso_week:02d}",
        }


class TreatmentRecoveryService:
    """FR-E7.5-FR-E7.7 (US37-US39): the unscheduled-treatment re-engagement
    queue — scan, campaign dispatch, decline, and conversion to a booking.
    """

    def __init__(
        self,
        treatment_repo: "UnscheduledTreatmentRepository",
        outreach_service: "OutreachService",
        audit: "AuditService",
    ) -> None:
        self._treatment_repo = treatment_repo
        self._outreach_service = outreach_service
        self._audit = audit

    async def scan(self, db: "AsyncSession") -> "list[UnscheduledTreatment]":
        """FR-E7.5 (US37): returns every ``"unscheduled"`` treatment, ordered
        by valuation descending with nulls last (enforced by
        ``UnscheduledTreatmentRepository.list_by_valuation`` itself) — a row
        with ``valuation_amount IS NULL`` is still included, flagged
        incomplete by the caller (route/service response), never mutated or
        omitted here.
        """
        return await self._treatment_repo.list_by_valuation(db, status=UnscheduledTreatmentStatus.unscheduled.value)

    async def dispatch_campaign(
        self, db: "AsyncSession", unscheduled_treatment_ids: "list[UUID]", actor_id: "UUID | None"
    ) -> dict:
        """FR-E7.6 (US38): dispatches a treatment re-engagement outreach
        message per eligible (consenting) treatment's patient via
        ``OutreachService.dispatch``, and sets ``status="re_engaged"`` on
        every one actually dispatched — a treatment whose patient has no
        consented channel comes back ``"suppressed"`` (US21 Alternate Flow)
        and is left ``"unscheduled"``. Watch out: same language-resolution
        gap as ``RecallCampaignService.dispatch`` — no ``PatientRepository``
        is injected here either, so ``Language.en`` is used uniformly.
        """
        campaign_id = uuid4()
        dispatched_count = 0

        treatments = await self._treatment_repo.list_by_ids(db, unscheduled_treatment_ids)
        for treatment in treatments:
            message = await self._outreach_service.dispatch(
                db,
                patient_id=treatment.patient_id,
                campaign_type=CampaignType.treatment_reengagement.value,
                language=Language.en.value,
                related_entity_type="unscheduled_treatment",
                related_entity_id=treatment.id,
            )
            if _enum_value(message.status) == OutreachMessageStatus.suppressed.value:
                continue

            await self._treatment_repo.update_status(db, treatment.id, UnscheduledTreatmentStatus.re_engaged.value)
            dispatched_count += 1

            await self._audit.record(
                db,
                actor_staff_id=actor_id,
                actor_type=ActorType.staff if actor_id is not None else ActorType.system,
                action_type="treatment.dispatch_campaign",
                entity_type="unscheduled_treatment",
                entity_id=treatment.id,
                override_payload={"campaign_id": str(campaign_id), "status": "re_engaged"},
            )

        return {"campaign_id": campaign_id, "dispatched_count": dispatched_count, "status": "re_engaged"}

    async def decline(self, db: "AsyncSession", id: "UUID", actor_id: "UUID") -> "UnscheduledTreatment":
        """FR-E7.6 (US38): marks ``"declined"`` — excluded from
        ``scan()``'s default ``status="unscheduled"`` filter, so it is never
        picked up by a future automated re-engagement cycle. The row's own
        existence is enforced by ``UnscheduledTreatmentRepository.
        update_status`` itself (resolved via ``scalar_one``) — no separate
        existence pre-check is performed here. Per the Public surface, only
        ``convert`` documents an explicit ``AuditService.record`` call in
        this class; ``actor_id`` is accepted (per the frozen signature) but
        this method itself does not audit-log the decline.
        """
        return await self._treatment_repo.update_status(db, id, UnscheduledTreatmentStatus.declined.value)

    async def convert(
        self, db: "AsyncSession", id: "UUID", appointment_id: "UUID", partial: bool, actor_id: "UUID"
    ) -> "UnscheduledTreatment":
        """FR-E7.7 (US39): links ``appointment_id`` and sets ``"booked"``
        unless ``partial=True``, in which case the record stays
        ``"unscheduled"`` for the remaining portion (US39 AC2) — this design
        keeps a single row rather than splitting it (see file spec's Watch
        out: the BRD names no field to represent a true partial-booking
        split). Interaction contract: the caller (route layer) is
        responsible for validating ``appointment_id`` via
        ``SchedulingService.get_by_id`` before calling this method —
        ``SchedulingService`` itself never calls back into this module. The
        treatment row's own existence is enforced by
        ``UnscheduledTreatmentRepository.link_appointment``/``update_status``
        (both resolve via ``scalar_one``) — no separate existence pre-check
        is performed here.
        """
        linked = await self._treatment_repo.link_appointment(db, id, appointment_id)
        if partial:
            updated = linked
        else:
            updated = await self._treatment_repo.update_status(db, id, UnscheduledTreatmentStatus.booked.value)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="treatment.convert",
            entity_type="unscheduled_treatment",
            entity_id=id,
            override_payload={"appointment_id": str(appointment_id), "partial": partial},
        )
        return updated
