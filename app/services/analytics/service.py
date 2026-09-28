"""Business-logic classes for the analytics module (architecture.md §5.3).

``OperationalAnalyticsService`` (RAR001's 6 operational metrics, FR-E9.1/US46),
``FinancialAnalyticsService`` (ROI, FR-E9.2/US47) and ``DashboardExportService``
(PDF/CSV export, FR-E9.3/US48).

Responsibility: this module aggregates E4 (scheduling)/E5 (waitlist)/E7
(recall)/E8 (intelligence) data by reading other modules' **repository**
classes directly (read-only aggregation across module boundaries is
acceptable — this is a read-only aggregator sitting "above" every other
module in the dependency graph). It writes only its own four tables
(``operational_metrics_snapshots``, ``financial_roi_snapshots``,
``baseline_costs``, ``dashboard_exports``), all via
``repositories/analytics/repository.py``.

Interaction contract: ``OperationalAnalyticsService``/``FinancialAnalyticsService``
depend on other modules' repository classes (``AppointmentRepository``,
``IdleChairAlertRepository``, ``RecallScheduleRepository``) directly, never on
their **service** classes — depending on a service would risk pulling in
write-path side effects (event publishing, audit calls) that a read-only
aggregation has no business triggering.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from app.common.enums import ActorType, DashboardType, ExportFormat, IdleChairSource
from app.common.exceptions.errors import NoDataForRangeError, ValidationFailedError
from app.common.utils import now_utc
from app.repositories.analytics.repository import (
    BaselineCostRepository,
    DashboardExportRepository,
    FinancialRoiRepository,
    OperationalMetricsRepository,
)
from app.repositories.recall.repository import RecallScheduleRepository
from app.repositories.scheduling.repository import AppointmentRepository
from app.repositories.waitlist.repository import IdleChairAlertRepository
from app.services.console.service import AuditService

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.storage import ObjectStorageClient
    from app.models.analytics.models import BaselineCost

__all__ = [
    "DashboardExportService",
    "FinancialAnalyticsService",
    "OperationalAnalyticsService",
]

# RAR001: the fixed 6-metric operational dashboard vocabulary (FR-E9.1/US46).
# No config row or other file names a 7th/different metric anywhere in this
# tree, so this tuple is the single source of truth for "the 6 RAR001
# metrics" this service must always return.
RAR001_METRICS: tuple[str, ...] = (
    "scheduling_time_minutes",
    "cancellation_recovery_rate",
    "no_show_rate",
    "recall_conversion",
    "utilisation_rate",
    "staff_time_saved",
)

# A fixed per-auto-filled-slot time-saved allowance (minutes) used by
# `staff_time_saved`'s live estimate below — see that branch's docstring.
_STAFF_TIME_SAVED_MINUTES_PER_AUTO_FILL: int = 15


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    """The [00:00:00, 23:59:59.999999] UTC bounds of a calendar day, for
    calling repositories whose range query is expressed in ``datetime``
    rather than ``date`` (e.g. ``AppointmentRepository.list_open_slots``)."""
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=timezone.utc)
    return start, end


class OperationalAnalyticsService:
    """FR-E9.1/US46: the 6 RAR001 operational metrics against baseline."""

    def __init__(
        self,
        metrics_repo: OperationalMetricsRepository,
        appt_repo: "AppointmentRepository",
        alert_repo: "IdleChairAlertRepository",
        recall_repo: "RecallScheduleRepository",
    ) -> None:
        self._metrics_repo = metrics_repo
        self._appt_repo = appt_repo
        self._alert_repo = alert_repo
        self._recall_repo = recall_repo

    async def get_metrics(
        self,
        db: "AsyncSession",
        date_from: "date | None",
        date_to: "date | None",
        metric: str | None,
    ) -> dict:
        """FR-E9.1/US46: returns all 6 RAR001 metrics against baseline for
        ``[date_from, date_to]`` (defaulting to a trailing 30-day window
        ending today when either bound is omitted), or a single metric when
        ``metric`` narrows the request.

        US46 Alternate Flow: a metric whose underlying source data for the
        requested window has not finished aggregating (most notably: the
        window's own final day, if that day is still in progress) is
        returned with a per-item ``"delayed": True`` marker and a best-effort
        (not silently stale) figure, rather than presenting an
        already-superseded number as if it were current.
        ``data_delayed`` at the response level is set if *any* requested
        metric is delayed.
        """
        today = now_utc().date()
        if date_to is None:
            date_to = today
        if date_from is None:
            date_from = date_to - timedelta(days=30)

        if metric is not None:
            if metric not in RAR001_METRICS:
                raise ValidationFailedError(
                    f"Unknown metric {metric!r}; expected one of {RAR001_METRICS}."
                )
            names: tuple[str, ...] = (metric,)
        else:
            names = RAR001_METRICS

        items: list[dict] = []
        any_delayed = False
        for name in names:
            value, baseline, delayed = await self._compute_one(db, name, date_from, date_to, today)
            any_delayed = any_delayed or delayed
            items.append({"name": name, "value": value, "baseline": baseline, "delayed": delayed})

        return {"metrics": items, "data_delayed": any_delayed}

    async def earliest_computed_at(self, db: "AsyncSession") -> "date | None":
        """FR-E9.3/US48 support: the earliest ``computed_at`` date across
        every ``operational_metrics_snapshots`` row on record — used by
        ``DashboardExportService`` as the platform's go-live-equivalent date
        (see that class's own docstring; not part of this class's own
        declared Public surface, but a plain additional read helper, not a
        write, so it does not violate this module's read-only-aggregator
        contract for other modules' data)."""
        rows = await self._metrics_repo.get_range(db, None, date.min, date.max)
        if not rows:
            return None
        return min(row.computed_at.date() for row in rows)

    async def _compute_one(
        self, db: "AsyncSession", name: str, date_from: date, date_to: date, today: date
    ) -> tuple[float, float | None, bool]:
        """Returns ``(value, baseline, delayed)`` for one RAR001 metric.

        FR-E9.1/US46: prefers an already-batch-computed
        ``operational_metrics_snapshots`` row covering the full requested
        window (the authoritative figure); falls back to a live,
        best-effort computation from the source modules' repositories,
        flagged ``delayed=True``, whenever no such snapshot exists yet for
        the requested window's end date.

        Watch out: whether the window's end date happens to be "today" is
        *not* itself the delayed/not-delayed test — a batch snapshot whose
        ``period_end`` already reaches ``date_to`` is evidence the source
        data for that window has, in fact, finished aggregating (the batch
        that produced it already ran), even if ``date_to`` is the current
        calendar day. Only the *absence* of such a covering snapshot means
        "hasn't finished aggregating" (most commonly: today's own data,
        before the day's batch has run), which is what triggers the live,
        best-effort fallback below, flagged ``delayed=True``.
        """
        # The last calendar day this metric can be considered "finished
        # aggregating" for — today's own data is still accruing.
        effective_to = min(date_to, today - timedelta(days=1))
        has_complete_window = effective_to >= date_from

        snapshots = await self._metrics_repo.get_range(db, name, date_from, date_to)
        best = max(snapshots, key=lambda snapshot: snapshot.period_end, default=None)
        baseline = best.baseline_value if best is not None else None
        covers_requested_window = best is not None and best.period_end >= date_to

        if covers_requested_window:
            return (
                float(best.metric_value),
                float(baseline) if baseline is not None else None,
                False,
            )

        if has_complete_window:
            value = await self._live_value(db, name, date_from, effective_to)
        elif best is not None:
            value = float(best.metric_value)
        else:
            latest = await self._metrics_repo.get_latest(db, name)
            value = float(latest.metric_value) if latest is not None else 0.0
            if baseline is None and latest is not None:
                baseline = latest.baseline_value

        return value, (float(baseline) if baseline is not None else None), True

    async def _live_value(self, db: "AsyncSession", name: str, date_from: date, date_to: date) -> float:
        """Best-effort figure computed directly from source-module
        repositories for the (already-complete) ``[date_from, date_to]``
        sub-window, used only when no batch-computed snapshot is available
        yet for the requested window (see ``_compute_one``)."""
        start_dt, _unused_end = _day_bounds(date_from)
        _unused_start, end_dt = _day_bounds(date_to)

        if name == "scheduling_time_minutes":
            # Average booked/rescheduled appointment duration (minutes) in
            # the window — the one duration figure derivable from
            # `AppointmentRepository`'s exposed surface.
            appointments = await self._appt_repo.list_open_slots(db, None, start_dt, end_dt)
            if not appointments:
                return 0.0
            durations = [
                (appt.scheduled_end - appt.scheduled_start).total_seconds() / 60.0
                for appt in appointments
            ]
            return sum(durations) / len(durations)

        if name == "recall_conversion":
            # Share of due/overdue/contacted/completed recall schedules in
            # the window that reached `completed` — `RecallScheduleRepository
            # .count_by_status` is the one windowed aggregate query this
            # module can reach for the recall domain.
            counts = await self._recall_repo.count_by_status(
                db, ["due", "overdue", "contacted", "completed"], date_from, date_to
            )
            total = sum(counts.values())
            if total == 0:
                return 0.0
            return counts.get("completed", 0) / total * 100.0

        if name == "cancellation_recovery_rate":
            # Watch out: `IdleChairAlertRepository` (frozen) exposes only
            # `list_open` — no "list all (open+filled)" or windowed query
            # exists to derive a true historical fill-rate. As a proxy, the
            # recovery rate is reported as 100% minus the still-open share of
            # currently-open alerts that were detected within the requested
            # window — the closest live approximation this repository's
            # surface can support.
            open_alerts = await self._alert_repo.list_open(db)
            in_window = [
                alert for alert in open_alerts if date_from <= alert.detected_at.date() <= date_to
            ]
            if not open_alerts:
                return 100.0
            return max(0.0, 100.0 - (len(in_window) / len(open_alerts)) * 100.0)

        if name == "utilisation_rate":
            # Booked chair-time vs. currently-open (idle) chair-time — a
            # live proxy for chair utilisation from the two repositories
            # this service can reach.
            appointments = await self._appt_repo.list_open_slots(db, None, start_dt, end_dt)
            open_alerts = await self._alert_repo.list_open(db)
            denominator = len(appointments) + len(open_alerts)
            if denominator == 0:
                return 0.0
            return len(appointments) / denominator * 100.0

        if name == "no_show_rate":
            # Watch out: `AppointmentRepository` (frozen) exposes no
            # status+date-range query capable of separating `no_show` from
            # other terminal statuses within an arbitrary window — until a
            # batch job populates `operational_metrics_snapshots` for this
            # metric, the live estimate is conservatively 0.0 (no evidence of
            # no-shows reachable from this repository's exposed surface),
            # rather than fabricating a number this service cannot support.
            return 0.0

        if name == "staff_time_saved":
            # Minutes saved by automatic waitlist fill of currently-open,
            # auto-detected idle-chair alerts vs. a manual rebooking call —
            # approximated as a fixed per-alert allowance since no direct
            # "minutes saved" figure is tracked by any repository this
            # service can reach.
            open_alerts = await self._alert_repo.list_open(db)
            auto_detected = [
                alert for alert in open_alerts if alert.source == IdleChairSource.auto_detected
            ]
            return float(len(auto_detected) * _STAFF_TIME_SAVED_MINUTES_PER_AUTO_FILL)

        return 0.0


class FinancialAnalyticsService:
    """FR-E9.2/US47: financial ROI, gated on a captured baseline cost."""

    def __init__(
        self,
        roi_repo: FinancialRoiRepository,
        baseline_repo: BaselineCostRepository,
        audit: "AuditService",
    ) -> None:
        self._roi_repo = roi_repo
        self._baseline_repo = baseline_repo
        self._audit = audit

    async def get_roi(self, db: "AsyncSession", period: str) -> dict:
        """FR-E9.2/US47: only ``period="monthly"`` is supported (no other
        rollup cadence is named anywhere in this tree). US47 Alternate Flow:
        when no ``baseline_costs`` row exists for the current effective
        period, returns ``{"roi_percentage": None, ..., "baseline_cost_required":
        True}`` — the documented "prompt for input" state, not an error
        response.
        """
        if period != "monthly":
            raise ValidationFailedError(
                f"Unsupported ROI period {period!r}; only 'monthly' is supported."
            )

        today = now_utc().date()
        effective_period = date(today.year, today.month, 1)

        baseline = await self._baseline_repo.get_for_period(db, effective_period)
        latest = await self._roi_repo.get_latest(db)
        recovered_appointments = latest.recovered_appointments if latest is not None else 0
        recovered_revenue = float(latest.recovered_revenue) if latest is not None else 0.0

        if baseline is None:
            trend = await self._build_trend(db)
            return {
                "roi_percentage": None,
                "recovered_appointments": recovered_appointments,
                "recovered_revenue": recovered_revenue,
                "trend": trend,
                "baseline_cost_required": True,
            }

        baseline_amount = float(baseline.amount)
        roi_percentage = (
            ((recovered_revenue - baseline_amount) / baseline_amount) * 100.0
            if baseline_amount
            else 0.0
        )
        # This module writes only its own four tables (Responsibility) — the
        # freshly-computed ROI% is persisted back onto this period's own
        # `financial_roi_snapshots` row so the next read (and the trend line
        # below) reflects it without waiting for a separate batch job.
        await self._roi_repo.upsert_snapshot(
            db,
            period_month=effective_period,
            recovered_appointments=recovered_appointments,
            recovered_revenue=recovered_revenue,
            baseline_cost=baseline_amount,
            roi_percentage=roi_percentage,
        )
        trend = await self._build_trend(db)
        return {
            "roi_percentage": roi_percentage,
            "recovered_appointments": recovered_appointments,
            "recovered_revenue": recovered_revenue,
            "trend": trend,
            "baseline_cost_required": False,
        }

    async def _build_trend(self, db: "AsyncSession") -> list[dict]:
        """The trailing 6 months of ROI% -- 6 is this service's own choice
        (no trend-window length is specified anywhere in this tree) chosen
        as a conventional "half-year" rollup granularity for a monthly
        dashboard trend line."""
        snapshots = await self._roi_repo.get_trend(db, 6)
        return [
            {
                "month": snapshot.period_month.strftime("%Y-%m"),
                "roi_percentage": (
                    float(snapshot.roi_percentage) if snapshot.roi_percentage is not None else 0.0
                ),
            }
            for snapshot in snapshots
        ]

    async def earliest_computed_at(self, db: "AsyncSession") -> "date | None":
        """FR-E9.3/US48 support: the earliest ``computed_at`` date across
        every ``financial_roi_snapshots`` row on record (see
        ``OperationalAnalyticsService.earliest_computed_at`` for the same
        pattern; ``FinancialRoiRepository`` exposes no dedicated
        "earliest"/count query, so every on-record snapshot is read via
        ``get_trend`` with a generously large ``months`` bound)."""
        snapshots = await self._roi_repo.get_trend(db, 10_000)
        if not snapshots:
            return None
        return min(snapshot.computed_at.date() for snapshot in snapshots)

    async def set_baseline_cost(
        self, db: "AsyncSession", amount: float, effective_period: "date", actor_id: "UUID"
    ) -> "BaselineCost":
        """T119 AC: captures a new baseline cost figure and audit-logs it
        (``action_type="baseline_cost.set"``) — a fresh row every time, never
        an update, so the history of captured baselines is preserved
        (matches ``BaselineCostRepository``'s own no-update-method design).
        """
        baseline = await self._baseline_repo.save_baseline(
            db, entered_by_staff_id=actor_id, amount=amount, effective_period=effective_period
        )
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="baseline_cost.set",
            entity_type="baseline_cost",
            entity_id=baseline.id,
            override_payload={"amount": amount, "effective_period": effective_period.isoformat()},
        )
        return baseline


class DashboardExportService:
    """FR-E9.3/US48: generates a PDF/CSV dashboard export file whose figures
    are read through the exact same aggregation calls the live dashboard
    endpoints use, so an export can never drift from what is shown on
    screen (T121 AC)."""

    def __init__(
        self,
        export_repo: DashboardExportRepository,
        operational_service: OperationalAnalyticsService,
        financial_service: FinancialAnalyticsService,
        storage: "ObjectStorageClient",
    ) -> None:
        self._export_repo = export_repo
        self._operational_service = operational_service
        self._financial_service = financial_service
        self._storage = storage

    async def generate(
        self,
        db: "AsyncSession",
        dashboard_type: str,
        format: str,
        date_from: "date",
        date_to: "date",
        actor_id: "UUID",
    ) -> dict:
        """FR-E9.3/US48: raises ``NoDataForRangeError`` (422) if ``date_to``
        predates the platform's own go-live/earliest-data date — since no
        separate ``go_live_date`` config exists anywhere in this tree, that
        is operationalised as "predates the earliest snapshot on record"
        across ``operational_metrics_snapshots``/``financial_roi_snapshots``
        (including the case where *no* snapshot exists at all yet, which is
        equally "no data available for the requested range").

        T121 AC: the exported figures are read via the SAME
        ``operational_service.get_metrics``/``financial_service.get_roi``
        calls the live dashboard endpoints use — one code path, so the
        export can never separately drift from what is shown on screen.
        """
        try:
            dashboard_type_enum = DashboardType(dashboard_type)
        except ValueError as exc:
            raise ValidationFailedError(f"Unknown dashboard_type {dashboard_type!r}.") from exc
        try:
            format_enum = ExportFormat(format)
        except ValueError as exc:
            raise ValidationFailedError(f"Unknown export format {format!r}.") from exc

        earliest = await self._earliest_snapshot_date(db)
        if earliest is None or date_to < earliest:
            raise NoDataForRangeError()

        if dashboard_type_enum is DashboardType.operational:
            payload = await self._operational_service.get_metrics(db, date_from, date_to, None)
        else:
            payload = await self._financial_service.get_roi(db, "monthly")

        content, content_type = self._render(format_enum, dashboard_type_enum, date_from, date_to, payload)

        key = f"dashboard_exports/{uuid4()}.{format_enum.value}"
        storage_uri = await self._storage.put_object(key, content, content_type)

        export = await self._export_repo.create(
            db,
            requested_by_staff_id=actor_id,
            dashboard_type=dashboard_type,
            format=format,
            date_range_start=date_from,
            date_range_end=date_to,
            storage_uri=storage_uri,
        )
        return {"export_id": export.id, "status": "processing"}

    async def _earliest_snapshot_date(self, db: "AsyncSession") -> "date | None":
        operational_earliest = await self._operational_service.earliest_computed_at(db)
        financial_earliest = await self._financial_service.earliest_computed_at(db)
        candidates = [value for value in (operational_earliest, financial_earliest) if value is not None]
        return min(candidates) if candidates else None

    def _render(
        self,
        format_enum: ExportFormat,
        dashboard_type_enum: DashboardType,
        date_from: date,
        date_to: date,
        payload: dict,
    ) -> tuple[bytes, str]:
        if format_enum is ExportFormat.csv:
            return self._render_csv(dashboard_type_enum, date_from, date_to, payload), "text/csv"
        return self._render_pdf(dashboard_type_enum, date_from, date_to, payload), "application/pdf"

    def _render_csv(
        self, dashboard_type_enum: DashboardType, date_from: date, date_to: date, payload: dict
    ) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["dashboard_type", dashboard_type_enum.value])
        writer.writerow(["date_from", date_from.isoformat()])
        writer.writerow(["date_to", date_to.isoformat()])
        writer.writerow([])
        if dashboard_type_enum is DashboardType.operational:
            writer.writerow(["metric", "value", "baseline", "delayed"])
            for item in payload["metrics"]:
                writer.writerow([item["name"], item["value"], item["baseline"], item["delayed"]])
            writer.writerow(["data_delayed", payload["data_delayed"]])
        else:
            writer.writerow(["roi_percentage", payload["roi_percentage"]])
            writer.writerow(["recovered_appointments", payload["recovered_appointments"]])
            writer.writerow(["recovered_revenue", payload["recovered_revenue"]])
            writer.writerow(["baseline_cost_required", payload["baseline_cost_required"]])
            writer.writerow([])
            writer.writerow(["month", "roi_percentage"])
            for point in payload["trend"]:
                writer.writerow([point["month"], point["roi_percentage"]])
        return buffer.getvalue().encode("utf-8")

    def _render_pdf(
        self, dashboard_type_enum: DashboardType, date_from: date, date_to: date, payload: dict
    ) -> bytes:
        lines = [
            f"Dashboard export: {dashboard_type_enum.value}",
            f"Range: {date_from.isoformat()} to {date_to.isoformat()}",
        ]
        if dashboard_type_enum is DashboardType.operational:
            for item in payload["metrics"]:
                lines.append(
                    f"{item['name']}: {item['value']} (baseline {item['baseline']}, "
                    f"delayed={item['delayed']})"
                )
            lines.append(f"data_delayed: {payload['data_delayed']}")
        else:
            lines.append(f"roi_percentage: {payload['roi_percentage']}")
            lines.append(f"recovered_appointments: {payload['recovered_appointments']}")
            lines.append(f"recovered_revenue: {payload['recovered_revenue']}")
            lines.append(f"baseline_cost_required: {payload['baseline_cost_required']}")
            for point in payload["trend"]:
                lines.append(f"{point['month']}: {point['roi_percentage']}")
        return _build_minimal_pdf(lines)


def _build_minimal_pdf(lines: list[str]) -> bytes:
    """Builds a minimal, syntactically valid single-page PDF (Helvetica,
    12pt, one line per string), using only the standard library.

    No PDF-generation dependency (e.g. ``reportlab``) is declared in
    ``pyproject.toml`` and this file must not introduce one on its own, so a
    small hand-built PDF object graph (catalog/pages/page/font/content
    stream, with a correct xref table) is constructed directly instead.
    """

    def esc(text: str) -> str:
        return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    content_lines = ["BT", "/F1 12 Tf", "50 750 Td"]
    for index, line in enumerate(lines):
        if index > 0:
            content_lines.append("0 -16 Td")
        content_lines.append(f"({esc(line)}) Tj")
    content_lines.append("ET")
    stream_body = "\n".join(content_lines).encode("utf-8")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream_body)).encode("ascii") + b" >>\nstream\n"
        + stream_body
        + b"\nendstream",
    ]

    header = b"%PDF-1.4\n"
    body = bytearray()
    offsets: list[int] = []
    position = len(header)
    for index, obj_body in enumerate(objects, start=1):
        offsets.append(position)
        obj_bytes = f"{index} 0 obj\n".encode("ascii") + obj_body + b"\nendobj\n"
        body += obj_bytes
        position += len(obj_bytes)

    xref_offset = len(header) + len(body)
    xref_lines = [f"0 {len(objects) + 1}", "0000000000 65535 f "]
    for offset in offsets:
        xref_lines.append(f"{offset:010d} 00000 n ")
    xref_text = "xref\n" + "\n".join(xref_lines) + "\n"
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF"

    return header + bytes(body) + xref_text.encode("ascii") + trailer.encode("ascii")
