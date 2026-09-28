"""SQLAlchemy ORM declarations for the recall module's three tables
(architecture.md §4.2 recall module).

Pure schema — no query methods, no business logic.

Interaction contract: ``patient_id``/``appointment_id`` are plain FK columns,
no cross-module ORM ``relationship()`` — ``repositories/recall/repository.py``
is the only file that imports these classes; ``services/analytics/
service.py`` depends directly on ``RecallScheduleRepository`` for read-only
aggregation, never this ORM class.
"""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import Date, DateTime, ForeignKey, Numeric, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import IntervalSource, RecallStatus, UnscheduledTreatmentStatus
from app.core.database import Base


class RecallSchedule(Base):
    """A per-patient recall due-date schedule, driven by risk classification
    or the default fallback interval.

    FR-E7.4: ``RecallComplianceBaseline`` existing (or not) for
    ``metric_name="recall_compliance_rate"`` is what
    ``RecallCampaignService``'s compliance endpoint reads to decide between
    reporting a rate-vs-baseline or a "baseline pending" raw-count state
    (US36 Alternate Flow) — see ``RecallComplianceBaseline`` below.
    """

    __tablename__ = "recall_schedules"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), nullable=False)
    # Set on completion.
    appointment_id: Mapped[UUID | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    risk_classification: Mapped[str | None] = mapped_column(String(50), nullable=True)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    last_recall_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[RecallStatus] = mapped_column(SAEnum(RecallStatus, native_enum=False), nullable=False)
    interval_source: Mapped[IntervalSource] = mapped_column(
        SAEnum(IntervalSource, native_enum=False), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class UnscheduledTreatment(Base):
    """An accepted-but-unscheduled treatment plan item eligible for
    re-engagement outreach."""

    __tablename__ = "unscheduled_treatments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id"), nullable=False)
    treatment_code: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    accepted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Flagged incomplete if null.
    valuation_amount: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    status: Mapped[UnscheduledTreatmentStatus] = mapped_column(
        SAEnum(UnscheduledTreatmentStatus, native_enum=False),
        nullable=False,
        default=UnscheduledTreatmentStatus.unscheduled,
        server_default=UnscheduledTreatmentStatus.unscheduled.value,
    )
    # Conversion link.
    appointment_id: Mapped[UUID | None] = mapped_column(ForeignKey("appointments.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class RecallComplianceBaseline(Base):
    """A captured baseline value for a named recall metric (e.g.
    ``"recall_compliance_rate"``), used by FR-E7.4's rate-vs-baseline
    reporting."""

    __tablename__ = "recall_compliance_baselines"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    baseline_value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
