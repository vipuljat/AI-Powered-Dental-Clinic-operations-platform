"""Pydantic v2 request/response models for every `/recall/*` endpoint
(architecture.md §5.2).

Field names/types here are exactly what `services/recall/service.py`'s return
dicts must match key-for-key, and exactly what `routes/recall/routes.py`
validates requests against. No additional out-of-band agreement exists beyond
what these signatures carry.
"""

from datetime import date
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import RecallStatus, UnscheduledTreatmentStatus


class RecallQueueItem(BaseModel):
    patient_id: UUID
    risk_classification: str | None = None
    due_date: date
    overdue_days: int
    status: RecallStatus


class RecallQueueResponse(BaseModel):
    items: list[RecallQueueItem]


class TriggerRecallCampaignRequest(BaseModel):
    recall_schedule_ids: list[UUID]


class TriggerCampaignResponse(BaseModel):
    campaign_id: UUID
    dispatched_count: int
    status: str


class RecallComplianceResponse(BaseModel):
    # architecture.md §5.2 GET /recall/compliance: `baseline_pending: true`
    # accompanies `baseline_rate: null` (US36 Alternate Flow "baseline
    # pending... compliance shown as raw counts") — `baseline_rate` is
    # nullable specifically to carry that state; `compliance_rate` is still
    # populated (raw rate) even when a baseline is pending.
    compliance_rate: float
    baseline_rate: float | None = None
    baseline_pending: bool
    period: str


class UnscheduledTreatmentItem(BaseModel):
    id: UUID
    patient_id: UUID
    treatment_code: str
    valuation_amount: float | None = None
    days_unscheduled: int
    status: UnscheduledTreatmentStatus


class UnscheduledTreatmentListResponse(BaseModel):
    items: list[UnscheduledTreatmentItem]


class UpdateValuationRequest(BaseModel):
    valuation_amount: float


class TriggerTreatmentCampaignRequest(BaseModel):
    unscheduled_treatment_ids: list[UUID]


class ConvertTreatmentRequest(BaseModel):
    appointment_id: UUID
    partial: bool = False


class ConvertTreatmentResponse(BaseModel):
    id: UUID
    status: UnscheduledTreatmentStatus
    appointment_id: UUID
