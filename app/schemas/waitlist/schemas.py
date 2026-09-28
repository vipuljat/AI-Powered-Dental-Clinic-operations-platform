"""Pydantic v2 request/response models for every ``/waitlist/*`` endpoint
(architecture.md §5.2).

These schemas are the whole contract for ``routes/waitlist/routes.py``:
``services/waitlist/service.py``'s return values are shaped to match these
field names directly, so no separate interaction-contract document exists
beyond this module. Contains no behaviour beyond field-level type shaping —
alert merging, offer ranking, and every other business rule live in
``services/waitlist/service.py``.
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import (
    IdleChairSource,
    IdleChairStatus,
    Urgency,
    WaitlistEntryStatus,
    WaitlistOfferStatus,
)


class FlagIdleChairRequest(BaseModel):
    provider_id: UUID
    chair_id: UUID
    slot_start: datetime
    slot_end: datetime


class FlagIdleChairResponse(BaseModel):
    """architecture.md §5.2 ``POST /waitlist/idle-chairs/flag``: 201 (new
    alert) or 200 (merged) — the same shape serves both status codes, the
    route distinguishes only by the ``merged`` flag and the HTTP status it
    sets based on the service's return value.
    """

    id: UUID
    status: IdleChairStatus
    merged: bool


class IdleChairAlertItem(BaseModel):
    id: UUID
    provider_id: UUID
    slot_start: datetime
    source: IdleChairSource
    status: IdleChairStatus


class IdleChairAlertListResponse(BaseModel):
    items: list[IdleChairAlertItem]


class AddWaitlistEntryRequest(BaseModel):
    patient_id: UUID
    desired_provider_id: UUID | None = None
    desired_timeframe_start: date | None = None
    desired_timeframe_end: date | None = None
    urgency: Urgency


class WaitlistEntryResponse(BaseModel):
    id: UUID
    priority_score: float
    status: WaitlistEntryStatus


class WaitlistEntryRankedItem(BaseModel):
    id: UUID
    patient_id: UUID
    priority_score: float
    rank: int
    status: WaitlistEntryStatus


class WaitlistEntryListResponse(BaseModel):
    items: list[WaitlistEntryRankedItem]


class RespondToOfferRequest(BaseModel):
    response: str  # "accept" | "decline"


class RespondToOfferResponse(BaseModel):
    id: UUID
    status: WaitlistOfferStatus
    appointment_id: UUID | None


class RecoveryOfferStatus(BaseModel):
    waitlist_entry_id: UUID
    status: WaitlistOfferStatus
    offered_at: datetime


class RecoveryStatusResponse(BaseModel):
    alert_id: UUID
    offers: list[RecoveryOfferStatus]
    final_status: str  # "filled" | "exhausted" | "open"
