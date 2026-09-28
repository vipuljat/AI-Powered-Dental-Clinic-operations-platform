"""Pydantic v2 request/response models for every `/outreach/*` endpoint
(architecture.md §5.2).

Responsibility: this module contains only schema definitions — no
validation logic beyond field-level constraints, no DB access, no service
calls. `services/outreach/service.py`'s return values are shaped to match
these field names directly; there is no separate interaction contract
document for this pairing.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import (
    CampaignType,
    Channel,
    ChannelConfigStatus,
    ConsentStatus,
    Language,
    OutreachMessageStatus,
)


class OutreachMessageItem(BaseModel):
    id: UUID
    campaign_type: CampaignType
    channel: Channel
    language: Language
    status: OutreachMessageStatus
    sent_at: datetime | None = None


class OutreachMessageListResponse(BaseModel):
    items: list[OutreachMessageItem]


class ManualOutcomeRequest(BaseModel):
    outcome: str
    notes: str | None = None


class ManualOutcomeResponse(BaseModel):
    id: UUID
    manual_outcome: str


class CaptureConsentRequest(BaseModel):
    patient_id: UUID
    channel: Channel
    # Accepts ConsentStatus.granted or .declined only; ConsentStatus.withdrawn
    # is reachable exclusively via POST /outreach/consent/withdraw, never as a
    # direct request value. Enforced at the service layer (not narrowed here)
    # so this single shared schema keeps the request shape identical to
    # architecture.md's documented example for both grant and decline.
    status: ConsentStatus
    source: str | None = None


class ConsentResponse(BaseModel):
    id: UUID
    channel: Channel
    status: ConsentStatus
    effective_at: datetime


class WithdrawConsentRequest(BaseModel):
    patient_id: UUID
    channel: Channel
    source: str | None = None


class WithdrawConsentResponse(BaseModel):
    id: UUID
    status: ConsentStatus
    queued_messages_halted: int


class ConsentLedgerItem(BaseModel):
    channel: Channel
    status: ConsentStatus
    effective_at: datetime


class ConsentLedgerResponse(BaseModel):
    items: list[ConsentLedgerItem]


class ConfigureWhatsappRequest(BaseModel):
    bsp_provider: str
    phone_number_id: str
    access_token: str


class ConfigureWhatsappResponse(BaseModel):
    id: UUID
    status: ChannelConfigStatus


class TestSendWhatsappRequest(BaseModel):
    to_phone: str


class TestSendWhatsappResponse(BaseModel):
    # Per architecture.md §5.2, a BSP delivery failure on this endpoint is a
    # 502 error response (ChannelDeliveryError) — this schema is therefore
    # only ever returned on success, with status="verified"; it never
    # carries a "pending"/failure value in a 200 body.
    status: ChannelConfigStatus
