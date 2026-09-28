"""Pydantic v2 request/response models for every `/education/*` endpoint
(architecture.md §5.2).

Contains no DB access, no business logic beyond field-level shaping; the
service layer computes every value these models carry.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import ContentDeliveryStatus, EducationTriggerType

# FR-E10.3 / architecture.md §5.2 `GET /education/tracking`: a static,
# documented technical-limitation table of which channels cannot report
# "opened" events — not derived from delivery data. SMS carriers give no
# open-tracking signal; WhatsApp and email both support delivery receipts
# with open/read tracking.
CHANNEL_OPEN_TRACKING_CAPABILITY: dict[str, bool] = {
    "sms": False,
    "email": True,
    "whatsapp": True,
}


class TrackingFunnelResponse(BaseModel):
    """FR-E10.3: the delivered -> opened -> completed funnel for education
    content, plus the list of channels whose opens cannot be tracked."""

    delivered: int
    opened: int
    completed: int
    channel_limited: list[str]


class ContentDeliveryItem(BaseModel):
    content_item_id: UUID
    trigger_type: EducationTriggerType
    status: ContentDeliveryStatus
    # Required but nullable: the field must always be supplied by the
    # service layer (a scheduled-but-not-yet-delivered item still reports
    # this key with an explicit null), so no default is set here — omitting
    # the key entirely is a distinct validation error from passing None.
    delivered_at: datetime | None


class ContentDeliveryListResponse(BaseModel):
    items: list[ContentDeliveryItem]
