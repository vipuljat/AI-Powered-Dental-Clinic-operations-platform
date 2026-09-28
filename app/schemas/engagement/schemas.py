"""Pydantic v2 request/response models for every ``/engagement/*`` endpoint.

Covers the webchat session lifecycle (start/message/verify), the inbound
call-routing webhook, and the read-only call-interaction console views.
Contains no behaviour beyond validation/shape declaration — routing,
verification logic and persistence all live in
app/services/engagement/service.py and app/repositories/engagement/repository.py.
"""

from datetime import date
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import CallScoreStatus, WebchatVerificationStatus


class StartWebchatSessionResponse(BaseModel):
    id: UUID
    # Interaction contract: visitor_token is the one value a client must echo
    # back (as the session's implicit identity) on every subsequent
    # .../messages and .../verify call via the {id} path parameter — these
    # Auth: none endpoints carry no separate bearer/auth token.
    visitor_token: str
    verification_status: WebchatVerificationStatus


class SendWebchatMessageRequest(BaseModel):
    body: str


class SendWebchatMessageResponse(BaseModel):
    reply: str
    requires_verification: bool


class VerifyIdentityRequest(BaseModel):
    first_name: str
    last_name: str
    dob: date


class VerifyIdentityResponse(BaseModel):
    """FR-E11.2/US51: only ever returned on a successful match.

    A non-match is a 404 domain error (standard error envelope) whose
    `details` offer the new-patient-registration path (US51 Alternate
    Flow) — there is deliberately no 200-with-null-patient_id shape.
    """

    verification_status: WebchatVerificationStatus
    patient_id: UUID


class InboundCallRequest(BaseModel):
    caller_phone: str
    call_sid: str


class InboundCallResponse(BaseModel):
    call_interaction_id: UUID
    routing: str  # "ai_voice" | "staff" | "callback_queue"


class CallInteractionItem(BaseModel):
    id: UUID
    caller_phone: str
    scenario_type: str | None
    outcome: str | None
    sentiment_score: float | None


class CallInteractionListResponse(BaseModel):
    items: list[CallInteractionItem]


class CallInteractionDetailResponse(BaseModel):
    id: UUID
    sentiment_score: float | None
    quality_score: float | None
    score_status: CallScoreStatus
    # DR002: deliberately named "excerpt" — a short, de-identified snippet,
    # never the raw full transcript (which lives in object storage,
    # referenced only by call_interactions.transcript_uri, never returned
    # inline over this API).
    transcript_excerpt: str | None
