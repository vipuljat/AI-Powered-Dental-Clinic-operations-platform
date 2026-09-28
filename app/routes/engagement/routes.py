"""Thin HTTP adapter over ``EngagementWebChatService``/``VoiceHandlingService``/
``CallScoringService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema (or FastAPI's own required-path/query validation), delegates to the
service layer, and shapes the response. This module's ``router`` carries
only its own ``/engagement`` sub-prefix; the shared ``/api/v1`` prefix is
added on top by ``app/main.py`` when it calls ``app.include_router``
(Interaction contract).
"""

from __future__ import annotations

import hashlib
import hmac
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request

# Watch out: `app/repositories/waitlist/repository.py` (frozen) registers a
# minimal id-only stand-in "patients" Table on `Base.metadata` as a no-op
# fallback for its own FK targets, *only* when the real `patients` table
# is not already registered there — but `app.core.dependencies` transitively
# imports that waitlist repository (via `services.console.service`) before
# this module otherwise would import the real `app.models.patients.models`.
# Importing the real patients repository/model first, here, ensures the real
# declarative `Patient` table claims the "patients" name before that stub
# loop ever runs, so its `if _table_name not in Base.metadata.tables` check
# correctly no-ops instead of racing a second, conflicting declaration
# (mirrors `app/routes/patients/routes.py`'s identical fix — this module
# also needs a real `PatientRepository` for `EngagementWebChatService`).
from app.repositories.patients.repository import PatientRepository  # isort: skip

from app.common.exceptions.errors import NoMatchingPatientError, NotFoundError, UnauthorizedError
from app.core.config import get_settings
from app.core.dependencies import CurrentUser, get_db, require_role
from app.core.messaging import get_broker
from app.repositories.console.repository import AuditLogRepository
from app.repositories.engagement.repository import (
    CallInteractionRepository,
    CallTranscriptSegmentRepository,
    WebChatMessageRepository,
    WebChatSessionRepository,
)
from app.repositories.intelligence.repository import EscalationRepository, TriageSessionRepository
from app.repositories.rules.repository import RuleDefinitionRepository, RuleSetRepository
from app.schemas.engagement.schemas import (
    CallInteractionDetailResponse,
    CallInteractionItem,
    CallInteractionListResponse,
    InboundCallRequest,
    InboundCallResponse,
    SendWebchatMessageRequest,
    SendWebchatMessageResponse,
    StartWebchatSessionResponse,
    VerifyIdentityRequest,
    VerifyIdentityResponse,
)
from app.services.console.service import AuditService
from app.services.engagement.service import (
    CallScoringService,
    EngagementWebChatService,
    VoiceHandlingService,
)
from app.services.intelligence.service import EscalationService, TriageService
from app.services.rules.service import RuleSetService, RuleValidationService

router = APIRouter(prefix="/engagement")

# Watch out (this file's own): the telephony carrier's shared-secret header
# name. No specific vendor is named anywhere in this tree
# (open_questions[Q-a4c64a5b], restated in app/services/engagement/service.py's
# module docstring) so a conventional header name is used here, documented at
# the point of use rather than silently assumed.
_CARRIER_SIGNATURE_HEADER = "X-Carrier-Signature"
_TRANSCRIPT_EXCERPT_MAX_CHARS = 280


# --- Per-request service/repository graph construction ---------------------
# Each request builds its own thin service/repository graph over the
# request-scoped `AsyncSession` (`Depends(get_db)`) -- no collaborator here is
# a module-level singleton, matching the rest of this codebase's
# `require_permission`/`AuthService` wiring (app/core/dependencies.py) and
# app/routes/intelligence/routes.py's identical per-request wiring pattern.


def _audit_service() -> AuditService:
    return AuditService(AuditLogRepository())


def _rule_set_service() -> RuleSetService:
    rule_def_repo = RuleDefinitionRepository()
    validator = RuleValidationService(rule_def_repo)
    return RuleSetService(RuleSetRepository(), rule_def_repo, validator, _audit_service())


def _escalation_service() -> EscalationService:
    settings = get_settings()
    return EscalationService(EscalationRepository(), _audit_service(), get_broker(settings))


def _triage_service() -> TriageService:
    settings = get_settings()
    return TriageService(
        TriageSessionRepository(), _rule_set_service(), _escalation_service(), get_broker(settings)
    )


def _webchat_service() -> EngagementWebChatService:
    return EngagementWebChatService(
        WebChatSessionRepository(), WebChatMessageRepository(), PatientRepository(), _triage_service()
    )


def _voice_service() -> VoiceHandlingService:
    return VoiceHandlingService(CallInteractionRepository(), _escalation_service(), get_settings())


def _scoring_service() -> CallScoringService:
    # Constructed for symmetry with the other per-request service builders
    # above; no route in this file calls `CallScoringService.score` directly
    # (FR-E11.5 scoring is triggered out of this REST surface's synchronous
    # path, mirroring `VoiceHandlingService.handle_scenario`'s own Watch out)
    # — retained so the collaborator graph documented in this module's
    # docstring is complete and ready for a future scoring-trigger endpoint.
    return CallScoringService(CallInteractionRepository(), CallTranscriptSegmentRepository())


def _verify_carrier_signature(raw_body: bytes, provided_signature: str | None) -> None:
    """This route's own request-signature check (this file's Watch out) --
    deliberately NOT a `core/dependencies.py` `require_role`/
    `require_permission` gate, since `POST /engagement/calls/inbound` stays
    `Auth: none` to every caller passing a request (project_rules.auth); this
    is a distinct, carrier-specific concern checked inline here instead.

    Compares the `X-Carrier-Signature` header against an HMAC-SHA256 (hex)
    of the raw request body, keyed by the carrier's shared secret.

    Config note (documented open gap, reported in this task's summary):
    `app/core/config.py` (frozen, not owned by this file) exposes no
    dedicated `telephony_webhook_secret` field, so `jwt_secret_key` is reused
    as the shared HMAC secret here -- the conventional fallback that keeps
    this check concrete without editing the frozen Settings object.
    """
    settings = get_settings()
    secret = getattr(settings, "telephony_webhook_secret", None) or settings.jwt_secret_key
    expected_signature = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    if not provided_signature or not hmac.compare_digest(expected_signature, provided_signature):
        raise UnauthorizedError("Invalid or missing carrier webhook signature.")


def _build_transcript_excerpt(segments: list) -> str | None:
    """DR002: builds a short excerpt only from already de-identified segment
    text (`CallScoringService.score` de-identifies before any
    `call_transcript_segments` row is persisted) -- the raw full transcript
    is never returned inline, only via `call_interactions.transcript_uri`
    (object storage), which this response never exposes.
    """
    if not segments:
        return None
    combined = " ".join(segment.text for segment in segments)
    if len(combined) <= _TRANSCRIPT_EXCERPT_MAX_CHARS:
        return combined
    return combined[:_TRANSCRIPT_EXCERPT_MAX_CHARS].rstrip() + "..."


@router.post("/webchat/sessions", response_model=StartWebchatSessionResponse, status_code=201)
async def start_webchat_session(db=Depends(get_db)) -> StartWebchatSessionResponse:
    """architecture.md §5.2 `POST /engagement/webchat/sessions`: 201, `Auth: none`.

    FR-E11.2/US51: starts a new anonymous, unverified webchat session.
    """
    service = _webchat_service()
    session = await service.start_session(db)
    return StartWebchatSessionResponse(
        id=session.id,
        visitor_token=session.visitor_token,
        verification_status=session.verification_status,
    )


@router.post("/webchat/sessions/{id}/messages", response_model=SendWebchatMessageResponse)
async def send_webchat_message(
    id: UUID, body: SendWebchatMessageRequest, db=Depends(get_db)
) -> SendWebchatMessageResponse:
    """architecture.md §5.2 `POST /engagement/webchat/sessions/{id}/messages`: 200, `Auth: none`.

    FR-E11.1/US51: `NotFoundError` (unknown session id) propagates unchanged
    to the registered global handler (404), never caught here.
    """
    service = _webchat_service()
    result = await service.send_message(db, id, body.body)
    return SendWebchatMessageResponse(**result)


@router.post("/webchat/sessions/{id}/verify", response_model=VerifyIdentityResponse)
async def verify_webchat_identity(
    id: UUID, body: VerifyIdentityRequest, db=Depends(get_db)
) -> VerifyIdentityResponse:
    """architecture.md §5.2 `POST /engagement/webchat/sessions/{id}/verify`: 200/404, `Auth: none`.

    FR-E11.2/US51 Alternate Flow: on `NoMatchingPatientError` (no
    first_name+last_name+dob match), this route layer -- not
    `EngagementWebChatService` -- attaches the new-patient-registration path
    (`POST /patients`, per this file's Interaction contract with
    `app/routes/patients/routes.py`) to the 404 envelope's `details`, per
    `VerifyIdentityResponse`'s own schema docstring.
    """
    service = _webchat_service()
    try:
        result = await service.verify_identity(db, id, body.first_name, body.last_name, body.dob)
    except NoMatchingPatientError as exc:
        raise NoMatchingPatientError(
            details=[{"action": "register_new_patient", "path": "/patients"}]
        ) from exc
    return VerifyIdentityResponse(**result)


@router.post("/calls/inbound", response_model=InboundCallResponse)
async def inbound_call_webhook(
    body: InboundCallRequest,
    request: Request,
    db=Depends(get_db),
    x_carrier_signature: str | None = Header(default=None, alias=_CARRIER_SIGNATURE_HEADER),
) -> InboundCallResponse:
    """architecture.md §5.2 `POST /engagement/calls/inbound`: 200, `Auth: none`
    (carrier-signature-verified webhook).

    Per this file's own Watch out: no role/permission dependency gates this
    endpoint (it is genuinely `Auth: none` to any caller); instead,
    `_verify_carrier_signature` checks the raw request body against the
    `X-Carrier-Signature` header before anything is created, raising
    `UnauthorizedError` (401) on a missing/invalid signature.

    FR-E11.3/US52: creates the `call_interactions` row via
    `VoiceHandlingService.route_inbound` -- the routing decision returned
    here is that initial `handled_by` (`"ai_voice"`); the basic-scenario vs.
    staff-escalation decision itself happens later, out of this synchronous
    webhook path, via `VoiceHandlingService.handle_scenario` (per that
    method's own docstring).
    """
    raw_body = await request.body()
    _verify_carrier_signature(raw_body, x_carrier_signature)

    service = _voice_service()
    call = await service.route_inbound(db, body.caller_phone, body.call_sid)
    return InboundCallResponse(call_interaction_id=call.id, routing=call.handled_by.value)


@router.get("/calls", response_model=CallInteractionListResponse)
async def list_calls(
    handled_by: str | None = None,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> CallInteractionListResponse:
    """architecture.md §5.2 `GET /engagement/calls`: 200."""
    call_repo = CallInteractionRepository()
    calls = await call_repo.list(db, handled_by)
    return CallInteractionListResponse(
        items=[
            CallInteractionItem(
                id=call.id,
                caller_phone=call.caller_phone,
                scenario_type=call.scenario_type,
                outcome=call.outcome,
                sentiment_score=(
                    float(call.sentiment_score) if call.sentiment_score is not None else None
                ),
            )
            for call in calls
        ]
    )


@router.get("/calls/{id}", response_model=CallInteractionDetailResponse)
async def get_call(
    id: UUID,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> CallInteractionDetailResponse:
    """architecture.md §5.2 `GET /engagement/calls/{id}`: 200.

    FR-E11.5/DR002/US53: `transcript_excerpt` is derived here (read-only)
    from the already de-identified `call_transcript_segments` rows --
    `call_interactions` itself carries no excerpt column
    (app/models/engagement/models.py).
    """
    call_repo = CallInteractionRepository()
    call = await call_repo.get_by_id(db, id)
    if call is None:
        raise NotFoundError("Call interaction not found.")

    segment_repo = CallTranscriptSegmentRepository()
    segments = await segment_repo.list_by_call(db, id)

    return CallInteractionDetailResponse(
        id=call.id,
        sentiment_score=float(call.sentiment_score) if call.sentiment_score is not None else None,
        quality_score=float(call.quality_score) if call.quality_score is not None else None,
        score_status=call.score_status,
        transcript_excerpt=_build_transcript_excerpt(segments),
    )
