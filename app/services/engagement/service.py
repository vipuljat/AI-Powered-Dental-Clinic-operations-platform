"""Business logic for the ``engagement`` module.

``EngagementWebChatService`` (FR-E11.1-FR-E11.2/US51) runs the anonymous
webchat session lifecycle — start, general conversation, and the one
identity-verification path that ever attaches a real patient to a session.
``VoiceHandlingService`` (FR-E11.3-FR-E11.4/US52) routes an inbound call and
classifies it into one of the three configured "basic" AI-voice scenarios or
escalates to staff via the shared ``EscalationService``. ``CallScoringService``
(FR-E11.5/DR002/US53) de-identifies and persists a call's transcript segments
and computes its sentiment/quality scores, or explicitly marks them
unavailable when there is nothing to analyse.

Watch out: no concrete LLM/telephony/STT/TTS vendor is named anywhere in this
tree (``open_questions[Q-a4c64a5b]``) — every "AI reply"/"transcription"/
"scoring" computation in this file is deterministic stub logic (keyword/
intent matching, canned text, a fixed heuristic), sufficient to satisfy this
tree's acceptance checks without a real LLM/telephony call. Wiring an actual
vendor is future work outside this backend spec's scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from app.common.enums import CallHandledBy, CallScoreStatus, CallSpeaker, WebchatSender
from app.common.exceptions.errors import NoMatchingPatientError, NotFoundError
from app.common.utils import de_identify_transcript_text, now_utc
from app.repositories.engagement.repository import (
    CallInteractionRepository,
    CallTranscriptSegmentRepository,
    WebChatMessageRepository,
    WebChatSessionRepository,
)
from app.repositories.patients.repository import PatientRepository

if TYPE_CHECKING:
    from datetime import date
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.core.config import Settings
    from app.models.engagement.models import CallInteraction, WebchatSession
    from app.services.intelligence.service import EscalationService, TriageService

__all__ = ["CallScoringService", "EngagementWebChatService", "VoiceHandlingService"]


# --- EngagementWebChatService keyword/reply configuration -------------------
# FR-E11.1/US51: the BRD never fixes an exact NLP intent classifier — this is
# the documented, conventional stand-in (keyword matching against a small
# configured phrase list), per this file's own Watch out.
_BOOKING_KEYWORDS = ("book", "appointment", "reschedule", "schedule", "cancel my")
_RECORD_VIEW_KEYWORDS = ("my record", "my file", "my history", "my chart", "my treatment")
_TRIAGE_INTENT_KEYWORDS = ("triage", "symptom", "hurts", "pain", "toothache", "emergency")
_SCHEDULING_INTENT_KEYWORDS = ("waitlist", "availability", "slot", "next opening")
_PATIENT_SPECIFIC_KEYWORDS = (
    _BOOKING_KEYWORDS + _RECORD_VIEW_KEYWORDS + _TRIAGE_INTENT_KEYWORDS + _SCHEDULING_INTENT_KEYWORDS
)

_VERIFICATION_REQUEST_REPLY = (
    "I can help with that, but first I need to verify who I'm speaking with. "
    "Could you share your first name, last name, and date of birth?"
)
# No LLM integration is specced here (see module docstring Watch out) — a
# single canned conversational reply stands in for a real general-purpose
# chat response.
_GENERAL_REPLY = (
    "Thanks for reaching out! I'm happy to answer general questions about our "
    "clinic. Let me know how I can help."
)


def _implies_patient_specific_action(body: str) -> bool:
    """FR-E11.1/US51: keyword match deciding whether ``body`` implies a
    booking/record-view action or an explicit triage/scheduling intent.
    """
    normalized = body.lower()
    return any(keyword in normalized for keyword in _PATIENT_SPECIFIC_KEYWORDS)


class EngagementWebChatService:
    """FR-E11.1-FR-E11.2/US51: anonymous-until-verified webchat sessions."""

    def __init__(
        self,
        session_repo: WebChatSessionRepository,
        message_repo: WebChatMessageRepository,
        patient_repo: PatientRepository,
        triage_service: "TriageService",
    ) -> None:
        self._session_repo = session_repo
        self._message_repo = message_repo
        self._patient_repo = patient_repo
        # Per this file's Interaction contract / T129: patient-specific action
        # delegation (booking/record-view/triage continuation) happens at the
        # ROUTE layer, gated on `session.verification_status == "verified"` —
        # this service never calls into `scheduling`/`patients`/`intelligence`
        # itself. Retained only to match this class's frozen constructor
        # Public surface (same pattern as `TriageService.__init__`'s unused
        # `broker` in app/services/intelligence/service.py).
        self._triage_service = triage_service

    async def start_session(self, db: "AsyncSession") -> "WebchatSession":
        """Creates a new anonymous ``webchat_sessions`` row.

        FR-E11.2/US51: the session starts unverified — no patient is
        attached until `verify_identity` succeeds.
        """
        visitor_token = f"vt_{uuid4().hex}"
        return await self._session_repo.create(db, visitor_token)

    async def send_message(self, db: "AsyncSession", session_id: "UUID", body: str) -> dict:
        """FR-E11.1/US51: appends the visitor message, then either requests
        identifying details (patient-specific action implied + unverified)
        or replies conversationally — `requires_verification` is never
        `True` for an ordinary conversational turn.
        """
        session = await self._session_repo.get_by_id(db, session_id)
        if session is None:
            raise NotFoundError("Webchat session not found.")

        await self._message_repo.append_message(db, session_id, WebchatSender.visitor.value, body)

        patient_action_implied = _implies_patient_specific_action(body)
        is_verified = session.verification_status.value == "verified" if hasattr(
            session.verification_status, "value"
        ) else session.verification_status == "verified"

        if patient_action_implied and not is_verified:
            reply = _VERIFICATION_REQUEST_REPLY
            requires_verification = True
        else:
            # General conversation is never gated behind verification
            # (FR-E11.1) — a stub/canned reply, no LLM integration specced.
            reply = _GENERAL_REPLY
            requires_verification = False

        await self._message_repo.append_message(db, session_id, WebchatSender.ai_agent.value, reply)
        return {"reply": reply, "requires_verification": requires_verification}

    async def verify_identity(
        self, db: "AsyncSession", session_id: "UUID", first_name: str, last_name: str, dob: "date"
    ) -> dict:
        """FR-E11.2/US51: the one path that ever attaches a real patient to a
        webchat session — exact first_name+last_name+dob match (per
        architecture.md §5.2's `VerifyIdentityRequest` example, the BRD names
        name+DOB, not phone, as the chat verification fields).

        Raises `NoMatchingPatientError` (404) on no match — the route layer
        (not this service) offers the new-patient-registration path in its
        error `details`, per this file's Interaction contract.
        """
        session = await self._session_repo.get_by_id(db, session_id)
        if session is None:
            raise NotFoundError("Webchat session not found.")

        candidates = await self._patient_repo.search(db, q=f"{first_name} {last_name}")
        match = next(
            (
                patient
                for patient in candidates
                if patient.first_name.strip().casefold() == first_name.strip().casefold()
                and patient.last_name.strip().casefold() == last_name.strip().casefold()
                and patient.dob == dob
            ),
            None,
        )
        if match is None:
            raise NoMatchingPatientError()

        verified_session = await self._session_repo.verify(db, session_id, match.id)
        return {
            "verification_status": verified_session.verification_status,
            "patient_id": match.id,
        }


# --- VoiceHandlingService scenario/escalation configuration ------------------
# FR-E11.3/US52: exactly the three configured "basic" scenarios named in the
# BRD — anything else (including any of these matching an escalation-trigger
# keyword) routes to staff (FR-E11.4).
_BASIC_SCENARIOS = frozenset({"confirmation", "reminder", "simple_reschedule"})
# FR-E11.4/FR-E8.5's "100% escalation" guarantee, reused uniformly here: any
# scenario mentioning one of these keywords always escalates, even if its
# name would otherwise match a basic scenario.
_ESCALATION_TRIGGER_KEYWORDS = (
    "emergency",
    "pain",
    "bleeding",
    "swelling",
    "trauma",
    "urgent",
    "complaint",
)


class _StubVoiceAdapter:
    """Canned stand-in for the vendor-unnamed telephony/STT/TTS/LLM stack.

    Interaction contract: same substitution pattern as
    ``services/outreach/channel_gateway.py``'s ``StubAdapter`` — but unlike
    that file's WhatsApp adapter, no real vendor is named for voice/STT/TTS/
    LLM in any environment (``open_questions[Q-a4c64a5b]``), so this stub
    always stands in, not only under ``settings.environment == "test"``.
    """

    async def respond_to_scenario(self, scenario_type: str) -> str:
        return f"Automated voice response delivered for scenario '{scenario_type}'."


def _get_voice_adapter(settings: "Settings") -> _StubVoiceAdapter:
    # project_rules.testing item (4): a real carrier/LLM client must never be
    # constructed when `settings.environment == "test"`; since no vendor is
    # named for production either (see `_StubVoiceAdapter`'s docstring), the
    # same canned adapter is returned in every environment until one is
    # selected — `settings` is threaded through so that switch point exists
    # here, ready for a real construction branch once a vendor is chosen.
    return _StubVoiceAdapter()


class VoiceHandlingService:
    """FR-E11.3-FR-E11.4/US52: inbound call routing and basic-scenario
    handling, escalating anything out of scope."""

    def __init__(
        self,
        call_repo: CallInteractionRepository,
        escalation_service: "EscalationService",
        settings: "Settings",
    ) -> None:
        self._call_repo = call_repo
        self._escalation_service = escalation_service
        self._settings = settings

    async def route_inbound(
        self, db: "AsyncSession", caller_phone: str, call_sid: str
    ) -> "CallInteraction":
        """Creates the `call_interactions` row for a newly connected call.

        `handled_by` starts as `"ai_voice"` — the routing decision itself
        happens in `handle_scenario`, called separately/asynchronously by the
        telephony integration once the call is actually connected (out of
        this REST endpoint's synchronous path, per this class's Watch out).
        `call_sid` is the carrier's own call identifier, used only to
        correlate this REST call with the telephony webhook that later calls
        `handle_scenario` — `call_interactions` carries no such column
        (models/engagement/models.py), so it is not persisted here.
        """
        return await self._call_repo.create(
            db,
            caller_phone=caller_phone,
            handled_by=CallHandledBy.ai_voice,
            started_at=now_utc(),
        )

    async def handle_scenario(
        self, db: "AsyncSession", call_interaction_id: "UUID", scenario_type: str
    ) -> "CallInteraction":
        """FR-E11.3: handles exactly the three configured basic scenarios
        when no escalation-trigger keyword matches; FR-E11.4: anything else
        (out of scope, or an escalation-trigger keyword match even on a
        basic-scenario name) routes to staff via the same escalation
        mechanism triage uses (FR-E8.5's "100% escalation" guarantee applied
        uniformly to both entry points).
        """
        call = await self._call_repo.get_by_id(db, call_interaction_id)
        if call is None:
            raise NotFoundError("Call interaction not found.")

        normalized = scenario_type.lower()
        is_basic_scenario = normalized in _BASIC_SCENARIOS
        escalation_matched = any(
            keyword in normalized for keyword in _ESCALATION_TRIGGER_KEYWORDS
        )

        if is_basic_scenario and not escalation_matched:
            adapter = _get_voice_adapter(self._settings)
            outcome = await adapter.respond_to_scenario(scenario_type)
            return await self._call_repo.update_fields(
                db,
                call_interaction_id,
                scenario_type=scenario_type,
                outcome=outcome,
                handled_by=CallHandledBy.ai_voice,
            )

        trigger_reason = scenario_type if escalation_matched else f"out_of_scope:{scenario_type}"
        await self._escalation_service.route(
            db, trigger_reason=trigger_reason, call_interaction_id=call_interaction_id
        )
        return await self._call_repo.update_fields(
            db,
            call_interaction_id,
            scenario_type=scenario_type,
            outcome="escalated_to_staff",
            handled_by=CallHandledBy.staff,
        )


# --- CallScoringService stub STT/scoring configuration -----------------------
# Watch out (module docstring): no real STT/sentiment-analysis vendor is
# integrated anywhere in this tree — these are documented, deterministic
# placeholders.
_STUB_QUALITY_SCORE = 0.9
_POSITIVE_SENTIMENT_WORDS = ("thank", "great", "good", "appreciate", "happy")
_NEGATIVE_SENTIMENT_WORDS = ("angry", "upset", "bad", "terrible", "unhappy", "frustrated")


def _stub_transcribe_call(call: "CallInteraction") -> list[dict]:
    """Canned transcript segments standing in for a real STT call.

    `call.scenario_type` (not PII) is folded into the canned text purely so
    the stub isn't identical for every call; no caller/patient PII is ever
    included here.
    """
    scenario = call.scenario_type or "a general inquiry"
    return [
        {
            "speaker": CallSpeaker.ai_agent,
            "text": "Thank you for calling our clinic. How can I help you today?",
            "start_ms": 0,
            "end_ms": 4000,
        },
        {
            "speaker": CallSpeaker.patient,
            "text": f"I'm calling about {scenario}.",
            "start_ms": 4000,
            "end_ms": 8000,
        },
        {
            "speaker": CallSpeaker.ai_agent,
            "text": "Understood, thank you for confirming. Have a great day.",
            "start_ms": 8000,
            "end_ms": 12000,
        },
    ]


def _stub_score_transcript(segments: list[dict]) -> "tuple[float, float]":
    """Fixed-heuristic sentiment/quality scoring over de-identified text."""
    combined_text = " ".join(segment["text"].lower() for segment in segments)
    positive_hits = sum(combined_text.count(word) for word in _POSITIVE_SENTIMENT_WORDS)
    negative_hits = sum(combined_text.count(word) for word in _NEGATIVE_SENTIMENT_WORDS)
    net = positive_hits - negative_hits
    sentiment_score = max(0.0, min(1.0, 0.5 + 0.1 * net))
    return sentiment_score, _STUB_QUALITY_SCORE


class CallScoringService:
    """FR-E11.5/DR002/US53: de-identifies and persists transcript segments,
    then scores sentiment/quality — or explicitly marks scoring
    `"unavailable"` when there is nothing usable to analyse."""

    def __init__(
        self,
        call_repo: CallInteractionRepository,
        segment_repo: CallTranscriptSegmentRepository,
    ) -> None:
        self._call_repo = call_repo
        self._segment_repo = segment_repo

    async def score(self, db: "AsyncSession", call_interaction_id: "UUID") -> "CallInteraction":
        """DR002/US53: de-identifies transcript text (via
        `app.common.utils.de_identify_transcript_text`) BEFORE any
        `call_transcript_segments` row is persisted — never after.

        FR-E11.5/US53 Alternate Flow: a call with no recording to analyse
        (nothing to transcribe/score) always writes an explicit
        `score_status="unavailable"`, leaving `sentiment_score`/
        `quality_score` `None` rather than guessing a number.
        """
        call = await self._call_repo.get_by_id(db, call_interaction_id)
        if call is None:
            raise NotFoundError("Call interaction not found.")

        if not getattr(call, "recording_uri", None):
            return await self._call_repo.update_score(
                db,
                call_interaction_id,
                sentiment_score=None,
                quality_score=None,
                score_status=CallScoreStatus.unavailable.value,
            )

        raw_segments = _stub_transcribe_call(call)
        # De-identify BEFORE persistence (DR002) — no raw PII-bearing
        # transcript segment is ever written to the DB.
        de_identified_segments = [
            {**segment, "text": de_identify_transcript_text(segment["text"])}
            for segment in raw_segments
        ]
        await self._segment_repo.bulk_insert(db, call_interaction_id, de_identified_segments)

        sentiment_score, quality_score = _stub_score_transcript(de_identified_segments)
        return await self._call_repo.update_score(
            db,
            call_interaction_id,
            sentiment_score=sentiment_score,
            quality_score=quality_score,
            score_status=CallScoreStatus.scored.value,
        )
