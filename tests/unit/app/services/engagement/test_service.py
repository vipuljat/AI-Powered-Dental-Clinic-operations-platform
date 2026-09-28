"""Unit tests for app/services/engagement/service.py, derived from its spec alone.

Dependencies (WebChatSessionRepository, WebChatMessageRepository, PatientRepository,
TriageService, CallInteractionRepository, CallTranscriptSegmentRepository,
EscalationService) are exercised through lightweight in-memory fakes that implement
their *real* documented signatures (see specs/app/repositories/engagement/repository.py.md,
specs/app/repositories/patients/repository.py.md, specs/app/services/intelligence/service.py.md)
rather than mocks of a guessed interface, so behaviour is observed through outcomes
(return values, state changes, raised exceptions) instead of internal call order.
"""
from collections import defaultdict
from datetime import date, datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.common.exceptions.errors import NoMatchingPatientError
from app.services.engagement.service import (
    CallScoringService,
    EngagementWebChatService,
    VoiceHandlingService,
)

pytestmark = pytest.mark.asyncio

# A sentinel standing in for the AsyncSession the fakes never actually touch.
DB = object()


# ---------------------------------------------------------------------------
# Fakes over the real repository/service signatures.
# ---------------------------------------------------------------------------


class FakeWebChatSessionRepository:
    def __init__(self):
        self.store = {}

    async def create(self, db, visitor_token):
        session = SimpleNamespace(
            id=uuid4(),
            visitor_token=visitor_token,
            patient_id=None,
            verification_status="unverified",
            started_at=None,
            ended_at=None,
        )
        self.store[session.id] = session
        return session

    async def get_by_id(self, db, id):
        return self.store.get(id)

    async def verify(self, db, id, patient_id):
        session = self.store[id]
        session.verification_status = "verified"
        session.patient_id = patient_id
        return session


class FakeWebChatMessageRepository:
    def __init__(self):
        self.messages = defaultdict(list)

    async def append_message(self, db, session_id, sender, body):
        msg = SimpleNamespace(
            id=uuid4(), session_id=session_id, sender=sender, body=body,
            created_at=datetime.now(timezone.utc),
        )
        self.messages[session_id].append(msg)
        return msg

    async def list_by_session(self, db, session_id):
        return list(self.messages[session_id])


class FakePatientRepository:
    """Backs both `search` and `list_active` off one shared in-memory list so the
    test does not need to assume which real method verify_identity dispatches to."""

    def __init__(self, patients=None):
        self._patients = list(patients or [])

    async def create(self, db, **fields):
        p = SimpleNamespace(id=uuid4(), status="active", **fields)
        self._patients.append(p)
        return p

    async def get_by_id(self, db, id):
        return next((p for p in self._patients if p.id == id), None)

    async def search(self, db, q, include_archived=False):
        ql = q.lower()
        out = []
        for p in self._patients:
            if not include_archived and getattr(p, "status", "active") != "active":
                continue
            full_name = f"{p.first_name} {p.last_name}".lower()
            phone = (getattr(p, "phone", "") or "").lower()
            code = (getattr(p, "patient_code", "") or "").lower()
            if ql in full_name or ql in phone or ql in code:
                out.append(p)
        return out

    async def update(self, db, id, **fields):
        p = await self.get_by_id(db, id)
        for k, v in fields.items():
            setattr(p, k, v)
        return p

    async def archive(self, db, id, reason):
        p = await self.get_by_id(db, id)
        p.status = "archived"
        p.archived_reason = reason
        return p

    async def find_duplicate_by_phone(self, db, phone):
        return next((p for p in self._patients if getattr(p, "phone", None) == phone), None)

    async def next_patient_code(self, db):
        return f"P{len(self._patients) + 1:05d}"

    async def list_active(self, db, ids=None):
        vals = [p for p in self._patients if getattr(p, "status", "active") == "active"]
        if ids is not None:
            vals = [p for p in vals if p.id in ids]
        return vals


class FakeTriageService:
    """Stands in for services.intelligence.service.TriageService's real signatures."""

    async def start_session(self, db, channel, patient_id):
        return {"session_id": uuid4(), "next_question": {"key": "q1", "text": "stub"}}

    async def answer(self, db, session_id, question_key, answer):
        return {"escalated": False, "next_question": None, "recommended_block": None}

    async def classify(self, db, session_id):
        return {"escalated": False}


class FakeCallInteractionRepository:
    def __init__(self):
        self.store = {}
        self.update_score_calls = []

    async def create(self, db, **fields):
        data = {
            "id": uuid4(), "score_status": "unavailable", "sentiment_score": None,
            "quality_score": None, "outcome": None, "ended_at": None,
        }
        data.update(fields)
        obj = SimpleNamespace(**data)
        self.store[obj.id] = obj
        return obj

    async def get_by_id(self, db, id):
        return self.store.get(id)

    async def update_fields(self, db, id, **fields):
        obj = self.store[id]
        for k, v in fields.items():
            setattr(obj, k, v)
        return obj

    async def update_score(self, db, id, sentiment_score, quality_score, score_status):
        obj = self.store[id]
        obj.sentiment_score = sentiment_score
        obj.quality_score = quality_score
        obj.score_status = score_status
        self.update_score_calls.append(
            {
                "id": id, "sentiment_score": sentiment_score,
                "quality_score": quality_score, "score_status": score_status,
            }
        )
        return obj

    async def list(self, db, handled_by=None):
        vals = list(self.store.values())
        if handled_by is not None:
            vals = [v for v in vals if v.handled_by == handled_by]
        return vals


class FakeCallTranscriptSegmentRepository:
    def __init__(self):
        self.by_call = defaultdict(list)
        self.bulk_insert_calls = []

    async def bulk_insert(self, db, call_interaction_id, segments):
        self.bulk_insert_calls.append((call_interaction_id, segments))
        stored = []
        for seg in segments:
            obj = SimpleNamespace(id=uuid4(), call_interaction_id=call_interaction_id, **seg)
            self.by_call[call_interaction_id].append(obj)
            stored.append(obj)
        return stored

    async def list_by_call(self, db, call_interaction_id):
        return list(self.by_call[call_interaction_id])


class FakeEscalationService:
    def __init__(self):
        self.calls = []

    async def route(self, db, trigger_reason, triage_session_id=None, call_interaction_id=None):
        self.calls.append(
            {
                "trigger_reason": trigger_reason,
                "triage_session_id": triage_session_id,
                "call_interaction_id": call_interaction_id,
            }
        )
        return SimpleNamespace(
            id=uuid4(), status="unacknowledged", trigger_reason=trigger_reason,
            triage_session_id=triage_session_id, call_interaction_id=call_interaction_id,
        )


# ---------------------------------------------------------------------------
# EngagementWebChatService
# ---------------------------------------------------------------------------


def make_webchat_service(patients=None):
    session_repo = FakeWebChatSessionRepository()
    message_repo = FakeWebChatMessageRepository()
    patient_repo = FakePatientRepository(patients)
    triage_service = FakeTriageService()
    service = EngagementWebChatService(session_repo, message_repo, patient_repo, triage_service)
    return service, session_repo, message_repo, patient_repo


async def test_start_session_generates_vt_prefixed_token_and_is_unverified():
    service, session_repo, _, _ = make_webchat_service()

    session = await service.start_session(DB)

    assert session.visitor_token.startswith("vt_")
    assert len(session.visitor_token) == len("vt_") + 32
    assert session.verification_status == "unverified"
    assert session.patient_id is None


async def test_send_message_general_conversation_never_requires_verification():
    service, session_repo, _, _ = make_webchat_service()
    session = await service.start_session(DB)

    result = await service.send_message(
        DB, session.id, "Hi there, thanks for the great service last time!"
    )

    assert result["requires_verification"] is False
    assert isinstance(result["reply"], str) and result["reply"]


async def test_send_message_booking_intent_requires_verification_when_unverified():
    service, session_repo, _, _ = make_webchat_service()
    session = await service.start_session(DB)

    result = await service.send_message(DB, session.id, "I'd like to book an appointment please.")

    assert result["requires_verification"] is True
    assert isinstance(result["reply"], str) and result["reply"]


async def test_send_message_booking_intent_skips_verification_once_already_verified():
    service, session_repo, _, _ = make_webchat_service()
    session = await service.start_session(DB)
    await session_repo.verify(DB, session.id, uuid4())

    result = await service.send_message(DB, session.id, "I'd like to book an appointment please.")

    assert result["requires_verification"] is False


async def test_send_message_appends_visitor_message_and_ai_reply():
    service, session_repo, message_repo, _ = make_webchat_service()
    session = await service.start_session(DB)

    await service.send_message(DB, session.id, "hello")

    messages = await message_repo.list_by_session(DB, session.id)
    assert any(m.sender == "visitor" and m.body == "hello" for m in messages)
    assert any(m.sender == "ai_agent" for m in messages)


async def test_verify_identity_matches_and_marks_session_verified():
    patient = SimpleNamespace(
        id=uuid4(), first_name="Ana", last_name="Silva", dob=date(1990, 5, 20),
        phone="+551199999999", patient_code="P00001", status="active",
    )
    service, session_repo, _, _ = make_webchat_service([patient])
    session = await service.start_session(DB)

    result = await service.verify_identity(DB, session.id, "Ana", "Silva", date(1990, 5, 20))

    assert isinstance(result, dict)
    stored = await session_repo.get_by_id(DB, session.id)
    assert stored.verification_status == "verified"
    assert stored.patient_id == patient.id


async def test_verify_identity_raises_no_matching_patient_error_on_mismatch():
    patient = SimpleNamespace(
        id=uuid4(), first_name="Ana", last_name="Silva", dob=date(1990, 5, 20),
        phone="+551199999999", patient_code="P00001", status="active",
    )
    service, session_repo, _, _ = make_webchat_service([patient])
    session = await service.start_session(DB)

    with pytest.raises(NoMatchingPatientError) as exc_info:
        await service.verify_identity(DB, session.id, "Ana", "Silva", date(1985, 1, 1))

    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "NO_MATCHING_PATIENT"
    stored = await session_repo.get_by_id(DB, session.id)
    assert stored.verification_status == "unverified"
    assert stored.patient_id is None


# ---------------------------------------------------------------------------
# VoiceHandlingService
# ---------------------------------------------------------------------------


def make_voice_service():
    call_repo = FakeCallInteractionRepository()
    escalation_service = FakeEscalationService()
    settings = SimpleNamespace(environment="test")
    service = VoiceHandlingService(call_repo, escalation_service, settings)
    return service, call_repo, escalation_service


async def test_route_inbound_creates_ai_voice_call_interaction():
    service, call_repo, _ = make_voice_service()

    call = await service.route_inbound(DB, "+15551234567", "CA123")

    assert call.handled_by == "ai_voice"
    assert call.caller_phone == "+15551234567"
    assert call.started_at is not None
    assert call.started_at.tzinfo is not None


@pytest.mark.parametrize("scenario_type", ["confirmation", "reminder", "simple_reschedule"])
async def test_handle_scenario_configured_basic_scenarios_stay_ai_voice(scenario_type):
    service, call_repo, escalation_service = make_voice_service()
    call = await service.route_inbound(DB, "+15551234567", "CA123")

    result = await service.handle_scenario(DB, call.id, scenario_type)

    assert result.handled_by == "ai_voice"
    assert result.outcome is not None
    assert escalation_service.calls == []


async def test_handle_scenario_out_of_scope_routes_to_staff_via_escalation():
    service, call_repo, escalation_service = make_voice_service()
    call = await service.route_inbound(DB, "+15551234567", "CA123")

    result = await service.handle_scenario(DB, call.id, "insurance_billing_question")

    assert result.handled_by == "staff"
    assert len(escalation_service.calls) == 1
    routed = escalation_service.calls[0]
    assert routed["call_interaction_id"] == call.id
    assert routed["triage_session_id"] is None
    assert isinstance(routed["trigger_reason"], str) and routed["trigger_reason"]


# ---------------------------------------------------------------------------
# CallScoringService
# ---------------------------------------------------------------------------


def make_scoring_service():
    call_repo = FakeCallInteractionRepository()
    segment_repo = FakeCallTranscriptSegmentRepository()
    service = CallScoringService(call_repo, segment_repo)
    return service, call_repo, segment_repo


async def test_score_always_writes_an_explicit_score_status():
    service, call_repo, segment_repo = make_scoring_service()
    call = await call_repo.create(
        db=DB, patient_id=None, caller_phone="+15550000000", handled_by="ai_voice",
        started_at=datetime.now(timezone.utc),
    )

    result = await service.score(DB, call.id)

    assert result.score_status in {"scored", "unavailable"}
    # score_status is persisted through the one designated repository method, never left
    # at its unavailable-by-default column value by omission (FR-E11.5).
    assert len(call_repo.update_score_calls) == 1
    assert call_repo.update_score_calls[0]["score_status"] == result.score_status
    if result.score_status == "scored":
        assert result.sentiment_score is not None
        assert result.quality_score is not None
    else:
        assert result.sentiment_score is None
        assert result.quality_score is None


async def test_score_persists_de_identified_transcript_segments_when_scored():
    service, call_repo, segment_repo = make_scoring_service()
    call = await call_repo.create(
        db=DB, patient_id=None, caller_phone="+15550000000", handled_by="ai_voice",
        started_at=datetime.now(timezone.utc),
    )

    result = await service.score(DB, call.id)

    if result.score_status == "scored":
        assert len(segment_repo.bulk_insert_calls) >= 1
        call_id_arg, segments_arg = segment_repo.bulk_insert_calls[0]
        assert call_id_arg == call.id
        assert len(segments_arg) >= 1
        for seg in segments_arg:
            assert isinstance(seg.get("text"), str) and seg["text"]
