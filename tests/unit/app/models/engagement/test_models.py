"""Unit tests for app/models/engagement/models.py.

Exercises the four engagement ORM tables (webchat_sessions, webchat_messages,
call_interactions, call_transcript_segments) directly against an in-memory
SQLite database, mirroring the ENVIRONMENT=test substitution described in
project_rules.testing (app/core/database.py switches to
sqlite+aiosqlite:///:memory: so the same ORM models run unmodified against
both Postgres and SQLite).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# The real composition roots set this before ever importing app.core.database;
# do the same here defensively so importing it never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from sqlalchemy import Column, Table  # noqa: E402

from app.core.database import Base  # noqa: E402
from app.models.engagement.models import (  # noqa: E402
    CallInteraction,
    CallTranscriptSegment,
    WebchatMessage,
    WebchatSession,
)

# webchat_sessions.patient_id and call_interactions.patient_id declare a FK
# to patients.id (Data shapes), a table owned by another module's own
# models.py that this task has no spec for and must not import for its own
# sake. If the patients module happens to already be importable (and so
# already registered `patients` on the shared Base.metadata) reuse it as-is;
# otherwise register a minimal stand-in table with just the `id` primary key
# FK resolution needs, purely so this in-memory engine can create *this*
# spec's own four tables.
try:  # pragma: no cover - exercised implicitly by the fixtures below
    import app.models.patients.models  # noqa: F401
except ModuleNotFoundError:
    pass

if "patients" not in Base.metadata.tables:
    Table(
        "patients",
        Base.metadata,
        Column("id", WebchatSession.__table__.c.id.type, primary_key=True),
    )

# The four engagement tables this spec owns.
_ENGAGEMENT_TABLES = [
    WebchatSession.__table__,
    WebchatMessage.__table__,
    CallInteraction.__table__,
    CallTranscriptSegment.__table__,
]

pytestmark = pytest.mark.asyncio

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with eng.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=_ENGAGEMENT_TABLES)
        )
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


def make_webchat_session(**overrides):
    defaults = dict(visitor_token=f"visitor-{uuid.uuid4()}")
    defaults.update(overrides)
    return WebchatSession(**defaults)


def make_call_interaction(**overrides):
    defaults = dict(
        caller_phone="+15551234567",
        handled_by="ai_voice",
        started_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return CallInteraction(**defaults)


# --------------------------------------------------------------------------
# table identity / mapper-level facts
# --------------------------------------------------------------------------


def test_table_names():
    assert WebchatSession.__tablename__ == "webchat_sessions"
    assert WebchatMessage.__tablename__ == "webchat_messages"
    assert CallInteraction.__tablename__ == "call_interactions"
    assert CallTranscriptSegment.__tablename__ == "call_transcript_segments"


@pytest.mark.parametrize(
    "model",
    [WebchatSession, WebchatMessage, CallInteraction, CallTranscriptSegment],
)
def test_no_cross_module_orm_relationship_declared(model):
    # Interaction contract: patient_id / call_interaction_id are plain FK
    # columns, no cross-module relationship() -- repositories/engagement own
    # traversal, not the ORM classes themselves.
    mapper = inspect(model)
    assert list(mapper.relationships) == []


@pytest.mark.parametrize(
    "model",
    [WebchatSession, WebchatMessage, CallInteraction, CallTranscriptSegment],
)
def test_id_is_the_primary_key(model):
    mapper = inspect(model)
    assert [c.name for c in mapper.primary_key] == ["id"]


# --------------------------------------------------------------------------
# webchat_sessions
# --------------------------------------------------------------------------


async def test_webchat_session_patient_id_is_nullable_and_defaults_to_none(session):
    # FR-E11.2 / US51 AC: anonymous until verified -- no patient link exists
    # by default.
    ws = make_webchat_session()
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    assert ws.patient_id is None


async def test_webchat_session_verification_status_defaults_to_unverified(session):
    ws = make_webchat_session()
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    assert ws.verification_status == "unverified"


async def test_webchat_session_patient_id_settable_once_verified(session):
    # FR-E11.2: patient_id is set only on verification -- once
    # verification_status flips to "verified" the row can legitimately carry
    # a patient_id.
    patient_id = uuid.uuid4()
    ws = make_webchat_session(patient_id=patient_id, verification_status="verified")
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    assert ws.patient_id == patient_id
    assert ws.verification_status == "verified"


async def test_webchat_session_visitor_token_uniqueness_enforced(session):
    session.add(make_webchat_session(visitor_token="dupe-token"))
    await session.commit()

    session.add(make_webchat_session(visitor_token="dupe-token"))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_webchat_session_visitor_token_is_not_nullable(session):
    session.add(WebchatSession(visitor_token=None))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_webchat_session_started_and_ended_at_are_nullable(session):
    ws = make_webchat_session()
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    assert ws.started_at is None
    assert ws.ended_at is None


async def test_webchat_session_started_and_ended_at_round_trip(session):
    started = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
    ended = started + timedelta(minutes=6)
    ws = make_webchat_session(started_at=started, ended_at=ended)
    session.add(ws)
    await session.commit()
    await session.refresh(ws)
    # Compared naive-vs-naive: the SQLite test engine's DATETIME storage does
    # not itself round-trip a tzinfo offset (a driver-level limitation of the
    # substitution, not something this spec asserts) -- the wall-clock value
    # is what must match.
    assert ws.started_at.replace(tzinfo=None) == started.replace(tzinfo=None)
    assert ws.ended_at.replace(tzinfo=None) == ended.replace(tzinfo=None)


async def test_webchat_session_id_is_a_generated_uuid(session):
    ws = make_webchat_session()
    session.add(ws)
    await session.flush()
    assert isinstance(ws.id, uuid.UUID)


# --------------------------------------------------------------------------
# webchat_messages
# --------------------------------------------------------------------------


async def test_webchat_message_round_trips_visitor_and_ai_agent_senders(session):
    ws = make_webchat_session()
    session.add(ws)
    await session.commit()

    created_at = datetime.now(timezone.utc)
    visitor_msg = WebchatMessage(
        session_id=ws.id, sender="visitor", body="Hi, do you have an opening?", created_at=created_at
    )
    ai_msg = WebchatMessage(
        session_id=ws.id, sender="ai_agent", body="Yes, tomorrow at 10am.", created_at=created_at
    )
    session.add_all([visitor_msg, ai_msg])
    await session.commit()
    await session.refresh(visitor_msg)
    await session.refresh(ai_msg)

    assert visitor_msg.sender == "visitor"
    assert visitor_msg.body == "Hi, do you have an opening?"
    assert ai_msg.sender == "ai_agent"


async def test_webchat_message_session_id_is_not_nullable(session):
    session.add(
        WebchatMessage(
            session_id=None, sender="visitor", body="hello", created_at=datetime.now(timezone.utc)
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_webchat_message_body_is_not_nullable(session):
    ws = make_webchat_session()
    session.add(ws)
    await session.commit()

    session.add(
        WebchatMessage(session_id=ws.id, sender="visitor", body=None, created_at=datetime.now(timezone.utc))
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_webchat_message_created_at_is_not_nullable(session):
    ws = make_webchat_session()
    session.add(ws)
    await session.commit()

    session.add(WebchatMessage(session_id=ws.id, sender="visitor", body="hello", created_at=None))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


# --------------------------------------------------------------------------
# call_interactions
# --------------------------------------------------------------------------


async def test_call_interaction_patient_id_is_nullable(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()
    await session.refresh(ci)
    assert ci.patient_id is None


async def test_call_interaction_score_status_defaults_to_unavailable(session):
    # sentiment_score/quality_score are null when the score is unavailable.
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()
    await session.refresh(ci)
    assert ci.score_status == "unavailable"
    assert ci.sentiment_score is None
    assert ci.quality_score is None


async def test_call_interaction_scored_status_round_trips_decimal_scores(session):
    ci = make_call_interaction(
        score_status="scored",
        sentiment_score=Decimal("0.825"),
        quality_score=Decimal("0.910"),
    )
    session.add(ci)
    await session.commit()
    await session.refresh(ci)
    assert ci.score_status == "scored"
    assert ci.sentiment_score == Decimal("0.825")
    assert ci.quality_score == Decimal("0.910")


async def test_call_interaction_handled_by_round_trips_ai_voice_and_staff(session):
    ai_call = make_call_interaction(handled_by="ai_voice")
    staff_call = make_call_interaction(handled_by="staff")
    session.add_all([ai_call, staff_call])
    await session.commit()
    await session.refresh(ai_call)
    await session.refresh(staff_call)
    assert ai_call.handled_by == "ai_voice"
    assert staff_call.handled_by == "staff"


async def test_call_interaction_optional_fields_are_nullable(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()
    await session.refresh(ci)
    assert ci.scenario_type is None
    assert ci.outcome is None
    assert ci.transcript_uri is None
    assert ci.recording_uri is None
    assert ci.ended_at is None


async def test_call_interaction_transcript_and_recording_uri_round_trip_as_strings(session):
    ci = make_call_interaction(
        transcript_uri="s3://clinic-transcripts/call-1.json",
        recording_uri="s3://clinic-recordings/call-1.wav",
    )
    session.add(ci)
    await session.commit()
    await session.refresh(ci)
    assert ci.transcript_uri == "s3://clinic-transcripts/call-1.json"
    assert ci.recording_uri == "s3://clinic-recordings/call-1.wav"


async def test_call_interaction_caller_phone_is_not_nullable(session):
    session.add(
        CallInteraction(
            caller_phone=None, handled_by="ai_voice", started_at=datetime.now(timezone.utc)
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_call_interaction_handled_by_is_not_nullable(session):
    session.add(
        CallInteraction(caller_phone="+15551234567", handled_by=None, started_at=datetime.now(timezone.utc))
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_call_interaction_started_at_is_not_nullable(session):
    session.add(CallInteraction(caller_phone="+15551234567", handled_by="staff", started_at=None))
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


# --------------------------------------------------------------------------
# call_transcript_segments
# --------------------------------------------------------------------------


async def test_call_transcript_segment_round_trips_speaker_text_and_timing(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()

    seg = CallTranscriptSegment(
        call_interaction_id=ci.id,
        speaker="patient",
        text="I'd like to reschedule my appointment.",
        start_ms=0,
        end_ms=2500,
    )
    session.add(seg)
    await session.commit()
    await session.refresh(seg)

    assert seg.speaker == "patient"
    assert seg.text == "I'd like to reschedule my appointment."
    assert seg.start_ms == 0
    assert seg.end_ms == 2500


async def test_call_transcript_segment_speaker_round_trips_ai_agent(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()

    seg = CallTranscriptSegment(
        call_interaction_id=ci.id,
        speaker="ai_agent",
        text="Sure, let me check our next available slot.",
        start_ms=2500,
        end_ms=5200,
    )
    session.add(seg)
    await session.commit()
    await session.refresh(seg)
    assert seg.speaker == "ai_agent"


async def test_call_transcript_segment_embedding_is_nullable(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()

    seg = CallTranscriptSegment(
        call_interaction_id=ci.id, speaker="patient", text="hello", start_ms=0, end_ms=500
    )
    session.add(seg)
    await session.commit()
    await session.refresh(seg)
    assert seg.embedding is None


async def test_call_transcript_segment_embedding_round_trips_via_sqlite_json_fallback(session, session_factory):
    # VectorType (app/core/db_types.py) renders a JSON-encoded fallback on
    # SQLite -- the value must survive a real round trip through that
    # TypeDecorator as a plain Python list of floats, not a native pgvector
    # object (which the SQLite test engine never constructs).
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()

    embedding = [0.1, -0.2, 0.3]
    seg = CallTranscriptSegment(
        call_interaction_id=ci.id,
        speaker="patient",
        text="hello",
        start_ms=0,
        end_ms=500,
        embedding=embedding,
    )
    session.add(seg)
    await session.commit()
    seg_id = seg.id

    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(CallTranscriptSegment, seg_id)
        assert reloaded.embedding == embedding


async def test_call_transcript_segment_call_interaction_id_is_not_nullable(session):
    session.add(
        CallTranscriptSegment(call_interaction_id=None, speaker="patient", text="hello", start_ms=0, end_ms=10)
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_call_transcript_segment_text_is_not_nullable(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()

    session.add(
        CallTranscriptSegment(call_interaction_id=ci.id, speaker="patient", text=None, start_ms=0, end_ms=10)
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_call_transcript_segment_start_ms_is_not_nullable(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()
    call_interaction_id = ci.id

    session.add(
        CallTranscriptSegment(
            call_interaction_id=call_interaction_id, speaker="patient", text="hello", start_ms=None, end_ms=10
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()


async def test_call_transcript_segment_end_ms_is_not_nullable(session):
    ci = make_call_interaction()
    session.add(ci)
    await session.commit()
    call_interaction_id = ci.id

    session.add(
        CallTranscriptSegment(
            call_interaction_id=call_interaction_id, speaker="patient", text="hello", start_ms=0, end_ms=None
        )
    )
    with pytest.raises(IntegrityError):
        await session.commit()
    await session.rollback()
