"""Unit tests for app/repositories/engagement/repository.py.

Exercises the four data-access classes (WebChatSessionRepository,
WebChatMessageRepository, CallInteractionRepository,
CallTranscriptSegmentRepository) against an in-memory SQLite database,
mirroring the ENVIRONMENT=test substitution described in
project_rules.testing.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# The real composition roots set this before ever importing app.core.database;
# do the same here defensively so importing it never tries to dial a real,
# unreachable Postgres server using the class default `database_url`.
os.environ.setdefault("ENVIRONMENT", "test")

from sqlalchemy import Column, Table  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.core.database import Base  # noqa: E402
from app.models.engagement.models import (  # noqa: E402
    CallInteraction,
    CallTranscriptSegment,
    WebchatMessage,
    WebchatSession,
)
from app.repositories.engagement.repository import (  # noqa: E402
    CallInteractionRepository,
    CallTranscriptSegmentRepository,
    WebChatMessageRepository,
    WebChatSessionRepository,
)

# webchat_sessions.patient_id and call_interactions.patient_id declare a FK
# to patients.id (see app/models/engagement/models.py.md Data shapes), a
# table owned by another module's own models.py that this task has no spec
# for and must not import for its own sake. If the patients module happens
# to already be importable (and so already registered `patients` on the
# shared Base.metadata) reuse it as-is; otherwise register a minimal
# stand-in table with just the `id` primary key FK resolution needs, purely
# so this in-memory engine can create *this* spec's own four tables.
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
async def db(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


@pytest.fixture
def session_repo():
    return WebChatSessionRepository()


@pytest.fixture
def message_repo():
    return WebChatMessageRepository()


@pytest.fixture
def call_repo():
    return CallInteractionRepository()


@pytest.fixture
def segment_repo():
    return CallTranscriptSegmentRepository()


def make_call_fields(**overrides):
    defaults = dict(
        caller_phone="+15551234567",
        handled_by="ai_voice",
        started_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return defaults


# --------------------------------------------------------------------------
# WebChatSessionRepository
# --------------------------------------------------------------------------


async def test_create_persists_session_with_visitor_token_and_unverified_defaults(db, session_repo):
    ws = await session_repo.create(db, visitor_token="visitor-abc")
    assert isinstance(ws, WebchatSession)
    assert ws.visitor_token == "visitor-abc"
    # FR-E11.2: create() has no patient_id/verification-status parameter --
    # the row starts anonymous, no PII-bearing link exists yet.
    assert ws.patient_id is None
    assert ws.verification_status == "unverified"


async def test_get_by_id_returns_created_session(db, session_repo):
    created = await session_repo.create(db, visitor_token="visitor-xyz")
    fetched = await session_repo.get_by_id(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.visitor_token == "visitor-xyz"


async def test_get_by_id_returns_none_for_unknown_id(db, session_repo):
    assert await session_repo.get_by_id(db, uuid.uuid4()) is None


async def test_verify_sets_patient_id_and_verification_status(db, session_repo):
    created = await session_repo.create(db, visitor_token="visitor-verify")
    patient_id = uuid.uuid4()

    verified = await session_repo.verify(db, created.id, patient_id)

    assert verified.patient_id == patient_id
    assert verified.verification_status == "verified"

    # Persisted, not just returned in-memory on the same object.
    reloaded = await session_repo.get_by_id(db, created.id)
    assert reloaded.patient_id == patient_id
    assert reloaded.verification_status == "verified"


async def test_session_stays_unverified_until_verify_is_called(db, session_repo):
    # FR-E11.2: verify() is the only method anywhere that sets patient_id /
    # verification_status="verified" -- create() plus repeated reads must
    # never move a session out of "unverified" on their own.
    created = await session_repo.create(db, visitor_token="visitor-untouched")
    await session_repo.get_by_id(db, created.id)

    still_unverified = await session_repo.get_by_id(db, created.id)
    assert still_unverified.patient_id is None
    assert still_unverified.verification_status == "unverified"


# --------------------------------------------------------------------------
# WebChatMessageRepository
# --------------------------------------------------------------------------


async def test_append_message_persists_sender_and_body(db, session_repo, message_repo):
    ws = await session_repo.create(db, visitor_token="visitor-msg")

    msg = await message_repo.append_message(db, ws.id, "visitor", "Do you have an opening today?")

    assert isinstance(msg, WebchatMessage)
    assert msg.session_id == ws.id
    assert msg.sender == "visitor"
    assert msg.body == "Do you have an opening today?"


async def test_append_message_persists_ai_agent_sender(db, session_repo, message_repo):
    ws = await session_repo.create(db, visitor_token="visitor-msg-2")

    msg = await message_repo.append_message(db, ws.id, "ai_agent", "Yes, tomorrow at 10am works.")

    assert msg.sender == "ai_agent"
    assert msg.body == "Yes, tomorrow at 10am works."


async def test_list_by_session_returns_only_that_sessions_messages(db, session_repo, message_repo):
    ws_a = await session_repo.create(db, visitor_token="visitor-a")
    ws_b = await session_repo.create(db, visitor_token="visitor-b")

    await message_repo.append_message(db, ws_a.id, "visitor", "hello from a")
    await message_repo.append_message(db, ws_a.id, "ai_agent", "hi a, how can I help?")
    await message_repo.append_message(db, ws_b.id, "visitor", "hello from b")

    a_messages = await message_repo.list_by_session(db, ws_a.id)
    b_messages = await message_repo.list_by_session(db, ws_b.id)

    assert {m.body for m in a_messages} == {"hello from a", "hi a, how can I help?"}
    assert {m.body for m in b_messages} == {"hello from b"}


async def test_list_by_session_returns_empty_list_for_session_with_no_messages(db, session_repo, message_repo):
    ws = await session_repo.create(db, visitor_token="visitor-empty")
    assert await message_repo.list_by_session(db, ws.id) == []


# --------------------------------------------------------------------------
# CallInteractionRepository
# --------------------------------------------------------------------------


async def test_create_persists_call_interaction_fields(db, call_repo):
    ci = await call_repo.create(db, **make_call_fields(caller_phone="+15559876543", handled_by="staff"))
    assert isinstance(ci, CallInteraction)
    assert ci.caller_phone == "+15559876543"
    assert ci.handled_by == "staff"


async def test_get_by_id_returns_none_for_unknown_call(db, call_repo):
    assert await call_repo.get_by_id(db, uuid.uuid4()) is None


async def test_get_by_id_returns_created_call(db, call_repo):
    ci = await call_repo.create(db, **make_call_fields())
    fetched = await call_repo.get_by_id(db, ci.id)
    assert fetched is not None
    assert fetched.id == ci.id


async def test_update_fields_persists_given_fields(db, call_repo):
    ci = await call_repo.create(db, **make_call_fields())
    ended = datetime.now(timezone.utc)

    updated = await call_repo.update_fields(db, ci.id, outcome="rescheduled", ended_at=ended)
    assert updated.outcome == "rescheduled"

    reloaded = await call_repo.get_by_id(db, ci.id)
    assert reloaded.outcome == "rescheduled"


async def test_update_score_persists_explicit_scored_status(db, call_repo):
    ci = await call_repo.create(db, **make_call_fields())

    updated = await call_repo.update_score(
        db, ci.id, sentiment_score=0.75, quality_score=0.9, score_status="scored"
    )

    assert updated.score_status == "scored"
    assert float(updated.sentiment_score) == pytest.approx(0.75)
    assert float(updated.quality_score) == pytest.approx(0.9)


async def test_update_score_persists_explicit_unavailable_status_even_with_scores_present(db, call_repo):
    # FR-E11.5/US53: score_status is never inferred from whether the scores
    # are non-null -- CallScoringService.score decides, this repository
    # just persists exactly what it was told, even when that looks
    # contradictory (non-null scores alongside "unavailable").
    ci = await call_repo.create(db, **make_call_fields())

    updated = await call_repo.update_score(
        db, ci.id, sentiment_score=0.6, quality_score=0.5, score_status="unavailable"
    )

    assert updated.score_status == "unavailable"


async def test_update_score_persists_explicit_scored_status_even_with_null_scores(db, call_repo):
    # Same guarantee in the other direction: a "scored" status is stored as
    # given even though both scores passed in are None.
    ci = await call_repo.create(db, **make_call_fields())

    updated = await call_repo.update_score(
        db, ci.id, sentiment_score=None, quality_score=None, score_status="scored"
    )

    assert updated.score_status == "scored"
    assert updated.sentiment_score is None
    assert updated.quality_score is None


async def test_list_without_filter_returns_all_call_interactions(db, call_repo):
    await call_repo.create(db, **make_call_fields(handled_by="ai_voice"))
    await call_repo.create(db, **make_call_fields(handled_by="staff"))

    all_calls = await call_repo.list(db, handled_by=None)
    assert len(all_calls) == 2


async def test_list_filters_by_handled_by(db, call_repo):
    await call_repo.create(db, **make_call_fields(handled_by="ai_voice"))
    await call_repo.create(db, **make_call_fields(handled_by="staff"))

    staff_only = await call_repo.list(db, handled_by="staff")
    assert len(staff_only) == 1
    assert staff_only[0].handled_by == "staff"


# --------------------------------------------------------------------------
# CallTranscriptSegmentRepository
# --------------------------------------------------------------------------


async def test_bulk_insert_persists_all_segments_linked_to_call(db, call_repo, segment_repo):
    ci = await call_repo.create(db, **make_call_fields())
    segments = [
        {"speaker": "patient", "text": "I'd like to reschedule.", "start_ms": 0, "end_ms": 1200},
        {"speaker": "ai_agent", "text": "Sure, let me check.", "start_ms": 1200, "end_ms": 2400},
    ]

    inserted = await segment_repo.bulk_insert(db, ci.id, segments)

    assert len(inserted) == 2
    assert isinstance(inserted[0], CallTranscriptSegment)
    assert {s.call_interaction_id for s in inserted} == {ci.id}
    assert {s.text for s in inserted} == {"I'd like to reschedule.", "Sure, let me check."}


async def test_list_by_call_returns_only_that_calls_segments(db, call_repo, segment_repo):
    ci_a = await call_repo.create(db, **make_call_fields())
    ci_b = await call_repo.create(db, **make_call_fields())

    await segment_repo.bulk_insert(
        db, ci_a.id, [{"speaker": "patient", "text": "call a segment", "start_ms": 0, "end_ms": 500}]
    )
    await segment_repo.bulk_insert(
        db, ci_b.id, [{"speaker": "patient", "text": "call b segment", "start_ms": 0, "end_ms": 500}]
    )

    a_segments = await segment_repo.list_by_call(db, ci_a.id)
    b_segments = await segment_repo.list_by_call(db, ci_b.id)

    assert [s.text for s in a_segments] == ["call a segment"]
    assert [s.text for s in b_segments] == ["call b segment"]


async def test_list_by_call_returns_empty_list_for_call_with_no_segments(db, call_repo, segment_repo):
    ci = await call_repo.create(db, **make_call_fields())
    assert await segment_repo.list_by_call(db, ci.id) == []
