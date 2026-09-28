"""SQLAlchemy ORM declarations for the engagement module (architecture.md §4.2).

Owns exactly four tables: ``webchat_sessions``, ``webchat_messages``,
``call_interactions``, ``call_transcript_segments``.

Interaction contract: ``patient_id``/``call_interaction_id`` are plain FK
columns, no cross-module ORM ``relationship()`` — ``repositories/engagement/
repository.py`` is the only file that imports these classes; ``services/
engagement/service.py`` and ``services/intelligence/service.py``'s
``EscalationService`` (which references ``call_interactions.id`` via the
polymorphic ``escalations.call_interaction_id``) both go through that
repository or their own, never these ORM classes directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Numeric, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import (
    CallHandledBy,
    CallScoreStatus,
    CallSpeaker,
    WebchatSender,
    WebchatVerificationStatus,
)
from app.core.database import Base
from app.core.db_types import VectorType


class WebchatSession(Base):
    """One anonymous-until-verified webchat conversation.

    FR-E11.2: ``patient_id`` stays ``NULL`` — no PII-bearing patient record is
    created or modified via chat — until ``verification_status`` flips to
    ``"verified"`` (US51 AC); until then the session is only addressable by
    its anonymous ``visitor_token``.
    """

    __tablename__ = "webchat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    visitor_token: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    verification_status: Mapped[WebchatVerificationStatus] = mapped_column(
        Enum(WebchatVerificationStatus, native_enum=False, length=20),
        nullable=False,
        default=WebchatVerificationStatus.unverified,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class WebchatMessage(Base):
    """One message (visitor- or AI-authored) within a ``WebchatSession``."""

    __tablename__ = "webchat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("webchat_sessions.id"), nullable=False
    )
    sender: Mapped[WebchatSender] = mapped_column(
        Enum(WebchatSender, native_enum=False, length=20), nullable=False
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class CallInteraction(Base):
    """One AI-voice or staff-handled phone call.

    ``transcript_uri``/``recording_uri`` point at object storage — the audio
    and full transcript never live in the DB row itself (DR002: the full
    transcript is de-identified before any ML use); only a cached excerpt
    lives alongside this row via the JSONB-backed schema layer.
    """

    __tablename__ = "call_interactions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("patients.id"), nullable=True
    )
    caller_phone: Mapped[str] = mapped_column(String(30), nullable=False)
    scenario_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    handled_by: Mapped[CallHandledBy] = mapped_column(
        Enum(CallHandledBy, native_enum=False, length=20), nullable=False
    )
    outcome: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sentiment_score: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 3), nullable=True
    )
    quality_score: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 3), nullable=True
    )
    score_status: Mapped[CallScoreStatus] = mapped_column(
        Enum(CallScoreStatus, native_enum=False, length=20),
        nullable=False,
        default=CallScoreStatus.unavailable,
    )
    transcript_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    recording_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class CallTranscriptSegment(Base):
    """One speaker-attributed segment of a call transcript.

    ``embedding`` uses ``VectorType`` (app/core/db_types.py) — pgvector
    ``VECTOR(1536)`` on Postgres for semantic retrieval feeding
    escalation-pattern matching, a JSON-encoded fallback on SQLite; any
    similarity-search behaviour built on it is a documented no-op under the
    SQLite test engine (open_questions[Q-dad9c2c8]).
    """

    __tablename__ = "call_transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    call_interaction_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("call_interactions.id"), nullable=False
    )
    speaker: Mapped[CallSpeaker] = mapped_column(
        Enum(CallSpeaker, native_enum=False, length=20), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        VectorType(dimensions=1536), nullable=True
    )
