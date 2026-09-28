"""Data-access classes over the engagement module's four tables.

Owns ``webchat_sessions``, ``webchat_messages``, ``call_interactions`` and
``call_transcript_segments`` (architecture.md §4.2). Services never touch the
DB directly, only via these repository classes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import WebchatVerificationStatus
from app.models.engagement.models import (
    CallInteraction,
    CallTranscriptSegment,
    WebchatMessage,
    WebchatSession,
)


class WebChatSessionRepository:
    """Data access for ``webchat_sessions``."""

    async def create(self, db: AsyncSession, visitor_token: str) -> WebchatSession:
        session = WebchatSession(
            visitor_token=visitor_token,
            started_at=datetime.now(timezone.utc),
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        return session

    async def get_by_id(
        self, db: AsyncSession, id: uuid.UUID
    ) -> WebchatSession | None:
        result = await db.execute(
            select(WebchatSession).where(WebchatSession.id == id)
        )
        return result.scalar_one_or_none()

    async def verify(
        self, db: AsyncSession, id: uuid.UUID, patient_id: uuid.UUID
    ) -> WebchatSession:
        """Flip a session to verified, attaching the confirmed patient.

        FR-E11.2: this is the only method anywhere that sets
        ``webchat_sessions.patient_id``/``verification_status="verified"`` —
        no PII-bearing patient record is created or modified until identity
        is verified (US51 AC). No other write path in this repository
        touches those two fields.
        """
        session = await self.get_by_id(db, id)
        if session is None:
            raise ValueError(f"WebchatSession {id} not found")
        session.patient_id = patient_id
        session.verification_status = WebchatVerificationStatus.verified
        await db.commit()
        await db.refresh(session)
        return session


class WebChatMessageRepository:
    """Data access for ``webchat_messages``."""

    async def append_message(
        self, db: AsyncSession, session_id: uuid.UUID, sender: str, body: str
    ) -> WebchatMessage:
        message = WebchatMessage(
            session_id=session_id,
            sender=sender,
            body=body,
            created_at=datetime.now(timezone.utc),
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)
        return message

    async def list_by_session(
        self, db: AsyncSession, session_id: uuid.UUID
    ) -> list[WebchatMessage]:
        result = await db.execute(
            select(WebchatMessage)
            .where(WebchatMessage.session_id == session_id)
            .order_by(WebchatMessage.created_at)
        )
        return list(result.scalars().all())


class CallInteractionRepository:
    """Data access for ``call_interactions``."""

    async def create(self, db: AsyncSession, **fields) -> CallInteraction:
        call = CallInteraction(**fields)
        db.add(call)
        await db.commit()
        await db.refresh(call)
        return call

    async def get_by_id(
        self, db: AsyncSession, id: uuid.UUID
    ) -> CallInteraction | None:
        result = await db.execute(
            select(CallInteraction).where(CallInteraction.id == id)
        )
        return result.scalar_one_or_none()

    async def update_fields(
        self, db: AsyncSession, id: uuid.UUID, **fields
    ) -> CallInteraction:
        """Apply an arbitrary field patch to a call interaction row.

        Interaction contract: this is the method a console-style override
        flow over voice calls would use — not currently wired into
        ``services/console/service.py``'s ``OverrideService`` (its fixed set
        of four target repositories is appointment/outreach_message/
        waitlist_entry/triage_session; US5's examples are scheduling/
        communication/triage, not voice calls). Exposed here for
        completeness/future use only, in this Phase-1 scope (open question).
        """
        call = await self.get_by_id(db, id)
        if call is None:
            raise ValueError(f"CallInteraction {id} not found")
        for key, value in fields.items():
            setattr(call, key, value)
        await db.commit()
        await db.refresh(call)
        return call

    async def update_score(
        self,
        db: AsyncSession,
        id: uuid.UUID,
        sentiment_score: float | None,
        quality_score: float | None,
        score_status: str,
    ) -> CallInteraction:
        """Persist sentiment/quality scores plus their status.

        FR-E11.5/US53: ``score_status`` is always an explicit argument
        supplied by the caller (``CallScoringService.score``), which decides
        ``"scored"`` vs. ``"unavailable"`` — this repository never infers it,
        only persists whichever was decided.
        """
        call = await self.get_by_id(db, id)
        if call is None:
            raise ValueError(f"CallInteraction {id} not found")
        call.sentiment_score = sentiment_score
        call.quality_score = quality_score
        call.score_status = score_status
        await db.commit()
        await db.refresh(call)
        return call

    async def list(
        self, db: AsyncSession, handled_by: str | None
    ) -> list[CallInteraction]:
        stmt = select(CallInteraction)
        if handled_by is not None:
            stmt = stmt.where(CallInteraction.handled_by == handled_by)
        stmt = stmt.order_by(CallInteraction.started_at.desc())
        result = await db.execute(stmt)
        return list(result.scalars().all())


class CallTranscriptSegmentRepository:
    """Data access for ``call_transcript_segments``."""

    async def bulk_insert(
        self,
        db: AsyncSession,
        call_interaction_id: uuid.UUID,
        segments: list[dict],
    ) -> list[CallTranscriptSegment]:
        rows = [
            CallTranscriptSegment(call_interaction_id=call_interaction_id, **segment)
            for segment in segments
        ]
        db.add_all(rows)
        await db.commit()
        for row in rows:
            await db.refresh(row)
        return rows

    async def list_by_call(
        self, db: AsyncSession, call_interaction_id: uuid.UUID
    ) -> list[CallTranscriptSegment]:
        result = await db.execute(
            select(CallTranscriptSegment)
            .where(CallTranscriptSegment.call_interaction_id == call_interaction_id)
            .order_by(CallTranscriptSegment.start_ms)
        )
        return list(result.scalars().all())
