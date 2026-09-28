"""Consumer for the `escalation.triggered` domain event (architecture.md §5.1,
project_rules.other's event contract).

Responsibility: the `escalations` row itself is already inserted
synchronously by `EscalationService.route`
(app/services/intelligence/service.py) **before** the event is published
(Flow-4 diagram fact) — this worker's job is the notification/audit side:
logging the dispatch at an elevated severity for operational visibility and
recording that the fan-out happened, never a second write to `escalations`
and never a per-staff routing decision (open_questions[Q-5b4034f4]: whoever
acknowledges the unacknowledged queue first claims it).

Interaction contract: consumes `escalation.triggered`
`{triage_session_id_or_call_interaction_id, trigger_reason}`, published by
`EscalationService.route` — restated identically in that file's Interaction
Contract section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from app.common.enums import ActorType
from app.core.logging import get_logger
from app.repositories.console.repository import AuditLogRepository
from app.repositories.intelligence.repository import EscalationRepository
from app.services.console.service import AuditService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.core.messaging import MessageBroker

logger = get_logger(__name__)

# The queue name this worker binds `escalation.triggered` under
# (project_rules.layout: app/workers/*.py, one of the 5 named consumers).
_QUEUE_NAME = "escalation_dispatcher.escalation_triggered"
_ROUTING_KEY = "escalation.triggered"

# Set once by `run()` before the broker's consume loop starts, so
# `handle_escalation_triggered` — whose signature is fixed to exactly
# `(payload, headers)` by the `MessageHandler` protocol every broker.consume
# call is built against (app/core/messaging.py) — can still reach the
# session factory `run()` was given, without smuggling it through the
# handler's own parameter list.
_session_factory: "async_sessionmaker[AsyncSession] | None" = None


async def _find_escalation(
    db: "AsyncSession",
    escalation_repo: EscalationRepository,
    reference_id: str,
    trigger_reason: str,
):
    """Locates the `escalations` row `EscalationService.route` already
    inserted, matching `reference_id` against either
    `triage_session_id` or `call_interaction_id` (the payload carries a
    single id whose owning column depends on the originating channel).

    `EscalationRepository` (frozen) exposes no direct "find by reference id"
    lookup — only `create`/`get_by_id`/`acknowledge`/`resolve`/
    `list_unacknowledged` — so, mirroring the exact accommodation
    `services/console/service.py`'s `OverrideService._find_existing_override`
    makes for a frozen repository with no dedicated lookup: a collaborator
    that happens to expose a more direct by-reference lookup (an optional
    capability this method does not require but prefers when present) is
    tried first via `getattr`; otherwise this scans `list_unacknowledged`
    (ordered oldest-first) and keeps the last (most recently inserted)
    match, since the row this worker is chasing was inserted immediately
    before the event that triggered this handler was published.
    """

    direct_lookup = getattr(escalation_repo, "find_by_reference", None)
    if direct_lookup is not None:
        direct_match = await direct_lookup(db, reference_id, trigger_reason)
        if direct_match is not None:
            return direct_match

    try:
        reference_uuid = UUID(str(reference_id))
    except (ValueError, TypeError, AttributeError):
        reference_uuid = None

    candidates = await escalation_repo.list_unacknowledged(db)
    match = None
    for candidate in candidates:
        same_reference = (
            reference_uuid is not None
            and (
                candidate.triage_session_id == reference_uuid
                or candidate.call_interaction_id == reference_uuid
            )
        )
        if same_reference and candidate.trigger_reason == trigger_reason:
            match = candidate
        elif same_reference and match is None:
            match = candidate
    return match


async def handle_escalation_triggered(payload: dict, headers: dict) -> None:
    """FR-E8.5/SFR001/US44: logs an ERROR-level structured line (the
    "elevated-visibility queue" US44 calls for) and records a secondary
    `AuditService.record(action_type="escalation.notify", actor_type="system", ...)`
    entry — the `escalations` row's persistence itself is already guaranteed
    by `EscalationService.route`'s insert, which ran before this handler was
    ever invoked.
    """

    reference_id = payload.get("triage_session_id_or_call_interaction_id")
    trigger_reason = payload.get("trigger_reason")

    if not reference_id or not trigger_reason:
        # Both fields are required by the documented payload shape
        # (Interaction contract); a malformed message is surfaced loudly
        # rather than silently no-op'ing, so the broker's own retry/
        # dead-letter handling (app/core/messaging.py) takes over.
        raise ValueError(
            "escalation.triggered payload missing "
            "triage_session_id_or_call_interaction_id/trigger_reason."
        )

    assert _session_factory is not None, "run() must set the session factory before consuming"

    # Collaborators are constructed per-invocation, from the module-level
    # (patchable) class references imported above, rather than cached as
    # module-level singletons at import time -- the same "construct lazily,
    # never at import time" rule project_rules applies to real infra clients
    # (app/core/database.py, app/core/messaging.py, app/core/storage.py)
    # applies here to keep this handler a pure function of its inputs.
    escalation_repo = EscalationRepository()
    audit_repo = AuditLogRepository()
    audit_service = AuditService(audit_repo)

    async with _session_factory() as db:
        try:
            escalation = await _find_escalation(db, escalation_repo, reference_id, trigger_reason)
            if escalation is None:
                # US44 Watch out: the row MUST already exist (inserted by
                # EscalationService.route before publishing); if it cannot be
                # found the handler still surfaces this loudly rather than
                # silently no-op'ing, and re-raises so the broker's own
                # retry/dead-letter mechanism (app/core/messaging.py) takes
                # over — never a raw stack trace, never patient PII, only the
                # entity reference and reason.
                logger.error(
                    "escalation_row_missing",
                    extra={
                        "reference_id": str(reference_id),
                        "trigger_reason": trigger_reason,
                    },
                )
                raise LookupError(
                    "No escalations row found for the published escalation.triggered event."
                )

            logger.error(
                "escalation_triggered_dispatched",
                extra={
                    "escalation_id": str(escalation.id),
                    "trigger_reason": trigger_reason,
                    "reference_id": str(reference_id),
                    "status": escalation.status.value
                    if hasattr(escalation.status, "value")
                    else escalation.status,
                },
            )

            await audit_service.record(
                db,
                actor_staff_id=None,
                actor_type=ActorType.system,
                action_type="escalation.notify",
                entity_type="escalation",
                entity_id=escalation.id,
                original_payload={
                    "trigger_reason": trigger_reason,
                    "reference_id": str(reference_id),
                },
            )
        except Exception:
            await db.rollback()
            raise
        else:
            await db.commit()


async def run(
    broker: "MessageBroker", session_factory: "async_sessionmaker[AsyncSession]"
) -> None:
    """Composition-root entrypoint (called from `app/worker.py`): binds
    `handle_escalation_triggered` to the `escalation.triggered` routing key
    on the shared `MessageBroker`, and runs until cancelled.
    """

    global _session_factory
    _session_factory = session_factory

    await broker.consume(_QUEUE_NAME, _ROUTING_KEY, handle_escalation_triggered)
