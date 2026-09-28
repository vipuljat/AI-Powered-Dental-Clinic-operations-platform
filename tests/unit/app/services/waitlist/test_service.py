"""Unit tests for app/services/waitlist/service.py.

Tests exercise IdleChairPollingService, WaitlistService and WaitlistFillService
against their documented public surface and behaviour, using AsyncMock/MagicMock
stand-ins for their injected repository/service/broker dependencies (per the
project's test-substitution approach) rather than any real DB, broker, or HTTP
call.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.common.constants import WAITLIST_RESPONSE_WINDOW_MINUTES
from app.services.waitlist.service import (
    IdleChairPollingService,
    WaitlistFillService,
    WaitlistService,
)


def _flat_call_values(call):
    """Flatten a mock call's positional args + keyword values into one list."""
    if call is None:
        return []
    return list(call.args) + list(call.kwargs.values())


# ---------------------------------------------------------------------------
# IdleChairPollingService
# ---------------------------------------------------------------------------

class TestDetectFromSlot:
    @pytest.mark.asyncio
    async def test_creates_new_alert_when_no_existing_match(self):
        alert_repo = AsyncMock()
        alert_repo.find_matching.return_value = None
        new_alert = MagicMock()
        alert_repo.create.return_value = new_alert
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)
        db = AsyncMock()
        provider_id, chair_id = uuid4(), uuid4()
        slot_start = datetime.now(timezone.utc)
        slot_end = slot_start + timedelta(minutes=30)

        alert, created = await svc.detect_from_slot(
            db, provider_id, chair_id, slot_start, slot_end, source="auto_detected"
        )

        assert created is True
        assert alert is new_alert
        alert_repo.create.assert_awaited_once()
        kwargs = alert_repo.create.await_args.kwargs
        assert kwargs.get("provider_id") == provider_id
        assert kwargs.get("chair_id") == chair_id
        assert kwargs.get("source") == "auto_detected"

    @pytest.mark.asyncio
    async def test_returns_existing_matching_alert_without_creating_duplicate(self):
        existing = MagicMock()
        alert_repo = AsyncMock()
        alert_repo.find_matching.return_value = existing
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)
        db = AsyncMock()
        slot_start = datetime.now(timezone.utc)
        slot_end = slot_start + timedelta(minutes=30)

        alert, created = await svc.detect_from_slot(
            db, uuid4(), uuid4(), slot_start, slot_end, source="manual_flag"
        )

        assert created is False
        assert alert is existing
        alert_repo.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_appointment_id_through_when_supplied(self):
        alert_repo = AsyncMock()
        alert_repo.find_matching.return_value = None
        alert_repo.create.return_value = MagicMock()
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)
        db = AsyncMock()
        appointment_id = uuid4()
        slot_start = datetime.now(timezone.utc)
        slot_end = slot_start + timedelta(minutes=30)

        await svc.detect_from_slot(
            db, uuid4(), uuid4(), slot_start, slot_end,
            source="auto_detected", appointment_id=appointment_id,
        )

        kwargs = alert_repo.create.await_args.kwargs
        assert kwargs.get("appointment_id") == appointment_id


class TestFlagManual:
    @pytest.mark.asyncio
    async def test_dedups_against_existing_alert_and_records_audit(self):
        existing = MagicMock()
        alert_repo = AsyncMock()
        alert_repo.find_matching.return_value = existing
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)
        db = AsyncMock()
        actor_id = uuid4()
        slot_start = datetime.now(timezone.utc)
        slot_end = slot_start + timedelta(minutes=30)

        alert, created = await svc.flag_manual(
            db, uuid4(), uuid4(), slot_start, slot_end, actor_id
        )

        assert created is False
        assert alert is existing
        alert_repo.create.assert_not_awaited()
        audit.record.assert_awaited_once()
        assert actor_id in _flat_call_values(audit.record.await_args)

    @pytest.mark.asyncio
    async def test_creates_new_alert_with_manual_flag_source(self):
        alert_repo = AsyncMock()
        alert_repo.find_matching.return_value = None
        new_alert = MagicMock()
        alert_repo.create.return_value = new_alert
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)
        db = AsyncMock()
        slot_start = datetime.now(timezone.utc)
        slot_end = slot_start + timedelta(minutes=30)

        alert, created = await svc.flag_manual(
            db, uuid4(), uuid4(), slot_start, slot_end, uuid4()
        )

        assert created is True
        assert alert is new_alert
        assert alert_repo.create.await_args.kwargs.get("source") == "manual_flag"


class TestRunCycle:
    @staticmethod
    def _empty_db():
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=exec_result)
        db.scalars = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[])))
        return db

    @staticmethod
    def _db_with_one_idle_row(row):
        db = MagicMock()
        exec_result = MagicMock()
        exec_result.scalars.return_value.all.return_value = [row]
        db.execute = AsyncMock(return_value=exec_result)
        db.scalars = AsyncMock(return_value=MagicMock(all=MagicMock(return_value=[row])))
        return db

    @pytest.mark.asyncio
    async def test_clean_cycle_raises_no_alert_and_returns_zero(self):
        alert_repo = AsyncMock()
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)
        db = self._empty_db()

        with patch.object(
            IdleChairPollingService, "detect_from_slot", new=AsyncMock()
        ) as mock_detect:
            count = await svc.run_cycle(db)

        assert count == 0
        mock_detect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cycle_with_idle_slot_raises_one_new_alert(self):
        alert_repo = AsyncMock()
        audit = AsyncMock()
        svc = IdleChairPollingService(alert_repo, audit)

        fake_row = MagicMock()
        fake_row.id = uuid4()
        fake_row.provider_id = uuid4()
        fake_row.chair_id = uuid4()
        fake_row.scheduled_start = datetime.now(timezone.utc)
        fake_row.scheduled_end = fake_row.scheduled_start + timedelta(minutes=30)
        db = self._db_with_one_idle_row(fake_row)

        with patch.object(
            IdleChairPollingService,
            "detect_from_slot",
            new=AsyncMock(return_value=(MagicMock(), True)),
        ) as mock_detect:
            count = await svc.run_cycle(db)

        assert count == 1
        mock_detect.assert_awaited()


# ---------------------------------------------------------------------------
# WaitlistService
# ---------------------------------------------------------------------------

class TestAddEntry:
    @pytest.mark.asyncio
    async def test_add_entry_computes_priority_score_creates_and_audits(self):
        entry_repo = AsyncMock()
        created_entry = MagicMock()
        created_entry.id = uuid4()
        created_entry.status = "active"
        entry_repo.create.return_value = created_entry
        entry_repo.list_by_priority.return_value = [created_entry]

        rule_set_service = AsyncMock()
        rule_set_service.get_active_rules_by_category.return_value = [
            {"rule_key": "urgency_weight", "value": {"high": 10}}
        ]
        audit = AsyncMock()
        svc = WaitlistService(entry_repo, rule_set_service, audit)
        db = AsyncMock()
        actor_id = uuid4()
        patient_id = uuid4()

        result = await svc.add_entry(db, {"patient_id": patient_id}, actor_id)

        assert result is created_entry
        entry_repo.create.assert_awaited_once()
        create_kwargs = entry_repo.create.await_args.kwargs
        assert create_kwargs.get("patient_id") == patient_id
        assert isinstance(create_kwargs.get("priority_score"), (int, float))
        assert audit.record.await_count >= 1
        assert actor_id in _flat_call_values(audit.record.await_args)


class TestRecalculatePriority:
    @pytest.mark.asyncio
    async def test_rescores_every_active_entry_and_returns_count(self):
        entry_repo = AsyncMock()
        e1, e2, e3 = MagicMock(id=uuid4()), MagicMock(id=uuid4()), MagicMock(id=uuid4())
        entry_repo.list_by_priority.return_value = [e1, e2, e3]
        rule_set_service = AsyncMock()
        rule_set_service.get_active_rules_by_category.return_value = []
        audit = AsyncMock()
        svc = WaitlistService(entry_repo, rule_set_service, audit)
        db = AsyncMock()

        count = await svc.recalculate_priority(db, trigger="cancellation")

        assert count == 3
        assert entry_repo.update_score.await_count == 3

    @pytest.mark.asyncio
    async def test_recalculate_priority_returns_zero_for_no_active_entries(self):
        entry_repo = AsyncMock()
        entry_repo.list_by_priority.return_value = []
        rule_set_service = AsyncMock()
        rule_set_service.get_active_rules_by_category.return_value = []
        audit = AsyncMock()
        svc = WaitlistService(entry_repo, rule_set_service, audit)
        db = AsyncMock()

        count = await svc.recalculate_priority(db, trigger="rule_change")

        assert count == 0
        entry_repo.update_score.assert_not_awaited()


# ---------------------------------------------------------------------------
# WaitlistFillService
# ---------------------------------------------------------------------------

class TestOfferAndCascade:
    @pytest.mark.asyncio
    async def test_offers_to_top_candidate_and_publishes_event(self):
        alert_id = uuid4()
        candidate = MagicMock()
        candidate.id = uuid4()
        candidate.patient_id = uuid4()

        offer_repo = AsyncMock()
        offer_repo.list_by_alert.return_value = []
        created_offer = MagicMock()
        offer_repo.create.return_value = created_offer

        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        alert_repo.get_by_id.return_value = MagicMock(
            id=alert_id, status="open",
            provider_id=uuid4(), chair_id=uuid4(),
            slot_start=datetime.now(timezone.utc),
            slot_end=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()
        now = datetime.now(timezone.utc)

        with patch.object(
            WaitlistService, "match_candidates", new=AsyncMock(return_value=[candidate])
        ):
            result = await svc.offer_and_cascade(db, alert_id)

        assert result is created_offer
        offer_repo.create.assert_awaited_once()
        create_kwargs = offer_repo.create.await_args.kwargs
        assert create_kwargs.get("status") == "pending"
        assert create_kwargs.get("waitlist_entry_id") == candidate.id
        assert create_kwargs.get("idle_chair_alert_id") == alert_id
        expires_at = create_kwargs.get("response_window_expires_at")
        assert isinstance(expires_at, datetime)
        delta_minutes = (expires_at - now).total_seconds() / 60
        assert (WAITLIST_RESPONSE_WINDOW_MINUTES - 1) <= delta_minutes <= (
            WAITLIST_RESPONSE_WINDOW_MINUTES + 1
        )

        entry_repo.update_status.assert_awaited_once()
        entry_call_values = _flat_call_values(entry_repo.update_status.await_args)
        assert candidate.id in entry_call_values
        assert "offered" in entry_call_values
        assert entry_repo.update_status.await_args.kwargs.get("current_alert_id") == alert_id

        broker.publish.assert_awaited_once()
        publish_values = _flat_call_values(broker.publish.await_args)
        assert "waitlist_offer.created" in publish_values
        payloads = [v for v in publish_values if isinstance(v, dict)]
        assert payloads, "expected a dict payload to be published"
        assert payloads[0] == {
            "waitlist_entry_id": candidate.id,
            "idle_chair_alert_id": alert_id,
        }

    @pytest.mark.asyncio
    async def test_exhausts_alert_when_no_candidates_remain(self):
        alert_id = uuid4()
        offer_repo = AsyncMock()
        offer_repo.list_by_alert.return_value = []
        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        alert_repo.get_by_id.return_value = MagicMock(id=alert_id, status="open")
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        with patch.object(
            WaitlistService, "match_candidates", new=AsyncMock(return_value=[])
        ):
            result = await svc.offer_and_cascade(db, alert_id)

        assert result is None
        offer_repo.create.assert_not_awaited()
        broker.publish.assert_not_awaited()
        alert_repo.update_status.assert_awaited_once()
        assert "exhausted" in _flat_call_values(alert_repo.update_status.await_args)

    @pytest.mark.asyncio
    async def test_skips_candidates_already_declined_or_expired_for_this_alert(self):
        alert_id = uuid4()
        candidate1 = MagicMock(id=uuid4(), patient_id=uuid4())
        candidate2 = MagicMock(id=uuid4(), patient_id=uuid4())

        prior_offer = MagicMock(waitlist_entry_id=candidate1.id, status="declined")

        offer_repo = AsyncMock()
        offer_repo.list_by_alert.return_value = [prior_offer]
        created_offer = MagicMock()
        offer_repo.create.return_value = created_offer

        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        alert_repo.get_by_id.return_value = MagicMock(
            id=alert_id, status="open",
            provider_id=uuid4(), chair_id=uuid4(),
            slot_start=datetime.now(timezone.utc),
            slot_end=datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        with patch.object(
            WaitlistService,
            "match_candidates",
            new=AsyncMock(return_value=[candidate1, candidate2]),
        ):
            await svc.offer_and_cascade(db, alert_id)

        create_kwargs = offer_repo.create.await_args.kwargs
        assert create_kwargs.get("waitlist_entry_id") == candidate2.id


class TestRespond:
    @pytest.mark.asyncio
    async def test_accept_books_appointment_and_updates_state(self):
        offer_id = uuid4()
        alert_id = uuid4()
        entry_id = uuid4()
        patient_id = uuid4()

        accepted_offer = MagicMock(
            id=offer_id, status="accepted",
            idle_chair_alert_id=alert_id, waitlist_entry_id=entry_id,
        )
        offer_repo = AsyncMock()
        offer_repo.update_status_conditional.return_value = accepted_offer

        alert = MagicMock(
            id=alert_id,
            provider_id=uuid4(), chair_id=uuid4(),
            slot_start=datetime.now(timezone.utc),
            slot_end=datetime.now(timezone.utc) + timedelta(minutes=30),
            status="open",
        )
        alert_repo = AsyncMock()
        alert_repo.get_by_id.return_value = alert

        entry = MagicMock(id=entry_id, patient_id=patient_id)
        entry_repo = AsyncMock()
        entry_repo.get_by_id.return_value = entry

        new_appt = MagicMock(id=uuid4())
        scheduling_service = AsyncMock()
        scheduling_service.create.return_value = (new_appt, False)

        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        result = await svc.respond(db, offer_id, "accept")

        offer_repo.update_status_conditional.assert_awaited_once()
        cond_values = _flat_call_values(offer_repo.update_status_conditional.await_args)
        assert "pending" in cond_values
        assert "accepted" in cond_values

        scheduling_service.create.assert_awaited_once()
        sched_values = _flat_call_values(scheduling_service.create.await_args)
        assert None in sched_values  # actor_id=None (AI-agent-originated)
        assert any(
            isinstance(v, str) and str(offer_id) in v for v in sched_values
        ), "expected an idempotency key derived from the offer id"

        alert_repo.update_status.assert_awaited_once()
        assert "filled" in _flat_call_values(alert_repo.update_status.await_args)

        entry_repo.update_status.assert_awaited_once()
        entry_values = _flat_call_values(entry_repo.update_status.await_args)
        assert "booked" in entry_values
        assert new_appt.id in entry_values

        assert result is accepted_offer

    @pytest.mark.asyncio
    async def test_decline_marks_entry_declined_and_cascades_to_next(self):
        offer_id = uuid4()
        alert_id = uuid4()
        entry_id = uuid4()

        declined_offer = MagicMock(
            id=offer_id, status="declined",
            idle_chair_alert_id=alert_id, waitlist_entry_id=entry_id,
        )
        offer_repo = AsyncMock()
        offer_repo.update_status_conditional.return_value = declined_offer
        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        with patch.object(
            WaitlistFillService, "offer_and_cascade", new=AsyncMock(return_value=None)
        ) as mock_cascade:
            result = await svc.respond(db, offer_id, "decline")

        entry_repo.update_status.assert_awaited_once()
        assert "declined" in _flat_call_values(entry_repo.update_status.await_args)

        scheduling_service.create.assert_not_awaited()

        mock_cascade.assert_awaited_once()
        assert alert_id in _flat_call_values(mock_cascade.await_args)

        assert result is declined_offer

    @pytest.mark.asyncio
    async def test_respond_to_already_resolved_offer_is_idempotent_no_op(self):
        # update_status_conditional returning None means the WHERE status='pending'
        # guard matched zero rows (a concurrent/replayed response) -- respond must
        # not raise, and must not re-run the booking/state-mutation side effects.
        offer_id = uuid4()
        offer_repo = AsyncMock()
        offer_repo.update_status_conditional.return_value = None

        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        await svc.respond(db, offer_id, "accept")

        scheduling_service.create.assert_not_awaited()
        entry_repo.update_status.assert_not_awaited()
        alert_repo.update_status.assert_not_awaited()


class TestSweepExpiredOffers:
    @pytest.mark.asyncio
    async def test_expires_pending_offers_and_cascades_per_alert(self):
        alert_id = uuid4()
        entry_id = uuid4()
        expired_offer = MagicMock(
            id=uuid4(), idle_chair_alert_id=alert_id,
            waitlist_entry_id=entry_id, status="pending",
        )

        offer_repo = AsyncMock()
        offer_repo.list_expired_pending.return_value = [expired_offer]
        offer_repo.update_status_conditional.return_value = MagicMock(status="expired")

        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        with patch.object(
            WaitlistFillService, "offer_and_cascade", new=AsyncMock(return_value=None)
        ) as mock_cascade:
            count = await svc.sweep_expired_offers(db)

        assert count == 1
        offer_repo.update_status_conditional.assert_awaited_once()
        assert "expired" in _flat_call_values(
            offer_repo.update_status_conditional.await_args
        )
        mock_cascade.assert_awaited_once()
        assert alert_id in _flat_call_values(mock_cascade.await_args)

    @pytest.mark.asyncio
    async def test_returns_zero_when_nothing_expired(self):
        offer_repo = AsyncMock()
        offer_repo.list_expired_pending.return_value = []
        entry_repo = AsyncMock()
        alert_repo = AsyncMock()
        scheduling_service = AsyncMock()
        broker = AsyncMock()
        audit = AsyncMock()

        svc = WaitlistFillService(
            offer_repo, entry_repo, alert_repo, scheduling_service, broker, audit
        )
        db = AsyncMock()

        count = await svc.sweep_expired_offers(db)

        assert count == 0
        offer_repo.update_status_conditional.assert_not_awaited()
