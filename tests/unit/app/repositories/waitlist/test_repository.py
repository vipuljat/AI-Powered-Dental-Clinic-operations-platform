"""
Unit tests for app/repositories/waitlist/repository.py

These exercise IdleChairAlertRepository, WaitlistEntryRepository and
WaitlistOfferRepository against a real (sqlite+aiosqlite, in-memory) async
SQLAlchemy session -- the project's own DB test-substitution mechanism
(architecture rule: ENVIRONMENT=test swaps the DB URL to
sqlite+aiosqlite:///:memory:, and app/core/db_types.py's JSONB/VECTOR
TypeDecorators make the same ORM models run unmodified against both
Postgres and SQLite). No mocking of the session/ORM is used -- these are
real queries against real tables.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.waitlist.models import IdleChairAlert, WaitlistEntry, WaitlistOffer
from app.repositories.waitlist.repository import (
    IdleChairAlertRepository,
    WaitlistEntryRepository,
    WaitlistOfferRepository,
)


def _now():
    return datetime.now(timezone.utc)


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_maker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def alert_repo():
    return IdleChairAlertRepository()


@pytest.fixture
def entry_repo():
    return WaitlistEntryRepository()


@pytest.fixture
def offer_repo():
    return WaitlistOfferRepository()


# --------------------------------------------------------------------------
# IdleChairAlertRepository
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_chair_alert_create_and_get_by_id(db_session, alert_repo):
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    slot_start = _now()
    slot_end = slot_start + timedelta(minutes=30)

    created = await alert_repo.create(
        db_session,
        id=uuid.uuid4(),
        provider_id=provider_id,
        chair_id=chair_id,
        slot_start=slot_start,
        slot_end=slot_end,
        status="open",
    )

    fetched = await alert_repo.get_by_id(db_session, created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.provider_id == provider_id
    assert fetched.chair_id == chair_id
    assert fetched.status == "open"


@pytest.mark.asyncio
async def test_idle_chair_alert_get_by_id_returns_none_when_missing(db_session, alert_repo):
    result = await alert_repo.get_by_id(db_session, uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_find_matching_requires_exact_slot(db_session, alert_repo):
    """FR-E5.2/US24: only an exact (provider_id, chair_id, slot_start, slot_end)
    match is returned; a partially-overlapping slot is a distinct alert."""
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    slot_start = _now()
    slot_end = slot_start + timedelta(minutes=30)

    existing = await alert_repo.create(
        db_session,
        id=uuid.uuid4(),
        provider_id=provider_id,
        chair_id=chair_id,
        slot_start=slot_start,
        slot_end=slot_end,
        status="open",
    )

    exact_match = await alert_repo.find_matching(
        db_session, provider_id, chair_id, slot_start, slot_end
    )
    assert exact_match is not None
    assert exact_match.id == existing.id

    overlapping_start = slot_start + timedelta(minutes=10)
    overlapping_end = slot_end + timedelta(minutes=10)
    overlap_match = await alert_repo.find_matching(
        db_session, provider_id, chair_id, overlapping_start, overlapping_end
    )
    assert overlap_match is None


@pytest.mark.asyncio
async def test_find_matching_ignores_non_open_alerts(db_session, alert_repo):
    provider_id = uuid.uuid4()
    chair_id = uuid.uuid4()
    slot_start = _now()
    slot_end = slot_start + timedelta(minutes=30)

    await alert_repo.create(
        db_session,
        id=uuid.uuid4(),
        provider_id=provider_id,
        chair_id=chair_id,
        slot_start=slot_start,
        slot_end=slot_end,
        status="filled",
    )

    result = await alert_repo.find_matching(db_session, provider_id, chair_id, slot_start, slot_end)
    assert result is None


@pytest.mark.asyncio
async def test_idle_chair_alert_update_status(db_session, alert_repo):
    created = await alert_repo.create(
        db_session,
        id=uuid.uuid4(),
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        slot_start=_now(),
        slot_end=_now() + timedelta(minutes=30),
        status="open",
    )

    updated = await alert_repo.update_status(db_session, created.id, "filled")

    assert updated.id == created.id
    assert updated.status == "filled"

    refetched = await alert_repo.get_by_id(db_session, created.id)
    assert refetched.status == "filled"


@pytest.mark.asyncio
async def test_list_open_returns_only_open_alerts(db_session, alert_repo):
    open_one = await alert_repo.create(
        db_session,
        id=uuid.uuid4(),
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        slot_start=_now(),
        slot_end=_now() + timedelta(minutes=30),
        status="open",
    )
    await alert_repo.create(
        db_session,
        id=uuid.uuid4(),
        provider_id=uuid.uuid4(),
        chair_id=uuid.uuid4(),
        slot_start=_now(),
        slot_end=_now() + timedelta(minutes=30),
        status="filled",
    )

    open_alerts = await alert_repo.list_open(db_session)

    assert [a.id for a in open_alerts] == [open_one.id]
    assert all(a.status == "open" for a in open_alerts)


# --------------------------------------------------------------------------
# WaitlistEntryRepository
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_waitlist_entry_create_and_get_by_id(db_session, entry_repo):
    created = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=5.0,
        status="active",
        created_at=_now(),
    )

    fetched = await entry_repo.get_by_id(db_session, created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.priority_score == 5.0
    assert fetched.status == "active"


@pytest.mark.asyncio
async def test_list_by_priority_orders_by_score_desc_then_created_at_asc(db_session, entry_repo):
    """US26 AC2: deterministic ordering -- priority_score DESC, created_at ASC
    is the tiebreaker for equal scores."""
    base_time = _now()

    low_score = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=1.0,
        status="active",
        created_at=base_time,
    )
    high_score = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=9.0,
        status="active",
        created_at=base_time + timedelta(seconds=1),
    )
    tie_earlier = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=5.0,
        status="active",
        created_at=base_time + timedelta(seconds=2),
    )
    tie_later = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=5.0,
        status="active",
        created_at=base_time + timedelta(seconds=3),
    )

    ordered = await entry_repo.list_by_priority(db_session)

    assert [e.id for e in ordered] == [
        high_score.id,
        tie_earlier.id,
        tie_later.id,
        low_score.id,
    ]


@pytest.mark.asyncio
async def test_list_by_priority_filters_by_status(db_session, entry_repo):
    active = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=1.0,
        status="active",
        created_at=_now(),
    )
    await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=9.0,
        status="fulfilled",
        created_at=_now(),
    )

    result = await entry_repo.list_by_priority(db_session, status="active")

    assert [e.id for e in result] == [active.id]


@pytest.mark.asyncio
async def test_list_matching_slot_filters_by_provider_and_open_timeframe(db_session, entry_repo):
    """Candidates whose desired_provider_id is NULL or matches, and whose
    desired_timeframe is NULL ("any time"), should be returned; a candidate
    tied to a different provider must not be."""
    provider_id = uuid.uuid4()
    other_provider_id = uuid.uuid4()
    slot_start = _now()
    slot_end = slot_start + timedelta(minutes=30)

    any_provider_entry = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=1.0,
        status="active",
        created_at=_now(),
    )
    matching_provider_entry = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=provider_id,
        desired_timeframe=None,
        priority_score=2.0,
        status="active",
        created_at=_now(),
    )
    non_matching_provider_entry = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=other_provider_id,
        desired_timeframe=None,
        priority_score=3.0,
        status="active",
        created_at=_now(),
    )

    results = await entry_repo.list_matching_slot(db_session, provider_id, slot_start, slot_end)
    result_ids = {e.id for e in results}

    assert any_provider_entry.id in result_ids
    assert matching_provider_entry.id in result_ids
    assert non_matching_provider_entry.id not in result_ids


@pytest.mark.asyncio
async def test_list_matching_slot_none_provider_arg_still_returns_open_candidates(db_session, entry_repo):
    entry = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=1.0,
        status="active",
        created_at=_now(),
    )

    slot_start = _now()
    slot_end = slot_start + timedelta(minutes=30)
    results = await entry_repo.list_matching_slot(db_session, None, slot_start, slot_end)

    assert entry.id in {e.id for e in results}


@pytest.mark.asyncio
async def test_waitlist_entry_update_score(db_session, entry_repo):
    created = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=1.0,
        status="active",
        created_at=_now(),
    )

    updated = await entry_repo.update_score(db_session, created.id, 42.5)

    assert updated.id == created.id
    assert updated.priority_score == 42.5


@pytest.mark.asyncio
async def test_waitlist_entry_update_status_with_extra_fields(db_session, entry_repo):
    created = await entry_repo.create(
        db_session,
        id=uuid.uuid4(),
        desired_provider_id=None,
        desired_timeframe=None,
        priority_score=1.0,
        status="active",
        created_at=_now(),
    )

    updated = await entry_repo.update_status(db_session, created.id, "fulfilled", priority_score=0.0)

    assert updated.status == "fulfilled"
    assert updated.priority_score == 0.0


# --------------------------------------------------------------------------
# WaitlistOfferRepository
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_waitlist_offer_create(db_session, offer_repo):
    idle_chair_alert_id = uuid.uuid4()
    waitlist_entry_id = uuid.uuid4()
    expires_at = _now() + timedelta(minutes=15)

    created = await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=idle_chair_alert_id,
        waitlist_entry_id=waitlist_entry_id,
        status="pending",
        response_window_expires_at=expires_at,
    )

    assert created.idle_chair_alert_id == idle_chair_alert_id
    assert created.waitlist_entry_id == waitlist_entry_id
    assert created.status == "pending"


@pytest.mark.asyncio
async def test_update_status_conditional_succeeds_when_from_status_matches(db_session, offer_repo):
    created = await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=uuid.uuid4(),
        waitlist_entry_id=uuid.uuid4(),
        status="pending",
        response_window_expires_at=_now() + timedelta(minutes=15),
    )

    updated = await offer_repo.update_status_conditional(
        db_session, created.id, from_status="pending", to_status="accepted"
    )

    assert updated is not None
    assert updated.id == created.id
    assert updated.status == "accepted"


@pytest.mark.asyncio
async def test_update_status_conditional_is_noop_when_already_resolved(db_session, offer_repo):
    """project_rules.concurrency: a second response to an already-resolved
    offer must be a no-op (returns None), not a race condition."""
    created = await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=uuid.uuid4(),
        waitlist_entry_id=uuid.uuid4(),
        status="accepted",
        response_window_expires_at=_now() + timedelta(minutes=15),
    )

    result = await offer_repo.update_status_conditional(
        db_session, created.id, from_status="pending", to_status="declined"
    )

    assert result is None

    # and the row's status is unchanged by the no-op attempt
    unchanged = await offer_repo.list_by_alert(db_session, created.idle_chair_alert_id)
    assert unchanged[0].status == "accepted"


@pytest.mark.asyncio
async def test_update_status_conditional_returns_none_for_unknown_id(db_session, offer_repo):
    result = await offer_repo.update_status_conditional(
        db_session, uuid.uuid4(), from_status="pending", to_status="accepted"
    )
    assert result is None


@pytest.mark.asyncio
async def test_list_by_alert_returns_only_offers_for_that_alert(db_session, offer_repo):
    alert_id = uuid.uuid4()
    other_alert_id = uuid.uuid4()

    matching = await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=alert_id,
        waitlist_entry_id=uuid.uuid4(),
        status="pending",
        response_window_expires_at=_now() + timedelta(minutes=15),
    )
    await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=other_alert_id,
        waitlist_entry_id=uuid.uuid4(),
        status="pending",
        response_window_expires_at=_now() + timedelta(minutes=15),
    )

    results = await offer_repo.list_by_alert(db_session, alert_id)

    assert [o.id for o in results] == [matching.id]


@pytest.mark.asyncio
async def test_list_expired_pending_filters_by_status_and_window(db_session, offer_repo):
    """Interaction contract: app/workers/waitlist_poller.py polls this to
    drive cascade-on-timeout (US27 Alternate Flow)."""
    as_of = _now()

    expired_pending = await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=uuid.uuid4(),
        waitlist_entry_id=uuid.uuid4(),
        status="pending",
        response_window_expires_at=as_of - timedelta(minutes=1),
    )
    # not expired yet
    await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=uuid.uuid4(),
        waitlist_entry_id=uuid.uuid4(),
        status="pending",
        response_window_expires_at=as_of + timedelta(minutes=10),
    )
    # already resolved, even though its window has technically passed
    await offer_repo.create(
        db_session,
        id=uuid.uuid4(),
        idle_chair_alert_id=uuid.uuid4(),
        waitlist_entry_id=uuid.uuid4(),
        status="accepted",
        response_window_expires_at=as_of - timedelta(minutes=1),
    )

    results = await offer_repo.list_expired_pending(db_session, as_of)

    assert [o.id for o in results] == [expired_pending.id]
