"""Unit tests for app/repositories/education/repository.py.

Exercises ContentItemRepository/ContentDeliveryRepository directly against an
in-memory SQLite database, mirroring the ENVIRONMENT=test substitution
described in project_rules.testing (app/core/database.py switches to
sqlite+aiosqlite:///:memory: with a StaticPool at the composition root, so
the same ORM models run unmodified against both SQLite and Postgres).

Per the unit-test-authoring contract, only this file's own spec is available
here -- app/repositories/education/repository.py's own implementation, and
every other task's spec, are deliberately out of scope. `content_items` /
`content_deliveries` carry no cross-module ORM relationship() (per
app/models/education/models.py's own documented Interaction contract), so no
stand-in tables for another module are required to create this module's
tables in isolation.
"""

from __future__ import annotations

import uuid
from datetime import timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.education.models import ContentDelivery, ContentItem
from app.repositories.education.repository import (
    ContentDeliveryRepository,
    ContentItemRepository,
)

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
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


@pytest.fixture
def item_repo() -> ContentItemRepository:
    return ContentItemRepository()


@pytest.fixture
def delivery_repo() -> ContentDeliveryRepository:
    return ContentDeliveryRepository()


async def make_content_item(session: AsyncSession, **overrides) -> ContentItem:
    values = dict(
        id=uuid.uuid4(),
        developmental_stage="child",
        language="en",
        trigger_type="pre_appointment",
        body="Brush twice a day.",
    )
    values.update(overrides)
    item = ContentItem(**values)
    session.add(item)
    await session.commit()
    return item


# --------------------------------------------------------------------------
# ContentItemRepository.get_by_stage_language
# --------------------------------------------------------------------------


async def test_get_by_stage_language_returns_exact_match(session, item_repo):
    item = await make_content_item(
        session,
        developmental_stage="child",
        language="en",
        trigger_type="pre_appointment",
    )

    found = await item_repo.get_by_stage_language(session, "child", "en", "pre_appointment")

    assert found is not None
    assert found.id == item.id
    assert found.developmental_stage == "child"


async def test_get_by_stage_language_returns_none_when_no_match(session, item_repo):
    await make_content_item(
        session,
        developmental_stage="child",
        language="en",
        trigger_type="pre_appointment",
    )

    # Different developmental_stage -- no exact match exists.
    found = await item_repo.get_by_stage_language(session, "teen", "en", "pre_appointment")

    assert found is None


async def test_get_by_stage_language_is_scoped_by_language_and_trigger(session, item_repo):
    await make_content_item(
        session,
        developmental_stage="child",
        language="en",
        trigger_type="pre_appointment",
    )

    # Same stage, but a different language -- must not match.
    assert await item_repo.get_by_stage_language(session, "child", "es", "pre_appointment") is None
    # Same stage/language, but a different trigger_type -- must not match.
    assert (
        await item_repo.get_by_stage_language(session, "child", "en", "post_appointment") is None
    )


# --------------------------------------------------------------------------
# ContentItemRepository.get_general_fallback / FR-E10.1
# --------------------------------------------------------------------------


async def test_get_general_fallback_returns_general_tagged_row(session, item_repo):
    general_item = await make_content_item(
        session,
        developmental_stage="general",
        language="en",
        trigger_type="post_appointment",
        body="General aftercare instructions.",
    )

    found = await item_repo.get_general_fallback(session, "en", "post_appointment")

    assert found is not None
    assert found.id == general_item.id
    assert found.developmental_stage == "general"


async def test_get_general_fallback_returns_none_when_no_general_row(session, item_repo):
    await make_content_item(
        session,
        developmental_stage="child",
        language="en",
        trigger_type="post_appointment",
    )

    found = await item_repo.get_general_fallback(session, "en", "post_appointment")

    assert found is None


async def test_missing_stage_specific_content_falls_back_to_general(session, item_repo):
    """FR-E10.1 / T122 AC: missing age-appropriate content falls back to
    general instructions -- i.e. when no stage-specific ContentItem exists,
    get_by_stage_language finds nothing but get_general_fallback still
    resolves the general-tagged row for the same language/trigger_type."""

    general_item = await make_content_item(
        session,
        developmental_stage="general",
        language="pt",
        trigger_type="pre_appointment",
        body="Instrucoes gerais.",
    )

    stage_specific = await item_repo.get_by_stage_language(
        session, "toddler", "pt", "pre_appointment"
    )
    fallback = await item_repo.get_general_fallback(session, "pt", "pre_appointment")

    assert stage_specific is None
    assert fallback is not None
    assert fallback.id == general_item.id


# --------------------------------------------------------------------------
# ContentDeliveryRepository.record_delivery / list_by_patient
# --------------------------------------------------------------------------


async def test_record_delivery_creates_and_returns_delivery(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    patient_id = uuid.uuid4()
    appointment_id = uuid.uuid4()

    delivery = await delivery_repo.record_delivery(
        session, patient_id, appointment_id, item.id, "whatsapp", "delivered"
    )

    assert delivery.id is not None
    assert delivery.patient_id == patient_id
    assert delivery.appointment_id == appointment_id
    assert delivery.content_item_id == item.id
    assert delivery.channel == "whatsapp"
    assert delivery.status == "delivered"


async def test_record_delivery_accepts_null_appointment_id(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    patient_id = uuid.uuid4()

    delivery = await delivery_repo.record_delivery(
        session, patient_id, None, item.id, "sms", "unavailable"
    )

    assert delivery.appointment_id is None
    assert delivery.status == "unavailable"


async def test_list_by_patient_returns_only_that_patients_deliveries(
    session, item_repo, delivery_repo
):
    item = await make_content_item(session)
    patient_a = uuid.uuid4()
    patient_b = uuid.uuid4()

    await delivery_repo.record_delivery(session, patient_a, None, item.id, "whatsapp", "delivered")
    await delivery_repo.record_delivery(session, patient_a, None, item.id, "email", "delivered")
    await delivery_repo.record_delivery(session, patient_b, None, item.id, "sms", "delivered")

    results = await delivery_repo.list_by_patient(session, patient_a)

    assert len(results) == 2
    assert {d.patient_id for d in results} == {patient_a}


async def test_list_by_patient_returns_empty_list_when_no_deliveries(
    session, item_repo, delivery_repo
):
    results = await delivery_repo.list_by_patient(session, uuid.uuid4())

    assert results == []


# --------------------------------------------------------------------------
# ContentDeliveryRepository.record_open
# --------------------------------------------------------------------------


async def test_record_open_sets_status_and_opened_at(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    delivery = await delivery_repo.record_delivery(
        session, uuid.uuid4(), None, item.id, "whatsapp", "delivered"
    )

    updated = await delivery_repo.record_open(session, delivery.id)

    assert updated.status == "opened"
    assert updated.opened_at is not None
    assert updated.opened_at.tzinfo is not None
    assert updated.opened_at.utcoffset() == timezone.utc.utcoffset(None)


async def test_record_open_is_a_no_op_when_already_opened(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    delivery = await delivery_repo.record_delivery(
        session, uuid.uuid4(), None, item.id, "whatsapp", "delivered"
    )

    await delivery_repo.record_open(session, delivery.id)
    # Calling it again on an already-opened delivery must not raise, and the
    # delivery must remain in the "opened" state.
    twice_updated = await delivery_repo.record_open(session, delivery.id)

    assert twice_updated.status == "opened"


# --------------------------------------------------------------------------
# ContentDeliveryRepository.record_completed
# --------------------------------------------------------------------------


async def test_record_completed_sets_status_and_completed_at(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    delivery = await delivery_repo.record_delivery(
        session, uuid.uuid4(), None, item.id, "webchat", "delivered"
    )

    updated = await delivery_repo.record_completed(session, delivery.id)

    assert updated.status == "completed"
    assert updated.completed_at is not None
    assert updated.completed_at.tzinfo is not None


# --------------------------------------------------------------------------
# ContentDeliveryRepository.get_funnel
# --------------------------------------------------------------------------


async def test_get_funnel_scoped_to_content_item_counts_only_delivered(
    session, item_repo, delivery_repo
):
    item = await make_content_item(session)
    patient = uuid.uuid4()
    await delivery_repo.record_delivery(session, patient, None, item.id, "whatsapp", "delivered")
    await delivery_repo.record_delivery(session, patient, None, item.id, "sms", "delivered")

    funnel = await delivery_repo.get_funnel(session, item.id)

    assert funnel == {"delivered": 2, "opened": 0, "completed": 0}


async def test_get_funnel_counts_opened_deliveries(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    delivery = await delivery_repo.record_delivery(
        session, uuid.uuid4(), None, item.id, "whatsapp", "delivered"
    )
    await delivery_repo.record_open(session, delivery.id)

    funnel = await delivery_repo.get_funnel(session, item.id)

    assert funnel["opened"] == 1
    assert funnel["completed"] == 0


async def test_get_funnel_counts_completed_deliveries(session, item_repo, delivery_repo):
    item = await make_content_item(session)
    delivery = await delivery_repo.record_delivery(
        session, uuid.uuid4(), None, item.id, "whatsapp", "delivered"
    )
    await delivery_repo.record_open(session, delivery.id)
    await delivery_repo.record_completed(session, delivery.id)

    funnel = await delivery_repo.get_funnel(session, item.id)

    assert funnel["completed"] == 1


async def test_get_funnel_without_content_item_id_aggregates_across_items(
    session, item_repo, delivery_repo
):
    item_one = await make_content_item(session, developmental_stage="child")
    item_two = await make_content_item(session, developmental_stage="teen")
    patient = uuid.uuid4()

    await delivery_repo.record_delivery(
        session, patient, None, item_one.id, "whatsapp", "delivered"
    )
    await delivery_repo.record_delivery(
        session, patient, None, item_one.id, "sms", "delivered"
    )
    await delivery_repo.record_delivery(
        session, patient, None, item_two.id, "email", "delivered"
    )

    scoped_one = await delivery_repo.get_funnel(session, item_one.id)
    scoped_two = await delivery_repo.get_funnel(session, item_two.id)
    aggregate = await delivery_repo.get_funnel(session, None)

    assert scoped_one == {"delivered": 2, "opened": 0, "completed": 0}
    assert scoped_two == {"delivered": 1, "opened": 0, "completed": 0}
    assert aggregate == {"delivered": 3, "opened": 0, "completed": 0}
