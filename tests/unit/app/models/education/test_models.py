"""Unit tests for app/models/education/models.py.

Exercises the two ORM tables declared here (content_items, content_deliveries)
directly against an in-memory SQLite database, mirroring the ENVIRONMENT=test
substitution described in project_rules.testing (app/core/database.py
switches to sqlite+aiosqlite:///:memory: with a StaticPool at the composition
root, so the same ORM models run unmodified against both SQLite and
Postgres).

Per the unit-test-authoring contract, only this file's own spec is available
here -- app/models/education/models.py's own implementation, and every other
task's spec, are deliberately out of scope.
content_deliveries.patient_id and .appointment_id are plain FK columns
pointing at patients.id / appointments.id, tables owned by the patients and
scheduling modules respectively (per this file's own Interaction contract:
"no cross-module ORM relationship()"). Since every ORM model shares one
Base.metadata, those FK targets must be resolvable for
Base.metadata.create_all to compile this file's own tables' DDL. Register
minimal stand-in tables (id column only) here -- rather than importing
another module's ORM classes -- to keep this test scoped to this file's own
spec; the guards are no-ops if the real patients/scheduling models already
registered those names on the shared metadata (e.g. when this file runs
alongside the full suite).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.education.models import ContentDelivery, ContentItem

pytestmark = pytest.mark.asyncio

if "patients" not in Base.metadata.tables:
    Table("patients", Base.metadata, Column("id", Uuid(), primary_key=True))

if "appointments" not in Base.metadata.tables:
    Table("appointments", Base.metadata, Column("id", Uuid(), primary_key=True))


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


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# factories for minimal-valid rows (overridable per-test)
# --------------------------------------------------------------------------


def make_content_item(**overrides):
    values = dict(
        developmental_stage="child",
        language="en",
        trigger_type="pre_appointment",
        body="Brush twice a day and floss once a day.",
    )
    values.update(overrides)
    return ContentItem(**values)


def make_content_delivery(**overrides):
    values = dict(
        patient_id=uuid.uuid4(),
        appointment_id=None,
        content_item_id=uuid.uuid4(),
        channel="whatsapp",
        status="delivered",
    )
    values.update(overrides)
    return ContentDelivery(**values)


# --------------------------------------------------------------------------
# table names / public surface
# --------------------------------------------------------------------------


def test_tablenames():
    assert ContentItem.__tablename__ == "content_items"
    assert ContentDelivery.__tablename__ == "content_deliveries"


# --------------------------------------------------------------------------
# declared column shape (Data shapes tables)
# --------------------------------------------------------------------------


def test_content_item_declared_columns_and_nullability():
    cols = {c.name: c for c in ContentItem.__table__.columns}
    assert set(cols) == {"id", "developmental_stage", "language", "trigger_type", "body"}
    assert cols["developmental_stage"].nullable is False
    assert cols["language"].nullable is False
    assert cols["trigger_type"].nullable is False
    assert cols["body"].nullable is False


def test_content_delivery_declared_columns_and_nullability():
    cols = {c.name: c for c in ContentDelivery.__table__.columns}
    assert set(cols) == {
        "id",
        "patient_id",
        "appointment_id",
        "content_item_id",
        "channel",
        "status",
        "delivered_at",
        "opened_at",
        "completed_at",
    }
    assert cols["patient_id"].nullable is False
    assert cols["appointment_id"].nullable is True
    assert cols["content_item_id"].nullable is False
    assert cols["channel"].nullable is False
    assert cols["status"].nullable is False
    assert cols["delivered_at"].nullable is True
    assert cols["opened_at"].nullable is True
    assert cols["completed_at"].nullable is True


# --------------------------------------------------------------------------
# foreign keys (Data shapes tables)
# --------------------------------------------------------------------------


def test_content_delivery_patient_id_fk_targets_patients():
    fks = ContentDelivery.__table__.c.patient_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "patients.id"


def test_content_delivery_appointment_id_fk_targets_appointments():
    fks = ContentDelivery.__table__.c.appointment_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "appointments.id"


def test_content_delivery_content_item_id_fk_targets_content_items():
    fks = ContentDelivery.__table__.c.content_item_id.foreign_keys
    assert len(fks) == 1
    assert next(iter(fks)).target_fullname == "content_items.id"


# --------------------------------------------------------------------------
# no cross-module ORM relationship() at all (Interaction contract)
# --------------------------------------------------------------------------


def test_content_item_declares_no_orm_relationships():
    assert inspect(ContentItem).relationships.keys() == []


def test_content_delivery_declares_no_orm_relationships():
    assert inspect(ContentDelivery).relationships.keys() == []


# --------------------------------------------------------------------------
# content_items behaviour
# --------------------------------------------------------------------------


async def test_content_item_id_is_a_generated_uuid(session):
    item = make_content_item()
    session.add(item)
    await session.flush()
    assert isinstance(item.id, uuid.UUID)


async def test_content_item_missing_developmental_stage_raises_integrity_error(session):
    item = make_content_item(developmental_stage=None)
    session.add(item)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_item_missing_language_raises_integrity_error(session):
    item = make_content_item(language=None)
    session.add(item)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_item_missing_trigger_type_raises_integrity_error(session):
    item = make_content_item(trigger_type=None)
    session.add(item)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_item_missing_body_raises_integrity_error(session):
    item = make_content_item(body=None)
    session.add(item)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_item_language_supports_every_declared_language(session):
    for language in ("en", "es", "pt"):
        session.add(make_content_item(id=uuid.uuid4(), language=language))
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(ContentItem))).scalars().all()
    persisted = {getattr(row.language, "value", row.language) for row in fetched}
    assert persisted == {"en", "es", "pt"}


async def test_content_item_trigger_type_supports_both_declared_values(session):
    for trigger_type in ("pre_appointment", "post_appointment"):
        session.add(make_content_item(id=uuid.uuid4(), trigger_type=trigger_type))
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(ContentItem))).scalars().all()
    persisted = {getattr(row.trigger_type, "value", row.trigger_type) for row in fetched}
    assert persisted == {"pre_appointment", "post_appointment"}


async def test_content_item_body_roundtrips_free_text(session, session_factory):
    body_text = "Avoid sugary snacks before bedtime and schedule a follow-up in 6 months."
    item = make_content_item(body=body_text)
    session.add(item)
    await session.commit()
    item_id = item.id

    # re-query from a fresh session to force a real round trip rather than
    # relying on the identity map
    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(ContentItem, item_id)
        assert reloaded.body == body_text
        assert reloaded.developmental_stage == "child"


# --------------------------------------------------------------------------
# content_deliveries behaviour
# --------------------------------------------------------------------------


async def test_content_delivery_id_is_a_generated_uuid(session):
    delivery = make_content_delivery()
    session.add(delivery)
    await session.flush()
    assert isinstance(delivery.id, uuid.UUID)


async def test_content_delivery_missing_patient_id_raises_integrity_error(session):
    delivery = make_content_delivery(patient_id=None)
    session.add(delivery)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_delivery_missing_content_item_id_raises_integrity_error(session):
    delivery = make_content_delivery(content_item_id=None)
    session.add(delivery)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_delivery_missing_channel_raises_integrity_error(session):
    delivery = make_content_delivery(channel=None)
    session.add(delivery)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_delivery_missing_status_raises_integrity_error(session):
    delivery = make_content_delivery(status=None)
    session.add(delivery)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_content_delivery_appointment_id_nullable_when_not_appointment_linked(session):
    # e.g. a post_appointment reminder that is not tied to a specific
    # appointment row -- appointment_id must not be forced to a value.
    delivery = make_content_delivery(appointment_id=None)
    session.add(delivery)
    await session.commit()  # must not raise
    session.expunge_all()

    fetched = (
        await session.execute(
            select(ContentDelivery).where(ContentDelivery.id == delivery.id)
        )
    ).scalar_one()
    assert fetched.appointment_id is None


async def test_content_delivery_appointment_id_roundtrips_when_linked(session, session_factory):
    appointment_id = uuid.uuid4()
    delivery = make_content_delivery(appointment_id=appointment_id)
    session.add(delivery)
    await session.commit()
    delivery_id = delivery.id

    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(ContentDelivery, delivery_id)
        assert reloaded.appointment_id == appointment_id


async def test_content_delivery_channel_supports_every_declared_channel(session):
    for channel in ("whatsapp", "sms", "email"):
        session.add(make_content_delivery(id=uuid.uuid4(), channel=channel))
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(ContentDelivery))).scalars().all()
    persisted = {getattr(row.channel, "value", row.channel) for row in fetched}
    assert persisted == {"whatsapp", "sms", "email"}


async def test_content_delivery_status_supports_every_declared_status(session):
    for status in ("delivered", "opened", "completed", "unavailable"):
        session.add(make_content_delivery(id=uuid.uuid4(), status=status))
    await session.commit()
    session.expunge_all()

    fetched = (await session.execute(select(ContentDelivery))).scalars().all()
    persisted = {getattr(row.status, "value", row.status) for row in fetched}
    assert persisted == {"delivered", "opened", "completed", "unavailable"}


async def test_content_delivery_delivered_opened_completed_at_default_to_none(session):
    delivery = make_content_delivery()
    session.add(delivery)
    await session.commit()
    await session.refresh(delivery)
    assert delivery.delivered_at is None
    assert delivery.opened_at is None
    assert delivery.completed_at is None


async def test_content_delivery_timestamp_fields_roundtrip_when_populated(
    session, session_factory
):
    delivered_at = _now()
    opened_at = _now()
    completed_at = _now()
    delivery = make_content_delivery(
        status="completed",
        delivered_at=delivered_at,
        opened_at=opened_at,
        completed_at=completed_at,
    )
    session.add(delivery)
    await session.commit()
    delivery_id = delivery.id

    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(ContentDelivery, delivery_id)
        assert reloaded.delivered_at is not None
        assert reloaded.opened_at is not None
        assert reloaded.completed_at is not None


# --------------------------------------------------------------------------
# FR-E10.3: channel technical limitations are never fabricated as "opened"
# --------------------------------------------------------------------------


async def test_content_delivery_sms_channel_can_be_recorded_as_unavailable_not_opened(
    session, session_factory
):
    # FR-E10.3: SMS cannot report opens -- the schema-level support for "never
    # fabricate open data" is that status may be written as "unavailable"
    # (or left at "delivered") instead of being silently promoted to "opened".
    delivery = make_content_delivery(channel="sms", status="unavailable", opened_at=None)
    session.add(delivery)
    await session.commit()
    delivery_id = delivery.id

    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(ContentDelivery, delivery_id)
        assert reloaded.status == "unavailable"
        assert reloaded.opened_at is None


async def test_content_delivery_sms_channel_can_remain_delivered_without_open_tracking(session):
    # A channel that cannot report opens is only ever "delivered" or
    # "unavailable" -- never silently written as "opened" with no evidence.
    delivery = make_content_delivery(channel="sms", status="delivered", opened_at=None)
    session.add(delivery)
    await session.commit()  # must not raise: "delivered" with no opened_at is valid
    await session.refresh(delivery)
    assert delivery.status == "delivered"
    assert delivery.opened_at is None


# --------------------------------------------------------------------------
# two separate tables (Data shapes)
# --------------------------------------------------------------------------


def test_content_items_and_content_deliveries_are_declared_as_separate_tables():
    assert ContentItem.__tablename__ != ContentDelivery.__tablename__
