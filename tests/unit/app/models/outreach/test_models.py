"""Unit tests for app/models/outreach/models.py.

Exercises the four ORM tables declared here (consent_records,
outreach_messages, message_templates, channel_configurations) directly
against an in-memory SQLite database, mirroring the ENVIRONMENT=test
substitution described in project_rules.testing (app/core/database.py
switches to sqlite+aiosqlite:///:memory: with a StaticPool at the
composition root).

consent_records.patient_id, outreach_messages.patient_id and
channel_configurations.configured_by_staff_id are plain FK columns pointing
at patients.id / staff_users.id, both owned by other modules' models.py
(patients and console respectively -- neither is part of this task per the
spec's own Interaction contract). Every ORM model shares one Base.metadata
(per project_rules.testing), so those FK targets must be resolvable for
Base.metadata.create_all to compile this file's own tables' DDL. Registering
minimal stand-in tables here -- rather than importing the other modules'
ORM classes -- keeps this test scoped to this file's own spec; the guard is
a no-op if the real patients/console models already registered the name on
the shared metadata (e.g. when this file runs alongside the full suite).

outreach_messages.template_id is a real in-module FK to message_templates.id
(both declared in the module under test), so no stand-in is needed for it.
outreach_messages.related_entity_type/related_entity_id is a polymorphic
pair with no DB-level FK at all (per this file's own Watch out section), so
no stand-in is registered for "appointments"/"waitlist_entries"/etc. either
-- a row referencing a table name that isn't even registered on the shared
metadata must still insert cleanly, which is exactly what proves there is no
FK there.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy import Column, Table, Uuid, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.outreach.models import (
    ChannelConfiguration,
    ConsentRecord,
    MessageTemplate,
    OutreachMessage,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Stand-in tables for cross-module FK targets (see module docstring).
# ---------------------------------------------------------------------------
if "patients" not in Base.metadata.tables:
    Table("patients", Base.metadata, Column("id", Uuid, primary_key=True))
if "staff_users" not in Base.metadata.tables:
    Table("staff_users", Base.metadata, Column("id", Uuid, primary_key=True))


def _col(model, name):
    return model.__table__.c[name]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


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
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


def make_consent_record(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        patient_id=uuid.uuid4(),
        channel="whatsapp",
        status="granted",
        effective_at=now,
        created_at=now,
    )
    defaults.update(overrides)
    return ConsentRecord(**defaults)


def make_outreach_message(**overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        patient_id=uuid.uuid4(),
        campaign_type="confirmation",
        channel="sms",
        language="en",
        created_at=now,
    )
    defaults.update(overrides)
    return OutreachMessage(**defaults)


def make_message_template(**overrides):
    defaults = dict(
        type="appointment_confirmation",
        language="en",
        channel="whatsapp",
        body_template="Hi {{first_name}}, your appointment is confirmed.",
    )
    defaults.update(overrides)
    return MessageTemplate(**defaults)


def make_channel_configuration(**overrides):
    defaults = dict(
        channel="whatsapp",
        configured_by_staff_id=uuid.uuid4(),
    )
    defaults.update(overrides)
    return ChannelConfiguration(**defaults)


# ---------------------------------------------------------------------------
# table identity
# ---------------------------------------------------------------------------


def test_table_names():
    assert ConsentRecord.__tablename__ == "consent_records"
    assert OutreachMessage.__tablename__ == "outreach_messages"
    assert MessageTemplate.__tablename__ == "message_templates"
    assert ChannelConfiguration.__tablename__ == "channel_configurations"


def test_all_four_models_are_declarative_base_subclasses():
    assert issubclass(ConsentRecord, Base)
    assert issubclass(OutreachMessage, Base)
    assert issubclass(MessageTemplate, Base)
    assert issubclass(ChannelConfiguration, Base)


# ---------------------------------------------------------------------------
# consent_records -- structural shape
# ---------------------------------------------------------------------------


def test_consent_record_primary_key_is_id():
    assert [c.name for c in ConsentRecord.__table__.primary_key.columns] == ["id"]


def test_consent_record_patient_id_is_not_null_fk_to_patients():
    col = _col(ConsentRecord, "patient_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "patients.id"


def test_consent_record_channel_is_not_null_enum_of_expected_values():
    col = _col(ConsentRecord, "channel")
    assert col.nullable is False
    assert set(col.type.enums) == {"whatsapp", "sms", "email", "voice", "webchat"}


def test_consent_record_status_is_not_null_enum_of_expected_values():
    col = _col(ConsentRecord, "status")
    assert col.nullable is False
    assert set(col.type.enums) == {"granted", "withdrawn", "declined"}


def test_consent_record_source_is_nullable_string():
    col = _col(ConsentRecord, "source")
    assert col.nullable is True
    assert isinstance(col.type, sa.String)


def test_consent_record_effective_at_and_created_at_are_not_null_timezone_aware():
    for name in ("effective_at", "created_at"):
        col = _col(ConsentRecord, name)
        assert col.nullable is False
        assert isinstance(col.type, sa.DateTime)
        assert col.type.timezone is True


# ---------------------------------------------------------------------------
# consent_records -- required-field enforcement
# ---------------------------------------------------------------------------


async def test_consent_record_missing_patient_id_raises_integrity_error(session):
    session.add(make_consent_record(patient_id=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_consent_record_missing_channel_raises_integrity_error(session):
    session.add(make_consent_record(channel=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_consent_record_missing_status_raises_integrity_error(session):
    session.add(make_consent_record(status=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_consent_record_missing_effective_at_raises_integrity_error(session):
    session.add(make_consent_record(effective_at=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_consent_record_source_may_be_omitted(session):
    record = make_consent_record()
    session.add(record)
    await session.commit()
    await session.refresh(record)
    assert record.source is None


# ---------------------------------------------------------------------------
# FR-E6.6 / US32: consent_records is append-only -- every grant/withdraw/
# decline is a NEW row, never an UPDATE, so the full per-channel history for
# a patient is reconstructable.
# ---------------------------------------------------------------------------


def test_consent_record_has_no_unique_constraint_on_patient_and_channel():
    target = {"patient_id", "channel"}
    for constraint in ConsentRecord.__table__.constraints:
        if isinstance(constraint, sa.UniqueConstraint):
            assert {c.name for c in constraint.columns} != target
    for index in ConsentRecord.__table__.indexes:
        if index.unique:
            assert {c.name for c in index.columns} != target


async def test_consent_record_state_change_inserts_a_new_row_instead_of_updating(session):
    patient_id = uuid.uuid4()
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(days=30)

    granted = make_consent_record(
        patient_id=patient_id,
        channel="whatsapp",
        status="granted",
        source="staff_manual",
        effective_at=t0,
        created_at=t0,
    )
    session.add(granted)
    await session.commit()

    withdrawn = make_consent_record(
        patient_id=patient_id,
        channel="whatsapp",
        status="withdrawn",
        source="keyword_stop",
        effective_at=t1,
        created_at=t1,
    )
    session.add(withdrawn)
    await session.commit()

    rows = (
        await session.execute(
            select(ConsentRecord)
            .where(
                ConsentRecord.patient_id == patient_id,
                ConsentRecord.channel == "whatsapp",
            )
            .order_by(ConsentRecord.effective_at)
        )
    ).scalars().all()

    # Both the original grant and the later withdrawal survive as distinct
    # rows -- the first was never overwritten -- giving a reconstructable
    # chronological history.
    assert len(rows) == 2
    assert [r.status for r in rows] == ["granted", "withdrawn"]
    assert rows[0].id != rows[1].id


# ---------------------------------------------------------------------------
# outreach_messages -- structural shape
# ---------------------------------------------------------------------------


def test_outreach_message_primary_key_is_id():
    assert [c.name for c in OutreachMessage.__table__.primary_key.columns] == ["id"]


def test_outreach_message_patient_id_is_not_null_fk_to_patients():
    col = _col(OutreachMessage, "patient_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "patients.id"


def test_outreach_message_campaign_type_is_not_null_enum_of_expected_values():
    col = _col(OutreachMessage, "campaign_type")
    assert col.nullable is False
    assert set(col.type.enums) == {
        "confirmation",
        "waitlist_offer",
        "recall",
        "treatment_reengagement",
        "education",
    }


def test_outreach_message_channel_is_not_null_enum_of_expected_values():
    col = _col(OutreachMessage, "channel")
    assert col.nullable is False
    assert set(col.type.enums) == {"whatsapp", "sms", "email", "voice", "webchat"}


def test_outreach_message_language_is_not_null_enum_of_expected_values():
    col = _col(OutreachMessage, "language")
    assert col.nullable is False
    assert set(col.type.enums) == {"en", "es", "pt"}


def test_outreach_message_template_id_is_nullable_fk_to_message_templates():
    col = _col(OutreachMessage, "template_id")
    assert col.nullable is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "message_templates.id"


def test_outreach_message_status_is_not_null_enum_of_expected_values():
    col = _col(OutreachMessage, "status")
    assert col.nullable is False
    assert set(col.type.enums) == {
        "queued",
        "sent",
        "delivered",
        "failed",
        "suppressed",
        "unreachable",
    }


def test_outreach_message_retry_count_is_not_null_integer():
    col = _col(OutreachMessage, "retry_count")
    assert col.nullable is False
    assert isinstance(col.type, sa.Integer)


def test_outreach_message_next_retry_at_and_sent_at_are_nullable_timezone_aware():
    for name in ("next_retry_at", "sent_at"):
        col = _col(OutreachMessage, name)
        assert col.nullable is True
        assert isinstance(col.type, sa.DateTime)
        assert col.type.timezone is True


def test_outreach_message_created_at_is_not_null_timezone_aware():
    col = _col(OutreachMessage, "created_at")
    assert col.nullable is False
    assert isinstance(col.type, sa.DateTime)
    assert col.type.timezone is True


def test_outreach_message_related_entity_columns_are_nullable_with_no_fk():
    type_col = _col(OutreachMessage, "related_entity_type")
    id_col = _col(OutreachMessage, "related_entity_id")
    assert type_col.nullable is True
    assert isinstance(type_col.type, sa.String)
    assert id_col.nullable is True
    assert len(id_col.foreign_keys) == 0


# ---------------------------------------------------------------------------
# outreach_messages -- required-field enforcement and defaults
# ---------------------------------------------------------------------------


async def test_outreach_message_missing_patient_id_raises_integrity_error(session):
    session.add(make_outreach_message(patient_id=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_outreach_message_missing_campaign_type_raises_integrity_error(session):
    session.add(make_outreach_message(campaign_type=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_outreach_message_missing_channel_raises_integrity_error(session):
    session.add(make_outreach_message(channel=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_outreach_message_missing_language_raises_integrity_error(session):
    session.add(make_outreach_message(language=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_outreach_message_status_defaults_to_queued_when_omitted(session):
    message = make_outreach_message()
    session.add(message)
    await session.commit()
    await session.refresh(message)
    assert message.status == "queued"


async def test_outreach_message_retry_count_defaults_to_zero_when_omitted(session):
    message = make_outreach_message()
    session.add(message)
    await session.commit()
    await session.refresh(message)
    assert message.retry_count == 0


async def test_outreach_message_template_id_may_be_omitted(session):
    message = make_outreach_message()
    session.add(message)
    await session.commit()
    await session.refresh(message)
    assert message.template_id is None


# ---------------------------------------------------------------------------
# Watch out: related_entity_type/related_entity_id is polymorphic exactly
# like audit_log.entity_type/entity_id -- string table name + UUID, no
# DB-level FK -- so a row can point at a table name that does not even
# exist on the shared metadata and still insert cleanly.
# ---------------------------------------------------------------------------


async def test_outreach_message_related_entity_accepts_any_table_name_with_no_fk_check(
    session,
):
    assert "appointments" not in Base.metadata.tables
    message = make_outreach_message(
        related_entity_type="appointment",
        related_entity_id=uuid.uuid4(),
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    assert message.related_entity_type == "appointment"
    assert isinstance(message.related_entity_id, uuid.UUID)


# ---------------------------------------------------------------------------
# message_templates -- structural shape
# ---------------------------------------------------------------------------


def test_message_template_primary_key_is_id():
    assert [c.name for c in MessageTemplate.__table__.primary_key.columns] == ["id"]


def test_message_template_type_is_not_null_string():
    col = _col(MessageTemplate, "type")
    assert col.nullable is False
    assert isinstance(col.type, sa.String)


def test_message_template_language_is_not_null_enum_of_expected_values():
    col = _col(MessageTemplate, "language")
    assert col.nullable is False
    assert set(col.type.enums) == {"en", "es", "pt"}


def test_message_template_channel_is_not_null_enum_of_expected_values():
    col = _col(MessageTemplate, "channel")
    assert col.nullable is False
    assert set(col.type.enums) == {"whatsapp", "sms", "email", "voice", "webchat"}


def test_message_template_body_template_is_not_null_text():
    col = _col(MessageTemplate, "body_template")
    assert col.nullable is False
    assert isinstance(col.type, sa.Text)


async def test_message_template_missing_type_raises_integrity_error(session):
    session.add(make_message_template(type=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_message_template_missing_language_raises_integrity_error(session):
    session.add(make_message_template(language=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_message_template_missing_channel_raises_integrity_error(session):
    session.add(make_message_template(channel=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_message_template_missing_body_template_raises_integrity_error(session):
    session.add(make_message_template(body_template=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_message_template_roundtrips_body_template_text(session):
    template = make_message_template(
        body_template="Hi {{first_name}}, we have an opening on {{date}}."
    )
    session.add(template)
    await session.commit()
    await session.refresh(template)
    assert template.body_template == "Hi {{first_name}}, we have an opening on {{date}}."


# ---------------------------------------------------------------------------
# channel_configurations -- structural shape
# ---------------------------------------------------------------------------


def test_channel_configuration_primary_key_is_id():
    assert [c.name for c in ChannelConfiguration.__table__.primary_key.columns] == ["id"]


def test_channel_configuration_channel_is_not_null_enum_of_expected_values():
    col = _col(ChannelConfiguration, "channel")
    assert col.nullable is False
    # Note: unlike every other channel enum in this module, voice channel
    # configuration has no webchat member -- webchat has no provider/channel
    # config row of its own.
    assert set(col.type.enums) == {"whatsapp", "sms", "email", "voice"}


def test_channel_configuration_provider_is_nullable_string():
    col = _col(ChannelConfiguration, "provider")
    assert col.nullable is True
    assert isinstance(col.type, sa.String)


def test_channel_configuration_status_is_not_null_enum_of_expected_values():
    col = _col(ChannelConfiguration, "status")
    assert col.nullable is False
    assert set(col.type.enums) == {"verified", "pending"}


def test_channel_configuration_verified_at_is_nullable_timezone_aware():
    col = _col(ChannelConfiguration, "verified_at")
    assert col.nullable is True
    assert isinstance(col.type, sa.DateTime)
    assert col.type.timezone is True


def test_channel_configuration_configured_by_staff_id_is_not_null_fk_to_staff_users():
    col = _col(ChannelConfiguration, "configured_by_staff_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "staff_users.id"


async def test_channel_configuration_missing_channel_raises_integrity_error(session):
    session.add(make_channel_configuration(channel=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_channel_configuration_missing_configured_by_staff_id_raises_integrity_error(
    session,
):
    session.add(make_channel_configuration(configured_by_staff_id=None))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_channel_configuration_status_defaults_to_pending_when_omitted(session):
    config = make_channel_configuration()
    session.add(config)
    await session.commit()
    await session.refresh(config)
    assert config.status == "pending"


async def test_channel_configuration_provider_and_verified_at_may_be_omitted(session):
    config = make_channel_configuration()
    session.add(config)
    await session.commit()
    await session.refresh(config)
    assert config.provider is None
    assert config.verified_at is None


# ---------------------------------------------------------------------------
# Interaction contract: plain FK columns only, no cross-module (or in-module)
# ORM relationship() attributes declared on any of the four classes -- this
# module's tables are reached only via services/outreach/service.py, never
# a directly-navigated ORM relationship from another module's model.
# ---------------------------------------------------------------------------


def test_no_orm_relationships_are_declared_on_any_outreach_model():
    for model in (ConsentRecord, OutreachMessage, MessageTemplate, ChannelConfiguration):
        assert len(inspect(model).relationships) == 0
