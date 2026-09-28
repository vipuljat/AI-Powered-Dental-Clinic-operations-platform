"""Unit tests for app/repositories/outreach/repository.py.

Exercises ConsentRepository / OutreachMessageRepository /
MessageTemplateRepository / ChannelConfigurationRepository directly against
an in-memory SQLite database, mirroring the ENVIRONMENT=test substitution
described in project_rules.testing (app/core/database.py switches to
sqlite+aiosqlite:///:memory: with a StaticPool at the composition root, so
the same ORM models run unmodified against both SQLite and Postgres).

Per the unit-test-authoring contract, only this file's own spec is available
here -- app/repositories/outreach/repository.py's own implementation, and
every other task's spec, are deliberately out of scope. consent_records.
patient_id, outreach_messages.patient_id and channel_configurations.
configured_by_staff_id are plain FK columns pointing at patients.id /
staff_users.id (owned by other modules, not part of this task) -- minimal
stand-in tables are registered here purely so Base.metadata.create_all can
compile this module's own DDL, exactly as app/models/outreach/models.py's
own test file does.

ChannelConfigurationRepository.upsert's `access_token_ref` argument is
deliberately never asserted to land on any particular attribute of the
returned ChannelConfiguration: the spec's own Watch out section only
guarantees the raw secret is not persisted in this table, not which column
(if any) receives the reference, so pinning that down here would test an
implementation detail this spec does not commit to.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import Column, Table, Uuid, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.common.constants import OUTREACH_MAX_RETRY_ATTEMPTS
from app.core.database import Base
from app.models.outreach.models import (
    ChannelConfiguration,
    ConsentRecord,
    MessageTemplate,
    OutreachMessage,
)
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    MessageTemplateRepository,
    OutreachMessageRepository,
)

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Stand-in tables for cross-module FK targets (see module docstring).
# ---------------------------------------------------------------------------
if "patients" not in Base.metadata.tables:
    Table("patients", Base.metadata, Column("id", Uuid, primary_key=True))
if "staff_users" not in Base.metadata.tables:
    Table("staff_users", Base.metadata, Column("id", Uuid, primary_key=True))


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
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


@pytest.fixture
def consent_repo() -> ConsentRepository:
    return ConsentRepository()


@pytest.fixture
def message_repo() -> OutreachMessageRepository:
    return OutreachMessageRepository()


@pytest.fixture
def template_repo() -> MessageTemplateRepository:
    return MessageTemplateRepository()


@pytest.fixture
def channel_repo() -> ChannelConfigurationRepository:
    return ChannelConfigurationRepository()


async def seed_consent(session: AsyncSession, **overrides) -> ConsentRecord:
    """Seeds a ConsentRecord directly via the ORM (bypassing
    ConsentRepository.insert) so tests can pin exact effective_at values and
    assert precise ordering from the read-side query methods."""
    now = datetime.now(timezone.utc)
    defaults = dict(
        patient_id=uuid.uuid4(),
        channel="whatsapp",
        status="granted",
        effective_at=now,
        created_at=now,
    )
    defaults.update(overrides)
    record = ConsentRecord(**defaults)
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def seed_template(session: AsyncSession, **overrides) -> MessageTemplate:
    defaults = dict(
        type="appointment_confirmation",
        language="en",
        channel="whatsapp",
        body_template="Hi {{first_name}}, your appointment is confirmed.",
    )
    defaults.update(overrides)
    template = MessageTemplate(**defaults)
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


# ---------------------------------------------------------------------------
# ConsentRepository.insert
# ---------------------------------------------------------------------------


async def test_insert_persists_row_with_given_fields(session, consent_repo):
    patient_id = uuid.uuid4()

    record = await consent_repo.insert(session, patient_id, "whatsapp", "granted", "staff_manual")

    assert record.id is not None
    assert record.patient_id == patient_id
    assert record.channel == "whatsapp"
    assert record.status == "granted"
    assert record.source == "staff_manual"
    assert record.effective_at is not None
    assert record.effective_at.tzinfo is not None
    assert record.created_at is not None

    # Actually persisted, not just held in memory.
    fetched = (
        await session.execute(select(ConsentRecord).where(ConsentRecord.id == record.id))
    ).scalar_one()
    assert fetched.status == "granted"


async def test_insert_accepts_null_source(session, consent_repo):
    record = await consent_repo.insert(session, uuid.uuid4(), "sms", "withdrawn", None)

    assert record.source is None
    assert record.status == "withdrawn"


# ---------------------------------------------------------------------------
# ConsentRepository.get_latest_by_channel
# ---------------------------------------------------------------------------


async def test_get_latest_by_channel_returns_none_when_no_rows(session, consent_repo):
    found = await consent_repo.get_latest_by_channel(session, uuid.uuid4(), "whatsapp")
    assert found is None


async def test_get_latest_by_channel_returns_most_recent_row(session, consent_repo):
    patient_id = uuid.uuid4()
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(days=10)
    t2 = t0 + timedelta(days=20)

    await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="granted", effective_at=t0)
    await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="withdrawn", effective_at=t2)
    await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="declined", effective_at=t1)

    latest = await consent_repo.get_latest_by_channel(session, patient_id, "whatsapp")

    assert latest is not None
    assert latest.status == "withdrawn"
    assert latest.effective_at == t2


async def test_get_latest_by_channel_is_scoped_to_requested_channel_and_patient(session, consent_repo):
    patient_a = uuid.uuid4()
    patient_b = uuid.uuid4()
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(days=1)

    await seed_consent(session, patient_id=patient_a, channel="whatsapp", status="granted", effective_at=t0)
    await seed_consent(session, patient_id=patient_a, channel="sms", status="withdrawn", effective_at=t1)
    await seed_consent(session, patient_id=patient_b, channel="whatsapp", status="withdrawn", effective_at=t1)

    result = await consent_repo.get_latest_by_channel(session, patient_a, "whatsapp")

    assert result is not None
    assert result.patient_id == patient_a
    assert result.channel == "whatsapp"
    assert result.status == "granted"


# ---------------------------------------------------------------------------
# ConsentRepository.list_latest_all_channels
# ---------------------------------------------------------------------------


async def test_list_latest_all_channels_returns_empty_list_when_no_consent_rows(session, consent_repo):
    result = await consent_repo.list_latest_all_channels(session, uuid.uuid4())
    assert result == []


async def test_list_latest_all_channels_returns_one_row_per_channel_at_its_latest_status(
    session, consent_repo
):
    patient_id = uuid.uuid4()
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(days=1)

    # whatsapp: granted then withdrawn -- latest must be withdrawn.
    await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="granted", effective_at=t0)
    await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="withdrawn", effective_at=t1)
    # sms: single row, granted.
    await seed_consent(session, patient_id=patient_id, channel="sms", status="granted", effective_at=t0)

    result = await consent_repo.list_latest_all_channels(session, patient_id)

    by_channel = {r.channel: r.status for r in result}
    assert by_channel == {"whatsapp": "withdrawn", "sms": "granted"}


# ---------------------------------------------------------------------------
# ConsentRepository.list_ledger / FR-E6.6 append-only
# ---------------------------------------------------------------------------


async def test_list_ledger_returns_full_chronological_history_ascending(session, consent_repo):
    patient_id = uuid.uuid4()
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(days=5)
    t2 = t0 + timedelta(days=10)

    r0 = await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="granted", effective_at=t0)
    r2 = await seed_consent(session, patient_id=patient_id, channel="sms", status="granted", effective_at=t2)
    r1 = await seed_consent(session, patient_id=patient_id, channel="whatsapp", status="withdrawn", effective_at=t1)

    ledger = await consent_repo.list_ledger(session, patient_id)

    assert [row.id for row in ledger] == [r0.id, r1.id, r2.id]


async def test_list_ledger_excludes_other_patients_rows(session, consent_repo):
    patient_a = uuid.uuid4()
    patient_b = uuid.uuid4()

    await seed_consent(session, patient_id=patient_a, status="granted")
    await seed_consent(session, patient_id=patient_b, status="granted")

    ledger = await consent_repo.list_ledger(session, patient_a)

    assert len(ledger) == 1
    assert ledger[0].patient_id == patient_a


def test_consent_repository_exposes_no_update_or_delete_method():
    """FR-E6.6: the append-only guarantee is enforced by this class exposing
    no update/delete on ConsentRecord."""
    assert not hasattr(ConsentRepository, "update")
    assert not hasattr(ConsentRepository, "delete")


# ---------------------------------------------------------------------------
# OutreachMessageRepository.create / get_by_id
# ---------------------------------------------------------------------------


async def test_create_persists_and_returns_message_with_default_status(session, message_repo):
    patient_id = uuid.uuid4()

    message = await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="confirmation",
        channel="sms",
        language="en",
    )

    assert message.id is not None
    assert message.patient_id == patient_id
    assert message.campaign_type == "confirmation"
    assert message.channel == "sms"
    assert message.language == "en"
    # Model default applies when the caller doesn't set status explicitly.
    assert message.status == "queued"


async def test_get_by_id_returns_matching_message(session, message_repo):
    created = await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="recall",
        channel="email",
        language="es",
    )

    found = await message_repo.get_by_id(session, created.id)

    assert found is not None
    assert found.id == created.id
    assert found.campaign_type == "recall"


async def test_get_by_id_returns_none_when_missing(session, message_repo):
    found = await message_repo.get_by_id(session, uuid.uuid4())
    assert found is None


# ---------------------------------------------------------------------------
# OutreachMessageRepository.update_status
# ---------------------------------------------------------------------------


async def test_update_status_updates_status_and_extra_fields(session, message_repo):
    created = await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="confirmation",
        channel="whatsapp",
        language="en",
    )
    sent_at = datetime.now(timezone.utc)

    updated = await message_repo.update_status(session, created.id, "sent", sent_at=sent_at)

    assert updated.id == created.id
    assert updated.status == "sent"
    assert updated.sent_at == sent_at

    refetched = await message_repo.get_by_id(session, created.id)
    assert refetched.status == "sent"


# ---------------------------------------------------------------------------
# OutreachMessageRepository.list_retry_due
# ---------------------------------------------------------------------------


async def test_list_retry_due_includes_only_failed_messages_due_and_under_max_retries(
    session, message_repo
):
    now = datetime.now(timezone.utc)
    past = now - timedelta(minutes=5)
    future = now + timedelta(hours=1)

    due_and_eligible = await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="waitlist_offer",
        channel="sms",
        language="en",
        status="failed",
        next_retry_at=past,
        retry_count=OUTREACH_MAX_RETRY_ATTEMPTS - 1,
    )
    await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="waitlist_offer",
        channel="sms",
        language="en",
        status="failed",
        next_retry_at=future,
        retry_count=0,
    )
    await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="waitlist_offer",
        channel="sms",
        language="en",
        status="failed",
        next_retry_at=past,
        retry_count=OUTREACH_MAX_RETRY_ATTEMPTS,
    )
    await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="waitlist_offer",
        channel="sms",
        language="en",
        status="queued",
        next_retry_at=past,
        retry_count=0,
    )

    due = await message_repo.list_retry_due(session, now)

    assert [m.id for m in due] == [due_and_eligible.id]


async def test_list_retry_due_returns_empty_list_when_nothing_is_due(session, message_repo):
    now = datetime.now(timezone.utc)
    due = await message_repo.list_retry_due(session, now)
    assert due == []


# ---------------------------------------------------------------------------
# OutreachMessageRepository.list_queued_for_patient_channel / FR-E6.5
# ---------------------------------------------------------------------------


async def test_list_queued_for_patient_channel_includes_queued_and_failed_only(
    session, message_repo
):
    patient_id = uuid.uuid4()

    queued = await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="whatsapp",
        language="en",
        status="queued",
    )
    failed = await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="whatsapp",
        language="en",
        status="failed",
    )
    await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="whatsapp",
        language="en",
        status="delivered",
    )
    await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="whatsapp",
        language="en",
        status="sent",
    )

    result = await message_repo.list_queued_for_patient_channel(session, patient_id, "whatsapp")

    assert {m.id for m in result} == {queued.id, failed.id}


async def test_list_queued_for_patient_channel_is_scoped_to_patient_and_channel(
    session, message_repo
):
    patient_id = uuid.uuid4()

    # Same patient, different channel -- must not be returned for "sms".
    await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="whatsapp",
        language="en",
        status="queued",
    )
    # Different patient, same channel -- must not be returned either.
    await message_repo.create(
        session,
        patient_id=uuid.uuid4(),
        campaign_type="recall",
        channel="sms",
        language="en",
        status="queued",
    )

    result = await message_repo.list_queued_for_patient_channel(session, patient_id, "sms")

    assert result == []


# ---------------------------------------------------------------------------
# OutreachMessageRepository.list_by_patient
# ---------------------------------------------------------------------------


async def test_list_by_patient_with_no_filters_returns_all_that_patients_messages(
    session, message_repo
):
    patient_id = uuid.uuid4()

    a = await message_repo.create(
        session, patient_id=patient_id, campaign_type="recall", channel="sms", language="en"
    )
    b = await message_repo.create(
        session, patient_id=patient_id, campaign_type="education", channel="email", language="en"
    )
    await message_repo.create(
        session, patient_id=uuid.uuid4(), campaign_type="recall", channel="sms", language="en"
    )

    result = await message_repo.list_by_patient(session, patient_id, None, None)

    assert {m.id for m in result} == {a.id, b.id}


async def test_list_by_patient_filters_by_channel(session, message_repo):
    patient_id = uuid.uuid4()

    sms_message = await message_repo.create(
        session, patient_id=patient_id, campaign_type="recall", channel="sms", language="en"
    )
    await message_repo.create(
        session, patient_id=patient_id, campaign_type="recall", channel="email", language="en"
    )

    result = await message_repo.list_by_patient(session, patient_id, "sms", None)

    assert [m.id for m in result] == [sms_message.id]


async def test_list_by_patient_filters_by_status(session, message_repo):
    patient_id = uuid.uuid4()

    delivered = await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="sms",
        language="en",
        status="delivered",
    )
    await message_repo.create(
        session,
        patient_id=patient_id,
        campaign_type="recall",
        channel="sms",
        language="en",
        status="queued",
    )

    result = await message_repo.list_by_patient(session, patient_id, None, "delivered")

    assert [m.id for m in result] == [delivered.id]


async def test_list_by_patient_returns_empty_list_when_no_messages(session, message_repo):
    result = await message_repo.list_by_patient(session, uuid.uuid4(), None, None)
    assert result == []


# ---------------------------------------------------------------------------
# MessageTemplateRepository.get_by_type_language_channel
# ---------------------------------------------------------------------------


async def test_get_by_type_language_channel_returns_exact_match(session, template_repo):
    template = await seed_template(
        session, type="waitlist_offer", language="pt", channel="sms"
    )

    found = await template_repo.get_by_type_language_channel(session, "waitlist_offer", "pt", "sms")

    assert found is not None
    assert found.id == template.id
    assert found.body_template == template.body_template


async def test_get_by_type_language_channel_returns_none_when_no_exact_match(session, template_repo):
    await seed_template(session, type="waitlist_offer", language="pt", channel="sms")

    # Same type/channel, different language -- must not match.
    assert await template_repo.get_by_type_language_channel(session, "waitlist_offer", "en", "sms") is None
    # Same type/language, different channel -- must not match.
    assert await template_repo.get_by_type_language_channel(session, "waitlist_offer", "pt", "email") is None
    # Different type entirely -- must not match.
    assert await template_repo.get_by_type_language_channel(session, "recall", "pt", "sms") is None


# ---------------------------------------------------------------------------
# ChannelConfigurationRepository.upsert / get_status / update_status
# ---------------------------------------------------------------------------


async def test_upsert_creates_new_row_when_none_exists_for_channel(session, channel_repo):
    staff_id = uuid.uuid4()

    config = await channel_repo.upsert(session, "whatsapp", "meta_bsp", "ref-123", staff_id)

    assert config.id is not None
    assert config.channel == "whatsapp"
    assert config.provider == "meta_bsp"
    assert config.configured_by_staff_id == staff_id


async def test_upsert_called_again_for_same_channel_updates_rather_than_duplicates(
    session, channel_repo
):
    staff_id = uuid.uuid4()

    await channel_repo.upsert(session, "whatsapp", "meta_bsp_old", "ref-1", staff_id)
    updated = await channel_repo.upsert(session, "whatsapp", "meta_bsp_new", "ref-2", staff_id)

    assert updated.provider == "meta_bsp_new"

    rows = (
        await session.execute(
            select(ChannelConfiguration).where(ChannelConfiguration.channel == "whatsapp")
        )
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].provider == "meta_bsp_new"


async def test_get_status_returns_none_when_channel_never_configured(session, channel_repo):
    found = await channel_repo.get_status(session, "email")
    assert found is None


async def test_get_status_returns_existing_configuration(session, channel_repo):
    staff_id = uuid.uuid4()
    await channel_repo.upsert(session, "sms", "twilio", "ref-9", staff_id)

    found = await channel_repo.get_status(session, "sms")

    assert found is not None
    assert found.channel == "sms"
    assert found.provider == "twilio"


async def test_update_status_changes_status_field(session, channel_repo):
    staff_id = uuid.uuid4()
    created = await channel_repo.upsert(session, "voice", "twilio_voice", "ref-42", staff_id)
    assert created.status == "pending"

    updated = await channel_repo.update_status(session, "voice", "verified")

    assert updated.status == "verified"
    refetched = await channel_repo.get_status(session, "voice")
    assert refetched.status == "verified"


def test_channel_configuration_model_has_no_raw_access_token_column():
    """Watch out: the raw access_token secret is never persisted in this
    table -- there is no column on the ORM model that could hold it."""
    assert "access_token" not in ChannelConfiguration.__table__.c
