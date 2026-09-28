"""Unit tests for app/models/console/models.py.

Exercises the five ORM tables declared here (staff_users, role_permissions,
password_reset_tokens, sessions, audit_log) directly against an in-memory
SQLite database, mirroring the ENVIRONMENT=test substitution described in
project_rules.testing (app/core/database.py switches the DB URL to
sqlite+aiosqlite:///:memory: at the composition root; the same ORM models run
unmodified against both SQLite and Postgres per app/core/db_types.py).

All five tables are declared in this one file and self-referential /
intra-module FKs (password_reset_tokens.user_id -> staff_users.id,
sessions.user_id -> staff_users.id, audit_log.actor_staff_id -> staff_users.id,
audit_log.overridden_of_id -> audit_log.id) resolve against the same
Base.metadata without needing any cross-module stand-in table.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.console.models import (
    AuditLog,
    PasswordResetToken,
    RolePermission,
    Session,
    StaffUser,
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
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def session(session_factory) -> AsyncSession:
    async with session_factory() as s:
        yield s


def make_staff_user(**overrides):
    unique = uuid.uuid4().hex[:10]
    defaults = dict(
        username=f"user_{unique}",
        email=f"user_{unique}@clinic.example",
        password_hash="$2b$12$abcdefghijklmnopqrstuv",
        role="front_office_staff",
    )
    defaults.update(overrides)
    return StaffUser(**defaults)


def make_role_permission(**overrides):
    defaults = dict(
        role="delivery_team",
        resource="rules.rule_sets",
        action="write",
        allowed=True,
    )
    defaults.update(overrides)
    return RolePermission(**defaults)


def make_reset_token(user_id, **overrides):
    defaults = dict(
        user_id=user_id,
        token_hash="hashed-token-value",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    defaults.update(overrides)
    return PasswordResetToken(**defaults)


def make_session_row(user_id, **overrides):
    now = datetime.now(timezone.utc)
    defaults = dict(
        user_id=user_id,
        jti=f"jti-{uuid.uuid4()}",
        issued_at=now,
        expires_at=now + timedelta(minutes=15),
    )
    defaults.update(overrides)
    return Session(**defaults)


def make_audit_log(**overrides):
    defaults = dict(
        actor_type="staff",
        action_type="appointment.cancel",
        entity_type="appointment",
        entity_id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return AuditLog(**defaults)


# --------------------------------------------------------------------------
# table identity
# --------------------------------------------------------------------------


def test_table_names():
    assert StaffUser.__tablename__ == "staff_users"
    assert RolePermission.__tablename__ == "role_permissions"
    assert PasswordResetToken.__tablename__ == "password_reset_tokens"
    assert Session.__tablename__ == "sessions"
    assert AuditLog.__tablename__ == "audit_log"


async def test_all_five_tables_generate_a_uuid_id_by_default(session):
    staff = make_staff_user()
    session.add(staff)
    await session.flush()

    perm = make_role_permission()
    reset_token = make_reset_token(user_id=staff.id)
    sess_row = make_session_row(user_id=staff.id)
    audit = make_audit_log()
    session.add_all([perm, reset_token, sess_row, audit])
    await session.flush()

    for obj in (staff, perm, reset_token, sess_row, audit):
        assert isinstance(obj.id, uuid.UUID)


# --------------------------------------------------------------------------
# staff_users
# --------------------------------------------------------------------------


async def test_staff_user_username_required(session):
    staff = StaffUser(
        email="missing.username@clinic.example",
        password_hash="hash",
        role="front_office_staff",
    )
    session.add(staff)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_staff_user_email_required(session):
    staff = StaffUser(username="no_email_user", password_hash="hash", role="front_office_staff")
    session.add(staff)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_staff_user_password_hash_required(session):
    staff = StaffUser(
        username="no_hash_user", email="no_hash@clinic.example", role="front_office_staff"
    )
    session.add(staff)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_staff_user_role_required(session):
    staff = StaffUser(
        username="no_role_user", email="no_role@clinic.example", password_hash="hash"
    )
    session.add(staff)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_staff_user_username_uniqueness_enforced(session):
    session.add(make_staff_user(username="dup_name", email="a@clinic.example"))
    await session.commit()

    session.add(make_staff_user(username="dup_name", email="b@clinic.example"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_staff_user_email_uniqueness_enforced(session):
    session.add(make_staff_user(username="user_a", email="dup@clinic.example"))
    await session.commit()

    session.add(make_staff_user(username="user_b", email="dup@clinic.example"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_staff_user_status_defaults_to_active(session):
    # FR-E1.7 / US6
    staff = make_staff_user()
    session.add(staff)
    await session.commit()
    await session.refresh(staff)
    assert staff.status == "active"


async def test_staff_user_failed_attempt_count_defaults_to_zero(session):
    # US1 Alternate Flow / open_questions[Q-d84091fb]: lockout counter starts at 0
    staff = make_staff_user()
    session.add(staff)
    await session.commit()
    await session.refresh(staff)
    assert staff.failed_attempt_count == 0


async def test_staff_user_locked_until_and_last_login_default_to_none(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()
    await session.refresh(staff)
    assert staff.locked_until is None
    assert staff.last_login_at is None


async def test_staff_user_supports_failed_attempt_count_increment_and_lock(session):
    # US1 Alternate Flow: lockout state is stored, not derived
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    lock_until = datetime.now(timezone.utc) + timedelta(minutes=15)
    staff.failed_attempt_count = 5
    staff.locked_until = lock_until
    await session.commit()
    await session.refresh(staff)
    assert staff.failed_attempt_count == 5
    assert staff.locked_until is not None


@pytest.mark.parametrize("role", ["front_office_staff", "clinic_management", "delivery_team"])
async def test_staff_user_role_accepts_each_documented_value(session, role):
    staff = make_staff_user(role=role)
    session.add(staff)
    await session.commit()
    await session.refresh(staff)
    assert staff.role == role


async def test_staff_user_status_accepts_inactive(session):
    staff = make_staff_user(status="inactive")
    session.add(staff)
    await session.commit()
    await session.refresh(staff)
    assert staff.status == "inactive"


async def test_staff_user_created_and_updated_at_populated(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()
    await session.refresh(staff)
    assert staff.created_at is not None
    assert staff.updated_at is not None


# --------------------------------------------------------------------------
# role_permissions
# --------------------------------------------------------------------------


async def test_role_permission_role_required(session):
    perm = RolePermission(resource="rules.rule_sets", action="write", allowed=True)
    session.add(perm)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_role_permission_resource_required(session):
    perm = RolePermission(role="delivery_team", action="write", allowed=True)
    session.add(perm)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_role_permission_action_required(session):
    perm = RolePermission(role="delivery_team", resource="rules.rule_sets", allowed=True)
    session.add(perm)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_role_permission_allowed_required(session):
    perm = RolePermission(role="delivery_team", resource="rules.rule_sets", action="write")
    session.add(perm)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_role_permission_allowed_boolean_roundtrips_true_and_false(session):
    granted = make_role_permission(resource="scheduling.appointments", action="write", allowed=True)
    denied = make_role_permission(resource="console.audit_log", action="read", allowed=False)
    session.add_all([granted, denied])
    await session.commit()
    await session.refresh(granted)
    await session.refresh(denied)
    assert granted.allowed is True
    assert denied.allowed is False


async def test_role_permission_fields_roundtrip(session):
    perm = make_role_permission(
        role="clinic_management", resource="console.staff_users", action="write", allowed=True
    )
    session.add(perm)
    await session.commit()
    await session.refresh(perm)
    assert perm.role == "clinic_management"
    assert perm.resource == "console.staff_users"
    assert perm.action == "write"


# --------------------------------------------------------------------------
# password_reset_tokens
# --------------------------------------------------------------------------


async def test_reset_token_user_id_required(session):
    token = PasswordResetToken(
        token_hash="hash", expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    session.add(token)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_reset_token_token_hash_required(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    token = PasswordResetToken(
        user_id=staff.id, expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    session.add(token)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_reset_token_expires_at_required(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    token = PasswordResetToken(user_id=staff.id, token_hash="hash")
    session.add(token)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_reset_token_used_at_defaults_to_none(session):
    # single-use: unused tokens have no used_at until consumed
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    token = make_reset_token(user_id=staff.id)
    session.add(token)
    await session.commit()
    await session.refresh(token)
    assert token.used_at is None


async def test_reset_token_used_at_can_be_set_once_consumed(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    token = make_reset_token(user_id=staff.id)
    session.add(token)
    await session.commit()

    used_time = datetime.now(timezone.utc)
    token.used_at = used_time
    await session.commit()
    await session.refresh(token)
    assert token.used_at is not None


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


async def test_session_user_id_required(session):
    now = datetime.now(timezone.utc)
    sess_row = Session(jti="jti-x", issued_at=now, expires_at=now + timedelta(minutes=15))
    session.add(sess_row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_session_jti_required(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    now = datetime.now(timezone.utc)
    sess_row = Session(user_id=staff.id, issued_at=now, expires_at=now + timedelta(minutes=15))
    session.add(sess_row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_session_issued_at_required(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    sess_row = Session(
        user_id=staff.id,
        jti="jti-y",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
    )
    session.add(sess_row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_session_expires_at_required(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    sess_row = Session(user_id=staff.id, jti="jti-z", issued_at=datetime.now(timezone.utc))
    session.add(sess_row)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_session_jti_uniqueness_enforced(session):
    # get_current_user (app/core/dependencies.py) looks sessions up by jti for
    # revocation checks; a duplicate jti must be a DB-level IntegrityError so
    # a concurrent session-issuance race can never insert two rows for one jti.
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    session.add(make_session_row(user_id=staff.id, jti="shared-jti"))
    await session.commit()

    session.add(make_session_row(user_id=staff.id, jti="shared-jti"))
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_session_revoked_at_defaults_to_none(session):
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    sess_row = make_session_row(user_id=staff.id)
    session.add(sess_row)
    await session.commit()
    await session.refresh(sess_row)
    assert sess_row.revoked_at is None


async def test_session_can_be_force_revoked(session):
    # force-logout: get_current_user checks sessions.revoked_at
    staff = make_staff_user()
    session.add(staff)
    await session.commit()

    sess_row = make_session_row(user_id=staff.id)
    session.add(sess_row)
    await session.commit()

    sess_row.revoked_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(sess_row)
    assert sess_row.revoked_at is not None


# --------------------------------------------------------------------------
# audit_log
# --------------------------------------------------------------------------


async def test_audit_log_actor_type_required(session):
    entry = AuditLog(
        action_type="appointment.cancel",
        entity_type="appointment",
        entity_id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
    )
    session.add(entry)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_audit_log_action_type_required(session):
    entry = AuditLog(
        actor_type="staff",
        entity_type="appointment",
        entity_id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
    )
    session.add(entry)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_audit_log_entity_type_required(session):
    entry = AuditLog(
        actor_type="staff",
        action_type="appointment.cancel",
        entity_id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
    )
    session.add(entry)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_audit_log_entity_id_required(session):
    entry = AuditLog(
        actor_type="staff",
        action_type="appointment.cancel",
        entity_type="appointment",
        timestamp=datetime.now(timezone.utc),
    )
    session.add(entry)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_audit_log_timestamp_required(session):
    entry = AuditLog(
        actor_type="staff",
        action_type="appointment.cancel",
        entity_type="appointment",
        entity_id=uuid.uuid4(),
    )
    session.add(entry)
    with pytest.raises(IntegrityError):
        await session.commit()


async def test_audit_log_actor_staff_id_is_nullable_for_system_or_ai_actions(session):
    # FR-E1.6: null actor_staff_id represents a system/AI-originated action
    entry = make_audit_log(actor_type="ai_agent", actor_staff_id=None)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    assert entry.actor_staff_id is None
    assert entry.actor_type == "ai_agent"


@pytest.mark.parametrize("actor_type", ["staff", "ai_agent", "system"])
async def test_audit_log_actor_type_accepts_each_documented_value(session, actor_type):
    entry = make_audit_log(actor_type=actor_type)
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    assert entry.actor_type == actor_type


async def test_audit_log_original_and_override_payload_default_to_none(session):
    entry = make_audit_log()
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    assert entry.original_payload is None
    assert entry.override_payload is None


async def test_audit_log_override_payload_jsonb_roundtrips_through_fresh_session(
    session, session_factory
):
    # US5: override actions retain both the pre- and post-override payload
    original_payload = {"slot_start": "2026-10-01T09:00:00+00:00"}
    override_payload = {"slot_start": "2026-10-01T10:30:00+00:00"}
    entry = make_audit_log(
        action_type="override.apply",
        original_payload=original_payload,
        override_payload=override_payload,
    )
    session.add(entry)
    await session.commit()
    entry_id = entry.id

    async with session_factory() as fresh_session:
        reloaded = await fresh_session.get(AuditLog, entry_id)
        assert reloaded.original_payload == original_payload
        assert reloaded.override_payload == override_payload


async def test_audit_log_overridden_of_id_links_override_entry_to_original(session):
    original = make_audit_log(action_type="appointment.reschedule")
    session.add(original)
    await session.commit()

    override_entry = make_audit_log(
        action_type="override.apply", overridden_of_id=original.id
    )
    session.add(override_entry)
    await session.commit()
    await session.refresh(override_entry)
    assert override_entry.overridden_of_id == original.id


async def test_audit_log_overridden_of_id_defaults_to_none(session):
    entry = make_audit_log()
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    assert entry.overridden_of_id is None


async def test_audit_log_entity_id_has_no_referential_integrity_to_target_table(session):
    # entity_type/entity_id are a polymorphic pair with no DB-level FK: an
    # audited entity may later be archived/deleted while its audit trail is
    # retained (FR-E2.5), so a reference to a UUID that exists nowhere else
    # must commit without error.
    entry = make_audit_log(entity_type="patient", entity_id=uuid.uuid4())
    session.add(entry)
    await session.commit()  # must not raise
    await session.refresh(entry)
    assert entry.entity_type == "patient"


def test_audit_log_exposes_no_update_or_delete_method():
    # FR-E1.6: append-only audit trail -- enforced at the repository layer by
    # simply never defining an update/delete method on this ORM class itself
    # (as opposed to a DB trigger), per architecture.md Sec.4.2 [INFERRED].
    assert "update" not in AuditLog.__dict__
    assert "delete" not in AuditLog.__dict__
