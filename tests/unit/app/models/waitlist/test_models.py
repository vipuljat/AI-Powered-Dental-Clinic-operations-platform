"""Unit tests for app/models/waitlist/models.py.

Exercises the three declarative ORM classes (`IdleChairAlert`,
`WaitlistEntry`, `WaitlistOffer`) directly against a real SQLite engine (the
same `sqlite+aiosqlite:///:memory:`-style substitution PROJECT.md's Testing
section describes, minus the async driver -- a plain synchronous engine is
the lightest way to drive `Base.metadata.create_all` and real
insert/select round trips end to end without any application composition
root, network, or event loop).

These three tables carry foreign keys into other BRD-epic modules
(`appointments`, `providers`, `chairs`, `patients`) that this task's spec
does not own and that this test file therefore never imports. To let
SQLite compile the `REFERENCES ...` DDL clause for the tables actually
under test, minimal stand-in `Table` objects for those names are
registered on the same shared `Base.metadata` registry -- but only if a
table of that name isn't already registered there (e.g. because another
module's models.py has already been imported into the same interpreter),
so this never clobbers a real model.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, String, Table, create_engine, inspect, select
from sqlalchemy.exc import NoReferencedTableError

from app.core.database import Base
from app.models.waitlist.models import IdleChairAlert, WaitlistEntry, WaitlistOffer

# ---------------------------------------------------------------------------
# Stand-in tables for the cross-module FK targets this file's columns point
# at, registered once at import time (see module docstring).
# ---------------------------------------------------------------------------
for _name in ("appointments", "providers", "chairs", "patients"):
    if _name not in Base.metadata.tables:
        Table(_name, Base.metadata, Column("id", String(36), primary_key=True))


def _col(model, name):
    return model.__table__.c[name]


def _new_id(column):
    """A value valid for `column`'s python type, tolerant of whichever
    UUID representation (`uuid.UUID` vs plain string) the implementation
    picked for its primary/foreign-key columns."""
    try:
        py_type = column.type.python_type
    except NotImplementedError:
        py_type = str
    return uuid.uuid4() if py_type is uuid.UUID else str(uuid.uuid4())


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        eng,
        tables=[
            IdleChairAlert.__table__,
            WaitlistEntry.__table__,
            WaitlistOffer.__table__,
        ],
    )
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# Table names (public surface)
# ---------------------------------------------------------------------------

def test_idle_chair_alert_tablename():
    assert IdleChairAlert.__tablename__ == "idle_chair_alerts"


def test_waitlist_entry_tablename():
    assert WaitlistEntry.__tablename__ == "waitlist_entries"


def test_waitlist_offer_tablename():
    assert WaitlistOffer.__tablename__ == "waitlist_offers"


def test_all_three_tables_are_declarative_base_subclasses():
    assert issubclass(IdleChairAlert, Base)
    assert issubclass(WaitlistEntry, Base)
    assert issubclass(WaitlistOffer, Base)


# ---------------------------------------------------------------------------
# idle_chair_alerts data shape
# ---------------------------------------------------------------------------

def test_idle_chair_alert_primary_key_is_id():
    assert [c.name for c in IdleChairAlert.__table__.primary_key.columns] == ["id"]


def test_idle_chair_alert_appointment_id_is_nullable_fk_to_appointments():
    col = _col(IdleChairAlert, "appointment_id")
    assert col.nullable is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "appointments.id"


def test_idle_chair_alert_provider_id_is_not_null_fk_to_providers():
    col = _col(IdleChairAlert, "provider_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "providers.id"


def test_idle_chair_alert_chair_id_is_not_null_fk_to_chairs():
    col = _col(IdleChairAlert, "chair_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "chairs.id"


def test_idle_chair_alert_slot_columns_are_not_null_timezone_aware_datetimes():
    for name in ("slot_start", "slot_end", "detected_at"):
        col = _col(IdleChairAlert, name)
        assert col.nullable is False
        assert isinstance(col.type, sa.DateTime)
        assert col.type.timezone is True


def test_idle_chair_alert_source_is_not_null_enum_of_auto_detected_and_manual_flag():
    col = _col(IdleChairAlert, "source")
    assert col.nullable is False
    assert set(col.type.enums) == {"auto_detected", "manual_flag"}


def test_idle_chair_alert_status_is_not_null_enum_of_open_filled_exhausted():
    col = _col(IdleChairAlert, "status")
    assert col.nullable is False
    assert set(col.type.enums) == {"open", "filled", "exhausted"}


def test_idle_chair_alert_status_defaults_to_open_when_omitted_on_insert(engine):
    row_id = _new_id(_col(IdleChairAlert, "id"))
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            IdleChairAlert.__table__.insert().values(
                id=row_id,
                provider_id=_new_id(_col(IdleChairAlert, "provider_id")),
                chair_id=_new_id(_col(IdleChairAlert, "chair_id")),
                slot_start=now,
                slot_end=now + timedelta(hours=1),
                source="auto_detected",
                detected_at=now,
                # status intentionally omitted
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            select(IdleChairAlert.__table__).where(IdleChairAlert.__table__.c.id == row_id)
        ).one()
    assert row.status == "open"


# ---------------------------------------------------------------------------
# FR-E5.2: no DB-level uniqueness constraint on
# (provider_id, chair_id, slot_start, slot_end) -- deduplication between an
# auto-detected and a manually-flagged alert for the same slot is an
# application-layer check, not a DB constraint.
# ---------------------------------------------------------------------------

def test_idle_chair_alert_has_no_unique_constraint_on_provider_chair_slot_columns():
    target = {"provider_id", "chair_id", "slot_start", "slot_end"}
    for constraint in IdleChairAlert.__table__.constraints:
        if isinstance(constraint, sa.UniqueConstraint):
            assert {c.name for c in constraint.columns} != target
    for index in IdleChairAlert.__table__.indexes:
        if index.unique:
            assert {c.name for c in index.columns} != target


def test_idle_chair_alert_allows_two_rows_for_the_same_provider_chair_and_slot(engine):
    provider_id = _new_id(_col(IdleChairAlert, "provider_id"))
    chair_id = _new_id(_col(IdleChairAlert, "chair_id"))
    slot_start = datetime.now(timezone.utc)
    slot_end = slot_start + timedelta(hours=1)
    detected_at = datetime.now(timezone.utc)

    with engine.begin() as conn:
        # One auto-detected alert...
        conn.execute(
            IdleChairAlert.__table__.insert().values(
                id=_new_id(_col(IdleChairAlert, "id")),
                provider_id=provider_id,
                chair_id=chair_id,
                slot_start=slot_start,
                slot_end=slot_end,
                source="auto_detected",
                detected_at=detected_at,
            )
        )
        # ...and one manually-flagged alert for the exact same slot must
        # both be insertable without an IntegrityError.
        conn.execute(
            IdleChairAlert.__table__.insert().values(
                id=_new_id(_col(IdleChairAlert, "id")),
                provider_id=provider_id,
                chair_id=chair_id,
                slot_start=slot_start,
                slot_end=slot_end,
                source="manual_flag",
                detected_at=detected_at,
            )
        )

    with engine.connect() as conn:
        count = conn.execute(
            select(sa.func.count())
            .select_from(IdleChairAlert.__table__)
            .where(
                IdleChairAlert.__table__.c.provider_id == provider_id,
                IdleChairAlert.__table__.c.chair_id == chair_id,
                IdleChairAlert.__table__.c.slot_start == slot_start,
                IdleChairAlert.__table__.c.slot_end == slot_end,
            )
        ).scalar_one()
    assert count == 2


# ---------------------------------------------------------------------------
# waitlist_entries data shape
# ---------------------------------------------------------------------------

def test_waitlist_entry_primary_key_is_id():
    assert [c.name for c in WaitlistEntry.__table__.primary_key.columns] == ["id"]


def test_waitlist_entry_patient_id_is_not_null_fk_to_patients():
    col = _col(WaitlistEntry, "patient_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "patients.id"


def test_waitlist_entry_desired_provider_id_is_nullable_fk_to_providers():
    col = _col(WaitlistEntry, "desired_provider_id")
    assert col.nullable is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "providers.id"


def test_waitlist_entry_desired_timeframe_bounds_are_nullable():
    assert _col(WaitlistEntry, "desired_timeframe_start").nullable is True
    assert _col(WaitlistEntry, "desired_timeframe_end").nullable is True


def test_waitlist_entry_urgency_is_not_null_enum_of_low_medium_high():
    col = _col(WaitlistEntry, "urgency")
    assert col.nullable is False
    assert set(col.type.enums) == {"low", "medium", "high"}


def test_waitlist_entry_priority_score_is_not_null_numeric_6_2():
    col = _col(WaitlistEntry, "priority_score")
    assert col.nullable is False
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 6
    assert col.type.scale == 2


def test_waitlist_entry_current_alert_id_is_nullable_fk_to_idle_chair_alerts():
    col = _col(WaitlistEntry, "current_alert_id")
    assert col.nullable is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "idle_chair_alerts.id"


def test_waitlist_entry_appointment_id_is_nullable_fk_to_appointments():
    col = _col(WaitlistEntry, "appointment_id")
    assert col.nullable is True
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "appointments.id"


def test_waitlist_entry_status_is_not_null_enum_of_expected_values():
    col = _col(WaitlistEntry, "status")
    assert col.nullable is False
    assert set(col.type.enums) == {"active", "offered", "booked", "declined", "expired"}


def test_waitlist_entry_created_and_updated_at_are_not_null_timezone_aware_datetimes():
    for name in ("created_at", "updated_at"):
        col = _col(WaitlistEntry, name)
        assert col.nullable is False
        assert isinstance(col.type, sa.DateTime)
        assert col.type.timezone is True


def test_waitlist_entry_status_defaults_to_active_when_omitted_on_insert(engine):
    row_id = _new_id(_col(WaitlistEntry, "id"))
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            WaitlistEntry.__table__.insert().values(
                id=row_id,
                patient_id=_new_id(_col(WaitlistEntry, "patient_id")),
                urgency="medium",
                priority_score=Decimal("10.00"),
                created_at=now,
                updated_at=now,
                # status intentionally omitted
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            select(WaitlistEntry.__table__).where(WaitlistEntry.__table__.c.id == row_id)
        ).one()
    assert row.status == "active"


# ---------------------------------------------------------------------------
# waitlist_offers data shape
# ---------------------------------------------------------------------------

def test_waitlist_offer_primary_key_is_id():
    assert [c.name for c in WaitlistOffer.__table__.primary_key.columns] == ["id"]


def test_waitlist_offer_waitlist_entry_id_is_not_null_fk_to_waitlist_entries():
    col = _col(WaitlistOffer, "waitlist_entry_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "waitlist_entries.id"


def test_waitlist_offer_idle_chair_alert_id_is_not_null_fk_to_idle_chair_alerts():
    col = _col(WaitlistOffer, "idle_chair_alert_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "idle_chair_alerts.id"


def test_waitlist_offer_offered_at_and_expiry_are_not_null_timezone_aware_datetimes():
    for name in ("offered_at", "response_window_expires_at"):
        col = _col(WaitlistOffer, name)
        assert col.nullable is False
        assert isinstance(col.type, sa.DateTime)
        assert col.type.timezone is True


def test_waitlist_offer_status_is_not_null_enum_of_expected_values():
    col = _col(WaitlistOffer, "status")
    assert col.nullable is False
    assert set(col.type.enums) == {"pending", "accepted", "declined", "expired"}


def test_waitlist_offer_status_defaults_to_pending_when_omitted_on_insert(engine):
    now = datetime.now(timezone.utc)
    alert_id = _new_id(_col(IdleChairAlert, "id"))
    entry_id = _new_id(_col(WaitlistEntry, "id"))
    row_id = _new_id(_col(WaitlistOffer, "id"))

    with engine.begin() as conn:
        # Satisfy the two FK targets that live inside this same module.
        conn.execute(
            IdleChairAlert.__table__.insert().values(
                id=alert_id,
                provider_id=_new_id(_col(IdleChairAlert, "provider_id")),
                chair_id=_new_id(_col(IdleChairAlert, "chair_id")),
                slot_start=now,
                slot_end=now + timedelta(hours=1),
                source="auto_detected",
                detected_at=now,
            )
        )
        conn.execute(
            WaitlistEntry.__table__.insert().values(
                id=entry_id,
                patient_id=_new_id(_col(WaitlistEntry, "patient_id")),
                urgency="high",
                priority_score=Decimal("5.50"),
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            WaitlistOffer.__table__.insert().values(
                id=row_id,
                waitlist_entry_id=entry_id,
                idle_chair_alert_id=alert_id,
                offered_at=now,
                response_window_expires_at=now + timedelta(minutes=30),
                # status intentionally omitted
            )
        )

    with engine.connect() as conn:
        row = conn.execute(
            select(WaitlistOffer.__table__).where(WaitlistOffer.__table__.c.id == row_id)
        ).one()
    assert row.status == "pending"


# ---------------------------------------------------------------------------
# Interaction contract: plain FK columns only, no cross-module (or in-module)
# ORM `relationship()` attributes declared on any of the three classes.
# ---------------------------------------------------------------------------

def test_no_orm_relationships_are_declared_on_any_waitlist_model():
    for model in (IdleChairAlert, WaitlistEntry, WaitlistOffer):
        assert len(inspect(model).relationships) == 0


# ---------------------------------------------------------------------------
# Sanity check on the stub-table registration helper itself: proves the
# fixture setup in this file is exercising the real FK-target-resolution
# behaviour (i.e. it isn't silently no-op-ing) by showing the *unpatched*
# scenario would indeed fail this way.
# ---------------------------------------------------------------------------

def test_creating_idle_chair_alerts_table_requires_its_fk_targets_in_metadata():
    other_metadata = sa.MetaData()
    orphan = Table(
        "idle_chair_alerts_copy",
        other_metadata,
        Column("id", String(36), primary_key=True),
        Column("provider_id", String(36), sa.ForeignKey("providers_missing.id"), nullable=False),
    )
    orphan_engine = create_engine("sqlite:///:memory:")
    with pytest.raises(NoReferencedTableError):
        orphan.create(orphan_engine)
    orphan_engine.dispose()
