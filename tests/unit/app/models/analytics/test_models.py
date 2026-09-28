"""Unit tests for app/models/analytics/models.py.

Exercises the four declarative ORM classes (`OperationalMetricsSnapshot`,
`FinancialRoiSnapshot`, `BaselineCost`, `DashboardExport`) directly against a
real SQLite engine (the same `sqlite+aiosqlite:///:memory:`-style
substitution PROJECT.md's Testing section describes, minus the async driver
-- a plain synchronous engine is the lightest way to drive
`Base.metadata.create_all` and real insert/select round trips end to end
without any application composition root, network, or event loop).

`baseline_costs` and `dashboard_exports` carry a foreign key into
`staff_users`, a table this task's spec does not own and that this test file
therefore never imports. To let SQLite compile the `REFERENCES ...` DDL
clause for the tables actually under test, a minimal stand-in `Table` object
for that name is registered on the same shared `Base.metadata` registry --
but only if a table of that name isn't already registered there (e.g.
because another module's models.py has already been imported into the same
interpreter), so this never clobbers a real model.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
import sqlalchemy as sa
from sqlalchemy import Column, String, Table, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError, NoReferencedTableError

from app.core.database import Base
from app.models.analytics.models import (
    BaselineCost,
    DashboardExport,
    FinancialRoiSnapshot,
    OperationalMetricsSnapshot,
)

# ---------------------------------------------------------------------------
# Stand-in table for the cross-module FK target this file's columns point
# at, registered once at import time (see module docstring).
# ---------------------------------------------------------------------------
if "staff_users" not in Base.metadata.tables:
    Table("staff_users", Base.metadata, Column("id", String(36), primary_key=True))


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
            OperationalMetricsSnapshot.__table__,
            FinancialRoiSnapshot.__table__,
            BaselineCost.__table__,
            DashboardExport.__table__,
        ],
    )
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# Table names (public surface)
# ---------------------------------------------------------------------------

def test_operational_metrics_snapshot_tablename():
    assert OperationalMetricsSnapshot.__tablename__ == "operational_metrics_snapshots"


def test_financial_roi_snapshot_tablename():
    assert FinancialRoiSnapshot.__tablename__ == "financial_roi_snapshots"


def test_baseline_cost_tablename():
    assert BaselineCost.__tablename__ == "baseline_costs"


def test_dashboard_export_tablename():
    assert DashboardExport.__tablename__ == "dashboard_exports"


def test_all_four_models_are_declarative_base_subclasses():
    assert issubclass(OperationalMetricsSnapshot, Base)
    assert issubclass(FinancialRoiSnapshot, Base)
    assert issubclass(BaselineCost, Base)
    assert issubclass(DashboardExport, Base)


# ---------------------------------------------------------------------------
# operational_metrics_snapshots data shape
# ---------------------------------------------------------------------------

def test_operational_metrics_snapshot_primary_key_is_id():
    cols = [c.name for c in OperationalMetricsSnapshot.__table__.primary_key.columns]
    assert cols == ["id"]


def test_operational_metrics_snapshot_metric_name_is_not_null_varchar_100():
    col = _col(OperationalMetricsSnapshot, "metric_name")
    assert col.nullable is False
    assert isinstance(col.type, sa.String)
    assert col.type.length == 100


def test_operational_metrics_snapshot_metric_value_is_not_null_numeric_12_2():
    col = _col(OperationalMetricsSnapshot, "metric_value")
    assert col.nullable is False
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 12
    assert col.type.scale == 2


def test_operational_metrics_snapshot_period_start_and_end_are_not_null_dates():
    for name in ("period_start", "period_end"):
        col = _col(OperationalMetricsSnapshot, name)
        assert col.nullable is False
        assert isinstance(col.type, sa.Date)


def test_operational_metrics_snapshot_computed_at_is_not_null_timezone_aware_datetime():
    col = _col(OperationalMetricsSnapshot, "computed_at")
    assert col.nullable is False
    assert isinstance(col.type, sa.DateTime)
    assert col.type.timezone is True


# FR-E9.1: baseline_value is the one nullable field driving the
# data_delayed/baseline-pending distinction -- null means "baseline pending".
def test_operational_metrics_snapshot_baseline_value_is_nullable_numeric_12_2():
    col = _col(OperationalMetricsSnapshot, "baseline_value")
    assert col.nullable is True
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 12
    assert col.type.scale == 2


def test_operational_metrics_snapshot_can_be_inserted_with_null_baseline_value(engine):
    row_id = _new_id(_col(OperationalMetricsSnapshot, "id"))
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            OperationalMetricsSnapshot.__table__.insert().values(
                id=row_id,
                metric_name="no_show_rate",
                metric_value=Decimal("12.50"),
                baseline_value=None,
                period_start=date(2026, 1, 1),
                period_end=date(2026, 1, 31),
                computed_at=now,
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            select(OperationalMetricsSnapshot.__table__).where(
                OperationalMetricsSnapshot.__table__.c.id == row_id
            )
        ).one()
    assert row.baseline_value is None
    assert row.metric_value == Decimal("12.50")


def test_operational_metrics_snapshot_rejects_null_metric_value(engine):
    row_id = _new_id(_col(OperationalMetricsSnapshot, "id"))
    now = datetime.now(timezone.utc)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                OperationalMetricsSnapshot.__table__.insert().values(
                    id=row_id,
                    metric_name="no_show_rate",
                    metric_value=None,
                    period_start=date(2026, 1, 1),
                    period_end=date(2026, 1, 31),
                    computed_at=now,
                )
            )


# ---------------------------------------------------------------------------
# financial_roi_snapshots data shape
# ---------------------------------------------------------------------------

def test_financial_roi_snapshot_primary_key_is_id():
    cols = [c.name for c in FinancialRoiSnapshot.__table__.primary_key.columns]
    assert cols == ["id"]


def test_financial_roi_snapshot_period_month_is_not_null_date():
    col = _col(FinancialRoiSnapshot, "period_month")
    assert col.nullable is False
    assert isinstance(col.type, sa.Date)


def test_financial_roi_snapshot_recovered_appointments_is_not_null_integer():
    col = _col(FinancialRoiSnapshot, "recovered_appointments")
    assert col.nullable is False
    assert isinstance(col.type, sa.Integer)


def test_financial_roi_snapshot_recovered_revenue_is_not_null_numeric_14_2():
    col = _col(FinancialRoiSnapshot, "recovered_revenue")
    assert col.nullable is False
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 14
    assert col.type.scale == 2


def test_financial_roi_snapshot_baseline_cost_is_nullable_numeric_14_2():
    col = _col(FinancialRoiSnapshot, "baseline_cost")
    assert col.nullable is True
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 14
    assert col.type.scale == 2


def test_financial_roi_snapshot_roi_percentage_is_nullable_numeric_6_2():
    col = _col(FinancialRoiSnapshot, "roi_percentage")
    assert col.nullable is True
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 6
    assert col.type.scale == 2


def test_financial_roi_snapshot_computed_at_is_not_null_timezone_aware_datetime():
    col = _col(FinancialRoiSnapshot, "computed_at")
    assert col.nullable is False
    assert isinstance(col.type, sa.DateTime)
    assert col.type.timezone is True


def test_financial_roi_snapshot_can_be_inserted_with_null_baseline_cost_and_roi(engine):
    row_id = _new_id(_col(FinancialRoiSnapshot, "id"))
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            FinancialRoiSnapshot.__table__.insert().values(
                id=row_id,
                period_month=date(2026, 1, 1),
                recovered_appointments=5,
                recovered_revenue=Decimal("1000.00"),
                baseline_cost=None,
                roi_percentage=None,
                computed_at=now,
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            select(FinancialRoiSnapshot.__table__).where(
                FinancialRoiSnapshot.__table__.c.id == row_id
            )
        ).one()
    assert row.baseline_cost is None
    assert row.roi_percentage is None
    assert row.recovered_appointments == 5


# ---------------------------------------------------------------------------
# baseline_costs data shape
# ---------------------------------------------------------------------------

def test_baseline_cost_primary_key_is_id():
    cols = [c.name for c in BaselineCost.__table__.primary_key.columns]
    assert cols == ["id"]


def test_baseline_cost_entered_by_staff_id_is_not_null_fk_to_staff_users():
    col = _col(BaselineCost, "entered_by_staff_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "staff_users.id"


def test_baseline_cost_amount_is_not_null_numeric_14_2():
    col = _col(BaselineCost, "amount")
    assert col.nullable is False
    assert isinstance(col.type, sa.Numeric)
    assert col.type.precision == 14
    assert col.type.scale == 2


def test_baseline_cost_effective_period_is_not_null_date():
    col = _col(BaselineCost, "effective_period")
    assert col.nullable is False
    assert isinstance(col.type, sa.Date)


def test_baseline_cost_created_at_is_not_null_timezone_aware_datetime():
    col = _col(BaselineCost, "created_at")
    assert col.nullable is False
    assert isinstance(col.type, sa.DateTime)
    assert col.type.timezone is True


def test_baseline_cost_can_be_inserted_and_read_back(engine):
    staff_id = _new_id(_col(BaselineCost, "entered_by_staff_id"))
    row_id = _new_id(_col(BaselineCost, "id"))
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            BaselineCost.__table__.insert().values(
                id=row_id,
                entered_by_staff_id=staff_id,
                amount=Decimal("2500.00"),
                effective_period=date(2026, 1, 1),
                created_at=now,
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            select(BaselineCost.__table__).where(BaselineCost.__table__.c.id == row_id)
        ).one()
    assert row.amount == Decimal("2500.00")


def test_baseline_cost_rejects_null_entered_by_staff_id(engine):
    row_id = _new_id(_col(BaselineCost, "id"))
    now = datetime.now(timezone.utc)
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                BaselineCost.__table__.insert().values(
                    id=row_id,
                    entered_by_staff_id=None,
                    amount=Decimal("2500.00"),
                    effective_period=date(2026, 1, 1),
                    created_at=now,
                )
            )


# ---------------------------------------------------------------------------
# dashboard_exports data shape
# ---------------------------------------------------------------------------

def test_dashboard_export_primary_key_is_id():
    cols = [c.name for c in DashboardExport.__table__.primary_key.columns]
    assert cols == ["id"]


def test_dashboard_export_requested_by_staff_id_is_not_null_fk_to_staff_users():
    col = _col(DashboardExport, "requested_by_staff_id")
    assert col.nullable is False
    fks = list(col.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "staff_users.id"


def test_dashboard_export_dashboard_type_is_not_null_enum_of_operational_financial():
    col = _col(DashboardExport, "dashboard_type")
    assert col.nullable is False
    assert set(col.type.enums) == {"operational", "financial"}


def test_dashboard_export_format_is_not_null_enum_of_pdf_csv():
    col = _col(DashboardExport, "format")
    assert col.nullable is False
    assert set(col.type.enums) == {"pdf", "csv"}


def test_dashboard_export_date_range_bounds_are_not_null_dates():
    for name in ("date_range_start", "date_range_end"):
        col = _col(DashboardExport, name)
        assert col.nullable is False
        assert isinstance(col.type, sa.Date)


def test_dashboard_export_storage_uri_is_not_null_varchar_500():
    col = _col(DashboardExport, "storage_uri")
    assert col.nullable is False
    assert isinstance(col.type, sa.String)
    assert col.type.length == 500


def test_dashboard_export_requested_at_is_not_null_timezone_aware_datetime():
    col = _col(DashboardExport, "requested_at")
    assert col.nullable is False
    assert isinstance(col.type, sa.DateTime)
    assert col.type.timezone is True


def test_dashboard_export_stores_an_object_storage_uri_not_raw_bytes(engine):
    staff_id = _new_id(_col(DashboardExport, "requested_by_staff_id"))
    row_id = _new_id(_col(DashboardExport, "id"))
    now = datetime.now(timezone.utc)
    uri = "s3://exports-bucket/2026/01/dashboard-export.pdf"
    with engine.begin() as conn:
        conn.execute(
            DashboardExport.__table__.insert().values(
                id=row_id,
                requested_by_staff_id=staff_id,
                dashboard_type="operational",
                format="pdf",
                date_range_start=date(2026, 1, 1),
                date_range_end=date(2026, 1, 31),
                storage_uri=uri,
                requested_at=now,
            )
        )
    with engine.connect() as conn:
        row = conn.execute(
            select(DashboardExport.__table__).where(DashboardExport.__table__.c.id == row_id)
        ).one()
    assert row.storage_uri == uri


# ---------------------------------------------------------------------------
# Interaction contract: plain FK columns only, no cross-module (or in-module)
# ORM `relationship()` attributes declared on any of the four classes.
# ---------------------------------------------------------------------------

def test_no_orm_relationships_are_declared_on_any_analytics_model():
    for model in (
        OperationalMetricsSnapshot,
        FinancialRoiSnapshot,
        BaselineCost,
        DashboardExport,
    ):
        assert len(inspect(model).relationships) == 0


# ---------------------------------------------------------------------------
# Sanity check on the stub-table registration helper itself: proves the
# fixture setup in this file is exercising the real FK-target-resolution
# behaviour (i.e. it isn't silently no-op-ing) by showing the *unpatched*
# scenario would indeed fail this way.
# ---------------------------------------------------------------------------

def test_creating_baseline_costs_table_requires_its_fk_target_in_metadata():
    other_metadata = sa.MetaData()
    orphan = Table(
        "baseline_costs_copy",
        other_metadata,
        Column("id", String(36), primary_key=True),
        Column(
            "entered_by_staff_id",
            String(36),
            sa.ForeignKey("staff_users_missing.id"),
            nullable=False,
        ),
    )
    orphan_engine = create_engine("sqlite:///:memory:")
    with pytest.raises(NoReferencedTableError):
        orphan.create(orphan_engine)
    orphan_engine.dispose()
