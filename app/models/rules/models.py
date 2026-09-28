"""SQLAlchemy ORM declarations for the rules module's three tables.

Pure schema — no query methods, no business logic. This is the data-driven
rules/config backbone every other module reads (never writes) at runtime:
FR-E3.1/RMR004 requires clinical/operational rules to live as data
(``RuleDefinition.rule_value: JSONB``), never as hardcoded code constants —
the only sanctioned exception is the numeric fallback defaults in
``app.common.constants``, used solely when no matching rule/config row
exists.

Interaction contract: no cross-module ``relationship()`` is declared here —
``repositories/rules/repository.py`` is the only file that imports these
classes directly; every other module reaches "the active rule set"/"a live
config value" only through ``services/rules/service.py``'s
``RuleSetService.get_active_rules_by_category``/``ConfigurationService.get_live``,
never this file's ORM classes or ``repositories/rules/repository.py``
directly.

Watch out: ``rule_sets``/``rule_definitions`` (clinical, approved by
clinic_management) and ``configuration_parameters`` (outreach
eligibility/cadence/prioritisation, delivery_team-only, no approval step) are
deliberately two separate tables with two separate write paths (FR-E3.2 vs
FR-E3.4) — they are never merged into one "settings" table, since the
approval-gate requirement applies to one but not the other.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy import Enum as SAEnum
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import RuleCategory, RuleSetStatus
from app.core.database import Base
from app.core.db_types import JSONBType


class RuleSet(Base):
    """A versioned collection of clinical/operational rule definitions.

    FR-E3.5: only one row may have ``status="active"`` at a time. This is
    enforced at the repository layer (``RuleSetRepository.activate`` performs
    the supersede-then-activate transition atomically as a single
    transaction), not by a DB constraint, because ``superseded`` rows must
    remain queryable for audit/diff history.
    """

    __tablename__ = "rule_sets"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[RuleSetStatus] = mapped_column(SAEnum(RuleSetStatus, native_enum=False), nullable=False)
    # delivery_team only (FR-E3.2 authoring).
    created_by_staff_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
    # clinic_management only (FR-E3.2 approval gate).
    approved_by_staff_id: Mapped[UUID | None] = mapped_column(ForeignKey("staff_users.id"), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class RuleDefinition(Base):
    """A single named rule/threshold belonging to a ``RuleSet``.

    FR-E3.1/RMR004: ``rule_value`` is structured JSONB, not free-text — every
    heuristic/threshold a service needs at runtime (patient category,
    risk classification, recall interval, appointment type, scheduling
    priority, triage question, escalation trigger, family scheduling) is read
    from here, never hardcoded as a code constant.
    """

    __tablename__ = "rule_definitions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    rule_set_id: Mapped[UUID] = mapped_column(ForeignKey("rule_sets.id"), nullable=False)
    category: Mapped[RuleCategory] = mapped_column(SAEnum(RuleCategory, native_enum=False), nullable=False)
    rule_key: Mapped[str] = mapped_column(String(150), nullable=False)
    rule_value: Mapped[dict] = mapped_column(JSONBType, nullable=False)


class ConfigurationParameter(Base):
    """A versioned outreach eligibility/cadence/prioritisation config value.

    FR-E3.4: delivery_team-authored, no approval step (unlike ``RuleSet``).
    ``rolled_back_from_id`` records rollback lineage when a prior version is
    restored as a new, current row.
    """

    __tablename__ = "configuration_parameters"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    value: Mapped[dict] = mapped_column(JSONBType, nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # delivery_team only.
    created_by_staff_id: Mapped[UUID] = mapped_column(ForeignKey("staff_users.id"), nullable=False)
    rolled_back_from_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("configuration_parameters.id"), nullable=True
    )
