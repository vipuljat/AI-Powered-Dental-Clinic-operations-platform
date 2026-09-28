"""Data-access classes over ``rule_sets``, ``rule_definitions``,
``configuration_parameters``.

Every SQL/ORM statement the rules module issues lives here —
``services/rules/service.py`` never constructs a query itself. Each method
takes an ``AsyncSession`` explicitly; no repository owns/creates its own
session (that is the composition root's / ``get_db``'s job, per
app/core/database.py).

Interaction contract: every class here is used only from
``services/rules/service.py`` — no other module's service or repository
imports these classes directly; every cross-module read of "active
rules"/"live config" goes through ``RuleSetService.get_active_rules_by_category``/
``ConfigurationService.get_live`` instead.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import RuleSetStatus
from app.common.exceptions.errors import NotFoundError
from app.common.utils import now_utc
from app.models.rules.models import ConfigurationParameter, RuleDefinition, RuleSet

__all__ = [
    "ConfigurationParameterRepository",
    "RuleDefinitionRepository",
    "RuleSetRepository",
]


class RuleSetRepository:
    """Data access for ``rule_sets``."""

    async def create_draft(self, db: AsyncSession, created_by_staff_id: UUID) -> RuleSet:
        result = await db.execute(select(func.max(RuleSet.version_number)))
        current_max = result.scalar_one_or_none()
        next_version = (current_max or 0) + 1
        rule_set = RuleSet(
            version_number=next_version,
            status=RuleSetStatus.draft,
            created_by_staff_id=created_by_staff_id,
        )
        db.add(rule_set)
        await db.flush()
        await db.refresh(rule_set)
        return rule_set

    async def get_by_id(self, db: AsyncSession, id: UUID) -> RuleSet | None:
        result = await db.execute(select(RuleSet).where(RuleSet.id == id))
        return result.scalar_one_or_none()

    async def get_active(self, db: AsyncSession) -> RuleSet | None:
        """FR-E3.5: WHERE status='active'; at most one row ever matches — the
        invariant is maintained by ``activate``'s atomic supersede-then-activate
        transition, not by a DB constraint (see ``models/rules/models.py``).
        """
        result = await db.execute(select(RuleSet).where(RuleSet.status == RuleSetStatus.active))
        return result.scalars().first()

    async def get_diff(self, db: AsyncSession, draft_id: UUID) -> dict:
        """Compares the draft's ``rule_definitions`` against the current
        active set's, keyed by ``(category, rule_key)``.

        Returns a dict with ``added``/``removed``/``changed``/``unchanged``
        lists, each entry carrying ``category``, ``rule_key`` and the
        old/new ``rule_value`` where applicable. If there is no active set
        (first-ever draft), every draft rule is reported as ``added``.
        """
        draft_rules = await RuleDefinitionRepository().list_by_set(db, draft_id)
        draft_by_key = {(r.category, r.rule_key): r for r in draft_rules}

        active_set = await self.get_active(db)
        active_by_key: dict[tuple, RuleDefinition] = {}
        if active_set is not None:
            active_rules = await RuleDefinitionRepository().list_by_set(db, active_set.id)
            active_by_key = {(r.category, r.rule_key): r for r in active_rules}

        added: list[dict] = []
        removed: list[dict] = []
        changed: list[dict] = []
        unchanged: list[dict] = []

        for key, draft_rule in draft_by_key.items():
            category, rule_key = key
            active_rule = active_by_key.get(key)
            if active_rule is None:
                added.append(
                    {
                        "category": category,
                        "rule_key": rule_key,
                        "new_value": draft_rule.rule_value,
                    }
                )
            elif active_rule.rule_value != draft_rule.rule_value:
                changed.append(
                    {
                        "category": category,
                        "rule_key": rule_key,
                        "old_value": active_rule.rule_value,
                        "new_value": draft_rule.rule_value,
                    }
                )
            else:
                unchanged.append(
                    {
                        "category": category,
                        "rule_key": rule_key,
                        "value": draft_rule.rule_value,
                    }
                )

        for key, active_rule in active_by_key.items():
            if key not in draft_by_key:
                category, rule_key = key
                removed.append(
                    {
                        "category": category,
                        "rule_key": rule_key,
                        "old_value": active_rule.rule_value,
                    }
                )

        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
        }

    async def set_status(
        self, db: AsyncSession, id: UUID, status: str, approved_by_staff_id: UUID | None = None
    ) -> RuleSet:
        rule_set = await self.get_by_id(db, id)
        if rule_set is None:
            raise NotFoundError("Rule set not found.")
        rule_set.status = status
        if approved_by_staff_id is not None:
            rule_set.approved_by_staff_id = approved_by_staff_id
        await db.flush()
        await db.refresh(rule_set)
        return rule_set

    async def activate(self, db: AsyncSession, id: UUID, approved_by_staff_id: UUID) -> RuleSet:
        """FR-E3.5/US15: supersedes the prior active row and activates the new
        one in a single transaction — "the approved version becomes the
        active rule set atomically; prior version remains in force until
        approved" (T39 AC) is satisfied by doing both writes here, never as
        two separate calls a caller could interleave with another read.
        """
        rule_set = await self.get_by_id(db, id)
        if rule_set is None:
            raise NotFoundError("Rule set not found.")

        current_active = await self.get_active(db)
        if current_active is not None and current_active.id != id:
            current_active.status = RuleSetStatus.superseded

        rule_set.status = RuleSetStatus.active
        rule_set.approved_by_staff_id = approved_by_staff_id
        rule_set.approved_at = now_utc()

        await db.flush()
        await db.refresh(rule_set)
        return rule_set


class RuleDefinitionRepository:
    """Data access for ``rule_definitions``."""

    async def bulk_insert(self, db: AsyncSession, rule_set_id: UUID, rules: list[dict]) -> list[RuleDefinition]:
        definitions = [RuleDefinition(rule_set_id=rule_set_id, **rule) for rule in rules]
        db.add_all(definitions)
        await db.flush()
        for definition in definitions:
            await db.refresh(definition)
        return definitions

    async def list_by_set(self, db: AsyncSession, rule_set_id: UUID) -> list[RuleDefinition]:
        result = await db.execute(
            select(RuleDefinition).where(RuleDefinition.rule_set_id == rule_set_id)
        )
        return list(result.scalars().all())

    async def list_by_set_and_category(
        self, db: AsyncSession, rule_set_id: UUID, category: str
    ) -> list[RuleDefinition]:
        result = await db.execute(
            select(RuleDefinition).where(
                RuleDefinition.rule_set_id == rule_set_id,
                RuleDefinition.category == category,
            )
        )
        return list(result.scalars().all())


class ConfigurationParameterRepository:
    """Data access for ``configuration_parameters``."""

    async def save_version(
        self, db: AsyncSession, name: str, value, created_by_staff_id: UUID
    ) -> ConfigurationParameter:
        result = await db.execute(
            select(func.max(ConfigurationParameter.version_number)).where(
                ConfigurationParameter.name == name
            )
        )
        current_max = result.scalar_one_or_none()
        next_version = (current_max or 0) + 1
        parameter = ConfigurationParameter(
            name=name,
            value=value,
            version_number=next_version,
            effective_at=now_utc(),
            created_by_staff_id=created_by_staff_id,
        )
        db.add(parameter)
        await db.flush()
        await db.refresh(parameter)
        return parameter

    async def rollback(
        self, db: AsyncSession, version_number: int, created_by_staff_id: UUID
    ) -> ConfigurationParameter:
        """T43: "restores the exact prior configuration values" — reads the
        row at ``version_number``, inserts a NEW row copying its ``name``/
        ``value`` with a fresh version number.

        ``rolled_back_from_id`` names the version being *restored*, i.e. the
        row read at ``version_number`` (see the file spec's Watch out: it
        always reads as "copied from", never "the version being replaced").
        """
        result = await db.execute(
            select(ConfigurationParameter).where(
                ConfigurationParameter.version_number == version_number
            )
        )
        source = result.scalars().first()
        if source is None:
            raise NotFoundError("Configuration parameter version not found.")

        max_result = await db.execute(
            select(func.max(ConfigurationParameter.version_number)).where(
                ConfigurationParameter.name == source.name
            )
        )
        current_max = max_result.scalar_one_or_none()
        next_version = (current_max or 0) + 1

        parameter = ConfigurationParameter(
            name=source.name,
            value=source.value,
            version_number=next_version,
            effective_at=now_utc(),
            created_by_staff_id=created_by_staff_id,
            rolled_back_from_id=source.id,
        )
        db.add(parameter)
        await db.flush()
        await db.refresh(parameter)
        return parameter

    async def get_latest(self, db: AsyncSession, name: str) -> ConfigurationParameter | None:
        result = await db.execute(
            select(ConfigurationParameter)
            .where(ConfigurationParameter.name == name)
            .order_by(ConfigurationParameter.version_number.desc())
        )
        return result.scalars().first()
