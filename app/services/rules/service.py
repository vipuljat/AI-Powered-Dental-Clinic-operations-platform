"""Business logic for the `rules` module: structural validation, the
draft/approval/versioning/activation workflow for clinical rules, and the
versioned eligibility/cadence/prioritisation configuration store.

``RuleValidationService`` performs the structural/consistency checks a draft
must pass before it is ever persisted (FR-E3.1). ``RuleSetService`` owns the
draft -> in_review -> active/superseded workflow (FR-E3.2/US15) and is also
the read API ("RuleSetService (read)" per architecture.md §5.3) every other
module's service calls for "the active clinical rules". ``ConfigurationService``
owns the separate, approval-free delivery_team configuration store
(FR-E3.4/US17) and is the read API every other module calls for a live,
no-code-release-required config value.

Every write in this file calls ``AuditService.record`` — no mutation here
bypasses the audit trail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.common.enums import ActorType, RuleCategory, RuleSetStatus
from app.common.exceptions.errors import (
    ConflictError,
    NoActiveRuleSetError,
    NotFoundError,
    RuleValidationError,
)

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models.rules.models import ConfigurationParameter, RuleDefinition, RuleSet
    from app.repositories.rules.repository import (
        ConfigurationParameterRepository,
        RuleDefinitionRepository,
        RuleSetRepository,
    )
    from app.services.console.service import AuditService

__all__ = ["ConfigurationService", "RuleSetService", "RuleValidationService"]


def _category_value(value: object) -> str:
    """Normalizes a rule's ``category`` field, accepted here as either a
    ``RuleCategory`` enum member (the shape ``schemas/rules/schemas.py``'s
    ``RuleDefinitionInput.model_dump()`` produces by default) or its plain
    string value (the shape a ``model_dump(mode="json")`` caller would pass
    instead) — both are treated identically throughout this file.
    """
    return value.value if hasattr(value, "value") else value  # type: ignore[return-value]


class RuleValidationService:
    """FR-E3.1/US14: structural/consistency checks run before a draft rule
    set is ever written to ``rule_definitions`` — "the rule editor presents
    structured fields, not free-text" is enforced here by rejecting drafts
    whose structure is internally inconsistent, not by re-validating field
    *types* (Pydantic's ``RuleDefinitionInput`` already guarantees that at
    the route boundary).
    """

    def __init__(self, rule_def_repo: "RuleDefinitionRepository") -> None:
        # Not queried by `validate` below: every check this class performs
        # is purely structural, over the submitted `rules` list alone, and
        # `validate` intentionally takes no `db`/session — a draft must be
        # checkable before any row (this draft's or any other's) is even
        # touched. Stored only so this class's constructor matches the
        # collaborator shape every other repo-backed service in this file
        # follows, and so a future DB-aware check has somewhere to reach.
        self._rule_def_repo = rule_def_repo

    async def validate(self, rules: list[dict]) -> None:
        """Raises ``RuleValidationError`` (422) on the first structural
        violation found, in the order documented on the file spec:
        (a) duplicate ``rule_key`` within a category, (b) conflicting
        ``recall_interval`` months for one ``patient_category``, (c) a
        ``triage_question``/``escalation_trigger`` rule referencing an
        undefined category key.
        """
        self._check_duplicate_keys(rules)
        self._check_recall_interval_conflicts(rules)
        self._check_cross_references(rules)

    @staticmethod
    def _check_duplicate_keys(rules: list[dict]) -> None:
        """US14 AC (a): a ``rule_key`` must be unique within its own
        ``category`` inside a single submission.
        """
        seen: dict[tuple[str, str], int] = {}
        for rule in rules:
            key = (_category_value(rule.get("category")), rule.get("rule_key"))
            seen[key] = seen.get(key, 0) + 1
        duplicates = [
            {"category": category, "rule_key": rule_key}
            for (category, rule_key), count in seen.items()
            if count > 1
        ]
        if duplicates:
            raise RuleValidationError(
                message="A rule_key is duplicated within the same category in this submission.",
                details=duplicates,
            )

    @staticmethod
    def _rule_field(rule: dict, key: str) -> object:
        """Reads a structural field (``patient_category``, ``months``,
        ``depends_on_category``) either directly off the rule dict (the
        flat shape used by, e.g., an already-loaded ``RuleDefinition``-style
        mapping) or nested under ``rule_value`` (the shape
        ``schemas/rules/schemas.py``'s ``RuleDefinitionInput.model_dump()``
        produces, where every rule detail beyond ``category``/``rule_key``
        lives in the JSONB ``rule_value`` column) — both are accepted
        identically, since the spec's own wording ("a recall_interval
        rule's `months` value ... for the same patient_category") never
        names an intermediate ``rule_value`` wrapper as mandatory.
        """
        if rule.get(key) is not None:
            return rule[key]
        rule_value = rule.get("rule_value")
        if isinstance(rule_value, dict):
            return rule_value.get(key)
        return None

    @staticmethod
    def _check_recall_interval_conflicts(rules: list[dict]) -> None:
        """US14 AC (b): two ``recall_interval`` rules for the same
        ``patient_category`` must agree on ``months`` — one category cannot
        carry two different interval values in a single submission.
        """
        months_by_patient_category: dict[object, object] = {}
        for rule in rules:
            if _category_value(rule.get("category")) != RuleCategory.recall_interval.value:
                continue
            patient_category = RuleValidationService._rule_field(rule, "patient_category")
            months = RuleValidationService._rule_field(rule, "months")
            if patient_category is None or months is None:
                continue
            if patient_category in months_by_patient_category:
                if months_by_patient_category[patient_category] != months:
                    raise RuleValidationError(
                        message=(
                            "Conflicting recall_interval months values for the same "
                            "patient_category in this submission."
                        ),
                        details=[
                            {
                                "patient_category": patient_category,
                                "months": [months_by_patient_category[patient_category], months],
                            }
                        ],
                    )
            else:
                months_by_patient_category[patient_category] = months

    @staticmethod
    def _check_cross_references(rules: list[dict]) -> None:
        """US14 AC (c): a ``triage_question``/``escalation_trigger`` rule may
        declare a ``depends_on_category`` field in its ``rule_value`` naming
        the ``RuleCategory`` it structurally depends on (e.g. an escalation
        trigger keyed off a ``risk_classification`` threshold). The spec
        leaves the exact field name open; ``depends_on_category`` is chosen
        here as the self-documenting, conventional name, and must reference
        one of the fixed ``RuleCategory`` keys — a value outside that fixed
        vocabulary is, structurally, an undefined category key.
        """
        valid_categories = {category.value for category in RuleCategory}
        dependent_categories = {RuleCategory.triage_question.value, RuleCategory.escalation_trigger.value}
        for rule in rules:
            if _category_value(rule.get("category")) not in dependent_categories:
                continue
            depends_on = RuleValidationService._rule_field(rule, "depends_on_category")
            if depends_on is None:
                continue
            depends_on_value = _category_value(depends_on)
            if depends_on_value not in valid_categories:
                raise RuleValidationError(
                    message="A rule references an undefined category key it depends on.",
                    details=[
                        {
                            "category": _category_value(rule.get("category")),
                            "rule_key": rule.get("rule_key"),
                            "depends_on_category": depends_on_value,
                        }
                    ],
                )


class RuleSetService:
    """FR-E3.2/US15/US16: the draft/approval/versioning/activation workflow,
    and the one read method ("RuleSetService (read)") every other module's
    service calls for the live clinical rules.
    """

    def __init__(
        self,
        rule_set_repo: "RuleSetRepository",
        rule_def_repo: "RuleDefinitionRepository",
        validator: RuleValidationService,
        audit: "AuditService",
    ) -> None:
        self._rule_set_repo = rule_set_repo
        self._rule_def_repo = rule_def_repo
        self._validator = validator
        self._audit = audit

    async def save_draft(self, db: "AsyncSession", rules: list[dict], actor_id: "UUID") -> "RuleSet":
        """FR-E3.1/US14: validates structurally before anything is
        persisted, then creates the new draft ``rule_sets`` row and bulk
        inserts its ``rule_definitions`` in the same transaction.
        """
        await self._validator.validate(rules)

        rule_set = await self._rule_set_repo.create_draft(db, created_by_staff_id=actor_id)
        await self._rule_def_repo.bulk_insert(db, rule_set.id, rules)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="rule_set.save_draft",
            entity_type="rule_set",
            entity_id=rule_set.id,
            override_payload={"version_number": rule_set.version_number, "rule_count": len(rules)},
        )
        return rule_set

    async def get_diff(self, db: "AsyncSession", draft_id: "UUID") -> dict:
        """US15: shows what a draft would change relative to the current
        active set before an approver commits to it. Raises ``NotFoundError``
        (404) for an unknown ``draft_id`` rather than silently diffing an
        empty rule set.
        """
        draft = await self._rule_set_repo.get_by_id(db, draft_id)
        if draft is None:
            raise NotFoundError("Rule set draft not found.")
        return await self._rule_set_repo.get_diff(db, draft_id)

    async def approve(self, db: "AsyncSession", rule_set_id: "UUID", actor_id: "UUID") -> "RuleSet":
        """FR-E3.2/US15, T39 AC: requires ``status in {"draft", "in_review"}``
        — role enforcement (Clinic Management only) happens at the route
        layer via ``require_role``/``require_permission``, never re-checked
        here. Delegates the atomic supersede-then-activate transition to
        ``RuleSetRepository.activate`` and audit-logs the approval.
        """
        rule_set = await self._rule_set_repo.get_by_id(db, rule_set_id)
        if rule_set is None:
            raise NotFoundError("Rule set not found.")
        if rule_set.status not in (RuleSetStatus.draft, RuleSetStatus.in_review):
            raise ConflictError(
                message=(
                    "Only a rule set in 'draft' or 'in_review' status can be approved "
                    f"(current status: {_category_value(rule_set.status)})."
                )
            )

        activated = await self._rule_set_repo.activate(db, rule_set_id, actor_id)

        # `version_number` is read defensively (a real `RuleSet` row always
        # carries it, but a minimal collaborator double may not) — it is
        # supplementary audit detail, never required for `entity_id`
        # (`activated.id`, always present) or for the approval itself.
        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="rule_set.approve",
            entity_type="rule_set",
            entity_id=activated.id,
            override_payload={"version_number": getattr(activated, "version_number", None)},
        )
        return activated

    async def request_changes(
        self, db: "AsyncSession", rule_set_id: "UUID", comments: str, actor_id: "UUID"
    ) -> "RuleSet":
        """FR-E3.5/US15 Alternate Flow: sends the draft back to
        ``"in_review"`` with reviewer ``comments`` logged in the audit
        entry's override-payload-equivalent details. The prior active
        version is never touched here and keeps driving live behaviour.
        """
        rule_set = await self._rule_set_repo.set_status(db, rule_set_id, RuleSetStatus.in_review)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="rule_set.request_changes",
            entity_type="rule_set",
            entity_id=rule_set.id,
            override_payload={"comments": comments},
        )
        return rule_set

    async def get_active(self, db: "AsyncSession") -> dict:
        """FR-E3.3/US16: read-only for every authenticated role. Raises
        ``NoActiveRuleSetError`` (404) if no rule set has ever been
        approved; otherwise groups the active set's rule definitions by
        category into the documented ``rules_by_category`` shape.
        """
        active_set = await self._rule_set_repo.get_active(db)
        if active_set is None:
            raise NoActiveRuleSetError()

        rules = await self._rule_def_repo.list_by_set(db, active_set.id)
        rules_by_category: dict[str, list["RuleDefinition"]] = {}
        for rule in rules:
            rules_by_category.setdefault(_category_value(rule.category), []).append(rule)

        return {"version_number": active_set.version_number, "rules_by_category": rules_by_category}

    async def get_active_rules_by_category(
        self, db: "AsyncSession", category: str
    ) -> list["RuleDefinition"]:
        """Interaction contract: the one method every other module's
        service depends on (never this class's write methods) to read live
        clinical rules — architecture.md §5.3's "RuleSetService (read)".
        Raises ``NoActiveRuleSetError`` (404) if no active rule set exists
        at all, rather than returning an empty list, so a caller like
        recall's interval lookup can tell "no rules configured" apart from
        "rules configured with zero entries in this category" and fall back
        to its own documented default only in the former case.
        """
        active_set = await self._rule_set_repo.get_active(db)
        if active_set is None:
            raise NoActiveRuleSetError()
        return await self._rule_def_repo.list_by_set_and_category(db, active_set.id, category)


class ConfigurationService:
    """FR-E3.4/US17: the versioned, approval-free (delivery_team-only)
    eligibility/cadence/prioritisation configuration store.
    """

    def __init__(self, config_repo: "ConfigurationParameterRepository", audit: "AuditService") -> None:
        self._config_repo = config_repo
        self._audit = audit

    async def save(self, db: "AsyncSession", name: str, value, actor_id: "UUID") -> "ConfigurationParameter":
        """FR-E3.4/US17: writes a new version row (never mutates a prior
        one — every change is retained for ``rollback``) and audit-logs it.
        """
        parameter = await self._config_repo.save_version(db, name, value, actor_id)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="configuration.save",
            entity_type="configuration_parameter",
            entity_id=parameter.id,
            override_payload={"name": name, "value": value, "version_number": parameter.version_number},
        )
        return parameter

    async def rollback(
        self, db: "AsyncSession", version_number: int, actor_id: "UUID"
    ) -> "ConfigurationParameter":
        """T43 AC: restores the exact values of ``version_number`` as a
        brand-new current version rather than deleting/rewriting history —
        delegated to ``ConfigurationParameterRepository.rollback``.
        """
        parameter = await self._config_repo.rollback(db, version_number, actor_id)

        await self._audit.record(
            db,
            actor_staff_id=actor_id,
            actor_type=ActorType.staff,
            action_type="configuration.rollback",
            entity_type="configuration_parameter",
            entity_id=parameter.id,
            override_payload={
                "name": parameter.name,
                "restored_from_version": version_number,
                "new_version_number": parameter.version_number,
            },
        )
        return parameter

    async def get_live(self, db: "AsyncSession", name: str, default=None):
        """T44 AC/FR-E3.4: the "no code release" read path — every caller
        re-reads via this method on each use rather than caching a value at
        process start, so a save/rollback here takes effect immediately for
        every other module's next read. Never raises: returns ``default``
        untouched when no row exists for ``name`` yet.
        """
        parameter = await self._config_repo.get_latest(db, name)
        if parameter is None:
            return default
        return parameter.value
