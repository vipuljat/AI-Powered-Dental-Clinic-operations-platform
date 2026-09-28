"""Pydantic v2 request/response models for every ``/rules/*`` endpoint
(architecture.md §5.2).

Contains no behaviour beyond field validation — request/response shape only.
Never redeclares an enum value set locally; imports from app.common.enums.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import RuleCategory, RuleSetStatus


class RuleDefinitionInput(BaseModel):
    """A single rule definition submitted as part of a draft rule set.

    Interaction contract: ``category`` must be exactly one of the
    ``RuleCategory`` enum members — every other module's
    ``RuleSetService.get_active_rules_by_category(category)`` call passes one
    of these same string values, so a category spelled differently here than
    in a caller's lookup would silently return zero rules rather than
    erroring.
    """

    category: RuleCategory
    rule_key: str
    rule_value: dict


class SaveDraftRuleSetRequest(BaseModel):
    rules: list[RuleDefinitionInput]


class RuleSetSummaryResponse(BaseModel):
    id: UUID
    version_number: int
    status: RuleSetStatus


class RuleDiffEntry(BaseModel):
    rule_key: str


class RuleSetDiffResponse(BaseModel):
    added: list[RuleDiffEntry]
    changed: list[RuleDiffEntry]
    removed: list[RuleDiffEntry]


class ApproveRuleSetResponse(BaseModel):
    id: UUID
    status: RuleSetStatus
    approved_by: str
    approved_at: datetime


class RequestChangesRequest(BaseModel):
    comments: str


class RequestChangesResponse(BaseModel):
    id: UUID
    status: RuleSetStatus


class ActiveRuleSetResponse(BaseModel):
    """architecture.md §5.2 `GET /rules/rule-sets/active`.

    When no rule set is active the route raises ``NoActiveRuleSetError`` and
    the global exception handler (app.common.exceptions.handlers) wraps it in
    the standard error envelope with
    ``error.message = "no active rules configured"`` — the documented
    ``404`` body IS the envelope's ``error.message``, not a bespoke response
    shape, so this success model carries no special-casing for that path.
    """

    version_number: int
    rules_by_category: dict[str, list[RuleDefinitionInput]]


class UpdateConfigurationRequest(BaseModel):
    name: str
    value: int | float | str | bool | dict | list


class ConfigurationResponse(BaseModel):
    id: UUID
    version_number: int
    effective_at: datetime


class RollbackConfigurationResponse(BaseModel):
    id: UUID
    version_number: int
    rolled_back_from: int
