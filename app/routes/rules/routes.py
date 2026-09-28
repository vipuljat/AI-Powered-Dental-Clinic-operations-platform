"""Thin HTTP adapter over ``RuleValidationService``/``RuleSetService``/
``ConfigurationService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema, delegates to the service layer, and shapes the response. This
module's ``router`` carries only its own ``/rules`` sub-prefix; the shared
``/api/v1`` prefix is added on top by ``app/main.py`` when it calls
``app.include_router`` (Interaction contract).
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from app.core.dependencies import CurrentUser, get_current_user, get_db, require_role
from app.repositories.console.repository import AuditLogRepository
from app.repositories.rules.repository import (
    ConfigurationParameterRepository,
    RuleDefinitionRepository,
    RuleSetRepository,
)
from app.schemas.rules.schemas import (
    ActiveRuleSetResponse,
    ApproveRuleSetResponse,
    ConfigurationResponse,
    RequestChangesRequest,
    RequestChangesResponse,
    RollbackConfigurationResponse,
    RuleSetDiffResponse,
    RuleSetSummaryResponse,
    SaveDraftRuleSetRequest,
    UpdateConfigurationRequest,
)
from app.services.console.service import AuditService
from app.services.rules.service import ConfigurationService, RuleSetService, RuleValidationService

router = APIRouter(prefix="/rules")


def _audit_service() -> AuditService:
    # Each request builds its own thin service/repository graph over the
    # request-scoped `AsyncSession` (`Depends(get_db)`) — no collaborator
    # here is a module-level singleton, matching the rest of this codebase's
    # `require_permission`/`AuthService` wiring (app/core/dependencies.py).
    return AuditService(AuditLogRepository())


def _rule_set_service() -> RuleSetService:
    rule_def_repo = RuleDefinitionRepository()
    validator = RuleValidationService(rule_def_repo)
    return RuleSetService(RuleSetRepository(), rule_def_repo, validator, _audit_service())


def _configuration_service() -> ConfigurationService:
    return ConfigurationService(ConfigurationParameterRepository(), _audit_service())


@router.post("/rule-sets", response_model=RuleSetSummaryResponse, status_code=201)
async def save_draft_rule_set(
    body: SaveDraftRuleSetRequest,
    user: CurrentUser = Depends(require_role("delivery_team")),
    db=Depends(get_db),
) -> RuleSetSummaryResponse:
    """FR-E1.3/FR-E3.1/US14: ``delivery_team``-only — the one place this
    module enforces the BRD's "restrict rule-configuration actions to the
    47Billion Delivery Team role only". 201 on success, 422
    (``RuleValidationError``) on a structurally-invalid draft, left to
    propagate to the registered global handler.
    """
    service = _rule_set_service()
    rules = [rule.model_dump() for rule in body.rules]
    rule_set = await service.save_draft(db, rules, user.id)
    return RuleSetSummaryResponse(
        id=rule_set.id,
        version_number=rule_set.version_number,
        status=rule_set.status,
    )


@router.get("/rule-sets/{id}/diff", response_model=RuleSetDiffResponse)
async def get_rule_set_diff(
    id: UUID,
    user: CurrentUser = Depends(require_role("clinic_management", "delivery_team")),
    db=Depends(get_db),
) -> RuleSetDiffResponse:
    """US15: 404 (``NotFoundError``) for an unknown draft ``id`` propagates
    unchanged to the registered global handler.
    """
    service = _rule_set_service()
    diff = await service.get_diff(db, id)
    return RuleSetDiffResponse(
        added=[{"rule_key": entry["rule_key"]} for entry in diff["added"]],
        changed=[{"rule_key": entry["rule_key"]} for entry in diff["changed"]],
        removed=[{"rule_key": entry["rule_key"]} for entry in diff["removed"]],
    )


@router.post("/rule-sets/{id}/approve", response_model=ApproveRuleSetResponse)
async def approve_rule_set(
    id: UUID,
    user: CurrentUser = Depends(require_role("clinic_management")),
    db=Depends(get_db),
) -> ApproveRuleSetResponse:
    """FR-E3.2/US15, T39 AC: ``clinic_management``-only. 404
    (``NotFoundError``) / 409 (``ConflictError``) propagate unchanged to the
    registered global handler.
    """
    service = _rule_set_service()
    rule_set = await service.approve(db, id, user.id)
    return ApproveRuleSetResponse(
        id=rule_set.id,
        status=rule_set.status,
        approved_by=str(rule_set.approved_by_staff_id),
        approved_at=rule_set.approved_at,
    )


@router.post("/rule-sets/{id}/request-changes", response_model=RequestChangesResponse)
async def request_rule_set_changes(
    id: UUID,
    body: RequestChangesRequest,
    user: CurrentUser = Depends(require_role("clinic_management")),
    db=Depends(get_db),
) -> RequestChangesResponse:
    """FR-E3.5/US15 Alternate Flow: sends the draft back to
    ``"in_review"`` with reviewer comments logged in the audit trail.
    """
    service = _rule_set_service()
    rule_set = await service.request_changes(db, id, body.comments, user.id)
    return RequestChangesResponse(id=rule_set.id, status=rule_set.status)


@router.get("/rule-sets/active", response_model=ActiveRuleSetResponse)
async def get_active_rule_set(
    user: CurrentUser = Depends(get_current_user),
    db=Depends(get_db),
) -> ActiveRuleSetResponse:
    """FR-E3.3/US16: any authenticated role may read the active rule set —
    ``get_current_user`` only, no role restriction (architecture.md §5.2).
    404 (``NoActiveRuleSetError``) propagates unchanged to the registered
    global handler when no rule set has ever been approved.
    """
    service = _rule_set_service()
    active = await service.get_active(db)
    return ActiveRuleSetResponse(
        version_number=active["version_number"],
        rules_by_category={
            category: [
                {
                    "category": rule.category,
                    "rule_key": rule.rule_key,
                    "rule_value": rule.rule_value,
                }
                for rule in rules
            ]
            for category, rules in active["rules_by_category"].items()
        },
    )


@router.put("/configuration", response_model=ConfigurationResponse)
async def update_configuration(
    body: UpdateConfigurationRequest,
    user: CurrentUser = Depends(require_role("delivery_team")),
    db=Depends(get_db),
) -> ConfigurationResponse:
    """FR-E1.3/FR-E3.4/US17: ``delivery_team``-only, approval-free config
    write — takes effect immediately for every other module's next
    ``ConfigurationService.get_live`` read.
    """
    service = _configuration_service()
    parameter = await service.save(db, body.name, body.value, user.id)
    return ConfigurationResponse(
        id=parameter.id,
        version_number=parameter.version_number,
        effective_at=parameter.effective_at,
    )


@router.post("/configuration/{version}/rollback", response_model=RollbackConfigurationResponse)
async def rollback_configuration(
    version: int,
    user: CurrentUser = Depends(require_role("delivery_team")),
    db=Depends(get_db),
) -> RollbackConfigurationResponse:
    """FR-E1.3/T43 AC: ``delivery_team``-only. 404 (``NotFoundError``)
    propagates unchanged when ``version`` does not exist.
    """
    service = _configuration_service()
    parameter = await service.rollback(db, version, user.id)
    return RollbackConfigurationResponse(
        id=parameter.id,
        version_number=parameter.version_number,
        rolled_back_from=version,
    )
