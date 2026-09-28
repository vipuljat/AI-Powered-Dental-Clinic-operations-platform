"""Unit tests for app/schemas/rules/schemas.py.

These are pure Pydantic v2 models with no DB/HTTP access, so each is
constructed directly and checked for the exact shape/validation behaviour
its own spec declares. `RuleCategory`/`RuleSetStatus` are not defined by this
spec; per the Interaction contract, `RuleDefinitionInput.category` must
accept exactly the `RuleCategory` enum members (a caller passes the same
string values into `RuleSetService.get_active_rules_by_category`), so those
tests exercise the real enum via introspection rather than hard-coding a
guessed member name.

One Behaviour line -- the `GET /rules/rule-sets/active` 404 body being the
global error envelope's `error.message` -- describes the route + exception
handler's behaviour, not anything this schemas module exposes on its own
public surface (no route, no exception, no handler lives here), so it is not
exercised below; see the summary for this note.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from app.common.enums import RuleCategory, RuleSetStatus
from app.schemas.rules.schemas import (
    ActiveRuleSetResponse,
    ApproveRuleSetResponse,
    ConfigurationResponse,
    RequestChangesRequest,
    RequestChangesResponse,
    RollbackConfigurationResponse,
    RuleDefinitionInput,
    RuleDiffEntry,
    RuleSetDiffResponse,
    RuleSetSummaryResponse,
    SaveDraftRuleSetRequest,
    UpdateConfigurationRequest,
)


def _some_category() -> RuleCategory:
    return next(iter(RuleCategory))


def _some_status() -> RuleSetStatus:
    return next(iter(RuleSetStatus))


# ---------------------------------------------------------------------------
# RuleDefinitionInput
# ---------------------------------------------------------------------------

class TestRuleDefinitionInput:
    def test_accepts_a_real_rule_category_member_and_round_trips_its_value(self):
        category = _some_category()
        model = RuleDefinitionInput(
            category=category, rule_key="max_daily_offers", rule_value={"limit": 3}
        )
        assert model.category == category
        assert model.rule_key == "max_daily_offers"
        assert model.rule_value == {"limit": 3}

    def test_accepts_the_enum_members_raw_string_value_too(self):
        category = _some_category()
        model = RuleDefinitionInput(
            category=category.value, rule_key="k", rule_value={}
        )
        assert model.category == category

    def test_rejects_a_category_string_that_is_not_a_rule_category_member(self):
        bogus = "not_a_real_rule_category__xyz"
        assert bogus not in {member.value for member in RuleCategory}
        with pytest.raises(ValidationError):
            RuleDefinitionInput(category=bogus, rule_key="k", rule_value={})

    def test_rule_value_must_be_a_mapping_not_a_list(self):
        with pytest.raises(ValidationError):
            RuleDefinitionInput(
                category=_some_category(), rule_key="k", rule_value=["not", "a", "dict"]
            )

    def test_missing_required_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RuleDefinitionInput(category=_some_category(), rule_key="k")


# ---------------------------------------------------------------------------
# SaveDraftRuleSetRequest
# ---------------------------------------------------------------------------

class TestSaveDraftRuleSetRequest:
    def test_accepts_a_list_of_rule_definition_inputs(self):
        rule = RuleDefinitionInput(
            category=_some_category(), rule_key="k", rule_value={"a": 1}
        )
        model = SaveDraftRuleSetRequest(rules=[rule])
        assert model.rules == [rule]

    def test_coerces_a_list_of_plain_dicts_into_rule_definition_input(self):
        category = _some_category()
        model = SaveDraftRuleSetRequest(
            rules=[{"category": category.value, "rule_key": "k", "rule_value": {}}]
        )
        assert isinstance(model.rules[0], RuleDefinitionInput)
        assert model.rules[0].category == category

    def test_an_invalid_nested_rule_definition_fails_validation(self):
        with pytest.raises(ValidationError):
            SaveDraftRuleSetRequest(
                rules=[{"category": "bogus", "rule_key": "k", "rule_value": {}}]
            )


# ---------------------------------------------------------------------------
# RuleSetSummaryResponse
# ---------------------------------------------------------------------------

class TestRuleSetSummaryResponse:
    def test_constructs_with_uuid_int_version_and_status_enum(self):
        rule_set_id = uuid.uuid4()
        status = _some_status()
        model = RuleSetSummaryResponse(
            id=rule_set_id, version_number=3, status=status
        )
        assert model.id == rule_set_id
        assert model.version_number == 3
        assert model.status == status

    def test_rejects_a_status_string_that_is_not_a_rule_set_status_member(self):
        bogus = "not_a_real_rule_set_status__xyz"
        assert bogus not in {member.value for member in RuleSetStatus}
        with pytest.raises(ValidationError):
            RuleSetSummaryResponse(id=uuid.uuid4(), version_number=1, status=bogus)

    def test_rejects_a_non_uuid_id(self):
        with pytest.raises(ValidationError):
            RuleSetSummaryResponse(
                id="not-a-uuid", version_number=1, status=_some_status()
            )


# ---------------------------------------------------------------------------
# RuleDiffEntry / RuleSetDiffResponse
# ---------------------------------------------------------------------------

class TestRuleDiffEntry:
    def test_constructs_with_rule_key(self):
        entry = RuleDiffEntry(rule_key="max_daily_offers")
        assert entry.rule_key == "max_daily_offers"

    def test_missing_rule_key_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RuleDiffEntry()


class TestRuleSetDiffResponse:
    def test_constructs_with_added_changed_removed_lists(self):
        added = RuleDiffEntry(rule_key="new_rule")
        changed = RuleDiffEntry(rule_key="changed_rule")
        removed = RuleDiffEntry(rule_key="removed_rule")
        diff = RuleSetDiffResponse(added=[added], changed=[changed], removed=[removed])
        assert diff.added == [added]
        assert diff.changed == [changed]
        assert diff.removed == [removed]

    def test_all_three_lists_may_be_empty_for_a_no_op_diff(self):
        diff = RuleSetDiffResponse(added=[], changed=[], removed=[])
        assert diff.added == []
        assert diff.changed == []
        assert diff.removed == []

    def test_missing_a_required_list_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RuleSetDiffResponse(added=[], changed=[])


# ---------------------------------------------------------------------------
# ApproveRuleSetResponse
# ---------------------------------------------------------------------------

class TestApproveRuleSetResponse:
    def test_constructs_with_id_status_approver_and_timestamp(self):
        rule_set_id = uuid.uuid4()
        status = _some_status()
        approved_at = datetime(2026, 1, 5, 12, 30, tzinfo=timezone.utc)
        model = ApproveRuleSetResponse(
            id=rule_set_id,
            status=status,
            approved_by="dr_jones",
            approved_at=approved_at,
        )
        assert model.id == rule_set_id
        assert model.status == status
        assert model.approved_by == "dr_jones"
        assert model.approved_at == approved_at

    def test_missing_approved_by_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ApproveRuleSetResponse(
                id=uuid.uuid4(),
                status=_some_status(),
                approved_at=datetime.now(timezone.utc),
            )


# ---------------------------------------------------------------------------
# RequestChangesRequest / RequestChangesResponse
# ---------------------------------------------------------------------------

class TestRequestChangesRequest:
    def test_constructs_with_comments(self):
        model = RequestChangesRequest(comments="please tighten the cadence rule")
        assert model.comments == "please tighten the cadence rule"

    def test_missing_comments_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RequestChangesRequest()


class TestRequestChangesResponse:
    def test_constructs_with_id_and_status(self):
        rule_set_id = uuid.uuid4()
        status = _some_status()
        model = RequestChangesResponse(id=rule_set_id, status=status)
        assert model.id == rule_set_id
        assert model.status == status


# ---------------------------------------------------------------------------
# ActiveRuleSetResponse
# ---------------------------------------------------------------------------

class TestActiveRuleSetResponse:
    def test_constructs_with_version_number_and_rules_grouped_by_category(self):
        category = _some_category()
        model = ActiveRuleSetResponse(
            version_number=7,
            rules_by_category={
                category.value: [
                    {"category": category.value, "rule_key": "k", "rule_value": {"x": 1}}
                ]
            },
        )
        assert model.version_number == 7
        assert category.value in model.rules_by_category
        entry = model.rules_by_category[category.value][0]
        assert isinstance(entry, RuleDefinitionInput)
        assert entry.rule_key == "k"

    def test_an_empty_mapping_is_a_valid_rules_by_category(self):
        model = ActiveRuleSetResponse(version_number=1, rules_by_category={})
        assert model.rules_by_category == {}


# ---------------------------------------------------------------------------
# UpdateConfigurationRequest
# ---------------------------------------------------------------------------

class TestUpdateConfigurationRequest:
    @pytest.mark.parametrize(
        "value",
        [42, 3.14, "a string value", True, {"nested": "dict"}, [1, 2, 3]],
    )
    def test_accepts_every_declared_value_type_and_preserves_its_python_type(self, value):
        model = UpdateConfigurationRequest(name="max_daily_offers", value=value)
        assert model.value == value
        assert type(model.value) is type(value)

    def test_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            UpdateConfigurationRequest(value=1)


# ---------------------------------------------------------------------------
# ConfigurationResponse
# ---------------------------------------------------------------------------

class TestConfigurationResponse:
    def test_constructs_with_id_version_number_and_effective_at(self):
        config_id = uuid.uuid4()
        effective_at = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
        model = ConfigurationResponse(
            id=config_id, version_number=2, effective_at=effective_at
        )
        assert model.id == config_id
        assert model.version_number == 2
        assert model.effective_at == effective_at

    def test_missing_effective_at_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ConfigurationResponse(id=uuid.uuid4(), version_number=2)


# ---------------------------------------------------------------------------
# RollbackConfigurationResponse
# ---------------------------------------------------------------------------

class TestRollbackConfigurationResponse:
    def test_constructs_with_id_version_number_and_rolled_back_from(self):
        config_id = uuid.uuid4()
        model = RollbackConfigurationResponse(
            id=config_id, version_number=4, rolled_back_from=5
        )
        assert model.id == config_id
        assert model.version_number == 4
        assert model.rolled_back_from == 5

    def test_missing_rolled_back_from_raises_validation_error(self):
        with pytest.raises(ValidationError):
            RollbackConfigurationResponse(id=uuid.uuid4(), version_number=4)


# ---------------------------------------------------------------------------
# Every declared model really is a BaseModel (schemas.py's stated public
# surface: "Pydantic v2 request/response models for every /rules/* endpoint")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_cls",
    [
        RuleDefinitionInput,
        SaveDraftRuleSetRequest,
        RuleSetSummaryResponse,
        RuleDiffEntry,
        RuleSetDiffResponse,
        ApproveRuleSetResponse,
        RequestChangesRequest,
        RequestChangesResponse,
        ActiveRuleSetResponse,
        UpdateConfigurationRequest,
        ConfigurationResponse,
        RollbackConfigurationResponse,
    ],
)
def test_every_public_model_is_a_pydantic_base_model(model_cls):
    assert issubclass(model_cls, BaseModel)
