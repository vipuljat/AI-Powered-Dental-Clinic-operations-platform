"""Unit tests for app/services/rules/service.py.

Written from the spec alone (specs/app/services/rules/service.py.md) against the
declared public surface and the real signatures of its dependencies
(app/repositories/rules/repository.py, app/common/exceptions/errors.py).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.common.exceptions.errors import NoActiveRuleSetError, RuleValidationError
from app.services.rules.service import (
    ConfigurationService,
    RuleSetService,
    RuleValidationService,
)


# ---------------------------------------------------------------------------
# RuleValidationService
# ---------------------------------------------------------------------------

@pytest.fixture
def validator():
    rule_def_repo = MagicMock()
    return RuleValidationService(rule_def_repo=rule_def_repo)


class TestRuleValidationService:
    async def test_well_formed_rules_do_not_raise(self, validator):
        rules = [
            {"category": "eligibility", "rule_key": "min_age", "value": 5},
            {
                "category": "recall_interval",
                "rule_key": "adult_cleaning",
                "patient_category": "adult",
                "months": 6,
            },
        ]
        # should not raise
        await validator.validate(rules)

    async def test_duplicate_rule_key_within_same_category_raises(self, validator):
        rules = [
            {"category": "eligibility", "rule_key": "min_age", "value": 5},
            {"category": "eligibility", "rule_key": "min_age", "value": 10},
        ]
        with pytest.raises(RuleValidationError) as exc_info:
            await validator.validate(rules)
        assert exc_info.value.status_code == 422
        assert exc_info.value.code == "RULE_VALIDATION_FAILED"

    async def test_duplicate_rule_key_across_different_categories_is_allowed(self, validator):
        rules = [
            {"category": "eligibility", "rule_key": "min_age", "value": 5},
            {"category": "prioritisation", "rule_key": "min_age", "value": 1},
        ]
        # same rule_key but different category -> not a conflict
        await validator.validate(rules)

    async def test_conflicting_recall_interval_months_for_same_patient_category_raises(
        self, validator
    ):
        rules = [
            {
                "category": "recall_interval",
                "rule_key": "adult_cleaning_a",
                "patient_category": "adult",
                "months": 6,
            },
            {
                "category": "recall_interval",
                "rule_key": "adult_cleaning_b",
                "patient_category": "adult",
                "months": 12,
            },
        ]
        with pytest.raises(RuleValidationError):
            await validator.validate(rules)

    async def test_same_recall_interval_months_for_same_patient_category_is_allowed(
        self, validator
    ):
        rules = [
            {
                "category": "recall_interval",
                "rule_key": "adult_cleaning_a",
                "patient_category": "adult",
                "months": 6,
            },
            {
                "category": "recall_interval",
                "rule_key": "adult_cleaning_b",
                "patient_category": "adult",
                "months": 6,
            },
        ]
        await validator.validate(rules)

    async def test_undefined_category_cross_reference_raises(self, validator):
        rules = [
            {
                "category": "triage_question",
                "rule_key": "swelling_question",
                "depends_on_category": "nonexistent_category",
            },
        ]
        with pytest.raises(RuleValidationError):
            await validator.validate(rules)


# ---------------------------------------------------------------------------
# RuleSetService
# ---------------------------------------------------------------------------

@pytest.fixture
def rule_set_repo():
    return MagicMock(
        create_draft=AsyncMock(),
        get_by_id=AsyncMock(),
        get_active=AsyncMock(),
        get_diff=AsyncMock(),
        set_status=AsyncMock(),
        activate=AsyncMock(),
    )


@pytest.fixture
def rule_def_repo():
    return MagicMock(
        bulk_insert=AsyncMock(),
        list_by_set=AsyncMock(),
        list_by_set_and_category=AsyncMock(),
    )


@pytest.fixture
def fake_validator():
    return MagicMock(validate=AsyncMock(return_value=None))


@pytest.fixture
def fake_audit():
    return MagicMock(record=AsyncMock(return_value=SimpleNamespace(id=uuid4())))


@pytest.fixture
def rule_set_service(rule_set_repo, rule_def_repo, fake_validator, fake_audit):
    return RuleSetService(
        rule_set_repo=rule_set_repo,
        rule_def_repo=rule_def_repo,
        validator=fake_validator,
        audit=fake_audit,
    )


@pytest.fixture
def db():
    return MagicMock(name="AsyncSession")


class TestRuleSetServiceSaveDraft:
    async def test_save_draft_validates_then_persists_and_audits(
        self, rule_set_service, rule_set_repo, rule_def_repo, fake_validator, fake_audit, db
    ):
        actor_id = uuid4()
        rules = [{"category": "eligibility", "rule_key": "min_age", "value": 5}]
        new_set = SimpleNamespace(id=uuid4(), status="draft", version_number=4)
        rule_set_repo.create_draft.return_value = new_set

        result = await rule_set_service.save_draft(db, rules, actor_id)

        fake_validator.validate.assert_awaited_once_with(rules)
        rule_set_repo.create_draft.assert_awaited_once()
        rule_def_repo.bulk_insert.assert_awaited_once()
        fake_audit.record.assert_awaited_once()
        assert result is new_set

    async def test_save_draft_propagates_validation_error_without_persisting(
        self, rule_set_service, rule_set_repo, rule_def_repo, fake_validator, db
    ):
        actor_id = uuid4()
        rules = [
            {"category": "eligibility", "rule_key": "dup", "value": 1},
            {"category": "eligibility", "rule_key": "dup", "value": 2},
        ]
        fake_validator.validate.side_effect = RuleValidationError("bad rules")

        with pytest.raises(RuleValidationError):
            await rule_set_service.save_draft(db, rules, actor_id)

        rule_set_repo.create_draft.assert_not_awaited()
        rule_def_repo.bulk_insert.assert_not_awaited()


class TestRuleSetServiceGetDiff:
    async def test_get_diff_delegates_to_repository(self, rule_set_service, rule_set_repo, db):
        draft_id = uuid4()
        expected = {"added": [], "removed": [], "changed": []}
        rule_set_repo.get_diff.return_value = expected

        result = await rule_set_service.get_diff(db, draft_id)

        rule_set_repo.get_diff.assert_awaited_once_with(db, draft_id)
        assert result == expected


class TestRuleSetServiceApprove:
    async def test_approve_activates_and_audits(
        self, rule_set_service, rule_set_repo, fake_audit, db
    ):
        actor_id = uuid4()
        rule_set_id = uuid4()
        rule_set_repo.get_by_id.return_value = SimpleNamespace(
            id=rule_set_id, status="in_review"
        )
        activated = SimpleNamespace(id=rule_set_id, status="active")
        rule_set_repo.activate.return_value = activated

        result = await rule_set_service.approve(db, rule_set_id, actor_id)

        rule_set_repo.activate.assert_awaited_once()
        activate_args = rule_set_repo.activate.await_args
        assert rule_set_id in activate_args.args or rule_set_id in activate_args.kwargs.values()
        fake_audit.record.assert_awaited_once()
        _, audit_kwargs = fake_audit.record.await_args
        assert audit_kwargs.get("action_type") == "rule_set.approve"
        assert result is activated

    async def test_approve_from_draft_status_is_accepted(
        self, rule_set_service, rule_set_repo, fake_audit, db
    ):
        actor_id = uuid4()
        rule_set_id = uuid4()
        rule_set_repo.get_by_id.return_value = SimpleNamespace(id=rule_set_id, status="draft")
        rule_set_repo.activate.return_value = SimpleNamespace(id=rule_set_id, status="active")

        await rule_set_service.approve(db, rule_set_id, actor_id)

        rule_set_repo.activate.assert_awaited_once()


class TestRuleSetServiceRequestChanges:
    async def test_request_changes_sets_in_review_and_logs_comments(
        self, rule_set_service, rule_set_repo, fake_audit, db
    ):
        actor_id = uuid4()
        rule_set_id = uuid4()
        comments = "please add a fluoride-varnish eligibility rule"
        updated = SimpleNamespace(id=rule_set_id, status="in_review")
        rule_set_repo.set_status.return_value = updated

        result = await rule_set_service.request_changes(db, rule_set_id, comments, actor_id)

        rule_set_repo.set_status.assert_awaited_once()
        set_status_args = rule_set_repo.set_status.await_args
        assert "in_review" in set_status_args.args or "in_review" in set_status_args.kwargs.values()
        # the prior active version must remain untouched -> activate is never called
        rule_set_repo.activate.assert_not_awaited()
        fake_audit.record.assert_awaited_once()
        _, audit_kwargs = fake_audit.record.await_args
        recorded_values = list(audit_kwargs.values())
        assert any(
            comments in str(v) for v in recorded_values if v is not None
        )
        assert result is updated


class TestRuleSetServiceGetActive:
    async def test_get_active_raises_when_no_active_set(self, rule_set_service, rule_set_repo, db):
        rule_set_repo.get_active.return_value = None

        with pytest.raises(NoActiveRuleSetError) as exc_info:
            await rule_set_service.get_active(db)
        assert exc_info.value.status_code == 404
        assert exc_info.value.code == "NO_ACTIVE_RULE_SET"

    async def test_get_active_groups_rules_by_category(
        self, rule_set_service, rule_set_repo, rule_def_repo, db
    ):
        active_set = SimpleNamespace(id=uuid4(), version_number=3, status="active")
        rule_set_repo.get_active.return_value = active_set
        rule_def_repo.list_by_set.return_value = [
            SimpleNamespace(category="eligibility", rule_key="min_age"),
            SimpleNamespace(category="recall_interval", rule_key="adult_cleaning"),
            SimpleNamespace(category="eligibility", rule_key="max_age"),
        ]

        result = await rule_set_service.get_active(db)

        assert result["version_number"] == 3
        by_category = result["rules_by_category"]
        assert len(by_category["eligibility"]) == 2
        assert len(by_category["recall_interval"]) == 1


class TestRuleSetServiceGetActiveRulesByCategory:
    async def test_raises_no_active_rule_set_when_none_active(
        self, rule_set_service, rule_set_repo, db
    ):
        rule_set_repo.get_active.return_value = None

        with pytest.raises(NoActiveRuleSetError):
            await rule_set_service.get_active_rules_by_category(db, "eligibility")

    async def test_returns_empty_list_when_active_set_has_no_rules_in_category(
        self, rule_set_service, rule_set_repo, rule_def_repo, db
    ):
        active_set = SimpleNamespace(id=uuid4(), version_number=1, status="active")
        rule_set_repo.get_active.return_value = active_set
        rule_def_repo.list_by_set_and_category.return_value = []

        result = await rule_set_service.get_active_rules_by_category(db, "escalation_trigger")

        assert result == []

    async def test_returns_matching_rules_for_category(
        self, rule_set_service, rule_set_repo, rule_def_repo, db
    ):
        active_set = SimpleNamespace(id=uuid4(), version_number=1, status="active")
        rule_set_repo.get_active.return_value = active_set
        expected_rules = [SimpleNamespace(category="recall_interval", rule_key="adult_cleaning")]
        rule_def_repo.list_by_set_and_category.return_value = expected_rules

        result = await rule_set_service.get_active_rules_by_category(db, "recall_interval")

        assert result == expected_rules


# ---------------------------------------------------------------------------
# ConfigurationService
# ---------------------------------------------------------------------------

@pytest.fixture
def config_repo():
    return MagicMock(
        save_version=AsyncMock(),
        rollback=AsyncMock(),
        get_latest=AsyncMock(),
    )


@pytest.fixture
def config_audit():
    return MagicMock(record=AsyncMock(return_value=SimpleNamespace(id=uuid4())))


@pytest.fixture
def configuration_service(config_repo, config_audit):
    return ConfigurationService(config_repo=config_repo, audit=config_audit)


class TestConfigurationServiceSave:
    async def test_save_persists_new_version_and_audits(
        self, configuration_service, config_repo, config_audit, db
    ):
        actor_id = uuid4()
        saved = SimpleNamespace(name="outreach_retry_delay_minutes", value=30, version_number=2)
        config_repo.save_version.return_value = saved

        result = await configuration_service.save(
            db, "outreach_retry_delay_minutes", 30, actor_id
        )

        config_repo.save_version.assert_awaited_once()
        config_audit.record.assert_awaited_once()
        assert result is saved


class TestConfigurationServiceRollback:
    async def test_rollback_restores_prior_version_and_audits(
        self, configuration_service, config_repo, config_audit, db
    ):
        actor_id = uuid4()
        restored = SimpleNamespace(
            name="outreach_retry_delay_minutes", value=15, version_number=9, rolled_back_from_id=7
        )
        config_repo.rollback.return_value = restored

        result = await configuration_service.rollback(db, 7, actor_id)

        config_repo.rollback.assert_awaited_once()
        rollback_args = config_repo.rollback.await_args
        assert 7 in rollback_args.args or 7 in rollback_args.kwargs.values()
        config_audit.record.assert_awaited_once()
        assert result is restored


class TestConfigurationServiceGetLive:
    async def test_get_live_returns_default_when_no_row_exists(
        self, configuration_service, config_repo, db
    ):
        config_repo.get_latest.return_value = None

        result = await configuration_service.get_live(
            db, "outreach_retry_delay_minutes", default=45
        )

        assert result == 45

    async def test_get_live_never_raises_when_missing_and_no_default_given(
        self, configuration_service, config_repo, db
    ):
        config_repo.get_latest.return_value = None

        # must not raise NotFoundError/NoActiveRuleSetError/etc.
        result = await configuration_service.get_live(db, "some_unconfigured_name")

        assert result is None

    async def test_get_live_returns_configured_value_when_present(
        self, configuration_service, config_repo, db
    ):
        config_repo.get_latest.return_value = SimpleNamespace(
            name="outreach_retry_delay_minutes", value=10
        )

        result = await configuration_service.get_live(
            db, "outreach_retry_delay_minutes", default=45
        )

        assert result == 10

    async def test_get_live_re_reads_repository_on_every_call(
        self, configuration_service, config_repo, db
    ):
        config_repo.get_latest.side_effect = [
            SimpleNamespace(name="x", value=1),
            SimpleNamespace(name="x", value=2),
        ]

        first = await configuration_service.get_live(db, "x", default=0)
        second = await configuration_service.get_live(db, "x", default=0)

        assert first == 1
        assert second == 2
        assert config_repo.get_latest.await_count == 2
