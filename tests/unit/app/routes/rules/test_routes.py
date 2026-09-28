"""Unit tests for app/routes/rules/routes.py.

These tests exercise the FastAPI router directly (mounted on a throwaway
FastAPI() app with the global exception handlers registered), stubbing out
RuleValidationService / RuleSetService / ConfigurationService -- the routes
file is documented as a thin HTTP adapter over those three services -- and
overriding the auth dependencies (get_current_user / get_db) so each test
controls the caller's role without a real JWT or DB.
"""
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.rules.routes import router
from app.core.dependencies import CurrentUser, get_current_user, get_db
from app.common.exceptions.handlers import register_exception_handlers
from app.common.exceptions.errors import NoActiveRuleSetError, RuleValidationError


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    register_exception_handlers(app)
    return app


def _client_as(app: FastAPI, role: str, user_id=None) -> TestClient:
    user_id = user_id or uuid.uuid4()

    def _override_user():
        return CurrentUser(id=user_id, role=role)

    def _override_db():
        return object()

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /rules/rule-sets  (delivery_team only)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.RuleSetService")
def test_save_draft_rule_set_returns_201_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    rule_set_id = uuid.uuid4()
    mock_service_cls.return_value.save_draft = AsyncMock(
        return_value=SimpleNamespace(id=rule_set_id, version_number=3, status="draft")
    )

    resp = client.post(
        "/rules/rule-sets",
        json={"rules": [{"category": "recall_interval", "rule_key": "k1", "rule_value": {"months": 6}}]},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == str(rule_set_id)
    assert body["version_number"] == 3
    assert body["status"] == "draft"


@patch("app.routes.rules.routes.RuleSetService")
def test_save_draft_rule_set_forbidden_for_clinic_management(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="clinic_management")

    resp = client.post(
        "/rules/rule-sets",
        json={"rules": [{"category": "recall_interval", "rule_key": "k1", "rule_value": {"months": 6}}]},
    )

    assert resp.status_code == 403
    mock_service_cls.return_value.save_draft.assert_not_called()


@patch("app.routes.rules.routes.RuleSetService")
def test_save_draft_rule_set_forbidden_for_front_office_staff(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="front_office_staff")

    resp = client.post(
        "/rules/rule-sets",
        json={"rules": [{"category": "recall_interval", "rule_key": "k1", "rule_value": {"months": 6}}]},
    )

    assert resp.status_code == 403


@patch("app.routes.rules.routes.RuleSetService")
def test_save_draft_rule_set_returns_422_on_rule_validation_error(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    mock_service_cls.return_value.save_draft = AsyncMock(
        side_effect=RuleValidationError("duplicate rule_key in submission")
    )

    resp = client.post(
        "/rules/rule-sets",
        json={"rules": [{"category": "recall_interval", "rule_key": "k1", "rule_value": {"months": 6}}]},
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "RULE_VALIDATION_FAILED"


def test_save_draft_rule_set_returns_422_on_malformed_body():
    app = _build_app()
    client = _client_as(app, role="delivery_team")

    # invalid category value (not a RuleCategory member) -> pydantic validation failure
    resp = client.post(
        "/rules/rule-sets",
        json={"rules": [{"category": "not_a_real_category", "rule_key": "k1", "rule_value": {}}]},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_FAILED"


# ---------------------------------------------------------------------------
# GET /rules/rule-sets/{id}/diff  (clinic_management or delivery_team)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.RuleSetService")
def test_get_rule_set_diff_returns_200_for_clinic_management(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="clinic_management")
    rule_set_id = uuid.uuid4()
    mock_service_cls.return_value.get_diff = AsyncMock(
        return_value={
            "added": [{"rule_key": "new_rule"}],
            "changed": [],
            "removed": [{"rule_key": "old_rule"}],
        }
    )

    resp = client.get(f"/rules/rule-sets/{rule_set_id}/diff")

    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == [{"rule_key": "new_rule"}]
    assert body["changed"] == []
    assert body["removed"] == [{"rule_key": "old_rule"}]


@patch("app.routes.rules.routes.RuleSetService")
def test_get_rule_set_diff_returns_200_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    rule_set_id = uuid.uuid4()
    mock_service_cls.return_value.get_diff = AsyncMock(
        return_value={"added": [], "changed": [], "removed": []}
    )

    resp = client.get(f"/rules/rule-sets/{rule_set_id}/diff")

    assert resp.status_code == 200


@patch("app.routes.rules.routes.RuleSetService")
def test_get_rule_set_diff_forbidden_for_front_office_staff(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="front_office_staff")
    rule_set_id = uuid.uuid4()

    resp = client.get(f"/rules/rule-sets/{rule_set_id}/diff")

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /rules/rule-sets/{id}/approve  (clinic_management only)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.RuleSetService")
def test_approve_rule_set_returns_200_for_clinic_management(mock_service_cls):
    app = _build_app()
    actor_id = uuid.uuid4()
    client = _client_as(app, role="clinic_management", user_id=actor_id)
    rule_set_id = uuid.uuid4()
    approved_at = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
    mock_service_cls.return_value.approve = AsyncMock(
        return_value=SimpleNamespace(
            id=rule_set_id, status="active", approved_by=str(actor_id), approved_at=approved_at
        )
    )

    resp = client.post(f"/rules/rule-sets/{rule_set_id}/approve")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(rule_set_id)
    assert body["status"] == "active"
    assert body["approved_by"] == str(actor_id)
    mock_service_cls.return_value.approve.assert_awaited_once()
    awaited_args = mock_service_cls.return_value.approve.await_args
    all_values = list(awaited_args.args) + list(awaited_args.kwargs.values())
    assert rule_set_id in all_values


@patch("app.routes.rules.routes.RuleSetService")
def test_approve_rule_set_forbidden_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    rule_set_id = uuid.uuid4()

    resp = client.post(f"/rules/rule-sets/{rule_set_id}/approve")

    assert resp.status_code == 403


@patch("app.routes.rules.routes.RuleSetService")
def test_approve_rule_set_forbidden_for_front_office_staff(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="front_office_staff")
    rule_set_id = uuid.uuid4()

    resp = client.post(f"/rules/rule-sets/{rule_set_id}/approve")

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /rules/rule-sets/{id}/request-changes  (clinic_management only)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.RuleSetService")
def test_request_rule_set_changes_returns_200_and_passes_comments(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="clinic_management")
    rule_set_id = uuid.uuid4()
    mock_service_cls.return_value.request_changes = AsyncMock(
        return_value=SimpleNamespace(id=rule_set_id, status="in_review")
    )

    resp = client.post(
        f"/rules/rule-sets/{rule_set_id}/request-changes",
        json={"comments": "please revise recall interval for high risk"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "in_review"
    mock_service_cls.return_value.request_changes.assert_awaited_once()
    awaited_args = mock_service_cls.return_value.request_changes.await_args
    all_values = list(awaited_args.args) + list(awaited_args.kwargs.values())
    assert "please revise recall interval for high risk" in all_values


@patch("app.routes.rules.routes.RuleSetService")
def test_request_rule_set_changes_forbidden_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    rule_set_id = uuid.uuid4()

    resp = client.post(
        f"/rules/rule-sets/{rule_set_id}/request-changes",
        json={"comments": "not allowed"},
    )

    assert resp.status_code == 403


def test_request_rule_set_changes_missing_comments_is_422():
    app = _build_app()
    client = _client_as(app, role="clinic_management")
    rule_set_id = uuid.uuid4()

    resp = client.post(f"/rules/rule-sets/{rule_set_id}/request-changes", json={})

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /rules/rule-sets/active  (any authenticated role)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.RuleSetService")
def test_get_active_rule_set_returns_200_for_front_office_staff(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="front_office_staff")
    mock_service_cls.return_value.get_active = AsyncMock(
        return_value={
            "version_number": 5,
            "rules_by_category": {
                "recall_interval": [
                    {"category": "recall_interval", "rule_key": "adult", "rule_value": {"months": 6}}
                ]
            },
        }
    )

    resp = client.get("/rules/rule-sets/active")

    assert resp.status_code == 200
    body = resp.json()
    assert body["version_number"] == 5
    assert "recall_interval" in body["rules_by_category"]


@patch("app.routes.rules.routes.RuleSetService")
def test_get_active_rule_set_returns_200_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    mock_service_cls.return_value.get_active = AsyncMock(
        return_value={"version_number": 1, "rules_by_category": {}}
    )

    resp = client.get("/rules/rule-sets/active")

    assert resp.status_code == 200


@patch("app.routes.rules.routes.RuleSetService")
def test_get_active_rule_set_returns_404_when_none_active(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="clinic_management")
    mock_service_cls.return_value.get_active = AsyncMock(
        side_effect=NoActiveRuleSetError("no active rules configured")
    )

    resp = client.get("/rules/rule-sets/active")

    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "NO_ACTIVE_RULE_SET"
    assert body["error"]["message"] == "no active rules configured"


# ---------------------------------------------------------------------------
# PUT /rules/configuration  (delivery_team only)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.ConfigurationService")
def test_update_configuration_returns_200_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    config_id = uuid.uuid4()
    effective_at = datetime(2026, 9, 28, tzinfo=timezone.utc)
    mock_service_cls.return_value.save = AsyncMock(
        return_value=SimpleNamespace(id=config_id, version_number=2, effective_at=effective_at)
    )

    resp = client.put(
        "/rules/configuration",
        json={"name": "outreach_retry_delay_minutes", "value": 15},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(config_id)
    assert body["version_number"] == 2
    mock_service_cls.return_value.save.assert_awaited_once()


@patch("app.routes.rules.routes.ConfigurationService")
def test_update_configuration_forbidden_for_clinic_management(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="clinic_management")

    resp = client.put(
        "/rules/configuration",
        json={"name": "outreach_retry_delay_minutes", "value": 15},
    )

    assert resp.status_code == 403


@patch("app.routes.rules.routes.ConfigurationService")
def test_update_configuration_forbidden_for_front_office_staff(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="front_office_staff")

    resp = client.put(
        "/rules/configuration",
        json={"name": "outreach_retry_delay_minutes", "value": 15},
    )

    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# POST /rules/configuration/{version}/rollback  (delivery_team only)
# ---------------------------------------------------------------------------

@patch("app.routes.rules.routes.ConfigurationService")
def test_rollback_configuration_returns_200_for_delivery_team(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="delivery_team")
    config_id = uuid.uuid4()
    mock_service_cls.return_value.rollback = AsyncMock(
        return_value=SimpleNamespace(id=config_id, version_number=4, rolled_back_from=5)
    )

    resp = client.post("/rules/configuration/3/rollback")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(config_id)
    assert body["version_number"] == 4
    assert body["rolled_back_from"] == 5
    mock_service_cls.return_value.rollback.assert_awaited_once()
    awaited_args = mock_service_cls.return_value.rollback.await_args
    all_values = list(awaited_args.args) + list(awaited_args.kwargs.values())
    assert 3 in all_values


@patch("app.routes.rules.routes.ConfigurationService")
def test_rollback_configuration_forbidden_for_clinic_management(mock_service_cls):
    app = _build_app()
    client = _client_as(app, role="clinic_management")

    resp = client.post("/rules/configuration/3/rollback")

    assert resp.status_code == 403
