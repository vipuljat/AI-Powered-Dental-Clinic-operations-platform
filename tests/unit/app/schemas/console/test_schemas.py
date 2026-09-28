"""Unit tests for app/schemas/console/schemas.py.

Pure Pydantic v2 models (no DB/HTTP access), so each is constructed
directly and checked for the exact shape/validation behaviour its own
spec declares. `Role`/`StaffStatus` are defined in app.common.enums (not
this file), so those are imported and exercised through the real enum
rather than a hard-coded guessed member name. `validate_non_empty` /
`validate_password_complexity` are imported from app.common.validators
per the spec's "Public surface" (`_validate_reason = field_validator(...)`
etc.) and are exercised only through the schema fields that wire them in,
never called directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from app.common.enums import Role, StaffStatus
from app.schemas.console.schemas import (
    AuditLogItem,
    AuditLogQueryResponse,
    DashboardWidget,
    DashboardWidgetsResponse,
    DeactivateStaffResponse,
    LoginRequest,
    LoginResponse,
    LoginUser,
    OverrideRequest,
    OverrideResponse,
    PasswordResetConfirmRequest,
    PasswordResetRequestRequest,
    ProvisionStaffRequest,
    StaffResponse,
)


def _some_role() -> Role:
    return next(iter(Role))


def _some_staff_status() -> StaffStatus:
    return next(iter(StaffStatus))


# ---------------------------------------------------------------------------
# LoginRequest
# ---------------------------------------------------------------------------

class TestLoginRequest:
    def test_constructs_with_username_and_password(self):
        model = LoginRequest(username="jane.silva", password="s3cret123")
        assert model.username == "jane.silva"
        assert model.password == "s3cret123"

    def test_missing_password_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LoginRequest(username="jane.silva")

    def test_missing_username_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LoginRequest(password="s3cret123")


# ---------------------------------------------------------------------------
# LoginUser
# ---------------------------------------------------------------------------

class TestLoginUser:
    def test_constructs_with_id_username_and_role(self):
        user_id = uuid.uuid4()
        role = _some_role()
        model = LoginUser(id=user_id, username="jane.silva", role=role)
        assert model.id == user_id
        assert model.username == "jane.silva"
        assert model.role == role

    def test_role_accepts_the_enum_members_raw_string_value(self):
        model = LoginUser(id=uuid.uuid4(), username="jane.silva", role="front_office_staff")
        assert model.role == Role.front_office_staff

    def test_rejects_a_role_string_that_is_not_a_role_member(self):
        bogus = "not_a_real_role__xyz"
        assert bogus not in {member.value for member in Role}
        with pytest.raises(ValidationError):
            LoginUser(id=uuid.uuid4(), username="jane.silva", role=bogus)

    def test_rejects_a_non_uuid_id(self):
        with pytest.raises(ValidationError):
            LoginUser(id="not-a-uuid", username="jane.silva", role=_some_role())

    def test_declares_exactly_id_username_role_never_a_password_hash_field(self):
        # architecture.md §5.2 login response's nested `user` object has only
        # id/username/role, never password_hash.
        assert set(LoginUser.model_fields) == {"id", "username", "role"}


# ---------------------------------------------------------------------------
# LoginResponse
# ---------------------------------------------------------------------------

class TestLoginResponse:
    def test_constructs_with_full_architecture_5_2_shape(self):
        user = LoginUser(id=uuid.uuid4(), username="jane.silva", role=_some_role())
        model = LoginResponse(
            access_token="access.tok",
            refresh_token="refresh.tok",
            expires_in=900,
            user=user,
        )
        assert model.access_token == "access.tok"
        assert model.refresh_token == "refresh.tok"
        assert model.expires_in == 900
        assert model.user == user
        # token_type defaults to "bearer" per the declared public surface.
        assert model.token_type == "bearer"

    def test_token_type_can_be_overridden_explicitly(self):
        user = LoginUser(id=uuid.uuid4(), username="jane.silva", role=_some_role())
        model = LoginResponse(
            access_token="a",
            refresh_token="r",
            token_type="bearer",
            expires_in=900,
            user=user,
        )
        assert model.token_type == "bearer"

    def test_user_dict_is_coerced_into_a_login_user(self):
        model = LoginResponse(
            access_token="a",
            refresh_token="r",
            expires_in=900,
            user={"id": str(uuid.uuid4()), "username": "jane.silva", "role": "clinic_management"},
        )
        assert isinstance(model.user, LoginUser)
        assert model.user.role == Role.clinic_management

    def test_missing_expires_in_raises_validation_error(self):
        user = LoginUser(id=uuid.uuid4(), username="jane.silva", role=_some_role())
        with pytest.raises(ValidationError):
            LoginResponse(access_token="a", refresh_token="r", user=user)

    def test_missing_user_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LoginResponse(access_token="a", refresh_token="r", expires_in=900)


# ---------------------------------------------------------------------------
# PasswordResetRequestRequest
# ---------------------------------------------------------------------------

class TestPasswordResetRequestRequest:
    def test_accepts_a_valid_email(self):
        model = PasswordResetRequestRequest(email="jane.silva@example.com")
        assert model.email == "jane.silva@example.com"

    def test_rejects_an_invalid_email(self):
        with pytest.raises(ValidationError):
            PasswordResetRequestRequest(email="not-an-email")

    def test_missing_email_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PasswordResetRequestRequest()


# ---------------------------------------------------------------------------
# PasswordResetConfirmRequest -- validate_password_complexity
# ---------------------------------------------------------------------------

class TestPasswordResetConfirmRequest:
    def test_accepts_a_password_with_min_length_letter_and_digit(self):
        model = PasswordResetConfirmRequest(token="reset-tok", new_password="abcd1234")
        assert model.new_password == "abcd1234"
        assert model.token == "reset-tok"

    def test_rejects_a_password_shorter_than_the_minimum_length(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirmRequest(token="reset-tok", new_password="ab1")

    def test_rejects_a_password_with_no_digit(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirmRequest(token="reset-tok", new_password="abcdefgh")

    def test_rejects_a_password_with_no_letter(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirmRequest(token="reset-tok", new_password="12345678")

    def test_missing_token_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PasswordResetConfirmRequest(new_password="abcd1234")


# ---------------------------------------------------------------------------
# DashboardWidget / DashboardWidgetsResponse
# ---------------------------------------------------------------------------

class TestDashboardWidget:
    def test_constructs_with_key_label_and_visible(self):
        widget = DashboardWidget(key="waitlist_fill_rate", label="Waitlist Fill Rate", visible=True)
        assert widget.key == "waitlist_fill_rate"
        assert widget.label == "Waitlist Fill Rate"
        assert widget.visible is True

    def test_missing_visible_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DashboardWidget(key="k", label="L")


class TestDashboardWidgetsResponse:
    def test_constructs_with_a_list_of_widgets(self):
        widget = DashboardWidget(key="k", label="L", visible=False)
        model = DashboardWidgetsResponse(widgets=[widget])
        assert model.widgets == [widget]

    def test_coerces_a_list_of_plain_dicts_into_dashboard_widgets(self):
        model = DashboardWidgetsResponse(widgets=[{"key": "k", "label": "L", "visible": True}])
        assert isinstance(model.widgets[0], DashboardWidget)
        assert model.widgets[0].visible is True

    def test_an_empty_widget_list_is_valid(self):
        model = DashboardWidgetsResponse(widgets=[])
        assert model.widgets == []


# ---------------------------------------------------------------------------
# OverrideRequest -- validate_non_empty on `reason`
# ---------------------------------------------------------------------------

class TestOverrideRequest:
    def test_constructs_with_corrected_payload_and_a_non_empty_reason(self):
        model = OverrideRequest(
            corrected_payload={"risk_level": "high"}, reason="Patient re-assessed after intake"
        )
        assert model.corrected_payload == {"risk_level": "high"}
        assert model.reason == "Patient re-assessed after intake"

    def test_rejects_a_blank_reason(self):
        with pytest.raises(ValidationError):
            OverrideRequest(corrected_payload={"a": 1}, reason="")

    def test_rejects_a_whitespace_only_reason(self):
        with pytest.raises(ValidationError):
            OverrideRequest(corrected_payload={"a": 1}, reason="   ")

    def test_corrected_payload_must_be_a_mapping_not_a_list(self):
        with pytest.raises(ValidationError):
            OverrideRequest(corrected_payload=["not", "a", "dict"], reason="valid reason")

    def test_missing_reason_raises_validation_error(self):
        with pytest.raises(ValidationError):
            OverrideRequest(corrected_payload={"a": 1})


# ---------------------------------------------------------------------------
# OverrideResponse
# ---------------------------------------------------------------------------

class TestOverrideResponse:
    def test_constructs_with_id_original_action_id_applied_at_and_actor_id(self):
        override_id = uuid.uuid4()
        original_action_id = uuid.uuid4()
        actor_id = uuid.uuid4()
        applied_at = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
        model = OverrideResponse(
            id=override_id,
            original_action_id=original_action_id,
            applied_at=applied_at,
            actor_id=actor_id,
        )
        assert model.id == override_id
        assert model.original_action_id == original_action_id
        assert model.applied_at == applied_at
        assert model.actor_id == actor_id

    def test_rejects_a_non_uuid_actor_id(self):
        with pytest.raises(ValidationError):
            OverrideResponse(
                id=uuid.uuid4(),
                original_action_id=uuid.uuid4(),
                applied_at=datetime.now(timezone.utc),
                actor_id="not-a-uuid",
            )

    def test_missing_applied_at_raises_validation_error(self):
        with pytest.raises(ValidationError):
            OverrideResponse(
                id=uuid.uuid4(), original_action_id=uuid.uuid4(), actor_id=uuid.uuid4()
            )


# ---------------------------------------------------------------------------
# ProvisionStaffRequest
# ---------------------------------------------------------------------------

class TestProvisionStaffRequest:
    def test_constructs_with_name_email_and_role(self):
        role = _some_role()
        model = ProvisionStaffRequest(name="Jane Silva", email="jane.silva@example.com", role=role)
        assert model.name == "Jane Silva"
        assert model.email == "jane.silva@example.com"
        assert model.role == role

    def test_rejects_an_invalid_email(self):
        with pytest.raises(ValidationError):
            ProvisionStaffRequest(name="Jane Silva", email="not-an-email", role=_some_role())

    def test_rejects_a_role_string_that_is_not_a_role_member(self):
        bogus = "not_a_real_role__xyz"
        assert bogus not in {member.value for member in Role}
        with pytest.raises(ValidationError):
            ProvisionStaffRequest(name="Jane Silva", email="jane.silva@example.com", role=bogus)

    def test_declares_exactly_name_email_role_no_explicit_dedup_flag_field(self):
        # FR-E1.7/US6: the 409 DUPLICATE_STAFF_ACCOUNT reactivation path is a
        # service-layer concern (StaffAdminService.provision), not encoded on
        # this schema -- there is no `force`/`dedupe` field here.
        assert set(ProvisionStaffRequest.model_fields) == {"name", "email", "role"}

    def test_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ProvisionStaffRequest(email="jane.silva@example.com", role=_some_role())


# ---------------------------------------------------------------------------
# StaffResponse
# ---------------------------------------------------------------------------

class TestStaffResponse:
    def test_constructs_with_id_username_role_and_status(self):
        staff_id = uuid.uuid4()
        role = _some_role()
        status = _some_staff_status()
        model = StaffResponse(id=staff_id, username="jane.silva", role=role, status=status)
        assert model.id == staff_id
        assert model.username == "jane.silva"
        assert model.role == role
        assert model.status == status

    def test_rejects_a_status_string_that_is_not_a_staff_status_member(self):
        bogus = "not_a_real_staff_status__xyz"
        assert bogus not in {member.value for member in StaffStatus}
        with pytest.raises(ValidationError):
            StaffResponse(id=uuid.uuid4(), username="jane.silva", role=_some_role(), status=bogus)


# ---------------------------------------------------------------------------
# DeactivateStaffResponse
# ---------------------------------------------------------------------------

class TestDeactivateStaffResponse:
    def test_constructs_with_id_status_and_sessions_revoked_count(self):
        staff_id = uuid.uuid4()
        status = _some_staff_status()
        model = DeactivateStaffResponse(id=staff_id, status=status, sessions_revoked=3)
        assert model.id == staff_id
        assert model.status == status
        assert model.sessions_revoked == 3

    def test_sessions_revoked_of_zero_is_valid(self):
        model = DeactivateStaffResponse(
            id=uuid.uuid4(), status=_some_staff_status(), sessions_revoked=0
        )
        assert model.sessions_revoked == 0

    def test_missing_sessions_revoked_raises_validation_error(self):
        with pytest.raises(ValidationError):
            DeactivateStaffResponse(id=uuid.uuid4(), status=_some_staff_status())


# ---------------------------------------------------------------------------
# AuditLogItem
# ---------------------------------------------------------------------------

class TestAuditLogItem:
    def test_constructs_with_a_display_string_actor(self):
        item_id = uuid.uuid4()
        entity_id = uuid.uuid4()
        timestamp = datetime(2026, 4, 2, 8, 15, tzinfo=timezone.utc)
        model = AuditLogItem(
            id=item_id,
            actor="jane.silva",
            action_type="override",
            entity_type="triage_session",
            entity_id=entity_id,
            timestamp=timestamp,
        )
        assert model.actor == "jane.silva"
        assert model.action_type == "override"
        assert model.entity_type == "triage_session"
        assert model.entity_id == entity_id
        assert model.timestamp == timestamp

    def test_actor_may_be_none_for_a_non_staff_actor(self):
        model = AuditLogItem(
            id=uuid.uuid4(),
            actor=None,
            action_type="auto_detect",
            entity_type="idle_chair_alert",
            entity_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
        )
        assert model.actor is None

    def test_missing_action_type_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AuditLogItem(
                id=uuid.uuid4(),
                actor="jane.silva",
                entity_type="patient",
                entity_id=uuid.uuid4(),
                timestamp=datetime.now(timezone.utc),
            )


# ---------------------------------------------------------------------------
# AuditLogQueryResponse
# ---------------------------------------------------------------------------

class TestAuditLogQueryResponse:
    def test_constructs_with_items_page_page_size_and_total(self):
        item = AuditLogItem(
            id=uuid.uuid4(),
            actor="jane.silva",
            action_type="override",
            entity_type="triage_session",
            entity_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
        )
        model = AuditLogQueryResponse(items=[item], page=1, page_size=20, total=1)
        assert model.items == [item]
        assert model.page == 1
        assert model.page_size == 20
        assert model.total == 1

    def test_an_empty_items_list_is_a_valid_zero_result_page(self):
        model = AuditLogQueryResponse(items=[], page=1, page_size=20, total=0)
        assert model.items == []
        assert model.total == 0

    def test_missing_total_raises_validation_error(self):
        with pytest.raises(ValidationError):
            AuditLogQueryResponse(items=[], page=1, page_size=20)


# ---------------------------------------------------------------------------
# Interaction contract: LoginUser.role is spelled exactly per app.common.enums.Role
# ---------------------------------------------------------------------------

def test_role_enum_values_are_exactly_the_three_console_roles():
    assert {member.value for member in Role} == {
        "front_office_staff",
        "clinic_management",
        "delivery_team",
    }


# ---------------------------------------------------------------------------
# Every declared model really is a BaseModel (schemas.py's stated public
# surface: "Pydantic v2 request/response models for every console-module
# endpoint")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_cls",
    [
        LoginRequest,
        LoginResponse,
        LoginUser,
        PasswordResetRequestRequest,
        PasswordResetConfirmRequest,
        DashboardWidget,
        DashboardWidgetsResponse,
        OverrideRequest,
        OverrideResponse,
        ProvisionStaffRequest,
        StaffResponse,
        DeactivateStaffResponse,
        AuditLogItem,
        AuditLogQueryResponse,
    ],
)
def test_every_public_model_is_a_pydantic_base_model(model_cls):
    assert issubclass(model_cls, BaseModel)
