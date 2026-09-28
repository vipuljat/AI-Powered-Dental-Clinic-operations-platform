"""Unit tests for app.common.exceptions.errors — the domain-exception hierarchy.

Verifies each class's fixed `code` / `status_code`, its inheritance chain, and
that `message`/`details` are captured correctly by the base `AppError.__init__`.
"""

import pytest

from app.common.exceptions import errors


# ---- Base class behaviour -------------------------------------------------


def test_app_error_defaults_message_and_details():
    err = errors.NotFoundError()
    assert isinstance(err, Exception)
    # details defaults to None or an empty structure but must not raise
    assert err.details in (None, [])


def test_app_error_stores_custom_message_and_details():
    details = [{"field": "email", "issue": "required"}]
    err = errors.ValidationFailedError("Email is required", details=details)
    assert err.message == "Email is required"
    assert err.details == details


def test_app_error_is_raisable_and_catchable_as_exception():
    with pytest.raises(errors.AppError):
        raise errors.ConflictError("conflict")


# ---- Base status-code / code classes --------------------------------------


@pytest.mark.parametrize(
    "cls, expected_code, expected_status",
    [
        (errors.NotFoundError, "NOT_FOUND", 404),
        (errors.ValidationFailedError, "VALIDATION_FAILED", 422),
        (errors.ConflictError, "CONFLICT", 409),
        (errors.UnauthorizedError, "UNAUTHORIZED", 401),
        (errors.ForbiddenError, "FORBIDDEN", 403),
        (errors.LockedError, "LOCKED", 423),
        (errors.RateLimitedError, "RATE_LIMITED", 429),
        (errors.BadGatewayError, "BAD_GATEWAY", 502),
    ],
)
def test_base_error_classes_fix_code_and_status(cls, expected_code, expected_status):
    err = cls("some message")
    assert err.code == expected_code
    assert err.status_code == expected_status
    assert isinstance(err, errors.AppError)


# ---- Specific subclasses ----------------------------------------------------


@pytest.mark.parametrize(
    "cls, expected_code, expected_status, expected_base",
    [
        (errors.InvalidCredentialsError, "INVALID_CREDENTIALS", 401, errors.UnauthorizedError),
        (errors.AccountLockedError, "ACCOUNT_LOCKED", 423, errors.LockedError),
        (errors.DuplicatePatientError, "DUPLICATE_PATIENT", 409, errors.ConflictError),
        (errors.DuplicateStaffAccountError, "DUPLICATE_STAFF_ACCOUNT", 409, errors.ConflictError),
        (errors.RecordLockedError, "RECORD_LOCKED", 409, errors.ConflictError),
        (errors.NotLockHolderError, "NOT_LOCK_HOLDER", 423, errors.LockedError),
        (errors.RuleValidationError, "RULE_VALIDATION_FAILED", 422, errors.ValidationFailedError),
        (errors.ScheduleConflictError, "APPOINTMENT_CONFLICT", 409, errors.ConflictError),
        (errors.MissingReasonCodeError, "MISSING_REASON_CODE", 422, errors.ValidationFailedError),
        (
            errors.InsufficientPermissionsError,
            "INSUFFICIENT_PERMISSIONS",
            403,
            errors.ForbiddenError,
        ),
        (errors.IrreversibleActionError, "IRREVERSIBLE_ACTION", 409, errors.ConflictError),
        (errors.PasswordComplexityError, "PASSWORD_COMPLEXITY", 422, errors.ValidationFailedError),
        (errors.NoActiveRuleSetError, "NO_ACTIVE_RULE_SET", 404, errors.NotFoundError),
        (
            errors.ImportValidationError,
            "IMPORT_VALIDATION_FAILED",
            422,
            errors.ValidationFailedError,
        ),
        (errors.ChannelDeliveryError, "CHANNEL_DELIVERY_FAILED", 502, errors.BadGatewayError),
        (errors.NoDataForRangeError, "NO_DATA_FOR_RANGE", 422, errors.ValidationFailedError),
        (errors.RecommendationStaleError, "RECOMMENDATION_STALE", 409, errors.ConflictError),
        (errors.NoMatchingPatientError, "NO_MATCHING_PATIENT", 404, errors.NotFoundError),
    ],
)
def test_specific_subclasses_fix_code_status_and_inherit_from_expected_base(
    cls, expected_code, expected_status, expected_base
):
    err = cls("message")
    assert err.code == expected_code
    assert err.status_code == expected_status
    assert isinstance(err, expected_base)
    assert isinstance(err, errors.AppError)


def test_invalid_reset_token_error_uses_status_400_directly_on_app_error():
    err = errors.InvalidResetTokenError("bad token")
    assert err.code == "INVALID_RESET_TOKEN"
    assert err.status_code == 400
    assert isinstance(err, errors.AppError)


def test_record_locked_error_carries_lock_owner_in_details():
    details = [{"locked_by": "staff-42"}]
    err = errors.RecordLockedError("Record is locked", details=details)
    assert err.details == details
    assert err.code == "RECORD_LOCKED"


def test_import_validation_error_carries_row_level_errors_in_details():
    details = [{"row": 3, "error": "missing dob"}]
    err = errors.ImportValidationError("Import failed", details=details)
    assert err.details == details
    assert err.code == "IMPORT_VALIDATION_FAILED"


def test_idempotency_replay_error_status_code_is_200_sentinel():
    err = errors.IdempotencyReplayError()
    assert err.status_code == 200
    assert isinstance(err, errors.AppError)
