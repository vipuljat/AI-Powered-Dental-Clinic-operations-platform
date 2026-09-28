"""The single domain-exception hierarchy every service, repository, and worker
in this tree raises instead of returning ad-hoc error values or raising a bare
``fastapi.HTTPException``.

Each exception class fixes an ``error_code`` (matches architecture.md §8's
UPPER_SNAKE_CODE convention) and the HTTP status it maps to, so
app/common/exceptions/handlers.py can translate any of them mechanically.
This file does not perform that translation itself, and it does not touch the
DB, HTTP, or logging.
"""

from __future__ import annotations


class AppError(Exception):
    """Base of the domain-exception hierarchy.

    Every documented error outcome across architecture.md §5.2 (401/403/404/
    409/422/423/429/502, etc.) is raised through exactly one subclass of this
    class, everywhere in the codebase — never a bare ``Exception`` and never
    ``fastapi.HTTPException`` directly.
    """

    code: str = "APP_ERROR"
    status_code: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, details: list[dict] | None = None) -> None:
        self.message = message or self.default_message
        self.details = details
        super().__init__(self.message)


class NotFoundError(AppError):
    code = "NOT_FOUND"
    status_code = 404
    default_message = "The requested resource was not found."


class ValidationFailedError(AppError):
    code = "VALIDATION_FAILED"
    status_code = 422
    default_message = "Validation failed."


class ConflictError(AppError):
    code = "CONFLICT"
    status_code = 409
    default_message = "The request conflicts with the current state of the resource."


class UnauthorizedError(AppError):
    code = "UNAUTHORIZED"
    status_code = 401
    default_message = "Authentication is required."


class ForbiddenError(AppError):
    code = "FORBIDDEN"
    status_code = 403
    default_message = "You do not have permission to perform this action."


class LockedError(AppError):
    code = "LOCKED"
    status_code = 423
    default_message = "The resource is locked."


class RateLimitedError(AppError):
    code = "RATE_LIMITED"
    status_code = 429
    default_message = "Too many requests."


class BadGatewayError(AppError):
    code = "BAD_GATEWAY"
    status_code = 502
    default_message = "An upstream service failed."


# --- Specific subclasses used across modules --------------------------------
# Each fixes `code`, inherits its parent's status_code.


class InvalidCredentialsError(UnauthorizedError):
    code = "INVALID_CREDENTIALS"
    default_message = "Invalid email or password."


class AccountLockedError(LockedError):
    # 423 per FR-E1 login lockout
    code = "ACCOUNT_LOCKED"
    default_message = "This account is temporarily locked due to too many failed login attempts."


class DuplicatePatientError(ConflictError):
    code = "DUPLICATE_PATIENT"
    default_message = "A matching patient record already exists."


class DuplicateStaffAccountError(ConflictError):
    code = "DUPLICATE_STAFF_ACCOUNT"
    default_message = "A staff account with this email already exists."


class RecordLockedError(ConflictError):
    # includes lock owner in details
    code = "RECORD_LOCKED"
    default_message = "This record is currently locked by another user."


class NotLockHolderError(LockedError):
    code = "NOT_LOCK_HOLDER"
    default_message = "You are not the current holder of this record's lock."


class RuleValidationError(ValidationFailedError):
    code = "RULE_VALIDATION_FAILED"
    default_message = "Rule validation failed."


class ScheduleConflictError(ConflictError):
    code = "APPOINTMENT_CONFLICT"
    default_message = "The requested appointment slot conflicts with an existing booking."


class MissingReasonCodeError(ValidationFailedError):
    code = "MISSING_REASON_CODE"
    default_message = "A reason code is required for this action."


class InsufficientPermissionsError(ForbiddenError):
    code = "INSUFFICIENT_PERMISSIONS"
    default_message = "Your role does not have permission to perform this action."


class IrreversibleActionError(ConflictError):
    # override attempted post-irreversible
    code = "IRREVERSIBLE_ACTION"
    default_message = "This action can no longer be reversed or overridden."


class InvalidResetTokenError(AppError):
    code = "INVALID_RESET_TOKEN"
    status_code = 400
    default_message = "This password reset token is invalid or has expired."


class PasswordComplexityError(ValidationFailedError):
    code = "PASSWORD_COMPLEXITY"
    default_message = "Password does not meet the complexity requirements."


class NoActiveRuleSetError(NotFoundError):
    code = "NO_ACTIVE_RULE_SET"
    default_message = "No active rule set was found."


class ImportValidationError(ValidationFailedError):
    # details = row-level errors[]
    code = "IMPORT_VALIDATION_FAILED"
    default_message = "One or more rows failed validation."


class ChannelDeliveryError(BadGatewayError):
    code = "CHANNEL_DELIVERY_FAILED"
    default_message = "Delivery via the outbound channel failed."


class NoDataForRangeError(ValidationFailedError):
    code = "NO_DATA_FOR_RANGE"
    default_message = "No data is available for the requested range."


class RecommendationStaleError(ConflictError):
    # apply-time re-validation failed
    code = "RECOMMENDATION_STALE"
    default_message = "This recommendation is no longer valid and cannot be applied."


class NoMatchingPatientError(NotFoundError):
    code = "NO_MATCHING_PATIENT"
    default_message = "No matching patient record was found."


class IdempotencyReplayError(AppError):
    """Documented naming placeholder, not an error path per se.

    Not actually raised as an HTTP error. Some services may use this
    internally to short-circuit and return the original response for a
    repeated ``Idempotency-Key`` (RMR001): "returns the original result
    rather than creating a duplicate row" — routes that do catch this return
    the **original** success response, not a 4xx/5xx.

    Prefer implementing idempotency replay as a plain return value from the
    repository/service (e.g. ``create(...) -> (Appointment, is_replay: bool)``)
    rather than exception control flow if that proves simpler — this class
    exists only so the option is named.
    """

    code = "IDEMPOTENCY_REPLAY"
    status_code = 200
    default_message = "The original result for this idempotency key is being returned."
