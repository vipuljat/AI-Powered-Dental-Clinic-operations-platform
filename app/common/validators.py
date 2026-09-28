"""Reusable Pydantic-compatible field validators shared across every module's
``schemas.py``.

Pure function library: phone format, password complexity, non-empty string,
and date-range ordering. Contains no persistence or HTTP logic.

## Interaction contract
Every function here is designed to be used as a Pydantic ``field_validator``,
called automatically by Pydantic at model-construction time — a schema wires
it via ``_validate_x = field_validator("field_name")(validate_x)``, never
calls it manually. A ``ValueError`` raised here becomes part of FastAPI's
standard 422 ``RequestValidationError`` body via
``app/common/exceptions/handlers.py``'s ``RequestValidationError`` handler,
not a domain ``AppError``.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from app.common.constants import PASSWORD_MIN_LENGTH

# Characters that are permitted as separators/decoration in a phone number and
# are stripped out before the length/shape check runs.
_PHONE_SEPARATOR_CHARS = re.compile(r"[\s\-()]+")

# After separators are stripped, a plausible E.164-ish phone number is an
# optional leading '+' followed only by digits.
_PHONE_SHAPE_RE = re.compile(r"^\+?\d+$")

_PHONE_MIN_LENGTH = 8
_PHONE_MAX_LENGTH = 20


def validate_phone(value: str) -> str:
    """FR-E2.1 / US8: shape-validate a contact phone number.

    Accepts digits, spaces, ``+``, ``-`` and parentheses. After stripping
    separators the remainder must be 8-20 characters long and contain only
    an optional leading ``+`` followed by digits. This function only
    validates shape; it does not normalise the value — normalisation for
    duplicate-phone detection is ``PatientRepository.find_duplicate_by_phone``'s
    job.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("phone number must not be empty")

    stripped = _PHONE_SEPARATOR_CHARS.sub("", value.strip())

    if not _PHONE_SHAPE_RE.match(stripped):
        raise ValueError(
            "phone number must contain only digits, spaces, +, -, and "
            "parentheses"
        )

    if not (_PHONE_MIN_LENGTH <= len(stripped) <= _PHONE_MAX_LENGTH):
        raise ValueError(
            f"phone number must be between {_PHONE_MIN_LENGTH} and "
            f"{_PHONE_MAX_LENGTH} characters after stripping separators"
        )

    return value


def validate_password_complexity(value: str) -> str:
    """US2 AC / T3: reject passwords failing configured complexity rules.

    This project's own inferred default (BRD does not numerically specify
    complexity — see open_questions): at least ``PASSWORD_MIN_LENGTH``
    characters, containing at least one letter and one digit. A stricter
    rule imposed later must be changed in this one place.
    """
    if len(value) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"password must be at least {PASSWORD_MIN_LENGTH} characters long"
        )

    has_letter = any(ch.isalpha() for ch in value)
    has_digit = any(ch.isdigit() for ch in value)

    if not has_letter or not has_digit:
        raise ValueError(
            "password must contain at least one letter and one digit"
        )

    return value


def validate_non_empty(value: str) -> str:
    """FR-E2.1: back the "required as non-null" rule on fields such as
    ``first_name``/``last_name``/``phone`` in ``schemas/patients/schemas.py``.
    """
    if value.strip() == "":
        raise ValueError("value must not be empty")

    return value


def validate_date_range(start: date | datetime, end: date | datetime) -> None:
    """Raise ``ValueError`` if ``end`` precedes ``start``.

    Used to back date-range ordering rules (e.g. schedule windows, recall
    windows) across multiple modules' schemas.
    """
    if end < start:
        raise ValueError("end must not be before start")
