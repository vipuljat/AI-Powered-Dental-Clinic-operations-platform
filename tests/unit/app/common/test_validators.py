"""Unit tests for app/common/validators.py.

These are pure functions (no persistence/HTTP), used as Pydantic
`field_validator`s across every module's schemas.py. Each is exercised
directly here, exactly as its spec describes it, without going through any
Pydantic model (the spec says schemas wire them via
`field_validator("field_name")(validate_x)`, but the functions themselves
are plain callables and are tested as such).
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.common.validators import (
    validate_date_range,
    validate_non_empty,
    validate_password_complexity,
    validate_phone,
)

# `PASSWORD_MIN_LENGTH` is the real project constant validate_password_complexity
# is documented to check against (see the function's own spec comment). Importing
# it keeps the boundary tests correct regardless of its configured numeric value,
# while a dedicated test below still pins down the spec's stated default of 8.
from app.common.constants import PASSWORD_MIN_LENGTH


# ---------------------------------------------------------------------------
# validate_phone
# ---------------------------------------------------------------------------

class TestValidatePhone:
    def test_accepts_plausible_e164_style_number_and_returns_it_unchanged(self):
        value = "+1 (555) 123-4567"
        assert validate_phone(value) == value

    def test_accepts_digits_only_at_minimum_length_of_8(self):
        value = "12345678"
        assert validate_phone(value) == value

    def test_accepts_digits_only_at_maximum_length_of_20(self):
        value = "1" * 20
        assert validate_phone(value) == value

    def test_rejects_number_shorter_than_8_significant_characters(self):
        with pytest.raises(ValueError):
            validate_phone("1234567")

    def test_rejects_number_longer_than_20_significant_characters(self):
        with pytest.raises(ValueError):
            validate_phone("1" * 21)

    def test_length_check_is_applied_after_stripping_separators(self):
        # 20 digits wrapped in separator characters (spaces, parens, hyphen)
        # must still be accepted -- the 8-20 length bound applies to the
        # remainder after separators are stripped out, not to the longer raw
        # string length (which includes the separator characters too).
        digits = "1" * 20
        value = f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]} {digits[10:]}"
        assert len(value) > 20
        assert validate_phone(value) == value

    def test_rejects_value_containing_letters(self):
        with pytest.raises(ValueError):
            validate_phone("555-CALL-NOW")

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            validate_phone("")

    def test_does_not_normalise_the_value(self):
        # Per the file's spec, this function only validates shape; it is
        # explicitly *not* responsible for normalisation/dedup of phone
        # representations (that belongs to PatientRepository.find_duplicate_by_phone).
        value = "+1 (555) 123-4567"
        assert validate_phone(value) == value


# ---------------------------------------------------------------------------
# validate_password_complexity
# ---------------------------------------------------------------------------

class TestValidatePasswordComplexity:
    def test_spec_default_minimum_length_is_8(self):
        assert PASSWORD_MIN_LENGTH == 8

    def test_accepts_password_meeting_length_and_letter_and_digit_requirements(self):
        value = "a" * (PASSWORD_MIN_LENGTH - 1) + "1"
        assert validate_password_complexity(value) == value

    def test_rejects_password_shorter_than_minimum_length(self):
        too_short = "a" * (PASSWORD_MIN_LENGTH - 2) + "1"
        with pytest.raises(ValueError):
            validate_password_complexity(too_short)

    def test_rejects_password_with_no_digit(self):
        letters_only = "a" * PASSWORD_MIN_LENGTH
        with pytest.raises(ValueError):
            validate_password_complexity(letters_only)

    def test_rejects_password_with_no_letter(self):
        digits_only = "1" * PASSWORD_MIN_LENGTH
        with pytest.raises(ValueError):
            validate_password_complexity(digits_only)

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            validate_password_complexity("")


# ---------------------------------------------------------------------------
# validate_non_empty
# ---------------------------------------------------------------------------

class TestValidateNonEmpty:
    def test_accepts_non_empty_string_and_returns_it(self):
        assert validate_non_empty("John") == "John"

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError):
            validate_non_empty("")

    def test_rejects_whitespace_only_string(self):
        with pytest.raises(ValueError):
            validate_non_empty("   ")

    def test_rejects_tab_and_newline_only_string(self):
        with pytest.raises(ValueError):
            validate_non_empty("\t\n")


# ---------------------------------------------------------------------------
# validate_date_range
# ---------------------------------------------------------------------------

class TestValidateDateRange:
    def test_returns_none_when_end_after_start(self):
        assert validate_date_range(dt.date(2026, 1, 1), dt.date(2026, 1, 2)) is None

    def test_returns_none_when_end_equals_start(self):
        same = dt.date(2026, 1, 1)
        assert validate_date_range(same, same) is None

    def test_raises_when_end_before_start(self):
        with pytest.raises(ValueError):
            validate_date_range(dt.date(2026, 1, 2), dt.date(2026, 1, 1))

    def test_works_with_timezone_aware_datetimes(self):
        start = dt.datetime(2026, 1, 1, 9, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 1, 1, 10, 0, tzinfo=dt.timezone.utc)
        assert validate_date_range(start, end) is None

    def test_raises_when_end_datetime_before_start_datetime(self):
        start = dt.datetime(2026, 1, 1, 10, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2026, 1, 1, 9, 0, tzinfo=dt.timezone.utc)
        with pytest.raises(ValueError):
            validate_date_range(start, end)
