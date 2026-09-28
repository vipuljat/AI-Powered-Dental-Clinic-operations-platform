"""Unit tests for app.common.utils — timestamp helpers, patient code generation,
and PII de-identification helpers.
"""

from datetime import datetime, timedelta, timezone

from app.common.utils import (
    de_identify_patient_fields,
    de_identify_transcript_text,
    new_patient_code,
    now_utc,
    to_iso8601,
)


# ---- now_utc / to_iso8601 ---------------------------------------------------


def test_now_utc_is_timezone_aware_utc():
    result = now_utc()
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    assert result.tzinfo.utcoffset(result) == timedelta(0)


def test_now_utc_is_close_to_actual_current_time():
    before = datetime.now(timezone.utc)
    result = now_utc()
    after = datetime.now(timezone.utc)
    assert before - timedelta(seconds=5) <= result <= after + timedelta(seconds=5)


def test_to_iso8601_includes_explicit_utc_offset():
    dt = datetime(2026, 9, 24, 10, 0, 0, tzinfo=timezone.utc)
    result = to_iso8601(dt)
    assert result == "2026-09-24T10:00:00+00:00"


def test_to_iso8601_returns_string_type():
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert isinstance(to_iso8601(dt), str)


# ---- new_patient_code --------------------------------------------------------


def test_new_patient_code_formats_with_zero_padding():
    assert new_patient_code(231) == "PT-00231"


def test_new_patient_code_small_sequence_number():
    assert new_patient_code(1) == "PT-00001"


def test_new_patient_code_large_sequence_number_beyond_padding_width():
    assert new_patient_code(123456) == "PT-123456"


# ---- de_identify_patient_fields ----------------------------------------------


def test_de_identify_patient_fields_strips_pii_fields():
    record = {
        "patient_id": "uuid-123",
        "first_name": "Jane",
        "last_name": "Doe",
        "phone": "+15551234567",
        "email": "jane@example.com",
        "dob": "1990-01-01",
        "risk_level": "high",
    }
    result = de_identify_patient_fields(record)
    assert "first_name" not in result
    assert "last_name" not in result
    assert "phone" not in result
    assert "email" not in result
    assert "dob" not in result


def test_de_identify_patient_fields_retains_patient_id_and_clinical_fields():
    record = {
        "patient_id": "uuid-123",
        "first_name": "Jane",
        "last_name": "Doe",
        "phone": "555",
        "email": "jane@example.com",
        "dob": "1990-01-01",
        "risk_level": "high",
        "next_appointment": "2026-10-01",
    }
    result = de_identify_patient_fields(record)
    assert result["patient_id"] == "uuid-123"
    assert result["risk_level"] == "high"
    assert result["next_appointment"] == "2026-10-01"


def test_de_identify_patient_fields_does_not_mutate_input():
    record = {
        "patient_id": "uuid-123",
        "first_name": "Jane",
        "last_name": "Doe",
        "phone": "555",
        "email": "jane@example.com",
        "dob": "1990-01-01",
    }
    original = dict(record)
    de_identify_patient_fields(record)
    assert record == original


def test_de_identify_patient_fields_returns_new_dict_object():
    record = {"patient_id": "uuid-123", "first_name": "Jane"}
    result = de_identify_patient_fields(record)
    assert result is not record


# ---- de_identify_transcript_text ---------------------------------------------


def test_de_identify_transcript_text_strips_long_digit_sequences():
    text = "Please call me back at 5551234567 tomorrow."
    result = de_identify_transcript_text(text)
    assert "5551234567" not in result


def test_de_identify_transcript_text_strips_capitalized_two_word_names():
    text = "Patient John Smith called about a filling."
    result = de_identify_transcript_text(text)
    assert "John Smith" not in result


def test_de_identify_transcript_text_leaves_non_pii_content_intact():
    text = "the appointment is scheduled for next tuesday"
    result = de_identify_transcript_text(text)
    assert "appointment" in result
    assert "scheduled" in result


def test_de_identify_transcript_text_returns_string():
    result = de_identify_transcript_text("some free text with 1234567890 in it")
    assert isinstance(result, str)
