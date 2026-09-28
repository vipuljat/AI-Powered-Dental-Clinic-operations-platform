"""Small, stateless helpers with no natural home in a single module: UTC
timestamp helpers, human-readable ID generation (``patient_code``), and PII
de-identification used by both patient export (FR-E2.6) and call-transcript
ML prep (DR002).

Contains no persistence access — callers pass in whatever data needs
transforming.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

# --- PII fields stripped from a patient record for de-identified export ----
_PATIENT_PII_FIELDS = ("first_name", "last_name", "phone", "email", "dob")

# Digit runs of length >= 7 are treated as phone-like and stripped.
_PHONE_LIKE_RE = re.compile(r"\d{7,}")

# Two consecutive capitalized words (e.g. "John Smith") are treated as
# name-like and stripped. Best-effort only — see module docstring / Watch out.
_NAME_LIKE_RE = re.compile(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")


def now_utc() -> datetime:
    """Timezone-aware UTC now().

    Every timestamp written anywhere in this tree goes through this, per
    project_rules "all timestamps are timezone-aware UTC".
    """
    return datetime.now(timezone.utc)


def to_iso8601(value: datetime) -> str:
    """Serialize `value` with an explicit UTC offset, e.g.
    "2026-09-24T10:00:00+00:00".

    If `value` is naive, it is treated as already being UTC.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def new_patient_code(sequence: int) -> str:
    """Human-readable patient code: "PT-%05d" % sequence,
    e.g. sequence=231 -> "PT-00231".
    """
    return "PT-%05d" % sequence


def de_identify_patient_fields(record: dict) -> dict:
    """Return a shallow copy of `record` with first_name/last_name/phone/
    email/dob removed; `patient_id` is retained as an opaque UUID key for
    joins.

    Used by ExportService (de-identified export, FR-E2.6/DR004) and
    CallScoringService (transcript de-identification before ML use, DR002).

    This function never mutates its input — callers that need the original
    untouched dict for a non-exported use (e.g. logging the raw record
    server-side) must keep their own reference.
    """
    return {key: value for key, value in record.items() if key not in _PATIENT_PII_FIELDS}


def de_identify_transcript_text(text: str) -> str:
    """Best-effort strip of digit sequences length>=7 (phone-like) and
    capitalized two-word sequences (name-like) from free text before it is
    persisted to call_transcript_segments.text or used for embedding
    generation.

    DR002 (US53 AC): call recordings/transcripts are de-identified before any
    ML training use.

    Documented best-effort only, not a certified PII scrubber — sufficient
    for this project's synthetic/dev data (§11 item 22) but not a compliance
    guarantee.
    """
    scrubbed = _NAME_LIKE_RE.sub("[REDACTED_NAME]", text)
    scrubbed = _PHONE_LIKE_RE.sub("[REDACTED_NUMBER]", scrubbed)
    return scrubbed
