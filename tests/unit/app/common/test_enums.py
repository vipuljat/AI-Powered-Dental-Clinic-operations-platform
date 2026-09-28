"""Unit tests for app.common.enums — the single source of truth for every
fixed-vocabulary column value used across the schema.

Each enum's exact member set and str-serialization behaviour is asserted so
that a value renamed or dropped in one place (but not another) is caught here
first.
"""

import enum
import json

import pytest

from app.common import enums


ENUM_MEMBERS = {
    "Role": ["front_office_staff", "clinic_management", "delivery_team"],
    "StaffStatus": ["active", "inactive"],
    "ActorType": ["staff", "ai_agent", "system"],
    "PatientStatus": ["active", "archived"],
    "Language": ["en", "es", "pt"],
    "ImportBatchStatus": ["validating", "rejected", "committed"],
    "LockEntityType": ["patient", "appointment"],
    "RuleSetStatus": ["draft", "in_review", "active", "superseded"],
    "RuleCategory": [
        "patient_category",
        "risk_classification",
        "recall_interval",
        "appointment_type",
        "scheduling_priority",
        "triage_question",
        "escalation_trigger",
        "family_scheduling",
    ],
    "AppointmentStatus": ["booked", "rescheduled", "cancelled", "completed", "no_show"],
    "RiskLevel": ["low", "medium", "high"],
    "ScoreSource": ["rules", "ml"],
    "ConfirmationStatus": ["sent", "failed", "suppressed"],
    "IdleChairSource": ["auto_detected", "manual_flag"],
    "IdleChairStatus": ["open", "filled", "exhausted"],
    "WaitlistEntryStatus": ["active", "offered", "booked", "declined", "expired"],
    "WaitlistOfferStatus": ["pending", "accepted", "declined", "expired"],
    "Urgency": ["low", "medium", "high"],
    "Channel": ["whatsapp", "sms", "email", "voice", "webchat"],
    "ConsentStatus": ["granted", "withdrawn", "declined"],
    "CampaignType": [
        "confirmation",
        "waitlist_offer",
        "recall",
        "treatment_reengagement",
        "education",
    ],
    "OutreachMessageStatus": [
        "queued",
        "sent",
        "delivered",
        "failed",
        "suppressed",
        "unreachable",
    ],
    "ChannelConfigStatus": ["verified", "pending"],
    "RecallStatus": ["due", "overdue", "dormant", "contacted", "completed"],
    "IntervalSource": ["risk_based", "default_fallback"],
    "UnscheduledTreatmentStatus": ["unscheduled", "re_engaged", "booked", "declined"],
    "EscalationStatus": ["unacknowledged", "acknowledged", "resolved"],
    "UtilisationRecommendationStatus": ["pending", "applied", "dismissed"],
    "MlEvaluationStatus": ["passing", "underperforming"],
    "DashboardType": ["operational", "financial"],
    "ExportFormat": ["pdf", "csv"],
    "EducationTriggerType": ["pre_appointment", "post_appointment"],
    "ContentDeliveryStatus": ["delivered", "opened", "completed", "unavailable"],
    "WebchatVerificationStatus": ["unverified", "verified"],
    "WebchatSender": ["visitor", "ai_agent"],
    "CallHandledBy": ["ai_voice", "staff"],
    "CallScoreStatus": ["scored", "unavailable"],
    "CallSpeaker": ["patient", "ai_agent"],
}


@pytest.mark.parametrize("enum_name, expected_values", list(ENUM_MEMBERS.items()))
def test_enum_member_values_match_spec(enum_name, expected_values):
    enum_cls = getattr(enums, enum_name)
    actual_values = [member.value for member in enum_cls]
    assert actual_values == expected_values


@pytest.mark.parametrize("enum_name", list(ENUM_MEMBERS.keys()))
def test_enum_is_str_enum(enum_name):
    enum_cls = getattr(enums, enum_name)
    assert issubclass(enum_cls, str)
    assert issubclass(enum_cls, enum.Enum)


def test_enum_member_equals_raw_string():
    assert enums.Role.front_office_staff == "front_office_staff"
    assert enums.AppointmentStatus.no_show == "no_show"


def test_enum_member_json_serializes_as_plain_string():
    payload = {"role": enums.Role.clinic_management}
    # str, Enum members dump as their plain string value when passed through
    # str() / json with a default=str, matching architecture.md's JSON contract.
    assert json.dumps(payload, default=str) == json.dumps({"role": "clinic_management"})
    assert str(enums.Role.clinic_management.value) == "clinic_management"


def test_risklevel_and_urgency_share_same_vocabulary_but_are_distinct_classes():
    # RiskLevel and Urgency both use low/medium/high but must be separate enums
    assert [m.value for m in enums.RiskLevel] == [m.value for m in enums.Urgency]
    assert enums.RiskLevel is not enums.Urgency
