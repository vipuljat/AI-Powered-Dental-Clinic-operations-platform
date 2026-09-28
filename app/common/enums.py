"""The single source of truth for every fixed-vocabulary (ENUM(...)) column
value used anywhere in the schema (architecture.md §4.2).

Every models/*/models.py, schemas/*/schemas.py, services/*/service.py and
routes/*/routes.py file that reads or writes an enumerated field imports its
values from here — never redeclares the value set locally.

Contains no behaviour, validation logic, or DB/HTTP code.
"""

from enum import Enum


class Role(str, Enum):
    front_office_staff = "front_office_staff"
    clinic_management = "clinic_management"
    delivery_team = "delivery_team"


class StaffStatus(str, Enum):
    active = "active"
    inactive = "inactive"


class ActorType(str, Enum):
    staff = "staff"
    ai_agent = "ai_agent"
    system = "system"


class PatientStatus(str, Enum):
    active = "active"
    archived = "archived"


class Language(str, Enum):
    en = "en"
    es = "es"
    pt = "pt"


class ImportBatchStatus(str, Enum):
    validating = "validating"
    rejected = "rejected"
    committed = "committed"


class LockEntityType(str, Enum):
    patient = "patient"
    appointment = "appointment"


class RuleSetStatus(str, Enum):
    draft = "draft"
    in_review = "in_review"
    active = "active"
    superseded = "superseded"


class RuleCategory(str, Enum):
    patient_category = "patient_category"
    risk_classification = "risk_classification"
    recall_interval = "recall_interval"
    appointment_type = "appointment_type"
    scheduling_priority = "scheduling_priority"
    triage_question = "triage_question"
    escalation_trigger = "escalation_trigger"
    family_scheduling = "family_scheduling"


class AppointmentStatus(str, Enum):
    booked = "booked"
    rescheduled = "rescheduled"
    cancelled = "cancelled"
    completed = "completed"
    no_show = "no_show"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ScoreSource(str, Enum):
    rules = "rules"
    ml = "ml"


class ConfirmationStatus(str, Enum):
    sent = "sent"
    failed = "failed"
    suppressed = "suppressed"


class IdleChairSource(str, Enum):
    auto_detected = "auto_detected"
    manual_flag = "manual_flag"


class IdleChairStatus(str, Enum):
    open = "open"
    filled = "filled"
    exhausted = "exhausted"


class WaitlistEntryStatus(str, Enum):
    active = "active"
    offered = "offered"
    booked = "booked"
    declined = "declined"
    expired = "expired"


class WaitlistOfferStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    declined = "declined"
    expired = "expired"


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Channel(str, Enum):
    whatsapp = "whatsapp"
    sms = "sms"
    email = "email"
    voice = "voice"
    webchat = "webchat"


class ConsentStatus(str, Enum):
    granted = "granted"
    withdrawn = "withdrawn"
    declined = "declined"


class CampaignType(str, Enum):
    confirmation = "confirmation"
    waitlist_offer = "waitlist_offer"
    recall = "recall"
    treatment_reengagement = "treatment_reengagement"
    education = "education"


class OutreachMessageStatus(str, Enum):
    queued = "queued"
    sent = "sent"
    delivered = "delivered"
    failed = "failed"
    suppressed = "suppressed"
    unreachable = "unreachable"


class ChannelConfigStatus(str, Enum):
    verified = "verified"
    pending = "pending"


class RecallStatus(str, Enum):
    due = "due"
    overdue = "overdue"
    dormant = "dormant"
    contacted = "contacted"
    completed = "completed"


class IntervalSource(str, Enum):
    risk_based = "risk_based"
    default_fallback = "default_fallback"


class UnscheduledTreatmentStatus(str, Enum):
    unscheduled = "unscheduled"
    re_engaged = "re_engaged"
    booked = "booked"
    declined = "declined"


class EscalationStatus(str, Enum):
    unacknowledged = "unacknowledged"
    acknowledged = "acknowledged"
    resolved = "resolved"


class UtilisationRecommendationStatus(str, Enum):
    pending = "pending"
    applied = "applied"
    dismissed = "dismissed"


class MlEvaluationStatus(str, Enum):
    passing = "passing"
    underperforming = "underperforming"


class DashboardType(str, Enum):
    operational = "operational"
    financial = "financial"


class ExportFormat(str, Enum):
    pdf = "pdf"
    csv = "csv"


class EducationTriggerType(str, Enum):
    pre_appointment = "pre_appointment"
    post_appointment = "post_appointment"


class ContentDeliveryStatus(str, Enum):
    delivered = "delivered"
    opened = "opened"
    completed = "completed"
    unavailable = "unavailable"


class WebchatVerificationStatus(str, Enum):
    unverified = "unverified"
    verified = "verified"


class WebchatSender(str, Enum):
    visitor = "visitor"
    ai_agent = "ai_agent"


class CallHandledBy(str, Enum):
    ai_voice = "ai_voice"
    staff = "staff"


class CallScoreStatus(str, Enum):
    scored = "scored"
    unavailable = "unavailable"


class CallSpeaker(str, Enum):
    patient = "patient"
    ai_agent = "ai_agent"
