"""Pydantic v2 request/response models for every `/intelligence/*` endpoint
(architecture.md §5.2).

Field names/types/aliases here are exactly what `services/intelligence/service.py`'s
return values must match key-for-key, and exactly what `routes/intelligence/routes.py`
validates requests against. No additional out-of-band agreement exists beyond
what these signatures carry.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.common.enums import (
    EscalationStatus,
    MlEvaluationStatus,
    RiskLevel,
    ScoreSource,
    UtilisationRecommendationStatus,
)


class RiskScoreResponse(BaseModel):
    risk_level: RiskLevel
    score_value: float
    source: ScoreSource
    model_version: str | None = None


class RiskAnalyticsResponse(BaseModel):
    # {"low": 120, "medium": 40, "high": 12}
    distribution: dict[str, int]
    model_status: MlEvaluationStatus | None = None


class StartTriageSessionRequest(BaseModel):
    channel: str  # "webchat" | "voice"
    patient_id: UUID | None = None


class TriageQuestion(BaseModel):
    # FR-E8.4: the triage interface is structured multiple-choice/guided
    # questions, not free-text diagnosis (US43 Alternate Flow) — `options`
    # is always a non-empty list of fixed choices.
    key: str
    options: list[str]


class StartTriageSessionResponse(BaseModel):
    id: UUID
    next_question: TriageQuestion | None = None


class AnswerTriageQuestionRequest(BaseModel):
    # FR-E8.4: `answer` must match one of the previously-offered options for
    # `question_key` — checked at the service layer (TriageService.answer),
    # since the valid option set is only known once the active rule set is
    # read, not by this schema.
    question_key: str
    answer: str


class AnswerTriageQuestionResponse(BaseModel):
    # architecture.md §5.2 `.../answer`: on escalation, `next_question` and
    # `recommended_block` are both null (FR-E8.5: no non-diagnostic
    # recommendation is issued for an escalated case). This schema does not
    # itself enforce that mutual exclusivity — TriageService.answer is
    # responsible for never populating `recommended_block` when
    # `escalated` is true.
    escalated: bool
    next_question: TriageQuestion | None = None
    recommended_block: str | None = None


class AcknowledgeEscalationResponse(BaseModel):
    id: UUID
    status: EscalationStatus
    acknowledged_at: datetime


class ResolveEscalationRequest(BaseModel):
    resolution_notes: str


class ResolveEscalationResponse(BaseModel):
    id: UUID
    status: EscalationStatus


class UtilisationRecommendationItem(BaseModel):
    id: UUID
    rationale: str
    recommended_change: dict


class UtilisationRecommendationListResponse(BaseModel):
    items: list[UtilisationRecommendationItem]


class ApplyRecommendationResponse(BaseModel):
    id: UUID
    status: UtilisationRecommendationStatus


class DismissRecommendationResponse(BaseModel):
    id: UUID
    status: UtilisationRecommendationStatus
