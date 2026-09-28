"""Unit tests for app/schemas/intelligence/schemas.py.

Covers every Pydantic v2 request/response model that forms the
`/intelligence/*` request/response contract (per this file's own spec).

Several fields are typed against enums this spec's `depends_on` names as
living in `app/common/enums.py` (`RiskLevel`, `ScoreSource`,
`MlEvaluationStatus`, `EscalationStatus`, `UtilisationRecommendationStatus`).
This test never assumes a *specific* member name of any of those enums --
it only relies on each enum having at least one member, discovered
dynamically via `next(iter(...))`, so the test cannot be accidentally
coupled to enum values that belong to a different file's spec.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.common.enums import (
    EscalationStatus,
    MlEvaluationStatus,
    RiskLevel,
    ScoreSource,
    UtilisationRecommendationStatus,
)
from app.schemas.intelligence.schemas import (
    AcknowledgeEscalationResponse,
    AnswerTriageQuestionRequest,
    AnswerTriageQuestionResponse,
    ApplyRecommendationResponse,
    DismissRecommendationResponse,
    ResolveEscalationRequest,
    ResolveEscalationResponse,
    RiskAnalyticsResponse,
    RiskScoreResponse,
    StartTriageSessionRequest,
    StartTriageSessionResponse,
    TriageQuestion,
    UtilisationRecommendationItem,
    UtilisationRecommendationListResponse,
)

# A value guaranteed not to be a member of any of the enums under test, used
# to prove invalid values are rejected rather than silently accepted as an
# arbitrary string.
_NOT_A_MEMBER = "definitely-not-a-real-enum-member-xyz"


def _first(enum_cls):
    return next(iter(enum_cls))


# ---------------------------------------------------------------------------
# RiskScoreResponse
# ---------------------------------------------------------------------------


class TestRiskScoreResponse:
    def test_accepts_all_declared_fields(self):
        level = _first(RiskLevel)
        source = _first(ScoreSource)
        resp = RiskScoreResponse(
            risk_level=level,
            score_value=0.82,
            source=source,
            model_version="v1.3",
        )
        assert resp.risk_level == level
        assert resp.score_value == 0.82
        assert resp.source == source
        assert resp.model_version == "v1.3"

    def test_model_version_accepts_none(self):
        resp = RiskScoreResponse(
            risk_level=_first(RiskLevel),
            score_value=0.5,
            source=_first(ScoreSource),
            model_version=None,
        )
        assert resp.model_version is None

    def test_risk_level_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskScoreResponse(
                risk_level=_NOT_A_MEMBER,
                score_value=0.5,
                source=_first(ScoreSource),
                model_version=None,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("risk_level",) for e in errors)

    def test_source_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskScoreResponse(
                risk_level=_first(RiskLevel),
                score_value=0.5,
                source=_NOT_A_MEMBER,
                model_version=None,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("source",) for e in errors)

    def test_score_value_rejects_non_numeric_string(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskScoreResponse(
                risk_level=_first(RiskLevel),
                score_value="not-a-number",
                source=_first(ScoreSource),
                model_version=None,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("score_value",) for e in errors)

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskScoreResponse(
                score_value=0.5, source=_first(ScoreSource), model_version=None
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("risk_level",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# RiskAnalyticsResponse
# ---------------------------------------------------------------------------


class TestRiskAnalyticsResponse:
    def test_accepts_distribution_and_model_status(self):
        status = _first(MlEvaluationStatus)
        resp = RiskAnalyticsResponse(
            distribution={"low": 120, "medium": 40, "high": 12},
            model_status=status,
        )
        assert resp.distribution == {"low": 120, "medium": 40, "high": 12}
        assert resp.model_status == status

    def test_model_status_accepts_none(self):
        resp = RiskAnalyticsResponse(distribution={"low": 1}, model_status=None)
        assert resp.model_status is None

    def test_distribution_values_must_be_int(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalyticsResponse(
                distribution={"low": "not-an-int"}, model_status=None
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("distribution", "low") for e in errors)

    def test_model_status_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalyticsResponse(distribution={"low": 1}, model_status=_NOT_A_MEMBER)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("model_status",) for e in errors)

    def test_missing_distribution_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskAnalyticsResponse(model_status=None)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("distribution",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# StartTriageSessionRequest
# ---------------------------------------------------------------------------


class TestStartTriageSessionRequest:
    def test_accepts_channel_and_patient_id(self):
        pid = uuid4()
        req = StartTriageSessionRequest(channel="webchat", patient_id=pid)
        assert req.channel == "webchat"
        assert req.patient_id == pid

    def test_patient_id_defaults_to_none_when_omitted(self):
        req = StartTriageSessionRequest(channel="voice")
        assert req.patient_id is None

    def test_patient_id_rejects_non_uuid_string(self):
        with pytest.raises(ValidationError) as exc_info:
            StartTriageSessionRequest(channel="webchat", patient_id="not-a-uuid")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("patient_id",) for e in errors)

    def test_missing_channel_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            StartTriageSessionRequest()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("channel",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# TriageQuestion
# ---------------------------------------------------------------------------


class TestTriageQuestion:
    def test_accepts_key_and_options(self):
        # FR-E8.4: the triage interface is structured multiple-choice, not
        # free-text -- `options` carries the fixed choice set for `key`.
        question = TriageQuestion(key="pain_level", options=["mild", "moderate", "severe"])
        assert question.key == "pain_level"
        assert question.options == ["mild", "moderate", "severe"]

    def test_options_rejects_non_string_items(self):
        with pytest.raises(ValidationError) as exc_info:
            TriageQuestion(key="pain_level", options=[1, 2, 3])
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("options", 0) for e in errors)

    def test_missing_options_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            TriageQuestion(key="pain_level")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("options",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# StartTriageSessionResponse
# ---------------------------------------------------------------------------


class TestStartTriageSessionResponse:
    def test_accepts_id_and_next_question(self):
        session_id = uuid4()
        question = TriageQuestion(key="pain_level", options=["mild", "severe"])
        resp = StartTriageSessionResponse(id=session_id, next_question=question)
        assert resp.id == session_id
        assert resp.next_question == question

    def test_next_question_accepts_none(self):
        resp = StartTriageSessionResponse(id=uuid4(), next_question=None)
        assert resp.next_question is None

    def test_next_question_accepts_plain_dict(self):
        session_id = uuid4()
        resp = StartTriageSessionResponse(
            id=session_id,
            next_question={"key": "swelling", "options": ["yes", "no"]},
        )
        assert isinstance(resp.next_question, TriageQuestion)
        assert resp.next_question.key == "swelling"
        assert resp.next_question.options == ["yes", "no"]

    def test_id_rejects_non_uuid_string(self):
        with pytest.raises(ValidationError) as exc_info:
            StartTriageSessionResponse(id="not-a-uuid", next_question=None)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)


# ---------------------------------------------------------------------------
# AnswerTriageQuestionRequest
# ---------------------------------------------------------------------------


class TestAnswerTriageQuestionRequest:
    def test_accepts_question_key_and_answer(self):
        # Interaction contract: `question_key` is expected to echo a `key`
        # previously returned by `TriageQuestion`; the schema itself does
        # not validate that echo, nor that `answer` is one of the offered
        # options (that check happens at the service layer against the
        # live rule set) -- both an arbitrary key and an arbitrary answer
        # string are accepted here.
        req = AnswerTriageQuestionRequest(question_key="pain_level", answer="severe")
        assert req.question_key == "pain_level"
        assert req.answer == "severe"

    def test_answer_not_restricted_to_any_fixed_set_by_this_schema(self):
        req = AnswerTriageQuestionRequest(
            question_key="pain_level", answer="an-answer-not-in-any-option-list"
        )
        assert req.answer == "an-answer-not-in-any-option-list"

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            AnswerTriageQuestionRequest(question_key="pain_level")
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("answer",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# AnswerTriageQuestionResponse
# ---------------------------------------------------------------------------


class TestAnswerTriageQuestionResponse:
    def test_non_escalated_response_with_next_question(self):
        question = TriageQuestion(key="swelling", options=["yes", "no"])
        resp = AnswerTriageQuestionResponse(
            escalated=False, next_question=question, recommended_block=None
        )
        assert resp.escalated is False
        assert resp.next_question == question
        assert resp.recommended_block is None

    def test_escalated_response_with_both_fields_null(self):
        # architecture.md §5.2: on escalation, next_question and
        # recommended_block are both null.
        resp = AnswerTriageQuestionResponse(
            escalated=True, next_question=None, recommended_block=None
        )
        assert resp.escalated is True
        assert resp.next_question is None
        assert resp.recommended_block is None

    def test_final_non_escalated_response_with_recommended_block(self):
        resp = AnswerTriageQuestionResponse(
            escalated=False, next_question=None, recommended_block="30min_general"
        )
        assert resp.escalated is False
        assert resp.next_question is None
        assert resp.recommended_block == "30min_general"

    def test_schema_itself_does_not_enforce_mutual_exclusivity(self):
        # Per this file's own Behaviour section: this schema does not
        # enforce that escalated=True implies next_question/
        # recommended_block are null -- that invariant is
        # TriageService.answer's responsibility, not this model's. A
        # (deliberately) contradictory combination must still validate
        # here so a regression that adds unspecified schema-level
        # enforcement is caught by this failing.
        question = TriageQuestion(key="swelling", options=["yes", "no"])
        resp = AnswerTriageQuestionResponse(
            escalated=True, next_question=question, recommended_block="30min_general"
        )
        assert resp.escalated is True
        assert resp.next_question == question
        assert resp.recommended_block == "30min_general"

    def test_missing_escalated_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            AnswerTriageQuestionResponse(next_question=None, recommended_block=None)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("escalated",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# AcknowledgeEscalationResponse
# ---------------------------------------------------------------------------


class TestAcknowledgeEscalationResponse:
    def test_accepts_all_declared_fields(self):
        esc_id = uuid4()
        status = _first(EscalationStatus)
        acknowledged_at = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
        resp = AcknowledgeEscalationResponse(
            id=esc_id, status=status, acknowledged_at=acknowledged_at
        )
        assert resp.id == esc_id
        assert resp.status == status
        assert resp.acknowledged_at == acknowledged_at

    def test_status_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            AcknowledgeEscalationResponse(
                id=uuid4(),
                status=_NOT_A_MEMBER,
                acknowledged_at=datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc),
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("status",) for e in errors)

    def test_missing_acknowledged_at_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            AcknowledgeEscalationResponse(id=uuid4(), status=_first(EscalationStatus))
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("acknowledged_at",) and e["type"] == "missing" for e in errors)

    def test_json_mode_serializes_id_and_acknowledged_at_as_strings(self):
        esc_id = uuid4()
        acknowledged_at = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
        resp = AcknowledgeEscalationResponse(
            id=esc_id, status=_first(EscalationStatus), acknowledged_at=acknowledged_at
        )
        dumped = resp.model_dump(mode="json")
        assert dumped["id"] == str(esc_id)
        assert dumped["acknowledged_at"] == "2026-03-01T09:30:00Z"


# ---------------------------------------------------------------------------
# ResolveEscalationRequest / ResolveEscalationResponse
# ---------------------------------------------------------------------------


class TestResolveEscalationRequest:
    def test_accepts_resolution_notes(self):
        req = ResolveEscalationRequest(resolution_notes="Contacted patient, no urgent care needed.")
        assert req.resolution_notes == "Contacted patient, no urgent care needed."

    def test_missing_resolution_notes_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ResolveEscalationRequest()
        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("resolution_notes",) and e["type"] == "missing" for e in errors
        )


class TestResolveEscalationResponse:
    def test_accepts_id_and_status(self):
        esc_id = uuid4()
        status = _first(EscalationStatus)
        resp = ResolveEscalationResponse(id=esc_id, status=status)
        assert resp.id == esc_id
        assert resp.status == status

    def test_status_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            ResolveEscalationResponse(id=uuid4(), status=_NOT_A_MEMBER)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("status",) for e in errors)


# ---------------------------------------------------------------------------
# UtilisationRecommendationItem / UtilisationRecommendationListResponse
# ---------------------------------------------------------------------------


class TestUtilisationRecommendationItem:
    def test_accepts_all_declared_fields(self):
        item_id = uuid4()
        change = {"provider_id": "abc-123", "shift": "afternoon"}
        item = UtilisationRecommendationItem(
            id=item_id,
            rationale="Chair 3 idle Tue/Thu afternoons over the last 4 weeks.",
            recommended_change=change,
        )
        assert item.id == item_id
        assert item.rationale == "Chair 3 idle Tue/Thu afternoons over the last 4 weeks."
        assert item.recommended_change == change

    def test_recommended_change_rejects_non_dict(self):
        with pytest.raises(ValidationError) as exc_info:
            UtilisationRecommendationItem(
                id=uuid4(), rationale="x", recommended_change="not-a-dict"
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("recommended_change",) for e in errors)

    def test_missing_rationale_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            UtilisationRecommendationItem(id=uuid4(), recommended_change={})
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("rationale",) and e["type"] == "missing" for e in errors)


class TestUtilisationRecommendationListResponse:
    def test_wraps_a_list_of_items(self):
        item = UtilisationRecommendationItem(
            id=uuid4(), rationale="x", recommended_change={"a": 1}
        )
        resp = UtilisationRecommendationListResponse(items=[item])
        assert resp.items == [item]

    def test_empty_items_list_is_valid(self):
        resp = UtilisationRecommendationListResponse(items=[])
        assert resp.items == []

    def test_missing_items_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            UtilisationRecommendationListResponse()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("items",) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# ApplyRecommendationResponse / DismissRecommendationResponse
# ---------------------------------------------------------------------------


class TestApplyRecommendationResponse:
    def test_accepts_id_and_status(self):
        rec_id = uuid4()
        status = _first(UtilisationRecommendationStatus)
        resp = ApplyRecommendationResponse(id=rec_id, status=status)
        assert resp.id == rec_id
        assert resp.status == status

    def test_status_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            ApplyRecommendationResponse(id=uuid4(), status=_NOT_A_MEMBER)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("status",) for e in errors)


class TestDismissRecommendationResponse:
    def test_accepts_id_and_status(self):
        rec_id = uuid4()
        status = _first(UtilisationRecommendationStatus)
        resp = DismissRecommendationResponse(id=rec_id, status=status)
        assert resp.id == rec_id
        assert resp.status == status

    def test_status_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            DismissRecommendationResponse(id=uuid4(), status=_NOT_A_MEMBER)
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("status",) for e in errors)
