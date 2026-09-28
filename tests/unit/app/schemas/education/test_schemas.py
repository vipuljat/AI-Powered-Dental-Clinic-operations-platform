"""Unit tests for app/schemas/education/schemas.py.

Covers the three Pydantic v2 models that form the entire request/response
contract for `/education/*` (per this file's own Interaction contract
section, no other file's spec is consulted here).

`ContentDeliveryItem.trigger_type` / `.status` are typed against
`EducationTriggerType` / `ContentDeliveryStatus`, which this spec's
`depends_on` names as living in `app/common/enums.py`. This test never
assumes a *specific* member name of either enum (the schemas.py spec does
not enumerate them) -- it only relies on each enum having at least one
member, discovered dynamically via `list(...)`, so the test cannot be
accidentally coupled to enum values that belong to a different file's spec.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.common.enums import ContentDeliveryStatus, EducationTriggerType
from app.schemas.education.schemas import (
    ContentDeliveryItem,
    ContentDeliveryListResponse,
    TrackingFunnelResponse,
)

# A trigger_type/status value guaranteed not to be a member of either enum,
# used to prove that invalid values are rejected rather than silently
# accepted as an arbitrary string.
_NOT_A_MEMBER = "definitely-not-a-real-enum-member-xyz"


def _first_trigger_type() -> EducationTriggerType:
    return next(iter(EducationTriggerType))


def _first_status() -> ContentDeliveryStatus:
    return next(iter(ContentDeliveryStatus))


# ---------------------------------------------------------------------------
# TrackingFunnelResponse
# ---------------------------------------------------------------------------


class TestTrackingFunnelResponse:
    def test_accepts_all_declared_fields(self):
        resp = TrackingFunnelResponse(
            delivered=100,
            opened=40,
            completed=25,
            channel_limited=["sms"],
        )
        assert resp.delivered == 100
        assert resp.opened == 40
        assert resp.completed == 25
        assert resp.channel_limited == ["sms"]

    def test_channel_limited_can_list_multiple_channels(self):
        # FR-E10.3 / architecture.md §5.2: channel_limited communicates
        # which channels in the funnel cannot report opens; nothing in the
        # schema restricts it to a single entry.
        resp = TrackingFunnelResponse(
            delivered=1, opened=1, completed=1, channel_limited=["sms", "ivr"]
        )
        assert resp.channel_limited == ["sms", "ivr"]

    def test_channel_limited_can_be_empty(self):
        resp = TrackingFunnelResponse(
            delivered=1, opened=1, completed=1, channel_limited=[]
        )
        assert resp.channel_limited == []

    def test_counts_reject_non_numeric_string(self):
        with pytest.raises(ValidationError) as exc_info:
            TrackingFunnelResponse(
                delivered="not-a-number", opened=1, completed=1, channel_limited=[]
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("delivered",) for e in errors)

    def test_channel_limited_rejects_non_string_items(self):
        with pytest.raises(ValidationError) as exc_info:
            TrackingFunnelResponse(
                delivered=1, opened=1, completed=1, channel_limited=[123]
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("channel_limited", 0) for e in errors)

    @pytest.mark.parametrize("missing", ["delivered", "opened", "completed", "channel_limited"])
    def test_missing_required_field_raises(self, missing):
        payload = {
            "delivered": 1,
            "opened": 1,
            "completed": 1,
            "channel_limited": ["sms"],
        }
        del payload[missing]
        with pytest.raises(ValidationError) as exc_info:
            TrackingFunnelResponse(**payload)
        errors = exc_info.value.errors()
        assert any(e["loc"] == (missing,) and e["type"] == "missing" for e in errors)


# ---------------------------------------------------------------------------
# ContentDeliveryItem
# ---------------------------------------------------------------------------


class TestContentDeliveryItem:
    def test_accepts_all_declared_fields_with_delivered_at(self):
        item_id = uuid4()
        trigger = _first_trigger_type()
        status = _first_status()
        delivered_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        item = ContentDeliveryItem(
            content_item_id=item_id,
            trigger_type=trigger,
            status=status,
            delivered_at=delivered_at,
        )
        assert item.content_item_id == item_id
        assert item.trigger_type == trigger
        assert item.status == status
        assert item.delivered_at == delivered_at

    def test_delivered_at_accepts_none(self):
        item = ContentDeliveryItem(
            content_item_id=uuid4(),
            trigger_type=_first_trigger_type(),
            status=_first_status(),
            delivered_at=None,
        )
        assert item.delivered_at is None

    def test_content_item_id_coerces_uuid_string(self):
        raw = "12345678-1234-5678-1234-567812345678"
        item = ContentDeliveryItem(
            content_item_id=raw,
            trigger_type=_first_trigger_type(),
            status=_first_status(),
            delivered_at=None,
        )
        assert isinstance(item.content_item_id, UUID)
        assert item.content_item_id == UUID(raw)

    def test_content_item_id_rejects_non_uuid_string(self):
        with pytest.raises(ValidationError) as exc_info:
            ContentDeliveryItem(
                content_item_id="not-a-uuid",
                trigger_type=_first_trigger_type(),
                status=_first_status(),
                delivered_at=None,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("content_item_id",) for e in errors)

    def test_trigger_type_accepts_enum_member_value(self):
        trigger = _first_trigger_type()
        item = ContentDeliveryItem(
            content_item_id=uuid4(),
            trigger_type=trigger.value,
            status=_first_status(),
            delivered_at=None,
        )
        assert item.trigger_type == trigger
        assert isinstance(item.trigger_type, EducationTriggerType)

    def test_trigger_type_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            ContentDeliveryItem(
                content_item_id=uuid4(),
                trigger_type=_NOT_A_MEMBER,
                status=_first_status(),
                delivered_at=None,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("trigger_type",) for e in errors)

    def test_status_rejects_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            ContentDeliveryItem(
                content_item_id=uuid4(),
                trigger_type=_first_trigger_type(),
                status=_NOT_A_MEMBER,
                delivered_at=None,
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("status",) for e in errors)

    def test_missing_delivered_at_raises_since_field_is_required_but_nullable(self):
        # `delivered_at: datetime | None` declares the field as nullable,
        # not optional-with-default -- omitting it entirely is still a
        # missing-field error distinct from passing an explicit None.
        with pytest.raises(ValidationError) as exc_info:
            ContentDeliveryItem(
                content_item_id=uuid4(),
                trigger_type=_first_trigger_type(),
                status=_first_status(),
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("delivered_at",) and e["type"] == "missing" for e in errors)

    def test_json_mode_serializes_uuid_and_datetime_as_strings(self):
        item_id = uuid4()
        delivered_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        item = ContentDeliveryItem(
            content_item_id=item_id,
            trigger_type=_first_trigger_type(),
            status=_first_status(),
            delivered_at=delivered_at,
        )
        dumped = item.model_dump(mode="json")
        assert dumped["content_item_id"] == str(item_id)
        assert dumped["delivered_at"] == "2026-01-01T12:00:00Z"


# ---------------------------------------------------------------------------
# ContentDeliveryListResponse
# ---------------------------------------------------------------------------


class TestContentDeliveryListResponse:
    def test_wraps_a_list_of_content_delivery_items(self):
        item = ContentDeliveryItem(
            content_item_id=uuid4(),
            trigger_type=_first_trigger_type(),
            status=_first_status(),
            delivered_at=None,
        )
        resp = ContentDeliveryListResponse(items=[item])
        assert resp.items == [item]

    def test_accepts_plain_dicts_and_parses_them_into_items(self):
        item_id = uuid4()
        trigger = _first_trigger_type()
        status = _first_status()
        resp = ContentDeliveryListResponse(
            items=[
                {
                    "content_item_id": str(item_id),
                    "trigger_type": trigger.value,
                    "status": status.value,
                    "delivered_at": None,
                }
            ]
        )
        assert len(resp.items) == 1
        parsed = resp.items[0]
        assert isinstance(parsed, ContentDeliveryItem)
        assert parsed.content_item_id == item_id
        assert parsed.trigger_type == trigger
        assert parsed.status == status

    def test_empty_items_list_is_valid(self):
        resp = ContentDeliveryListResponse(items=[])
        assert resp.items == []

    def test_missing_items_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ContentDeliveryListResponse()
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("items",) and e["type"] == "missing" for e in errors)

    def test_invalid_item_in_list_raises_with_index_in_location(self):
        with pytest.raises(ValidationError) as exc_info:
            ContentDeliveryListResponse(
                items=[
                    {
                        "content_item_id": "not-a-uuid",
                        "trigger_type": _first_trigger_type().value,
                        "status": _first_status().value,
                        "delivered_at": None,
                    }
                ]
            )
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("items", 0, "content_item_id") for e in errors)
