"""Unit tests for app/services/outreach/channel_gateway.py.

This is the one external-system adapter file in the tree: it sends an
outbound message on a given channel and reports success/failure back to
its callers via `DeliveryResult`. Only the WhatsApp (Meta Cloud API) call
shape is concretely specced -- SMS/email are intentionally vendor-agnostic
per open_questions[Q-a4c64a5b], so this file does not assert on their
internal HTTP behaviour, only on `get_channel_adapter`'s dispatch to them.

No live infrastructure is used: the one real network call this file makes
(WhatsAppAdapter.send's POST to the Meta Cloud API) is exercised against a
mocked `httpx.AsyncClient.post`, never a real HTTP connection.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.common.exceptions.errors import ChannelDeliveryError
from app.services.outreach.channel_gateway import (
    DeliveryResult,
    EmailAdapter,
    SmsAdapter,
    StubAdapter,
    WhatsAppAdapter,
    get_channel_adapter,
)

pytestmark = pytest.mark.asyncio

META_URL = "https://graph.facebook.com/v19.0/1234567890/messages"


def _settings(environment: str) -> SimpleNamespace:
    """A minimal duck-typed stand-in for app.core.config.Settings.

    get_channel_adapter only ever reads `.environment` off of it per this
    file's own public surface, so a SimpleNamespace is sufficient without
    reaching into app.core.config (owned by a different task).
    """
    return SimpleNamespace(environment=environment)


@pytest.fixture
def mock_post():
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mocked:
        yield mocked


def _get_post_kwargs(mock_post) -> tuple[tuple, dict]:
    args, kwargs = mock_post.await_args
    return args, kwargs


# ---------------------------------------------------------------------------
# DeliveryResult
# ---------------------------------------------------------------------------


class TestDeliveryResult:
    def test_success_result_round_trips_all_fields(self):
        result = DeliveryResult(
            success=True, provider_message_id="wamid.HBGDYQ", failure_reason=None
        )
        assert result.success is True
        assert result.provider_message_id == "wamid.HBGDYQ"
        assert result.failure_reason is None

    def test_failure_result_round_trips_all_fields(self):
        result = DeliveryResult(
            success=False, provider_message_id=None, failure_reason="Invalid parameter"
        )
        assert result.success is False
        assert result.provider_message_id is None
        assert result.failure_reason == "Invalid parameter"


# ---------------------------------------------------------------------------
# WhatsAppAdapter -- the one concretely-specced adapter (Meta Cloud API)
# ---------------------------------------------------------------------------


class TestWhatsAppAdapterSend:
    async def test_posts_to_metas_documented_send_message_endpoint_and_payload(self, mock_post):
        request = httpx.Request("POST", META_URL)
        mock_post.return_value = httpx.Response(
            200, json={"messages": [{"id": "wamid.HBGDYQVAAAA"}]}, request=request
        )
        adapter = WhatsAppAdapter(phone_number_id="1234567890", access_token="secret-token")

        await adapter.send(to="+15551234567", body="Your appointment is confirmed")

        assert mock_post.await_count == 1
        args, kwargs = _get_post_kwargs(mock_post)
        url = args[0] if args else kwargs.get("url")
        assert url == "https://graph.facebook.com/v19.0/1234567890/messages"
        assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
        assert kwargs["json"] == {
            "messaging_product": "whatsapp",
            "to": "+15551234567",
            "type": "text",
            "text": {"body": "Your appointment is confirmed"},
        }

    async def test_2xx_response_returns_successful_delivery_result(self, mock_post):
        request = httpx.Request("POST", META_URL)
        mock_post.return_value = httpx.Response(
            200, json={"messages": [{"id": "wamid.HBGDYQVAAAA"}]}, request=request
        )
        adapter = WhatsAppAdapter(phone_number_id="1234567890", access_token="secret-token")

        result = await adapter.send(to="+15551234567", body="hello")

        assert result.success is True
        assert result.failure_reason is None
        # Meta's documented send-message response echoes the created message's
        # id under messages[0].id -- the one id this adapter has to surface.
        assert result.provider_message_id == "wamid.HBGDYQVAAAA"

    async def test_non_2xx_response_raises_channel_delivery_error(self, mock_post):
        request = httpx.Request("POST", META_URL)
        mock_post.return_value = httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid parameter",
                    "type": "OAuthException",
                    "code": 100,
                }
            },
            request=request,
        )
        adapter = WhatsAppAdapter(phone_number_id="1234567890", access_token="secret-token")

        with pytest.raises(ChannelDeliveryError):
            await adapter.send(to="+15551234567", body="hello")

    async def test_5xx_response_also_raises_channel_delivery_error(self, mock_post):
        request = httpx.Request("POST", META_URL)
        mock_post.return_value = httpx.Response(
            500, json={"error": {"message": "Internal error", "code": 1}}, request=request
        )
        adapter = WhatsAppAdapter(phone_number_id="1234567890", access_token="secret-token")

        with pytest.raises(ChannelDeliveryError):
            await adapter.send(to="+15551234567", body="hello")


# ---------------------------------------------------------------------------
# StubAdapter -- used for every channel in test mode
# ---------------------------------------------------------------------------


class TestStubAdapter:
    async def test_send_always_returns_a_canned_success_result(self):
        adapter = StubAdapter()

        result = await adapter.send(to="+15551234567", body="anything")

        assert result.success is True
        assert result.failure_reason is None
        assert isinstance(result.provider_message_id, str)
        assert result.provider_message_id.startswith("stub-")
        # the suffix after "stub-" must be a real uuid4, per this file's spec
        suffix = result.provider_message_id[len("stub-") :]
        parsed = uuid.UUID(suffix)
        assert parsed.version == 4

    async def test_send_does_not_reuse_the_same_provider_message_id_across_calls(self):
        adapter = StubAdapter()

        first = await adapter.send(to="+15551234567", body="one")
        second = await adapter.send(to="+15551234567", body="two")

        assert first.provider_message_id != second.provider_message_id


# ---------------------------------------------------------------------------
# get_channel_adapter -- the single factory / test-mode substitution point
# ---------------------------------------------------------------------------


class TestGetChannelAdapter:
    @pytest.mark.parametrize("channel", ["whatsapp", "sms", "email"])
    def test_test_environment_returns_a_stub_adapter_for_every_named_channel(self, channel):
        adapter = get_channel_adapter(
            channel=channel, settings=_settings("test"), channel_config=None
        )
        assert isinstance(adapter, StubAdapter)

    def test_test_environment_returns_a_stub_adapter_even_for_an_unrecognised_channel(self):
        # project_rules.testing: EVERY channel is a StubAdapter in test mode --
        # channel selection among the concrete adapters only matters once
        # settings.environment != "test".
        adapter = get_channel_adapter(
            channel="carrier_pigeon", settings=_settings("test"), channel_config=None
        )
        assert isinstance(adapter, StubAdapter)

    def test_non_test_environment_selects_whatsapp_adapter_for_whatsapp_channel(self):
        adapter = get_channel_adapter(
            channel="whatsapp", settings=_settings("production"), channel_config=MagicMock()
        )
        assert isinstance(adapter, WhatsAppAdapter)

    def test_non_test_environment_selects_sms_adapter_for_sms_channel(self):
        adapter = get_channel_adapter(
            channel="sms", settings=_settings("production"), channel_config=MagicMock()
        )
        assert isinstance(adapter, SmsAdapter)

    def test_non_test_environment_selects_email_adapter_for_email_channel(self):
        adapter = get_channel_adapter(
            channel="email", settings=_settings("production"), channel_config=MagicMock()
        )
        assert isinstance(adapter, EmailAdapter)
