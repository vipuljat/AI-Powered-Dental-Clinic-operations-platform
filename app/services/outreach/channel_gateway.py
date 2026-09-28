"""The one external-system adapter file in this tree.

Sends an outbound message on a given channel (WhatsApp, SMS, email — voice's
text-to-speech trigger in app/services/engagement/service.py uses the same
factory) and reports success/failure back to the caller
(``OutreachService.dispatch``/``.retry``, ``ChannelConfigService.test_send``,
``AuthService.request_reset``'s email path).

Only the WhatsApp Business (Meta Cloud API) call shape is concretely specced
(architecture.md §5.2's ``configure``/``test-send`` payloads name it
explicitly, transcribed from Meta's documented Cloud API send-message call).
SMS/email are written against the same generic ``ChannelAdapter`` interface
and are intentionally vendor-unnamed per ``open_questions[Q-a4c64a5b]``
(architecture.md §11 items 17/18) — ``provider_config`` carries whatever the
eventually-selected vendor needs.

``get_channel_adapter`` is the single factory every caller uses to obtain an
adapter (Interaction contract) — none of them import ``WhatsAppAdapter``/
``SmsAdapter``/``EmailAdapter``/``StubAdapter`` directly, which is what keeps
the project_rules.testing substitution point exactly here: in test mode every
channel resolves to a ``StubAdapter`` returning a canned success payload
instead of calling Meta/SMS/telephony APIs.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Protocol

import httpx
from pydantic import BaseModel

from app.common.exceptions.errors import ChannelDeliveryError

if TYPE_CHECKING:
    # Forward-reference only: app/models/outreach/models.py is authored by a
    # different task in this wave. Guarding the import behind TYPE_CHECKING
    # keeps this module importable regardless of that file's write order,
    # while still giving type-checkers the real type for `channel_config`.
    from app.core.config import Settings
    from app.models.outreach.models import ChannelConfiguration

META_GRAPH_API_VERSION = "v19.0"
META_GRAPH_API_BASE_URL = "https://graph.facebook.com"


class DeliveryResult(BaseModel):
    """Outcome of a single `ChannelAdapter.send` call."""

    success: bool
    provider_message_id: str | None
    failure_reason: str | None


class ChannelAdapter(Protocol):
    """The generic outbound-channel interface every adapter implements.

    No caller depends on an adapter's internal HTTP shape (Watch out) —
    only on this `send` signature and the `DeliveryResult` it returns.
    """

    async def send(self, to: str, body: str) -> DeliveryResult: ...


class WhatsAppAdapter:
    """WhatsApp Business (Meta Cloud API) adapter.

    The only channel concretely specced against a named vendor API
    (open_questions[Q-a4c64a5b]) — endpoint/method/payload transcribed from
    Meta's documented Cloud API send-message call, not invented.
    """

    def __init__(self, phone_number_id: str, access_token: str) -> None:
        self._phone_number_id = phone_number_id
        self._access_token = access_token

    async def send(self, to: str, body: str) -> DeliveryResult:
        url = f"{META_GRAPH_API_BASE_URL}/{META_GRAPH_API_VERSION}/{self._phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {self._access_token}"}
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": body},
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                # architecture.md §5.2 POST /outreach/channels/whatsapp/test-send
                # Errors: "502 BSP delivery failure" — a transport-level
                # failure to reach the BSP maps to the same ChannelDeliveryError.
                raise ChannelDeliveryError(f"WhatsApp delivery failed: {exc}") from exc

        if response.is_success:
            data = response.json()
            message_id = None
            messages = data.get("messages") or []
            if messages:
                message_id = messages[0].get("id")
            return DeliveryResult(success=True, provider_message_id=message_id, failure_reason=None)

        # Map Meta's documented error payload shape ({"error": {"message": ...}})
        # into `failure_reason`, and raise so the caller/route surfaces the
        # documented 502 "BSP delivery failure".
        try:
            error_payload = response.json()
            failure_reason = (error_payload.get("error") or {}).get("message") or response.text
        except ValueError:
            failure_reason = response.text or f"HTTP {response.status_code}"

        raise ChannelDeliveryError(
            f"WhatsApp Cloud API delivery failed: {failure_reason}",
            details=[{"status_code": response.status_code, "failure_reason": failure_reason}],
        )


class SmsAdapter:
    """Vendor-agnostic SMS adapter.

    Not concretely specced against a named vendor API per
    open_questions[Q-a4c64a5b] — `provider_config` carries whatever the
    eventually-selected SMS vendor needs (account id, API key, sender id...).
    """

    def __init__(self, provider_config: dict) -> None:
        self._provider_config = provider_config

    async def send(self, to: str, body: str) -> DeliveryResult:
        base_url = self._provider_config.get("base_url")
        api_key = self._provider_config.get("api_key")
        if not base_url or not api_key:
            raise ChannelDeliveryError(
                "SMS provider is not configured (missing base_url/api_key).",
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"to": to, "body": body}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(base_url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise ChannelDeliveryError(f"SMS delivery failed: {exc}") from exc

        if not response.is_success:
            raise ChannelDeliveryError(
                f"SMS provider delivery failed with HTTP {response.status_code}",
                details=[{"status_code": response.status_code, "body": response.text}],
            )

        data: dict = {}
        try:
            data = response.json()
        except ValueError:
            pass

        return DeliveryResult(
            success=True,
            provider_message_id=data.get("id") or data.get("message_id"),
            failure_reason=None,
        )


class EmailAdapter:
    """Vendor-agnostic email adapter, same caveat as `SmsAdapter`."""

    def __init__(self, provider_config: dict) -> None:
        self._provider_config = provider_config

    async def send(self, to: str, subject: str, body: str) -> DeliveryResult:
        base_url = self._provider_config.get("base_url")
        api_key = self._provider_config.get("api_key")
        if not base_url or not api_key:
            raise ChannelDeliveryError(
                "Email provider is not configured (missing base_url/api_key).",
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"to": to, "subject": subject, "body": body}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(base_url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise ChannelDeliveryError(f"Email delivery failed: {exc}") from exc

        if not response.is_success:
            raise ChannelDeliveryError(
                f"Email provider delivery failed with HTTP {response.status_code}",
                details=[{"status_code": response.status_code, "body": response.text}],
            )

        data: dict = {}
        try:
            data = response.json()
        except ValueError:
            pass

        return DeliveryResult(
            success=True,
            provider_message_id=data.get("id") or data.get("message_id"),
            failure_reason=None,
        )


class StubAdapter:
    """Canned-success stand-in for every channel in test mode.

    project_rules.testing item (4): the outbound channel adapters are
    substituted for a stub returning a canned success payload instead of
    calling Meta/SMS/telephony APIs — this is that stub.
    """

    async def send(self, to: str, body: str) -> DeliveryResult:
        return DeliveryResult(
            success=True,
            provider_message_id=f"stub-{uuid.uuid4()}",
            failure_reason=None,
        )


def get_channel_adapter(
    channel: str,
    settings: "Settings",
    channel_config: "ChannelConfiguration | None",
) -> ChannelAdapter:
    """The single factory point every caller uses to obtain a channel adapter.

    project_rules.testing: returns a `StubAdapter` for every channel when
    `settings.environment == "test"` — a real Meta/SMS/telephony client is
    never constructed in test mode, keeping the substitution point exactly
    here rather than ad hoc inside `OutreachService`/`ChannelConfigService`/
    `AuthService`.
    """

    if settings.environment == "test":
        # No real Meta/SMS/telephony API is reachable (or desirable) under
        # test — every channel resolves to the canned-success stub here,
        # the one switch point per the Interaction contract.
        return StubAdapter()

    normalized_channel = channel.lower()

    if normalized_channel == "whatsapp":
        provider_config = getattr(channel_config, "provider_config", None) or {}
        phone_number_id = provider_config.get("phone_number_id", "")
        access_token = provider_config.get("access_token", "")
        return WhatsAppAdapter(phone_number_id=phone_number_id, access_token=access_token)

    if normalized_channel == "sms":
        provider_config = getattr(channel_config, "provider_config", None) or {}
        return SmsAdapter(provider_config=provider_config)

    if normalized_channel == "email":
        provider_config = getattr(channel_config, "provider_config", None) or {}
        return EmailAdapter(provider_config=provider_config)

    raise ChannelDeliveryError(f"Unsupported outreach channel: {channel!r}")
