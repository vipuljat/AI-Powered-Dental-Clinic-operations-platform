"""Thin HTTP adapter over ``OutreachService``/``ConsentService``/
``ChannelConfigService`` (architecture.md §5.2).

No business logic lives here — every handler validates via its Pydantic
schema, delegates to the service layer, and shapes the response. This
module's ``router`` carries only its own ``/outreach`` sub-prefix; the shared
``/api/v1`` prefix is added on top by ``app/main.py`` when it calls
``app.include_router`` (Interaction contract).

Watch out: no endpoint in this file's own Public surface calls into
``OutreachService`` (``dispatch``/``retry``/``mark_unreachable``) — every
other module's campaign trigger calls that service directly (Interaction
contract, ``services/outreach/service.py``'s own docstring), so this file
never constructs one. ``GET /outreach/messages`` and
``POST .../manual-outcome`` instead read/verify rows via the frozen
``OutreachMessageRepository`` directly, since no ``OutreachService`` method
covers either operation — a documented gap against the frozen contract,
reported in the task summary.
"""

from __future__ import annotations

from datetime import timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select

from app.common.enums import ConsentStatus
from app.common.exceptions.errors import NotFoundError, ValidationFailedError
from app.core.dependencies import CurrentUser, get_db, require_role
from app.models.outreach.models import OutreachMessage
from app.repositories.console.repository import AuditLogRepository
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    OutreachMessageRepository,
)
from app.schemas.outreach.schemas import (
    CaptureConsentRequest,
    ConfigureWhatsappRequest,
    ConfigureWhatsappResponse,
    ConsentLedgerItem,
    ConsentLedgerResponse,
    ConsentResponse,
    ManualOutcomeRequest,
    ManualOutcomeResponse,
    OutreachMessageItem,
    OutreachMessageListResponse,
    TestSendWhatsappRequest,
    TestSendWhatsappResponse,
    WithdrawConsentRequest,
    WithdrawConsentResponse,
)
from app.services.console.service import AuditService
from app.services.outreach.service import ChannelConfigService, ConsentService

router = APIRouter(prefix="/outreach")


def _audit_service() -> AuditService:
    # Each request builds its own thin service/repository graph over the
    # request-scoped `AsyncSession` (`Depends(get_db)`) — no collaborator
    # here is a module-level singleton, matching the rest of this codebase's
    # `require_permission`/`AuthService` wiring (app/core/dependencies.py).
    return AuditService(AuditLogRepository())


def _message_repo() -> OutreachMessageRepository:
    return OutreachMessageRepository()


def _channel_config_repo() -> ChannelConfigurationRepository:
    return ChannelConfigurationRepository()


def _consent_service() -> ConsentService:
    return ConsentService(ConsentRepository(), _message_repo(), _audit_service())


def _channel_config_service() -> ChannelConfigService:
    return ChannelConfigService(_channel_config_repo(), _audit_service())


@router.get("/messages", response_model=OutreachMessageListResponse)
async def list_outreach_messages(
    patient_id: "UUID | None" = None,
    channel: str | None = None,
    status: str | None = None,
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> OutreachMessageListResponse:
    """architecture.md §5.2 `GET /outreach/messages`: 200.

    Watch out: `OutreachService`/its repositories (frozen) expose only
    `OutreachMessageRepository.list_by_patient`, which requires a
    `patient_id` — there is no "list across every patient" repository
    method. When `patient_id` is supplied this delegates straight to that
    method; when it is omitted (this endpoint's own signature makes it
    optional) a direct, read-only `select()` against the frozen
    `OutreachMessage` ORM model is issued here instead, since no service/
    repository method exists to call for that case. This is a documented
    gap against the frozen `services/outreach/service.py` contract (no
    business logic added — filtering only), reported in the task summary.
    """
    if patient_id is not None:
        messages = await _message_repo().list_by_patient(db, patient_id, channel=channel, status=status)
    else:
        query = select(OutreachMessage)
        if channel is not None:
            query = query.where(OutreachMessage.channel == channel)
        if status is not None:
            query = query.where(OutreachMessage.status == status)
        result = await db.execute(query.order_by(OutreachMessage.created_at.desc()))
        messages = list(result.scalars().all())
        for message in messages:
            if message.sent_at is not None and message.sent_at.tzinfo is None:
                message.sent_at = message.sent_at.replace(tzinfo=timezone.utc)

    return OutreachMessageListResponse(
        items=[
            OutreachMessageItem(
                id=message.id,
                campaign_type=message.campaign_type,
                channel=message.channel,
                language=message.language,
                status=message.status,
                sent_at=message.sent_at,
            )
            for message in messages
        ]
    )


@router.post("/messages/{id}/manual-outcome", response_model=ManualOutcomeResponse)
async def log_manual_outcome(
    id: "UUID",
    body: ManualOutcomeRequest,
    user: CurrentUser = Depends(require_role("front_office_staff")),
    db=Depends(get_db),
) -> ManualOutcomeResponse:
    """architecture.md §5.2 `POST /outreach/messages/{id}/manual-outcome`:
    200; 404 (`NotFoundError`) if `id` does not exist.

    Watch out: the frozen `OutreachMessage` ORM model has no
    `manual_outcome`/`manual_outcome_notes` column and `OutreachService`
    exposes no method for recording one — there is no persisted store for
    this value to land in without editing either frozen file. The message's
    existence is still verified against the real repository (404 on a
    missing id), and the outcome is echoed back in the response exactly as
    `ManualOutcomeResponse` declares it; this non-persistence is a
    documented gap against the frozen contract, reported in the task
    summary.
    """
    message = await _message_repo().get_by_id(db, id)
    if message is None:
        raise NotFoundError("Outreach message not found.")
    return ManualOutcomeResponse(id=id, manual_outcome=body.outcome)


@router.post("/consent", response_model=ConsentResponse, status_code=201)
async def capture_consent(body: CaptureConsentRequest, db=Depends(get_db)) -> ConsentResponse:
    """architecture.md §5.2 `POST /outreach/consent`: 201.

    Auth: none — this route declares no `Depends(require_role(...))`/
    `Depends(get_current_user)` at all (Behaviour), so it is reachable both
    by an unauthenticated patient-facing call and an authenticated staff
    manual-entry call; caller identity is never inspected, so every grant/
    decline recorded here is attributed to `ActorType.system`/no staff actor
    by `ConsentService.grant`/`.decline` (their own frozen `actor_id`
    parameter is passed `None`).

    `body.status` must be `granted` or `declined` — `withdrawn` is only
    reachable via `POST /outreach/consent/withdraw` (schema docstring); a
    `withdrawn` value here raises `ValidationFailedError` (422).
    """
    service = _consent_service()
    if body.status == ConsentStatus.granted:
        record = await service.grant(db, body.patient_id, body.channel.value, body.source, None)
    elif body.status == ConsentStatus.declined:
        record = await service.decline(db, body.patient_id, body.channel.value, body.source, None)
    else:
        raise ValidationFailedError(
            "Consent status must be 'granted' or 'declined' here; use POST /outreach/consent/withdraw "
            "to record a withdrawal.",
            details=[{"field": "status", "value": body.status}],
        )
    return ConsentResponse(
        id=record.id,
        channel=record.channel,
        status=record.status,
        effective_at=record.effective_at,
    )


@router.post("/consent/withdraw", response_model=WithdrawConsentResponse)
async def withdraw_consent(body: WithdrawConsentRequest, db=Depends(get_db)) -> WithdrawConsentResponse:
    """architecture.md §5.2 `POST /outreach/consent/withdraw`: 200.

    Auth: none — webhook, e.g. "STOP" keyword (project_rules.auth). FR-E6.5:
    every queued/failed message on this exact channel is halted synchronously
    inside `ConsentService.withdraw`.
    """
    service = _consent_service()
    record, halted = await service.withdraw(db, body.patient_id, body.channel.value, body.source)
    return WithdrawConsentResponse(id=record.id, status=record.status, queued_messages_halted=halted)


@router.get("/consent/{patient_id}/ledger", response_model=ConsentLedgerResponse)
async def get_consent_ledger(
    patient_id: "UUID",
    user: CurrentUser = Depends(require_role("front_office_staff", "clinic_management")),
    db=Depends(get_db),
) -> ConsentLedgerResponse:
    """architecture.md §5.2 `GET /outreach/consent/{patient_id}/ledger`: 200.
    FR-E6.6: the full, chronological, append-only ledger for this patient.
    """
    service = _consent_service()
    records = await service.get_ledger(db, patient_id)
    return ConsentLedgerResponse(
        items=[
            ConsentLedgerItem(channel=record.channel, status=record.status, effective_at=record.effective_at)
            for record in records
        ]
    )


@router.get("/consent/{patient_id}/export")
async def export_consent_ledger(
    patient_id: "UUID",
    user: CurrentUser = Depends(require_role("clinic_management")),
    db=Depends(get_db),
) -> Response:
    """architecture.md §5.2 `GET /outreach/consent/{patient_id}/export`: 200,
    `clinic_management` only. T82 AC: returns a raw PDF byte stream,
    `Content-Type: application/pdf` — `ConsentService.export_ledger` also
    audit-logs the export request itself.
    """
    service = _consent_service()
    pdf_bytes = await service.export_ledger(db, patient_id, user.id)
    return Response(content=pdf_bytes, media_type="application/pdf")


@router.post("/channels/whatsapp/configure", response_model=ConfigureWhatsappResponse)
async def configure_whatsapp(
    body: ConfigureWhatsappRequest,
    user: CurrentUser = Depends(require_role("delivery_team")),
    db=Depends(get_db),
) -> ConfigureWhatsappResponse:
    """architecture.md §5.2 `POST /outreach/channels/whatsapp/configure`:
    200 (Watch out: not 201, even though this upserts a
    `channel_configurations` row — the endpoint is idempotent-by-channel).
    `delivery_team`-only (FR-E1.3).

    `body.phone_number_id` has no column on the frozen `ChannelConfiguration`
    ORM model / no parameter on `ChannelConfigService.configure` to carry it
    to, so it is accepted (schema-valid) but not forwarded — a documented
    gap against the frozen contract, reported in the task summary.
    """
    service = _channel_config_service()
    config = await service.configure(db, "whatsapp", body.bsp_provider, body.access_token, user.id)
    return ConfigureWhatsappResponse(id=config.id, status=config.status)


@router.post("/channels/whatsapp/test-send", response_model=TestSendWhatsappResponse)
async def test_send_whatsapp(
    body: TestSendWhatsappRequest,
    user: CurrentUser = Depends(require_role("delivery_team")),
    db=Depends(get_db),
) -> TestSendWhatsappResponse:
    """architecture.md §5.2 `POST /outreach/channels/whatsapp/test-send`:
    200/502. `delivery_team`-only (FR-E1.3). A BSP delivery failure raises
    `ChannelDeliveryError` (502) out of `ChannelConfigService.test_send`,
    left to propagate unchanged to the registered global handler.
    """
    service = _channel_config_service()
    config = await service.test_send(db, body.to_phone, user.id)
    return TestSendWhatsappResponse(status=config.status)
