"""Unit tests for app/services/education/service.py::EducationTriggerService.

These tests exercise EducationTriggerService against mocked repositories and a mocked
OutreachService, per the file's own spec (FR-E10.1/E10.2/E10.3). The service's declared
constructor only takes (content_repo, delivery_repo, patient_repo, outreach_service); per
its `depends_on` on app/repositories/scheduling/repository.py, appointment resolution goes
through AppointmentRepository (a stateless, no-arg-constructor data-access class per its own
spec) -- so we patch AppointmentRepository.get_by_id directly on the class object, which
intercepts the call regardless of how/where the service instantiates or imports that class.
"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.repositories.scheduling.repository import AppointmentRepository
from app.services.education.service import EducationTriggerService

pytestmark = pytest.mark.asyncio


def _flat(call):
    """Flatten a mock call's positional args + keyword values into one list."""
    args, kwargs = call
    return list(args) + list(kwargs.values())


def _build_service():
    content_repo = MagicMock()
    content_repo.get_by_stage_language = AsyncMock()
    content_repo.get_general_fallback = AsyncMock()

    delivery_repo = MagicMock()
    delivery_repo.record_delivery = AsyncMock()
    delivery_repo.get_funnel = AsyncMock()
    delivery_repo.list_by_patient = AsyncMock()

    patient_repo = MagicMock()
    patient_repo.get_by_id = AsyncMock()

    outreach_service = MagicMock()
    outreach_service.dispatch = AsyncMock()

    service = EducationTriggerService(content_repo, delivery_repo, patient_repo, outreach_service)
    return service, content_repo, delivery_repo, patient_repo, outreach_service


def _patient(**overrides):
    base = dict(
        id=uuid4(),
        developmental_stage="child",
        language="en",
        preferred_channel="sms",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _appointment(**overrides):
    base = dict(id=uuid4(), patient_id=uuid4(), status="booked")
    base.update(overrides)
    return SimpleNamespace(**base)


def _content_item(**overrides):
    base = dict(id=uuid4(), developmental_stage="child", language="en")
    base.update(overrides)
    return SimpleNamespace(**base)


def _outreach_message(**overrides):
    base = dict(id=uuid4(), status="sent", channel="sms")
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPreAppointment:

    async def test_dispatches_with_exact_stage_language_match(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient(developmental_stage="teen", language="es", preferred_channel="whatsapp")
        appointment = _appointment(id=appointment_id, patient_id=patient.id)
        content_item = _content_item(developmental_stage="teen", language="es")
        delivered = SimpleNamespace(id=uuid4(), status="delivered")

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = content_item
        outreach_service.dispatch.return_value = _outreach_message(status="sent", channel="whatsapp")
        delivery_repo.record_delivery.return_value = delivered

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            result = await service.pre_appointment(MagicMock(), appointment_id)

        # FR-E10.1: content selection keyed off patient's own developmental_stage + language
        stage_call_values = _flat(content_repo.get_by_stage_language.await_args)
        assert "teen" in stage_call_values
        assert "es" in stage_call_values
        assert "pre_appointment" in stage_call_values

        # no fallback needed since exact match found
        content_repo.get_general_fallback.assert_not_awaited()

        # dispatch is called with the documented keyword contract
        dispatch_kwargs = outreach_service.dispatch.await_args.kwargs
        assert dispatch_kwargs["campaign_type"] == "education"
        assert dispatch_kwargs["related_entity_type"] == "appointment"
        assert dispatch_kwargs["related_entity_id"] == appointment_id

        delivery_repo.record_delivery.assert_awaited_once()
        assert result is delivered

    async def test_falls_back_to_general_content_when_no_exact_match(self, caplog):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient(developmental_stage="senior", language="fr")
        appointment = _appointment(id=appointment_id, patient_id=patient.id)
        fallback_item = _content_item(developmental_stage="general", language="fr")

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = None
        content_repo.get_general_fallback.return_value = fallback_item
        outreach_service.dispatch.return_value = _outreach_message(status="sent")
        delivery_repo.record_delivery.return_value = SimpleNamespace(id=uuid4(), status="delivered")

        with caplog.at_level(logging.WARNING):
            with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
                result = await service.pre_appointment(MagicMock(), appointment_id)

        fallback_call_values = _flat(content_repo.get_general_fallback.await_args)
        assert "fr" in fallback_call_values
        assert "pre_appointment" in fallback_call_values

        # T122 AC: the content-library gap is logged (WARNING), not written to the audit log
        assert any(rec.levelno >= logging.WARNING for rec in caplog.records)

        outreach_service.dispatch.assert_awaited_once()
        assert result is not None

    async def test_returns_none_when_no_channel_capable_content_exists(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient()
        appointment = _appointment(id=appointment_id, patient_id=patient.id)

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = None
        content_repo.get_general_fallback.return_value = None

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            result = await service.pre_appointment(MagicMock(), appointment_id)

        outreach_service.dispatch.assert_not_awaited()
        delivery_repo.record_delivery.assert_not_awaited()
        assert result is None

    async def test_returns_none_when_no_consent_on_any_channel(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient()
        appointment = _appointment(id=appointment_id, patient_id=patient.id)
        content_item = _content_item()

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = content_item
        # OutreachService.dispatch's documented no-consent behaviour: a "suppressed" row, no raise
        outreach_service.dispatch.return_value = _outreach_message(status="suppressed")

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            result = await service.pre_appointment(MagicMock(), appointment_id)

        delivery_repo.record_delivery.assert_not_awaited()
        assert result is None

    async def test_never_raises_when_content_and_consent_both_missing(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient()
        appointment = _appointment(id=appointment_id, patient_id=patient.id)

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = None
        content_repo.get_general_fallback.return_value = None

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            # must complete without raising
            result = await service.pre_appointment(MagicMock(), appointment_id)

        assert result is None

    @pytest.mark.parametrize("channel", ["sms", "whatsapp", "email"])
    async def test_record_delivery_status_is_delivered_regardless_of_channel(self, channel):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient(preferred_channel=channel)
        appointment = _appointment(id=appointment_id, patient_id=patient.id)
        content_item = _content_item()

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = content_item
        outreach_service.dispatch.return_value = _outreach_message(status="sent", channel=channel)
        delivery_repo.record_delivery.return_value = SimpleNamespace(id=uuid4(), status="delivered")

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            await service.pre_appointment(MagicMock(), appointment_id)

        # FR-E10.3: record_delivery is always given status="delivered" at creation time, whether
        # the channel is SMS (no open-tracking) or WhatsApp/email (promotable later via webhook)
        call_values = _flat(delivery_repo.record_delivery.await_args)
        assert "delivered" in call_values
        assert "opened" not in call_values
        assert "completed" not in call_values


class TestPostAppointment:

    async def test_dispatches_when_appointment_is_completed(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        patient = _patient()
        appointment = _appointment(id=appointment_id, patient_id=patient.id, status="completed")
        content_item = _content_item()
        delivered = SimpleNamespace(id=uuid4(), status="delivered")

        patient_repo.get_by_id.return_value = patient
        content_repo.get_by_stage_language.return_value = content_item
        outreach_service.dispatch.return_value = _outreach_message(status="sent")
        delivery_repo.record_delivery.return_value = delivered

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            result = await service.post_appointment(MagicMock(), appointment_id)

        # trigger_type differentiates post- from pre-appointment content selection
        stage_call_values = _flat(content_repo.get_by_stage_language.await_args)
        assert "post_appointment" in stage_call_values

        outreach_service.dispatch.assert_awaited_once()
        assert result is delivered

    async def test_returns_none_when_status_is_no_show(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        appointment = _appointment(id=appointment_id, status="no_show")

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            result = await service.post_appointment(MagicMock(), appointment_id)

        # FR-E10.2/US49 Alternate Flow: no-show status suppresses post-appointment guidance entirely
        content_repo.get_by_stage_language.assert_not_awaited()
        outreach_service.dispatch.assert_not_awaited()
        delivery_repo.record_delivery.assert_not_awaited()
        assert result is None

    @pytest.mark.parametrize("status", ["booked", "cancelled", "rescheduled"])
    async def test_returns_none_for_any_non_completed_status(self, status):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        appointment_id = uuid4()
        appointment = _appointment(id=appointment_id, status=status)

        with patch.object(AppointmentRepository, "get_by_id", AsyncMock(return_value=appointment)):
            result = await service.post_appointment(MagicMock(), appointment_id)

        outreach_service.dispatch.assert_not_awaited()
        assert result is None


class TestTrackingWrappers:

    async def test_get_tracking_funnel_delegates_to_repository(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        content_item_id = uuid4()
        funnel = {"delivered": 10, "opened": 4, "completed": 2}
        delivery_repo.get_funnel.return_value = funnel
        db = MagicMock()

        result = await service.get_tracking_funnel(db, content_item_id)

        delivery_repo.get_funnel.assert_awaited_once_with(db, content_item_id)
        assert result == funnel

    async def test_get_tracking_funnel_accepts_none_content_item_id(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        funnel = {"delivered": 100, "opened": 40, "completed": 10}
        delivery_repo.get_funnel.return_value = funnel
        db = MagicMock()

        result = await service.get_tracking_funnel(db, None)

        delivery_repo.get_funnel.assert_awaited_once_with(db, None)
        assert result == funnel

    async def test_list_deliveries_for_patient_delegates_to_repository(self):
        service, content_repo, delivery_repo, patient_repo, outreach_service = _build_service()
        patient_id = uuid4()
        deliveries = [SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4())]
        delivery_repo.list_by_patient.return_value = deliveries
        db = MagicMock()

        result = await service.list_deliveries_for_patient(db, patient_id)

        delivery_repo.list_by_patient.assert_awaited_once_with(db, patient_id)
        assert result == deliveries
