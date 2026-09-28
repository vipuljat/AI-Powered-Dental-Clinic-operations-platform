"""Data-access classes over ``content_items``/``content_deliveries`` (architecture.md §4.2).

Interaction contract: every method here is called only from
``services/education/service.py``'s ``EducationTriggerService`` -- including
its two read-only wrapper methods (``get_tracking_funnel`` /
``list_deliveries_for_patient``) that back ``routes/education/routes.py``'s
``GET`` endpoints, keeping the "routes -> services -> repositories" layering
uniform even though those two calls perform no business logic beyond the
repository query itself.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import ContentDeliveryStatus
from app.common.utils import now_utc
from app.models.education.models import ContentDelivery, ContentItem


class ContentItemRepository:
    """Read access over ``content_items``."""

    async def get_by_stage_language(
        self,
        db: AsyncSession,
        developmental_stage: str,
        language: str,
        trigger_type: str,
    ) -> ContentItem | None:
        result = await db.execute(
            select(ContentItem).where(
                ContentItem.developmental_stage == developmental_stage,
                ContentItem.language == language,
                ContentItem.trigger_type == trigger_type,
            )
        )
        return result.scalars().first()

    async def get_general_fallback(
        self, db: AsyncSession, language: str, trigger_type: str
    ) -> ContentItem | None:
        """FR-E10.1: fallback lookup for ``developmental_stage="general"``
        content, used by ``EducationTriggerService`` when
        ``get_by_stage_language`` finds no exact stage-specific match --
        "missing age-appropriate content falls back to general instructions"
        (T122 AC).
        """
        result = await db.execute(
            select(ContentItem).where(
                ContentItem.developmental_stage == "general",
                ContentItem.language == language,
                ContentItem.trigger_type == trigger_type,
            )
        )
        return result.scalars().first()


class ContentDeliveryRepository:
    """Read/write access over ``content_deliveries``."""

    async def record_delivery(
        self,
        db: AsyncSession,
        patient_id: UUID,
        appointment_id: UUID | None,
        content_item_id: UUID,
        channel: str,
        status: str,
    ) -> ContentDelivery:
        delivery = ContentDelivery(
            patient_id=patient_id,
            appointment_id=appointment_id,
            content_item_id=content_item_id,
            channel=channel,
            status=status,
            delivered_at=now_utc(),
        )
        db.add(delivery)
        await db.commit()
        await db.refresh(delivery)
        return delivery

    async def record_open(self, db: AsyncSession, id: UUID) -> ContentDelivery:
        """Sets ``status="opened"``, ``opened_at=now_utc()``; a no-op on the
        already-recorded open timestamp if ``status`` was already ``opened``
        -- it does not raise on a repeated call.
        """
        result = await db.execute(select(ContentDelivery).where(ContentDelivery.id == id))
        delivery = result.scalar_one()
        if delivery.status != ContentDeliveryStatus.opened:
            delivery.status = ContentDeliveryStatus.opened
            delivery.opened_at = now_utc()
            await db.commit()
            await db.refresh(delivery)
        return delivery

    async def record_completed(self, db: AsyncSession, id: UUID) -> ContentDelivery:
        result = await db.execute(select(ContentDelivery).where(ContentDelivery.id == id))
        delivery = result.scalar_one()
        delivery.status = ContentDeliveryStatus.completed
        delivery.completed_at = now_utc()
        await db.commit()
        await db.refresh(delivery)
        return delivery

    async def get_funnel(self, db: AsyncSession, content_item_id: UUID | None) -> dict:
        """Returns ``{"delivered": n, "opened": n, "completed": n}`` counts,
        optionally scoped to one ``content_item_id``.

        ``delivered`` counts every row (a delivery that has since moved on to
        ``opened``/``completed`` was still delivered), while ``opened`` and
        ``completed`` count rows that reached at least that stage.
        """
        base = select(func.count()).select_from(ContentDelivery)
        if content_item_id is not None:
            base = base.where(ContentDelivery.content_item_id == content_item_id)

        delivered_result = await db.execute(base)
        delivered = delivered_result.scalar_one()

        opened_query = base.where(
            ContentDelivery.status.in_(
                [ContentDeliveryStatus.opened, ContentDeliveryStatus.completed]
            )
        )
        opened_result = await db.execute(opened_query)
        opened = opened_result.scalar_one()

        completed_query = base.where(ContentDelivery.status == ContentDeliveryStatus.completed)
        completed_result = await db.execute(completed_query)
        completed = completed_result.scalar_one()

        return {"delivered": delivered, "opened": opened, "completed": completed}

    async def list_by_patient(self, db: AsyncSession, patient_id: UUID) -> list[ContentDelivery]:
        result = await db.execute(
            select(ContentDelivery).where(ContentDelivery.patient_id == patient_id)
        )
        return list(result.scalars().all())
