"""SQLAlchemy ORM declarations for the education module (architecture.md §4.2).

Owns exactly two tables: ``content_items``, ``content_deliveries``.

Interaction contract: ``patient_id``/``appointment_id``/``content_item_id`` are
plain FK *columns* — real ``ForeignKey()`` targets (``patients.id`` /
``appointments.id`` / ``content_items.id``) but no cross-module ORM
``relationship()`` is declared on either end, so this module stays free of an
import-time coupling to the patients/scheduling ORM classes. ``repositories/
education/repository.py`` is the only file that imports these classes for
querying; ``services/education/service.py`` reads/writes exclusively through
that repository, never these ORM classes directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Enum, ForeignKey, String, Text, Uuid, types
from sqlalchemy.orm import Mapped, mapped_column

from app.common.enums import Channel, ContentDeliveryStatus, EducationTriggerType, Language
from app.core.database import Base


class _TimestampTZ(types.TypeDecorator):
    """``TIMESTAMPTZ`` in Postgres; a UTC-normalized naive ``DATETIME`` on
    SQLite that reattaches UTC ``tzinfo`` on load.

    Postgres's native ``TIMESTAMPTZ`` (via ``DateTime(timezone=True)``)
    round-trips ``tzinfo`` unmodified. SQLite's generic ``DATETIME`` storage
    format (project_rules.testing substitutes SQLite for every test run) has
    no timezone component at all, so a tz-aware value would otherwise store
    and load back *naive*. This local decorator normalizes any bound value to
    UTC and stores it naive on non-Postgres dialects, then reattaches UTC
    ``tzinfo`` on load, so ``delivered_at``/``opened_at``/``completed_at``
    still round-trip as ISO-8601 UTC-offset-aware ``datetime`` values end to
    end on both dialects, per project_rules.also's "All timestamps are
    timezone-aware UTC" rule.
    """

    impl = types.DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(types.DateTime(timezone=(dialect.name == "postgresql")))

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None or dialect.name == "postgresql":
            return value
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is not None and dialect.name != "postgresql" and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class ContentItem(Base):
    """One piece of patient-education content, keyed by the developmental
    stage/language/trigger combination it applies to."""

    __tablename__ = "content_items"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    developmental_stage: Mapped[str] = mapped_column(String(50), nullable=False)
    language: Mapped[Language] = mapped_column(
        Enum(Language, native_enum=False, length=10), nullable=False
    )
    trigger_type: Mapped[EducationTriggerType] = mapped_column(
        Enum(EducationTriggerType, native_enum=False, length=20), nullable=False
    )
    body: Mapped[str] = mapped_column(Text, nullable=False)


class ContentDelivery(Base):
    """One dispatch of a ``ContentItem`` to a patient over a specific channel.

    FR-E10.3: ``status`` is only ever written as ``delivered`` or
    ``unavailable`` for a channel that cannot report opens (e.g. SMS) — never
    fabricated as ``opened``. ``opened``/``completed`` are only ever set by a
    channel capable of reporting that signal (e.g. a WhatsApp read receipt or
    a webchat completion event), and ``unavailable`` is the schema-level
    representation of "this channel cannot report opens", not an error state.
    """

    __tablename__ = "content_deliveries"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Real FK to patients.id (patients module) -- no cross-module ORM
    # relationship() declared, per the Interaction contract.
    patient_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("patients.id"), nullable=False
    )
    # Real FK to appointments.id (scheduling module) -- no cross-module ORM
    # relationship() declared, per the Interaction contract.
    appointment_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("appointments.id"), nullable=True
    )
    content_item_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("content_items.id"), nullable=False
    )
    channel: Mapped[Channel] = mapped_column(
        Enum(Channel, native_enum=False, length=20), nullable=False
    )
    status: Mapped[ContentDeliveryStatus] = mapped_column(
        Enum(ContentDeliveryStatus, native_enum=False, length=20), nullable=False
    )
    delivered_at: Mapped[datetime | None] = mapped_column(_TimestampTZ(), nullable=True)
    opened_at: Mapped[datetime | None] = mapped_column(_TimestampTZ(), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(_TimestampTZ(), nullable=True)
