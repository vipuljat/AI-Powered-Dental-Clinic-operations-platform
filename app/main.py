"""The API composition root (architecture.md §1/§5: `boot.target` ==
`app.main:app`).

Builds the `FastAPI()` app object, wires every cross-cutting piece
(`core/*`, `common/*`), constructs every service+repository singleton
exactly once at startup, mounts every module's router(s) under the shared
`/api/v1` prefix, and defines the trivial `/health` liveness route inline
(no DB/broker/storage touched).

Responsibility: this file does not implement any business rule itself --
every rule lives in a `services/<module>/service.py` file this module only
constructs and wires. It depends on every module's `routes.py` (all 11)
purely to call `app.include_router(...)`, and on
`services/console/service.py` only for the one-time `role_permissions`
seed (`RolePermissionRepository.seed_defaults`) -- it does not otherwise
depend on any module's `service.py`/`repository.py` file for typing
purposes, even though it constructs an instance of every one of them below.

Watch out: no stray `main.py`/entrypoint file exists anywhere else in this
tree -- this is the single, sole place `FastAPI()` is constructed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.common.constants import API_PREFIX, PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR
from app.common.exceptions.handlers import register_exception_handlers
from app.common.middleware import InMemoryRateLimiter, register_middleware
from app.core.config import Settings, get_settings
from app.core.database import (
    build_engine,
    build_session_factory,
    configure_session_factory,
    create_all_tables,
)
from app.core.logging import configure_logging, get_logger
from app.core.messaging import MessageBroker, RabbitMQBroker, get_broker
from app.core.storage import get_storage_client

# Every module's models.py is imported here, purely for the side effect of
# registering its ORM tables on the shared `Base.metadata`
# (app/core/database.py), in the same alphabetical order `app/worker.py`
# uses for its own identical import block. This matters beyond mere
# completeness: `app/repositories/waitlist/repository.py`'s module-level
# FK-stand-in loop registers a minimal id-only stand-in table for any of
# "providers"/"chairs"/"appointments"/"patients" not already present on
# `Base.metadata` *at the moment that module is first imported* -- since
# `app.core.dependencies` (imported by nearly every routes.py below)
# transitively imports that waitlist repository, the real `patients`/
# `scheduling` models must already be registered before any routes.py is
# imported, or the later import of their real models.py would collide with
# the stand-in table ("Table 'patients' is already defined for this
# MetaData instance"). Importing every module's real models.py directly,
# first, here -- before any routes.py -- makes the stand-in loop a
# documented no-op regardless of which routes.py happens to be imported
# first below.
#
# Watch out: aliased (`import a.b.c as name`) rather than a bare
# `import a.b.c`, deliberately -- a bare `import app.models.analytics.models`
# would bind the name `app` (the top-level package) in *this* module's own
# namespace, colliding with the module-level `app = create_app()` FastAPI
# instance this file defines below.
import app.models.analytics.models as _analytics_models  # noqa: F401
import app.models.console.models as _console_models  # noqa: F401
import app.models.education.models as _education_models  # noqa: F401
import app.models.engagement.models as _engagement_models  # noqa: F401
import app.models.intelligence.models as _intelligence_models  # noqa: F401
import app.models.outreach.models as _outreach_models  # noqa: F401
import app.models.patients.models as _patients_models  # noqa: F401
import app.models.recall.models as _recall_models  # noqa: F401
import app.models.rules.models as _rules_models  # noqa: F401
import app.models.scheduling.models as _scheduling_models  # noqa: F401
import app.models.waitlist.models as _waitlist_models  # noqa: F401
from app.repositories.analytics.repository import (
    BaselineCostRepository,
    DashboardExportRepository,
    FinancialRoiRepository,
    OperationalMetricsRepository,
)
from app.repositories.console.repository import (
    AuditLogRepository,
    PasswordResetTokenRepository,
    RolePermissionRepository,
    SessionRepository,
    StaffUserRepository,
)
from app.repositories.education.repository import ContentDeliveryRepository, ContentItemRepository
from app.repositories.engagement.repository import (
    CallInteractionRepository,
    CallTranscriptSegmentRepository,
    WebChatMessageRepository,
    WebChatSessionRepository,
)
from app.repositories.intelligence.repository import (
    EscalationRepository,
    MlModelEvaluationRepository,
    RiskScoreRepository,
    TriageSessionRepository,
    UtilisationRecommendationRepository,
)
from app.repositories.outreach.repository import (
    ChannelConfigurationRepository,
    ConsentRepository,
    MessageTemplateRepository,
    OutreachMessageRepository,
)
from app.repositories.patients.repository import (
    PatientImportRepository,
    PatientRepository,
    RecordLockRepository,
)
from app.repositories.recall.repository import (
    RecallComplianceBaselineRepository,
    RecallScheduleRepository,
    UnscheduledTreatmentRepository,
)
from app.repositories.rules.repository import (
    ConfigurationParameterRepository,
    RuleDefinitionRepository,
    RuleSetRepository,
)
from app.repositories.scheduling.repository import (
    AppointmentHistoryRepository,
    AppointmentImportRepository,
    AppointmentRepository,
)
from app.repositories.waitlist.repository import (
    IdleChairAlertRepository,
    WaitlistEntryRepository,
    WaitlistOfferRepository,
)
from app.routes.analytics.routes import router as analytics_router
from app.routes.console.routes import auth_router as console_auth_router
from app.routes.console.routes import console_router
from app.routes.education.routes import router as education_router
from app.routes.engagement.routes import router as engagement_router
from app.routes.intelligence.routes import router as intelligence_router
from app.routes.outreach.routes import router as outreach_router
from app.routes.patients.routes import router as patients_router
from app.routes.recall.routes import router as recall_router
from app.routes.rules.routes import router as rules_router
from app.routes.scheduling.routes import router as scheduling_router
from app.routes.waitlist.routes import router as waitlist_router
from app.services.analytics.service import (
    DashboardExportService,
    FinancialAnalyticsService,
    OperationalAnalyticsService,
)
from app.services.console.service import (
    AuditService,
    AuthService,
    OverrideService,
    PermissionService,
    StaffAdminService,
)
from app.services.education.service import EducationTriggerService
from app.services.engagement.service import (
    CallScoringService,
    EngagementWebChatService,
    VoiceHandlingService,
)
from app.services.intelligence.service import (
    EscalationService,
    RiskScoringService,
    TriageService,
    UtilisationEngine,
)
from app.services.outreach.service import ChannelConfigService, ConsentService, OutreachService
from app.services.patients.service import (
    ExportService,
    PatientImportService,
    PatientService,
    RecordLockService,
)
from app.services.recall.service import (
    RecallCampaignService,
    RecallScanService,
    TreatmentRecoveryService,
)
from app.services.rules.service import ConfigurationService, RuleSetService, RuleValidationService
from app.services.scheduling.service import SchedulingService
from app.services.waitlist.service import (
    IdleChairPollingService,
    WaitlistFillService,
    WaitlistService,
)

logger = get_logger(__name__)


def _build_services(settings: "Settings", broker: "MessageBroker") -> dict[str, Any]:
    """Constructs every repository, then every service, exactly once, in
    dependency order -- `console` first, since almost every other module's
    service depends on `AuditService`/`PermissionService`.

    Returns a plain `dict` keyed by service class name -- the
    `app.state.services` namespace every `Depends(get_<x>_service)`-style
    callable in this tree is meant to read from (project_rules.layout's
    project-wide DI convention).
    """

    services: dict[str, Any] = {}

    # --- console (first: almost everything below depends on AuditService) --
    audit_repo = AuditLogRepository()
    role_permission_repo = RolePermissionRepository()
    staff_repo = StaffUserRepository()
    session_repo = SessionRepository()
    reset_repo = PasswordResetTokenRepository()

    audit_service = AuditService(audit_repo)
    permission_service = PermissionService(role_permission_repo)
    # project_rules.auth / InMemoryRateLimiter's own Interaction Contract
    # (app/common/middleware.py): constructed once here, at composition-root
    # startup, and shared across every request -- never rebuilt per-request.
    password_reset_rate_limiter = InMemoryRateLimiter(
        max_calls=PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR, per_seconds=3600
    )
    auth_service = AuthService(staff_repo, session_repo, reset_repo, audit_service, password_reset_rate_limiter)
    staff_admin_service = StaffAdminService(staff_repo, session_repo, audit_service)

    services["AuditService"] = audit_service
    services["PermissionService"] = permission_service
    services["AuthService"] = auth_service
    services["StaffAdminService"] = staff_admin_service

    # --- patients -------------------------------------------------------------
    patient_repo = PatientRepository()
    patient_import_repo = PatientImportRepository()
    record_lock_repo = RecordLockRepository()

    record_lock_service = RecordLockService(record_lock_repo)
    patient_service = PatientService(patient_repo, record_lock_repo, audit_service)
    patient_import_service = PatientImportService(patient_import_repo, patient_repo, audit_service)

    services["RecordLockService"] = record_lock_service
    services["PatientService"] = patient_service
    services["PatientImportService"] = patient_import_service

    # --- rules ------------------------------------------------------------------
    rule_set_repo = RuleSetRepository()
    rule_def_repo = RuleDefinitionRepository()
    config_param_repo = ConfigurationParameterRepository()

    rule_validation_service = RuleValidationService(rule_def_repo)
    rule_set_service = RuleSetService(rule_set_repo, rule_def_repo, rule_validation_service, audit_service)
    configuration_service = ConfigurationService(config_param_repo, audit_service)

    services["RuleValidationService"] = rule_validation_service
    services["RuleSetService"] = rule_set_service
    services["ConfigurationService"] = configuration_service

    # --- scheduling ---------------------------------------------------------------
    appointment_repo = AppointmentRepository()
    appointment_history_repo = AppointmentHistoryRepository()
    appointment_import_repo = AppointmentImportRepository()

    scheduling_service = SchedulingService(
        appointment_repo,
        appointment_history_repo,
        appointment_import_repo,
        record_lock_service,
        rule_set_service,
        audit_service,
        broker,
    )
    services["SchedulingService"] = scheduling_service

    # `OverrideService` depends on another module's REPOSITORY class only,
    # never that module's SERVICE class (the project-wide layering rule
    # preventing a console<->everything dependency cycle) -- constructed
    # here, once its repository collaborators exist.
    outreach_message_repo = OutreachMessageRepository()
    waitlist_entry_repo = WaitlistEntryRepository()
    triage_session_repo = TriageSessionRepository()
    override_service = OverrideService(
        audit_repo, appointment_repo, outreach_message_repo, waitlist_entry_repo, triage_session_repo
    )
    services["OverrideService"] = override_service

    # --- waitlist -------------------------------------------------------------------
    idle_chair_alert_repo = IdleChairAlertRepository()
    waitlist_offer_repo = WaitlistOfferRepository()

    idle_chair_polling_service = IdleChairPollingService(idle_chair_alert_repo, audit_service)
    waitlist_service = WaitlistService(waitlist_entry_repo, rule_set_service, audit_service)
    waitlist_fill_service = WaitlistFillService(
        waitlist_offer_repo, waitlist_entry_repo, idle_chair_alert_repo, scheduling_service, broker, audit_service
    )

    services["IdleChairPollingService"] = idle_chair_polling_service
    services["WaitlistService"] = waitlist_service
    services["WaitlistFillService"] = waitlist_fill_service

    # --- outreach ------------------------------------------------------------------
    consent_repo = ConsentRepository()
    template_repo = MessageTemplateRepository()
    channel_config_repo = ChannelConfigurationRepository()

    outreach_service = OutreachService(
        message_repo=outreach_message_repo,
        consent_repo=consent_repo,
        template_repo=template_repo,
        channel_config_repo=channel_config_repo,
        config_service=configuration_service,
        audit=audit_service,
        settings=settings,
    )
    consent_service = ConsentService(consent_repo, outreach_message_repo, audit_service)
    channel_config_service = ChannelConfigService(channel_config_repo, audit_service)

    services["OutreachService"] = outreach_service
    services["ConsentService"] = consent_service
    services["ChannelConfigService"] = channel_config_service

    # --- recall --------------------------------------------------------------------
    recall_schedule_repo = RecallScheduleRepository()
    unscheduled_treatment_repo = UnscheduledTreatmentRepository()
    recall_baseline_repo = RecallComplianceBaselineRepository()

    recall_scan_service = RecallScanService(recall_schedule_repo, patient_repo, rule_set_service)
    recall_campaign_service = RecallCampaignService(
        recall_schedule_repo, recall_baseline_repo, outreach_service, audit_service
    )
    treatment_recovery_service = TreatmentRecoveryService(
        unscheduled_treatment_repo, outreach_service, audit_service
    )

    services["RecallScanService"] = recall_scan_service
    services["RecallCampaignService"] = recall_campaign_service
    services["TreatmentRecoveryService"] = treatment_recovery_service

    # --- intelligence ----------------------------------------------------------------
    risk_score_repo = RiskScoreRepository()
    escalation_repo = EscalationRepository()
    utilisation_repo = UtilisationRecommendationRepository()
    ml_eval_repo = MlModelEvaluationRepository()

    risk_scoring_service = RiskScoringService(risk_score_repo, appointment_repo, ml_eval_repo, rule_set_service)
    escalation_service = EscalationService(escalation_repo, audit_service, broker)
    triage_service = TriageService(triage_session_repo, rule_set_service, escalation_service, broker)
    utilisation_engine = UtilisationEngine(
        utilisation_repo, risk_score_repo, triage_session_repo, appointment_repo, scheduling_service, audit_service
    )

    services["RiskScoringService"] = risk_scoring_service
    services["EscalationService"] = escalation_service
    services["TriageService"] = triage_service
    services["UtilisationEngine"] = utilisation_engine

    # --- analytics --------------------------------------------------------------------
    operational_metrics_repo = OperationalMetricsRepository()
    financial_roi_repo = FinancialRoiRepository()
    baseline_cost_repo = BaselineCostRepository()
    dashboard_export_repo = DashboardExportRepository()

    operational_analytics_service = OperationalAnalyticsService(
        operational_metrics_repo, appointment_repo, idle_chair_alert_repo, recall_schedule_repo
    )
    financial_analytics_service = FinancialAnalyticsService(financial_roi_repo, baseline_cost_repo, audit_service)
    # Object-storage client resolved once, here, at startup -- never at
    # import time (project_rules.testing item 3): a real boto3 client is
    # never constructed in test mode, only the in-memory fake.
    storage_client = get_storage_client(settings)
    dashboard_export_service = DashboardExportService(
        dashboard_export_repo, operational_analytics_service, financial_analytics_service, storage_client
    )

    services["OperationalAnalyticsService"] = operational_analytics_service
    services["FinancialAnalyticsService"] = financial_analytics_service
    services["DashboardExportService"] = dashboard_export_service

    # --- education -----------------------------------------------------------------------
    content_item_repo = ContentItemRepository()
    content_delivery_repo = ContentDeliveryRepository()

    education_service = EducationTriggerService(
        content_item_repo, content_delivery_repo, patient_repo, outreach_service
    )
    services["EducationTriggerService"] = education_service

    # --- engagement ------------------------------------------------------------------------
    webchat_session_repo = WebChatSessionRepository()
    webchat_message_repo = WebChatMessageRepository()
    call_interaction_repo = CallInteractionRepository()
    call_transcript_segment_repo = CallTranscriptSegmentRepository()

    webchat_service = EngagementWebChatService(
        webchat_session_repo, webchat_message_repo, patient_repo, triage_service
    )
    voice_handling_service = VoiceHandlingService(call_interaction_repo, escalation_service, settings)
    call_scoring_service = CallScoringService(call_interaction_repo, call_transcript_segment_repo)

    services["EngagementWebChatService"] = webchat_service
    services["VoiceHandlingService"] = voice_handling_service
    services["CallScoringService"] = call_scoring_service

    return services


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler: performs the whole startup sequence exactly
    once, then yields control until shutdown, when the real infra clients
    (if any were opened) are released.

    project_rules.testing: every infrastructure substitution below is wired
    here, at this composition root, never inside a route/service/repository
    body -- each carries its own one-line comment explaining why a real
    client isn't constructed when `settings.environment == "test"`.
    """

    settings = get_settings()

    # Structured JSON logging, configured once per process
    # (project_rules.logging).
    configure_logging(settings)

    # (1) `build_engine(settings)` + `create_all_tables(engine)` -- table
    # creation via `Base.metadata.create_all`; no Alembic migration files
    # exist in this tree. `build_engine` itself performs the
    # `ENVIRONMENT=test` -> `sqlite+aiosqlite:///:memory:` pivot
    # (project_rules.testing item 1) -- a real Postgres connection is never
    # opened in test mode, only here, lazily, at startup.
    engine = build_engine(settings)
    await create_all_tables(engine)
    session_factory = build_session_factory(engine)
    configure_session_factory(session_factory)

    # (2) Broker: `InMemoryBroker` (asyncio.Queue-backed fake, no real
    # network) when `settings.environment == "test"`, else a real
    # `RabbitMQBroker` that must `connect()` before use -- constructed
    # lazily here, never at import time (project_rules.testing item 2).
    broker = get_broker(settings)
    if isinstance(broker, RabbitMQBroker):
        await broker.connect()
    # else: InMemoryBroker requires no connection step -- a pure in-process
    # asyncio.Queue fake with nothing to open, so no-op.

    # (3) Object-storage client: in-memory dict-backed fake in test mode,
    # real boto3 S3-compatible client otherwise (project_rules.testing item
    # 3) -- resolved once inside `_build_services` below (analytics'
    # `DashboardExportService` is the one collaborator that needs it at
    # construction time); never a module-level connection at import time.

    # (4) Construct every repository, then every service, in dependency
    # order, storing the resulting singletons on `app.state.services`.
    app.state.services = _build_services(settings, broker)
    app.state.broker = broker
    app.state.engine = engine
    app.state.session_factory = session_factory

    # (5) Seed the data-driven `role_permissions` RBAC matrix, once,
    # idempotently, so a fresh database boots with a working permission
    # matrix (services/console/service.py's Watch out).
    role_permission_repo = RolePermissionRepository()
    async with session_factory() as db:
        try:
            await role_permission_repo.seed_defaults(db)
        except Exception:
            await db.rollback()
            raise
        else:
            await db.commit()

    logger.info("api_startup_complete", extra={"environment": settings.environment})

    try:
        yield
    finally:
        if isinstance(broker, RabbitMQBroker):
            await broker.close()
        await engine.dispose()


def create_app() -> FastAPI:
    """Builds and returns a fresh `FastAPI()` instance.

    project_rules.other: `title`/`version`/`description` are set explicitly
    from `settings.api_title`/`api_version`/`api_description`
    (architecture.md §1 + TDD "Version 1.0" header) -- never left at the
    framework default.

    Watch out: no client (DB engine, broker, storage) is constructed here --
    everything infra-related happens lazily inside `_lifespan`, at server
    startup, so importing this module (e.g. to mount a single router on a
    throwaway app for another file's own `verify` command) never requires
    any infrastructure to be present.
    """

    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=_lifespan,
    )

    register_exception_handlers(app)
    register_middleware(app)

    # `boot.api_prefix`: the shared "/api/v1" prefix (app.common.constants
    # .API_PREFIX) is added ONCE here, on top of each router's own
    # sub-prefix ("/auth", "/console", "/patients", "/rules", "/scheduling",
    # "/waitlist", "/outreach", "/recall", "/intelligence", "/analytics",
    # "/education", "/engagement") -- no router anywhere in this tree adds
    # "/api/v1" itself.
    app.include_router(console_auth_router, prefix=API_PREFIX)
    app.include_router(console_router, prefix=API_PREFIX)
    app.include_router(patients_router, prefix=API_PREFIX)
    app.include_router(rules_router, prefix=API_PREFIX)
    app.include_router(scheduling_router, prefix=API_PREFIX)
    app.include_router(waitlist_router, prefix=API_PREFIX)
    app.include_router(outreach_router, prefix=API_PREFIX)
    app.include_router(recall_router, prefix=API_PREFIX)
    app.include_router(intelligence_router, prefix=API_PREFIX)
    app.include_router(analytics_router, prefix=API_PREFIX)
    app.include_router(education_router, prefix=API_PREFIX)
    app.include_router(engagement_router, prefix=API_PREFIX)

    # `boot.health_path`: no prefix, answers 200 with no DB/broker/storage
    # access -- must succeed with none of those running.
    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    return app


# The module-level app instance uvicorn serves -- `create_app()` called once
# at import time, matching the target import path `app.main:app`
# (`boot.target`).
app = create_app()
