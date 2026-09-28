"""The single Pydantic Settings object every other file in this tree reads
environment-driven configuration from — database URL, RabbitMQ URL,
object-storage credentials/endpoint, JWT secret/algorithm, CORS origins, and
the `environment` flag that project_rules.testing pivots every infrastructure
substitution on.

This file only *exposes* `environment`; it never branches on it. Each
composition root (app/main.py, app/worker.py) is the one place that inspects
`settings.environment == "test"` and swaps in the in-memory fakes described in
project_rules.testing — never here, and never inside a route/service/
repository body.

`get_settings()` is `lru_cache`-wrapped so `.env` is parsed at most once per
process, giving every caller the same singleton instance.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.common.constants import ACCESS_TOKEN_TTL_MINUTES, REFRESH_TOKEN_TTL_DAYS


class Settings(BaseSettings):
    """Environment-driven configuration for the whole backend.

    Field defaults are the conventional local-dev fallbacks; every field is
    overridable via an environment variable of the same (upper-cased) name,
    or via a `.env` file in the project root.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Environment flag (project_rules.testing) ---------------------------
    # "development" | "test" | "production" — read from the ENVIRONMENT env
    # var. This is the one flag every composition root reads to decide
    # whether to construct real infrastructure clients or in-memory fakes.
    environment: str = "development"

    # --- Database -------------------------------------------------------------
    # Overridden to sqlite+aiosqlite:///:memory: by app/core/database.py's
    # engine-construction function when environment == "test" — this file
    # only carries the configured value, it does not itself substitute it.
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/dental_platform"

    # --- Messaging --------------------------------------------------------------
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    # --- Object storage -----------------------------------------------------------
    object_storage_endpoint: str | None = None
    object_storage_bucket: str = "dental-platform"
    object_storage_access_key: str | None = None
    object_storage_secret_key: str | None = None

    # --- Auth / JWT (architecture.md §11 item 5, open_questions[Q-d84091fb]) ------
    jwt_secret_key: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = ACCESS_TOKEN_TTL_MINUTES
    refresh_token_ttl_days: int = REFRESH_TOKEN_TTL_DAYS

    # --- CORS -----------------------------------------------------------------------
    cors_allow_origins: list[str] = ["*"]

    # --- API metadata (architecture.md §1 + TDD "Version 1.0" header) -----------------
    # These are the exact values app/main.py must pass into FastAPI(...) per
    # project_rules.other: "title/version/description must be set explicitly".
    api_title: str = "AI-Powered Dental Clinic Operations Platform"
    api_version: str = "v1.0"
    api_description: str = (
        "Unified staff console API for scheduling, patient management, "
        "multilingual outreach, and AI-assisted operational intelligence "
        "(Phase 1, single-clinic)."
    )


@lru_cache()
def get_settings() -> Settings:
    """Return the process-wide singleton `Settings` instance.

    `lru_cache` ensures `.env` (and the environment) is read at most once per
    process; every caller across the tree receives the same instance.
    """

    return Settings()
