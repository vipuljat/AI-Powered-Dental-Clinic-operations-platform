"""Unit tests for app/core/config.py.

`Settings` is the single Pydantic Settings object every other file in this
tree reads environment-driven configuration from, and `get_settings()` is the
`@lru_cache()`-wrapped singleton accessor that keeps `.env`/environment
parsing to at most once per process (per this file's own spec).

Every test here constructs `Settings` with `_env_file=None` -- an ordinary
pydantic-settings `BaseSettings.__init__` keyword, not anything internal to
the module under test -- so a real `.env` file that may or may not exist on
disk wherever the suite happens to run from can never leak into an assertion
about *default* values. Tests that care about environment-variable behaviour
instead use `monkeypatch.setenv`/`monkeypatch.delenv` explicitly and clear
`get_settings`'s cache before and after each test so the singleton never
leaks state between tests (mirroring `get_settings()`'s own documented
per-process caching behaviour).
"""
from __future__ import annotations

import pytest

from app.core.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Environment variables this module is documented to read. Cleared before
# every test so no ambient value (e.g. a real ENVIRONMENT=test exported by
# a wrapping CI/test-runner) can bleed into a default-value assertion.
# ---------------------------------------------------------------------------
_ENV_VARS = [
    "ENVIRONMENT",
    "DATABASE_URL",
    "RABBITMQ_URL",
    "OBJECT_STORAGE_ENDPOINT",
    "OBJECT_STORAGE_BUCKET",
    "OBJECT_STORAGE_ACCESS_KEY",
    "OBJECT_STORAGE_SECRET_KEY",
    "JWT_SECRET_KEY",
    "JWT_ALGORITHM",
    "ACCESS_TOKEN_TTL_MINUTES",
    "REFRESH_TOKEN_TTL_DAYS",
    "CORS_ALLOW_ORIGINS",
    "API_TITLE",
    "API_VERSION",
    "API_DESCRIPTION",
]


@pytest.fixture(autouse=True)
def _clean_env_and_cache(monkeypatch):
    """Strip every settings-relevant env var and clear the `get_settings()`
    cache both before and after each test, so tests are independent of the
    real process environment and of each other."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _settings(**overrides) -> Settings:
    """Construct a Settings instance with .env-file loading disabled so
    defaults are deterministic regardless of the real filesystem."""
    return Settings(_env_file=None, **overrides)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_environment_is_development():
    assert _settings().environment == "development"


def test_default_database_url_is_postgres_asyncpg():
    assert _settings().database_url.startswith("postgresql+asyncpg://")


def test_default_rabbitmq_url():
    assert _settings().rabbitmq_url == "amqp://guest:guest@localhost:5672/"


def test_default_object_storage_endpoint_is_none():
    assert _settings().object_storage_endpoint is None


def test_default_object_storage_bucket():
    assert _settings().object_storage_bucket == "dental-platform"


def test_default_object_storage_access_key_is_none():
    assert _settings().object_storage_access_key is None


def test_default_object_storage_secret_key_is_none():
    assert _settings().object_storage_secret_key is None


def test_default_jwt_secret_key():
    assert _settings().jwt_secret_key == "dev-secret-change-me"


def test_default_jwt_algorithm():
    assert _settings().jwt_algorithm == "HS256"


def test_default_cors_allow_origins():
    assert _settings().cors_allow_origins == ["*"]


# ---------------------------------------------------------------------------
# project_rules.other: FastAPI() app object title/version/description must
# be set explicitly from architecture.md §1 / the TDD's "Version 1.0" header
# -- these are the exact literal values app/main.py is expected to pass into
# FastAPI(...).
# ---------------------------------------------------------------------------

def test_default_api_title_matches_platform_name():
    assert (
        _settings().api_title
        == "AI-Powered Dental Clinic Operations Platform"
    )


def test_default_api_version_is_v1_0():
    assert _settings().api_version == "v1.0"


def test_default_api_description_matches_spec_text():
    assert _settings().api_description == (
        "Unified staff console API for scheduling, patient management, "
        "multilingual outreach, and AI-assisted operational intelligence "
        "(Phase 1, single-clinic)."
    )


# ---------------------------------------------------------------------------
# open_questions[Q-d84091fb]-derived TTL defaults, sourced from
# app.common.constants, but env-overridable.
# ---------------------------------------------------------------------------

def test_default_access_token_ttl_minutes_is_15():
    assert _settings().access_token_ttl_minutes == 15


def test_default_refresh_token_ttl_days_is_7():
    assert _settings().refresh_token_ttl_days == 7


def test_access_token_ttl_minutes_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN_TTL_MINUTES", "30")
    settings = _settings()
    assert settings.access_token_ttl_minutes == 30
    assert isinstance(settings.access_token_ttl_minutes, int)


def test_refresh_token_ttl_days_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("REFRESH_TOKEN_TTL_DAYS", "14")
    settings = _settings()
    assert settings.refresh_token_ttl_days == 14
    assert isinstance(settings.refresh_token_ttl_days, int)


# ---------------------------------------------------------------------------
# project_rules.testing: ENVIRONMENT env var drives Settings.environment --
# this file only exposes the flag, so the observable contract is limited to
# "the field reflects the env var", not any branching behaviour (that lives
# in app/core/database.py per the Interaction Contract).
# ---------------------------------------------------------------------------

def test_environment_is_read_from_env_var(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    assert _settings().environment == "test"


def test_environment_production_value_is_read_from_env_var(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert _settings().environment == "production"


def test_settings_does_not_itself_substitute_sqlite_url_for_test_environment(
    monkeypatch,
):
    """Per the Interaction Contract, Settings only exposes the flag -- the
    sqlite substitution is app/core/database.py's job, not Settings'. Setting
    ENVIRONMENT=test must not, by itself, change database_url."""
    monkeypatch.setenv("ENVIRONMENT", "test")
    settings = _settings()
    assert settings.environment == "test"
    assert settings.database_url.startswith("postgresql+asyncpg://")


# ---------------------------------------------------------------------------
# Env-var overrides for the remaining infrastructure fields.
# ---------------------------------------------------------------------------

def test_database_url_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@host/db")
    assert _settings().database_url == "postgresql+asyncpg://u:p@host/db"


def test_rabbitmq_url_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("RABBITMQ_URL", "amqp://user:pass@rabbit:5672/")
    assert _settings().rabbitmq_url == "amqp://user:pass@rabbit:5672/"


def test_object_storage_credentials_are_overridable_via_env(monkeypatch):
    monkeypatch.setenv("OBJECT_STORAGE_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("OBJECT_STORAGE_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("OBJECT_STORAGE_SECRET_KEY", "minioadminsecret")
    monkeypatch.setenv("OBJECT_STORAGE_BUCKET", "custom-bucket")
    settings = _settings()
    assert settings.object_storage_endpoint == "http://minio:9000"
    assert settings.object_storage_access_key == "minioadmin"
    assert settings.object_storage_secret_key == "minioadminsecret"
    assert settings.object_storage_bucket == "custom-bucket"


def test_jwt_secret_key_is_overridable_via_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "super-secret-value")
    assert _settings().jwt_secret_key == "super-secret-value"


# ---------------------------------------------------------------------------
# get_settings(): lru_cache-backed singleton, so `.env`/environment parsing
# happens at most once per process.
# ---------------------------------------------------------------------------

def test_get_settings_returns_a_settings_instance():
    assert isinstance(get_settings(), Settings)


def test_get_settings_is_cached_singleton():
    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_does_not_reread_env_after_first_call(monkeypatch):
    """Once cached, get_settings() must not reflect a later env var change --
    this is the "does not read `.env` more than once per process" contract.
    """
    monkeypatch.setenv("JWT_SECRET_KEY", "first-value")
    first = get_settings()
    assert first.jwt_secret_key == "first-value"

    monkeypatch.setenv("JWT_SECRET_KEY", "second-value")
    second = get_settings()

    assert second is first
    assert second.jwt_secret_key == "first-value"


def test_get_settings_cache_clear_allows_a_fresh_read(monkeypatch):
    """Sanity check on the caching mechanism itself: explicitly clearing the
    cache (as this suite's own fixture does between tests) does allow a new
    read, proving the earlier no-reread assertion is due to caching and not
    e.g. environment variables being ignored altogether."""
    monkeypatch.setenv("JWT_SECRET_KEY", "value-a")
    first = get_settings()
    assert first.jwt_secret_key == "value-a"

    get_settings.cache_clear()
    monkeypatch.setenv("JWT_SECRET_KEY", "value-b")
    second = get_settings()

    assert second.jwt_secret_key == "value-b"
    assert second is not first
