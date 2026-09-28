"""Unit tests for app.common.constants — the hard-coded fallback defaults.

Every value here is asserted verbatim against the spec so that any accidental
drift (someone "tuning" a number without updating the spec) is caught.
"""

from app.common import constants


def test_auth_token_ttls():
    assert constants.ACCESS_TOKEN_TTL_MINUTES == 15
    assert constants.REFRESH_TOKEN_TTL_DAYS == 7


def test_login_lockout_defaults():
    assert constants.LOGIN_LOCKOUT_THRESHOLD == 5
    assert constants.LOGIN_LOCKOUT_MINUTES == 15


def test_record_lock_timeout_minutes():
    assert constants.RECORD_LOCK_TIMEOUT_MINUTES == 10


def test_idle_chair_poll_interval_minutes():
    assert constants.IDLE_CHAIR_POLL_INTERVAL_MINUTES == 2


def test_waitlist_response_window_minutes():
    assert constants.WAITLIST_RESPONSE_WINDOW_MINUTES == 15


def test_outreach_retry_defaults():
    assert constants.OUTREACH_MAX_RETRY_ATTEMPTS == 3
    assert constants.OUTREACH_RETRY_DELAY_MINUTES == 60


def test_late_cancellation_window_hours():
    assert constants.LATE_CANCELLATION_WINDOW_HOURS == 24


def test_recall_defaults():
    assert constants.DEFAULT_RECALL_INTERVAL_MONTHS == 6
    assert constants.RECALL_DORMANT_THRESHOLD_MONTHS == 18


def test_ml_defaults():
    assert constants.ML_MIN_HISTORIC_DATA_YEARS == 2
    assert constants.ML_TARGET_PRECISION == 0.75
    assert constants.ML_LEAD_TIME_HOURS == 48


def test_utilisation_adoption_target():
    assert constants.UTILISATION_ADOPTION_TARGET == 0.50


def test_password_reset_defaults():
    assert constants.PASSWORD_RESET_TOKEN_TTL_HOURS == 1
    assert constants.PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR == 5


def test_audit_write_max_retries():
    assert constants.AUDIT_WRITE_MAX_RETRIES == 3


def test_csv_import_limits():
    assert constants.CSV_IMPORT_MAX_FILE_SIZE_MB == 10
    assert constants.CSV_IMPORT_MAX_ROWS == 5000


def test_password_min_length():
    assert constants.PASSWORD_MIN_LENGTH == 8


def test_education_pre_appointment_window_hours():
    assert constants.EDUCATION_PRE_APPOINTMENT_WINDOW_HOURS == 24


def test_appointment_completion_grace_minutes():
    assert constants.APPOINTMENT_COMPLETION_GRACE_MINUTES == 30


def test_default_page_size():
    assert constants.DEFAULT_PAGE_SIZE == 50


def test_api_prefix():
    assert constants.API_PREFIX == "/api/v1"


def test_constant_types_are_correct():
    # int-typed constants
    assert isinstance(constants.ACCESS_TOKEN_TTL_MINUTES, int)
    assert isinstance(constants.OUTREACH_MAX_RETRY_ATTEMPTS, int)
    assert isinstance(constants.DEFAULT_PAGE_SIZE, int)
    # float-typed constants
    assert isinstance(constants.ML_TARGET_PRECISION, float)
    assert isinstance(constants.UTILISATION_ADOPTION_TARGET, float)
    # str-typed constants
    assert isinstance(constants.API_PREFIX, str)
