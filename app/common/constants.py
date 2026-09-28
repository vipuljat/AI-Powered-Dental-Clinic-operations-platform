"""Hard-coded fallback defaults for every numeric/business value the BRD and TDD
left unquantified (architecture.md §11 assumptions table / open_questions[Q-d84091fb]
and the other numeric open questions).

These are plain, immutable module-level constants — every module reads the same
number instead of each service re-guessing its own default. Where a
`configuration_parameters` row (rules module, FR-E3.4) can override a value at
runtime, that is called out in the comment next to the constant. A file that
needs a *live-overridable* value must call
``ConfigurationService.get_live(name, default=<this constant>)`` rather than
reading the constant directly — see app/services/rules/service.py and
app/services/outreach/service.py's Interaction Contract sections for the exact
read order (live configuration row wins; this constant is only the fallback).

Contains no logic.
"""

# --- Auth / session (architecture.md §11 item 5's own inferred default) ----
ACCESS_TOKEN_TTL_MINUTES: int = 15
REFRESH_TOKEN_TTL_DAYS: int = 7
LOGIN_LOCKOUT_THRESHOLD: int = 5  # failed attempts before lockout
LOGIN_LOCKOUT_MINUTES: int = 15
RECORD_LOCK_TIMEOUT_MINUTES: int = 10

# --- Scheduling / waitlist ---------------------------------------------------
IDLE_CHAIR_POLL_INTERVAL_MINUTES: int = 2
WAITLIST_RESPONSE_WINDOW_MINUTES: int = 15
LATE_CANCELLATION_WINDOW_HOURS: int = 24

# --- Outreach -----------------------------------------------------------------
# overridable via configuration_parameters["outreach_max_retry_attempts"]
OUTREACH_MAX_RETRY_ATTEMPTS: int = 3
# overridable via configuration_parameters["outreach_retry_delay_minutes"]
OUTREACH_RETRY_DELAY_MINUTES: int = 60

# --- Recall -------------------------------------------------------------------
DEFAULT_RECALL_INTERVAL_MONTHS: int = 6
RECALL_DORMANT_THRESHOLD_MONTHS: int = 18

# --- ML / intelligence ---------------------------------------------------------
ML_MIN_HISTORIC_DATA_YEARS: int = 2
ML_TARGET_PRECISION: float = 0.75
ML_LEAD_TIME_HOURS: int = 48

# --- Analytics ------------------------------------------------------------------
UTILISATION_ADOPTION_TARGET: float = 0.50

# --- Auth / password reset --------------------------------------------------------
PASSWORD_RESET_TOKEN_TTL_HOURS: int = 1
# per-email rate limit backing the 429 on POST /auth/password-reset/request
PASSWORD_RESET_REQUEST_RATE_LIMIT_PER_HOUR: int = 5

# --- Console / audit -----------------------------------------------------------------
AUDIT_WRITE_MAX_RETRIES: int = 3

# --- Patients / CSV import -------------------------------------------------------------
CSV_IMPORT_MAX_FILE_SIZE_MB: int = 10
CSV_IMPORT_MAX_ROWS: int = 5000

# --- Auth / password policy --------------------------------------------------------------
PASSWORD_MIN_LENGTH: int = 8

# --- Education -----------------------------------------------------------------------------
EDUCATION_PRE_APPOINTMENT_WINDOW_HOURS: int = 24

# --- Scheduling completion sweep --------------------------------------------------------------
# how long after scheduled_end a still-booked appointment is swept to completed
APPOINTMENT_COMPLETION_GRACE_MINUTES: int = 30

# --- API / pagination ---------------------------------------------------------------------------
DEFAULT_PAGE_SIZE: int = 50
API_PREFIX: str = "/api/v1"
