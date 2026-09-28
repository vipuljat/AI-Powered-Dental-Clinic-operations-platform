"""Password hashing/verification and JWT issuance/decoding.

These are the two primitives ``AuthService`` (console module) and
``core/dependencies.py`` build on. Contains no DB access, no knowledge of
``staff_users``/``sessions`` tables, and no role/permission logic (that is
``PermissionService``, services/console).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID

import bcrypt
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel

from app.common.exceptions.errors import UnauthorizedError
from app.core.config import get_settings

if TYPE_CHECKING:
    pass

# Password hashing: project_rules names "passlib/bcrypt", but passlib 1.7.4's
# bcrypt backend-detection reads `bcrypt.__about__.__version__`, an attribute
# removed by bcrypt>=4.1 — with the bcrypt version resolved into this
# environment, passlib's CryptContext crashes on the very first hash/verify
# call. `bcrypt` itself (the underlying hashing library passlib would have
# delegated to) works correctly, so this file calls it directly rather than
# through the broken passlib wrapper. This is the one place a plaintext
# password is transformed before persistence (FR-E1 NFR: never displayed or
# logged in plaintext); nothing in this file logs the plaintext or the hash.
_BCRYPT_ROUNDS = 12

_TOKEN_TYPE_ACCESS = "access"
_TOKEN_TYPE_REFRESH = "refresh"


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password with bcrypt."""

    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt(rounds=_BCRYPT_ROUNDS))
    return hashed.decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""

    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed/foreign hash format — never a valid match.
        return False


class TokenPayload(BaseModel):
    """The decoded/validated claim set of an access or refresh JWT."""

    sub: str  # staff_users.id, as str(UUID)
    role: str  # Role value
    jti: str  # sessions.jti
    type: str  # "access" | "refresh"
    exp: int  # unix timestamp


def _encode_token(user_id: UUID, role: str, jti: str, token_type: str, ttl: timedelta) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        # FR-E1.1 / architecture.md US1 AC2: the resolved role is embedded
        # directly in the token claims so every downstream permission check
        # reads the role from the validated token, not a second DB lookup.
        "role": role,
        "jti": jti,
        "type": token_type,
        "exp": int((now + ttl).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(user_id: UUID, role: str, jti: str) -> str:
    """Issue an access token.

    architecture.md §5.2 Login response: ``expires_in: 900`` (15 minutes) —
    ``exp = now_utc() + timedelta(minutes=settings.access_token_ttl_minutes)``,
    default 900s from ``app.common.constants.ACCESS_TOKEN_TTL_MINUTES``.
    """

    settings = get_settings()
    ttl = timedelta(minutes=settings.access_token_ttl_minutes)
    return _encode_token(user_id, role, jti, _TOKEN_TYPE_ACCESS, ttl)


def create_refresh_token(user_id: UUID, role: str, jti: str) -> str:
    """Issue a refresh token (7-day default TTL, per project_rules.auth)."""

    settings = get_settings()
    ttl = timedelta(days=settings.refresh_token_ttl_days)
    return _encode_token(user_id, role, jti, _TOKEN_TYPE_REFRESH, ttl)


def decode_token(token: str) -> TokenPayload:
    """Decode and validate an access/refresh token's signature and expiry.

    Raises ``UnauthorizedError`` (not a bare exception) on an expired,
    malformed, or invalid-signature token — this is what lets
    ``core/dependencies.py``'s ``get_current_user`` propagate a clean 401
    through the global handler without its own try/except duplicating the
    mapping.

    Watch out: this validates signature and expiry only; it does **not**
    check revocation. A token that decodes successfully but whose ``jti``
    maps to a revoked session (force-logout, US6) must still be rejected —
    that check is always the caller's (``core/dependencies.py``)
    responsibility, since this file has no DB access.
    """

    settings = get_settings()
    try:
        raw_payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except ExpiredSignatureError as exc:
        raise UnauthorizedError("This session has expired.") from exc
    except JWTError as exc:
        raise UnauthorizedError("Invalid authentication token.") from exc

    try:
        return TokenPayload.model_validate(raw_payload)
    except Exception as exc:  # pydantic ValidationError on a malformed payload
        raise UnauthorizedError("Invalid authentication token.") from exc


def new_jti() -> str:
    """Generate a new session identifier (``sessions.jti``), a uuid4 hex string."""

    return uuid.uuid4().hex
