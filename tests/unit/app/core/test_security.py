"""Unit tests for app/core/security.py.

Covers password hashing/verification, JWT issuance embedding `role` directly
in the token claims (US1 AC2), the documented 15-minute access-token /
7-day refresh-token lifetimes, and `decode_token` mapping every
expired/malformed/invalid-signature token to `UnauthorizedError` (never a
bare exception) so `core/dependencies.py` can propagate a clean 401.

For the expired/invalid-signature cases we hand-craft JWTs with
`python-jose` (already a project runtime dependency) directly, rather than
waiting on a real clock or reaching into this module's internals. The
expired-token case signs with the `jwt_secret_key`/`jwt_algorithm` *class
defaults* documented in specs/app/core/config.py.py.md
(`"dev-secret-change-me"` / `"HS256"`) since `Settings` reads no `.env` file
and no `JWT_SECRET_KEY`/`JWT_ALGORITHM` environment variables in this test
process, so `decode_token` resolves to those same defaults.
"""
from __future__ import annotations

import time
from uuid import uuid4

import pytest
from jose import jwt as jose_jwt

from app.common.exceptions.errors import UnauthorizedError
from app.core.security import (
    TokenPayload,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    new_jti,
    verify_password,
)

_DEFAULT_JWT_SECRET = "dev-secret-change-me"  # config.py.py.md Settings default
_DEFAULT_JWT_ALGORITHM = "HS256"              # config.py.py.md Settings default


# ---------------------------------------------------------------------------
# hash_password / verify_password
# ---------------------------------------------------------------------------

def test_hash_password_does_not_return_the_plaintext():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert "correct horse battery staple" not in hashed


def test_hash_password_produces_a_bcrypt_hash():
    hashed = hash_password("s3cr3t-Passw0rd")
    assert hashed.startswith(("$2a$", "$2b$", "$2y$"))


def test_hash_password_salts_each_call_differently():
    first = hash_password("same-password")
    second = hash_password("same-password")
    assert first != second


def test_verify_password_true_for_matching_password():
    hashed = hash_password("my-password-123")
    assert verify_password("my-password-123", hashed) is True


def test_verify_password_false_for_non_matching_password():
    hashed = hash_password("my-password-123")
    assert verify_password("a-totally-different-password", hashed) is False


# ---------------------------------------------------------------------------
# new_jti
# ---------------------------------------------------------------------------

def test_new_jti_is_a_uuid4_hex_string():
    jti = new_jti()
    assert isinstance(jti, str)
    assert len(jti) == 32
    int(jti, 16)  # raises ValueError if not valid hex


def test_new_jti_is_unique_per_call():
    assert new_jti() != new_jti()


# ---------------------------------------------------------------------------
# create_access_token / create_refresh_token / decode_token round trip
# ---------------------------------------------------------------------------

def test_create_access_token_round_trips_through_decode_token():
    user_id = uuid4()
    jti = new_jti()

    token = create_access_token(user_id, "clinic_management", jti)
    payload = decode_token(token)

    assert isinstance(payload, TokenPayload)
    assert payload.sub == str(user_id)
    assert payload.role == "clinic_management"
    assert payload.jti == jti
    assert payload.type == "access"


def test_create_access_token_embeds_role_directly_in_claims():
    # US1 AC2: the resolved role drives the permission check from the token
    # claim itself, not a second DB lookup.
    token = create_access_token(uuid4(), "delivery_team", new_jti())
    payload = decode_token(token)
    assert payload.role == "delivery_team"


def test_create_access_token_expires_in_900_seconds():
    before = int(time.time())
    token = create_access_token(uuid4(), "front_office_staff", new_jti())
    payload = decode_token(token)

    assert 895 <= payload.exp - before <= 905


def test_create_refresh_token_round_trips_and_has_refresh_type():
    user_id = uuid4()
    jti = new_jti()

    token = create_refresh_token(user_id, "front_office_staff", jti)
    payload = decode_token(token)

    assert payload.sub == str(user_id)
    assert payload.jti == jti
    assert payload.type == "refresh"


def test_create_refresh_token_expires_in_about_seven_days():
    before = int(time.time())
    token = create_refresh_token(uuid4(), "front_office_staff", new_jti())
    payload = decode_token(token)

    seven_days_seconds = 7 * 24 * 60 * 60
    assert abs((payload.exp - before) - seven_days_seconds) <= 60


def test_create_access_token_returns_a_three_part_jwt_string():
    token = create_access_token(uuid4(), "front_office_staff", new_jti())
    assert isinstance(token, str)
    assert token.count(".") == 2


# ---------------------------------------------------------------------------
# decode_token error paths -> UnauthorizedError
# ---------------------------------------------------------------------------

def test_decode_token_raises_unauthorized_for_malformed_token():
    with pytest.raises(UnauthorizedError):
        decode_token("this-is-not-a-jwt-at-all")


def test_decode_token_raises_unauthorized_for_invalid_signature():
    token = jose_jwt.encode(
        {
            "sub": str(uuid4()),
            "role": "front_office_staff",
            "jti": new_jti(),
            "type": "access",
            "exp": int(time.time()) + 900,
        },
        "a-completely-wrong-secret-key",
        algorithm="HS256",
    )
    with pytest.raises(UnauthorizedError):
        decode_token(token)


def test_decode_token_raises_unauthorized_for_expired_token():
    token = jose_jwt.encode(
        {
            "sub": str(uuid4()),
            "role": "front_office_staff",
            "jti": new_jti(),
            "type": "access",
            "exp": int(time.time()) - 60,
        },
        _DEFAULT_JWT_SECRET,
        algorithm=_DEFAULT_JWT_ALGORITHM,
    )
    with pytest.raises(UnauthorizedError):
        decode_token(token)


def test_decode_token_does_not_raise_a_bare_exception_type():
    # Interaction contract: callers (core/dependencies.py) rely on catching
    # exactly UnauthorizedError, not a bare Exception/JWTError, to map to 401.
    try:
        decode_token("garbage")
    except UnauthorizedError:
        pass
    except Exception as exc:  # pragma: no cover - failure path
        pytest.fail(f"expected UnauthorizedError, got {type(exc).__name__}")
