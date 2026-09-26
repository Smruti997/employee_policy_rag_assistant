"""HS256 JWT verification and authorization-claim validation."""

from __future__ import annotations

from typing import Any

import jwt

from app.auth.models import AuthenticatedUser
from app.auth.rbac import MAX_ACCESS_LEVEL, MIN_ACCESS_LEVEL, VALID_DEPARTMENTS
from app.core.exceptions import AuthenticationError


ALGORITHM = "HS256"


def decode_access_token(token: str, secret: str) -> AuthenticatedUser:
    """Verify an access token and return authorization-safe claims."""
    if not token or not token.strip():
        raise AuthenticationError("Missing access token")

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            options={"require": ["sub", "email", "department", "level", "iat", "exp"]},
        )
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Invalid or expired access token") from exc

    return _user_from_payload(payload)


def _user_from_payload(payload: dict[str, Any]) -> AuthenticatedUser:
    user_id = payload.get("sub")
    email = payload.get("email")
    department = payload.get("department")
    level = payload.get("level")

    if not isinstance(user_id, str) or not user_id:
        raise AuthenticationError("Token subject is invalid")
    if not isinstance(email, str) or not email:
        raise AuthenticationError("Token email is invalid")
    if department not in VALID_DEPARTMENTS:
        raise AuthenticationError("Token department is invalid")
    if isinstance(level, bool) or not isinstance(level, int):
        raise AuthenticationError("Token access level is invalid")
    if not MIN_ACCESS_LEVEL <= level <= MAX_ACCESS_LEVEL:
        raise AuthenticationError("Token access level is out of range")

    return AuthenticatedUser(
        user_id=user_id,
        email=email,
        department=department,
        level=level,
    )
