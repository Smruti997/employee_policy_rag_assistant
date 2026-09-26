"""Trusted identity models used across application layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """The only identity object application layers receive after authentication."""

    user_id: str
    email: str
    department: str
    level: int

    @property
    def access_scope(self) -> "AccessScope":
        from app.auth.rbac import AccessScope

        return AccessScope.for_user(self)
