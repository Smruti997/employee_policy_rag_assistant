"""Role-based retrieval policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.auth.models import AuthenticatedUser


VALID_DEPARTMENTS = frozenset({"hr", "finance", "exec"})
MIN_ACCESS_LEVEL = 1
MAX_ACCESS_LEVEL = 3


@dataclass(frozen=True, slots=True)
class AccessScope:
    """A retrieval-ready representation of the authenticated user's permissions."""

    departments: tuple[str, ...]
    maximum_access_level: int

    @classmethod
    def for_user(cls, user: AuthenticatedUser) -> "AccessScope":
        # HR content is company-wide; a user's own department is the only other scope.
        departments = ("hr",) if user.department == "hr" else ("hr", user.department)
        return cls(departments=departments, maximum_access_level=user.level)

    def qdrant_filter(self) -> dict[str, Any]:
        """Return the payload filter the data layer must pass to Qdrant.search/query."""
        return {
            "must": [
                {"key": "department", "match": {"any": list(self.departments)}},
                {
                    "key": "access_level",
                    "range": {"lte": self.maximum_access_level},
                },
            ]
        }
