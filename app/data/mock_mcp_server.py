"""In-process mock MCP operations backed by keyed sample employee records."""

from __future__ import annotations

import asyncio
from typing import Any

# emp-001 preserves the assignment's expected sample values. Other entries make
# it clear that the tool is scoped to the authenticated subject, not a constant.
EMPLOYEE_RECORDS: dict[str, dict[str, Any]] = {
    "emp-001": {
        "profile": {"name": "Jane Doe", "grade": "Senior"},
        "manager": {"manager": "John Smith"},
        "team": {"team_size": 8, "team_name": "Platform"},
    },
    "mgr-002": {
        "profile": {"name": "Alex Morgan", "grade": "Manager"},
        "manager": {"manager": "Priya Shah"},
        "team": {"team_size": 8, "team_name": "People Operations"},
    },
    "exec-003": {
        "profile": {"name": "Samira Khan", "grade": "Executive"},
        "manager": {"manager": "Board of Directors"},
        "team": {"team_size": 12, "team_name": "Executive Leadership"},
    },
}


def _record(user_id: str) -> dict[str, Any]:
    try:
        return EMPLOYEE_RECORDS[user_id]
    except KeyError as exc:
        raise ValueError("No mock employee record exists for this authenticated user") from exc


async def get_profile(user_id: str) -> dict[str, Any]:
    """Return mock name and grade after simulating a one-second API call."""
    await asyncio.sleep(1)
    return dict(_record(user_id)["profile"])


async def get_manager(user_id: str) -> dict[str, Any]:
    """Return mock manager information after simulating a one-second API call."""
    await asyncio.sleep(1)
    return dict(_record(user_id)["manager"])


async def get_team(user_id: str) -> dict[str, Any]:
    """Return mock team information after simulating a one-second API call."""
    await asyncio.sleep(1)
    return dict(_record(user_id)["team"])
