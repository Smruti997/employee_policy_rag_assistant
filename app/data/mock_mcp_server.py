"""Mock MCP server exposing three employee-context operations.

Each operation reads its own table from ``app/sample_data`` and simulates a
one-second network call. Because the tables are independent, the three reads are
concurrent when the client fans them out with ``asyncio.gather``: ~1s in total
rather than ~3s.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.data.sample_data import get_row


MOCK_LATENCY_SECONDS = 1.0


async def _read(table: str, user_id: str) -> dict[str, Any]:
    """Simulate one table read, keeping the blocking parse off the event loop."""
    await asyncio.sleep(MOCK_LATENCY_SECONDS)
    return await asyncio.to_thread(get_row, table, user_id)


async def get_profile(user_id: str) -> dict[str, Any]:
    """Return name and grade from ``profile.xlsx`` after a mock one-second delay."""
    row = await _read("profile", user_id)
    return {"name": row["cust_name"], "grade": row["grade"]}


async def get_manager(user_id: str) -> dict[str, Any]:
    """Return the manager from ``manager.xlsx`` after a mock one-second delay."""
    row = await _read("manager", user_id)
    return {"manager": row["manager"]}


async def get_team(user_id: str) -> dict[str, Any]:
    """Return team name and size from ``team info.xlsx`` after a mock one-second delay."""
    row = await _read("team", user_id)
    return {"team_name": row["team_name"], "team_size": row["team_size"]}
