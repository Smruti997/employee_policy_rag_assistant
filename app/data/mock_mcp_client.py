"""Mock MCP client: fetches employee context in parallel.

Calls ``get_profile``, ``get_manager``, and ``get_team`` concurrently
so the combined latency is ~1 s instead of ~3 s.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.data.mock_mcp_server import get_manager, get_profile, get_team


async def get_employee_context(user_id: str) -> dict[str, Any]:
    """Fetch profile, manager, and team info concurrently and merge them."""
    profile, manager, team = await asyncio.gather(
        get_profile(user_id),
        get_manager(user_id),
        get_team(user_id),
    )
    return {**profile, **manager, **team}