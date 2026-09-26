"""Authentication adapters used by API transports."""

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket

from app.auth.jwt import decode_access_token
from app.auth.models import AuthenticatedUser
from app.core.config import Settings
from app.core.exceptions import AuthenticationError


async def authenticate_websocket(websocket: WebSocket) -> AuthenticatedUser:
    """Read and validate the mandatory first WebSocket authentication frame."""
    raw = await websocket.receive_text()
    try:
        message: Any = json.loads(raw)
    except ValueError:
        raise AuthenticationError("Auth frame is not valid JSON")

    if not isinstance(message, dict) or message.get("type") != "auth":
        raise AuthenticationError("First message must be an auth message")

    token = message.get("token")
    if not isinstance(token, str):
        raise AuthenticationError("Token is missing")

    return decode_access_token(token, Settings.from_environment().jwt_secret)
