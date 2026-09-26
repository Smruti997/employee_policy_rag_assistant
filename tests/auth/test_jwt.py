from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.auth.jwt import ALGORITHM, decode_access_token
from app.core.exceptions import AuthenticationError
from app.main import app


SECRET = "test-secret-that-is-longer-than-thirty-two-characters"


def make_token(**overrides: object) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "emp-001",
        "email": "emp@test.com",
        "department": "finance",
        "level": 1,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        **overrides,
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"department": "sales"}, "department"),
        ({"level": 0}, "range"),
        ({"level": True}, "invalid"),
        ({"sub": ""}, "subject"),
    ],
)
def test_rejects_invalid_authorization_claims(
    overrides: dict[str, object], error: str
) -> None:
    with pytest.raises(AuthenticationError, match=error):
        decode_access_token(make_token(**overrides), SECRET)


def test_rejects_expired_token() -> None:
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {
            "sub": "emp-001",
            "email": "emp@test.com",
            "department": "hr",
            "level": 1,
            "iat": now - timedelta(minutes=10),
            "exp": now - timedelta(minutes=1),
        },
        SECRET,
        algorithm=ALGORITHM,
    )

    with pytest.raises(AuthenticationError, match="Invalid or expired"):
        decode_access_token(token, SECRET)


def test_websocket_authenticates_first_frame_and_keeps_session_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"type": "auth", "token": make_token()})
            assert websocket.receive_json() == {
                "type": "auth_success",
                "user_id": "emp-001",
                "department": "finance",
                "level": 1,
            }

            websocket.send_json({"type": "message", "text": "How much sick leave can I take?"})
            frame = websocket.receive_json()
            assert frame["type"] == "stream"
            while frame.get("type") == "stream":
                frame = websocket.receive_json()
            assert frame == {"type": "done"}



def test_websocket_rejects_invalid_token_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"type": "auth", "token": "not-a-jwt"})
            assert websocket.receive_json()["type"] == "auth_failed"
            with pytest.raises(WebSocketDisconnect) as disconnected:
                websocket.receive_json()

    assert disconnected.value.code == 1008
