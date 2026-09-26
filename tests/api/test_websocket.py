from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.api import websocket as ws_module
from app.main import app
from app.orchestration.models import ChatEvent, DomainCategory, Route, RoutingDecision


SECRET = "test-secret-that-is-longer-than-thirty-two-characters"


def make_token() -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "emp-001",
        "email": "emp@test.com",
        "department": "hr",
        "level": 1,
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")


class _FakeOrchestrator:
    """Yields one error event, then a normal stream, to prove the socket survives."""

    def __init__(self) -> None:
        self._calls = 0

    async def respond(self, text, user, history=None):
        self._calls += 1
        decision = RoutingDecision(
            route=Route.RAG,
            reasoning="test",
            extracted_keywords=(),
            domain_category=DomainCategory.GENERAL,
        )
        if self._calls == 1:
            yield ChatEvent(
                text="I couldn't generate an answer right now. Please try again.",
                decision=decision,
                is_error=True,
            )
        else:
            yield ChatEvent(text="ok", decision=decision)


def test_tool_failure_emits_error_frame_without_done_and_keeps_socket_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setattr(ws_module, "orchestrator", _FakeOrchestrator())

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"type": "auth", "token": make_token()})
            assert websocket.receive_json()["type"] == "auth_success"

            # A failing turn emits an error frame, not stream + done.
            websocket.send_json({"type": "message", "text": "boom"})
            assert websocket.receive_json() == {
                "type": "error",
                "message": "I couldn't generate an answer right now. Please try again.",
            }

            # The connection stayed open: the next message is answered normally.
            websocket.send_json({"type": "message", "text": "hi"})
            assert websocket.receive_json() == {"type": "stream", "text": "ok"}
            assert websocket.receive_json() == {"type": "done"}
