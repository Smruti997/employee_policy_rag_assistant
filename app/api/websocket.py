"""WebSocket transport for authenticated chat sessions."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.auth.dependencies import authenticate_websocket
from app.core.exceptions import AuthenticationError
from app.orchestration.chat import ChatOrchestrator


router = APIRouter()
orchestrator = ChatOrchestrator()


@router.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    """Authenticate once, then retain only server-verified identity for the session."""
    await websocket.accept()

    try:
        user = await authenticate_websocket(websocket)
    except AuthenticationError:
        await websocket.send_json(
            {"type": "auth_failed", "message": "Invalid or expired access token"}
        )
        await websocket.close(code=1008)
        return
    except WebSocketDisconnect:
        return

    await websocket.send_json(
        {
            "type": "auth_success",
            "user_id": user.user_id,
            "department": user.department,
            "level": user.level,
        }
    )

    # Per-connection conversation history so follow-up questions have context.
    history: list[dict[str, str]] = []

    while True:
        try:
            message = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except ValueError:
            await websocket.send_json({"type": "error", "message": "Message must be valid JSON"})
            continue

        if not isinstance(message, dict) or message.get("type") != "message":
            await websocket.send_json({"type": "error", "message": "Unsupported message type"})
            continue

        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            await websocket.send_json({"type": "error", "message": "Message text is required"})
            continue

        reply_parts: list[str] = []
        errored = False
        async for event in orchestrator.respond(text, user, history=history):
            if event.is_error:
                errored = True
                await websocket.send_json({"type": "error", "message": event.text})
            else:
                reply_parts.append(event.text)
                await websocket.send_json({"type": "stream", "text": event.text})
        if not errored:
            await websocket.send_json({"type": "done"})

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": "".join(reply_parts)})
