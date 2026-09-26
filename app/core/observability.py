"""Structured per-turn logging.

Every chat turn emits one JSON object per line, so a single request can be
followed end to end:

``turn.start`` / ``router.result`` / ``tool.selected`` / ``rag.input`` /
``rag.retrieved`` / ``rag.output`` / ``employee_db.read.start`` /
``employee_db.read.done`` / ``turn.end``

Lines are appended to ``logs/chat_turns.jsonl`` (override with ``CHAT_LOG_FILE``)
and mirrored to stderr so they are visible while the server runs.

Events logged inside a turn carry the same ``turn_id``: it is held in a
``ContextVar`` and set once by the orchestrator, so tools do not need it passed
through their signatures.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_FILE = "logs/chat_turns.jsonl"

_turn_id: ContextVar[str] = ContextVar("turn_id", default="-")
_write_lock = threading.Lock()


def new_turn_id() -> str:
    """Return a short id that ties every log line of one turn together."""
    return uuid.uuid4().hex[:12]


def bind_turn(turn_id: str) -> None:
    """Tag all later ``log_event`` calls in this context with *turn_id*."""
    _turn_id.set(turn_id)


def log_path() -> Path:
    """Return the JSON log file, honouring ``CHAT_LOG_FILE``."""
    return Path(os.getenv("CHAT_LOG_FILE") or DEFAULT_LOG_FILE)


def elapsed_ms(started: float) -> float:
    """Return milliseconds since a ``time.monotonic()`` reading."""
    return round((time.monotonic() - started) * 1000, 1)


def log_event(event: str, **fields: Any) -> None:
    """Append one JSON log line. Never raises into the caller."""
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "turn_id": _turn_id.get(),
        "event": event,
    }
    record.update(fields)
    line = json.dumps(record, default=str)

    try:
        with _write_lock:
            path = log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as exc:  # a broken log file must not break a chat turn
        print(f"[observability] could not write {log_path()}: {exc}", file=sys.stderr)

    print(line, file=sys.stderr)
