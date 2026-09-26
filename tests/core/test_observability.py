from __future__ import annotations

import json
from pathlib import Path

from app.core.observability import bind_turn, elapsed_ms, log_event, log_path, new_turn_id


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_log_event_appends_one_json_object_per_line(tmp_path, monkeypatch) -> None:
    target = tmp_path / "nested" / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(target))

    log_event("turn.start", query="hello", level=1)
    log_event("turn.end", latency_ms=12.5)

    records = _read(target)
    assert [r["event"] for r in records] == ["turn.start", "turn.end"]
    assert records[0]["query"] == "hello"
    assert records[0]["level"] == 1
    assert records[1]["latency_ms"] == 12.5


def test_events_share_the_bound_turn_id(tmp_path, monkeypatch) -> None:
    target = tmp_path / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(target))

    turn_id = new_turn_id()
    bind_turn(turn_id)
    log_event("turn.start", query="hello")
    log_event("tool.selected", tool="policy_rag")

    records = _read(target)
    assert {r["turn_id"] for r in records} == {turn_id}


def test_unserialisable_values_do_not_raise(tmp_path, monkeypatch) -> None:
    target = tmp_path / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(target))

    log_event("weird", value=object())

    assert _read(target)[0]["event"] == "weird"


def test_unwritable_log_path_does_not_raise(monkeypatch) -> None:
    monkeypatch.setenv("CHAT_LOG_FILE", "/dev/null/not-a-directory/turns.jsonl")

    log_event("turn.start", query="still works")  # must not raise


def test_log_path_defaults_and_honours_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_LOG_FILE", raising=False)
    assert log_path() == Path("logs/chat_turns.jsonl")

    monkeypatch.setenv("CHAT_LOG_FILE", "/tmp/custom.jsonl")
    assert log_path() == Path("/tmp/custom.jsonl")


def test_elapsed_ms_is_a_rounded_positive_number() -> None:
    import time

    assert elapsed_ms(time.monotonic()) >= 0
