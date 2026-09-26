"""Shared test configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_turn_log(tmp_path, monkeypatch):
    """Keep turn logs out of the repository during tests."""
    monkeypatch.setenv("CHAT_LOG_FILE", str(tmp_path / "chat_turns.jsonl"))
