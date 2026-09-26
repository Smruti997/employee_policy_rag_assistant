from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.auth.models import AuthenticatedUser
from app.data.qdrant_store import SearchResult
from app.orchestration.chat import ChatOrchestrator
from app.orchestration.models import DomainCategory, Route, RoutingDecision
from app.orchestration.tools import PersonalAssistantTool


_RAG_DECISION = RoutingDecision(
    route=Route.RAG, reasoning="policy", extracted_keywords=(), domain_category=DomainCategory.HR
)
_PERSONAL_DECISION = RoutingDecision(
    route=Route.PERSONAL_INFO,
    reasoning="personal",
    extracted_keywords=(),
    domain_category=DomainCategory.PERSONAL,
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stream(*chunks: str):
    async def stream(messages):
        for chunk in chunks:
            yield chunk

    return stream


def _user(user_id: str = "emp-001", department: str = "hr", level: int = 1) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=user_id, email=f"{user_id}@test.com", department=department, level=level
    )


@pytest.mark.anyio
async def test_rag_turn_logs_router_tool_and_input_output(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(log_file))

    mock_router = AsyncMock()
    mock_router.route.return_value = _RAG_DECISION
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[
        SearchResult(
            chunk_id="c1",
            text="Employees get 15 days of sick leave.",
            source_file="leave_policy.pdf",
            department="hr",
            access_level=1,
            page=2,
            score=0.92,
        )
    ])
    mock_llm = MagicMock()
    mock_llm.stream = _stream("You get ", "15 days.")

    orchestrator = ChatOrchestrator(
        intent_router=mock_router, retriever=mock_retriever, llm_provider=mock_llm
    )
    [event async for event in orchestrator.respond("How many leave days?", _user())]

    records = _read(log_file)
    assert [record["event"] for record in records] == [
        "turn.start",
        "router.result",
        "tool.selected",
        "rag.input",
        "rag.retrieved",
        "rag.output",
        "turn.end",
    ]
    assert len({record["turn_id"] for record in records}) == 1

    assert records[2]["tool"] == "policy_rag"
    assert records[3]["question"] == "How many leave days?"
    assert records[3]["departments"] == ["hr"]
    assert records[3]["max_access_level"] == 1
    assert records[4]["chunk_count"] == 1
    assert records[4]["sources"] == [
        {"source_file": "leave_policy.pdf", "page": 2, "score": 0.92}
    ]
    assert records[5]["answer"] == "You get 15 days."


@pytest.mark.anyio
async def test_personal_turn_logs_the_db_read_before_and_after(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(log_file))

    mock_router = AsyncMock()
    mock_router.route.return_value = _PERSONAL_DECISION
    mock_llm = MagicMock()
    mock_llm.stream = _stream("Your manager is Dwayne Johnson.")

    orchestrator = ChatOrchestrator(
        intent_router=mock_router,
        personal_tool=PersonalAssistantTool(llm_provider=mock_llm),
    )
    [event async for event in orchestrator.respond("who is my manager", _user())]

    records = _read(log_file)
    assert [record["event"] for record in records] == [
        "turn.start",
        "router.result",
        "tool.selected",
        "employee_db.read.start",
        "employee_db.read.done",
        "personal.output",
        "turn.end",
    ]
    assert records[2]["tool"] == "personal_assistant"

    read_start, read_done = records[3], records[4]
    assert read_start["user_id"] == "emp-001"
    assert read_start["mode"] == "parallel"
    assert read_start["tables"] == ["profile", "manager", "team"]
    assert read_done["context"] == {
        "name": "John Doe",
        "grade": "senior",
        "manager": "Dwayne Johnson",
        "team_size": 8,
        "team_name": "Platform",
    }
    assert read_done["latency_ms"] >= 900


@pytest.mark.anyio
async def test_personal_turn_logs_the_db_read_failure(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(log_file))

    mock_router = AsyncMock()
    mock_router.route.return_value = _PERSONAL_DECISION

    orchestrator = ChatOrchestrator(intent_router=mock_router)
    [event async for event in orchestrator.respond("who is my manager", _user("mgr-002"))]

    events = [record["event"] for record in _read(log_file)]
    assert "employee_db.read.error" in events
    assert "employee_db.read.done" not in events
    assert events[-1] == "turn.end"

    failure = next(r for r in _read(log_file) if r["event"] == "employee_db.read.error")
    assert failure["user_id"] == "mgr-002"
    assert "mgr-002" in failure["error"]


@pytest.mark.anyio
async def test_lookup_only_routes_do_not_touch_the_database(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "turns.jsonl"
    monkeypatch.setenv("CHAT_LOG_FILE", str(log_file))

    mock_router = AsyncMock()
    mock_router.route.return_value = RoutingDecision(
        route=Route.GREETING,
        reasoning="greeting",
        extracted_keywords=(),
        domain_category=DomainCategory.GENERAL,
    )

    orchestrator = ChatOrchestrator(intent_router=mock_router)
    [event async for event in orchestrator.respond("Hi", _user())]

    records = _read(log_file)
    assert [record["event"] for record in records] == [
        "turn.start",
        "router.result",
        "tool.selected",
        "turn.end",
    ]
    assert records[2]["tool"] == "greeting"
