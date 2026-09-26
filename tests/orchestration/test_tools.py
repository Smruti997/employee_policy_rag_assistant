from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.auth.models import AuthenticatedUser
from app.orchestration.models import DomainCategory, Route, RoutingDecision
from app.orchestration.tools import PersonalAssistantTool, PolicyRAGTool


def _make_user() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="emp-001", email="emp@test.com", department="hr", level=1)


def _rag_decision() -> RoutingDecision:
    return RoutingDecision(
        route=Route.RAG,
        reasoning="Policy question",
        extracted_keywords=("leave",),
        domain_category=DomainCategory.HR,
    )


@pytest.mark.anyio
async def test_policy_rag_tool_execution():
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    mock_llm = MagicMock()

    async def mock_stream(messages):
        yield "Chunk 1"
        yield "Chunk 2"

    mock_llm.stream = mock_stream

    tool = PolicyRAGTool(retriever=mock_retriever, llm_provider=mock_llm)

    events = [e async for e in tool.execute("How much leave?", _make_user(), _rag_decision())]
    assert len(events) == 2
    assert events[0].text == "Chunk 1"
    assert events[1].text == "Chunk 2"
    assert all(not event.is_error for event in events)


@pytest.mark.anyio
async def test_rag_tool_retrieval_failure_is_a_safe_error_event():
    mock_retriever = MagicMock()

    async def boom(question, scope, limit=5):
        raise RuntimeError("qdrant exploded with secret internals")

    mock_retriever.search = boom
    tool = PolicyRAGTool(retriever=mock_retriever, llm_provider=MagicMock())

    events = [e async for e in tool.execute("How much leave?", _make_user(), _rag_decision())]

    assert len(events) == 1
    assert events[0].is_error is True
    assert "qdrant" not in events[0].text
    assert "secret internals" not in events[0].text
    assert "retrieve policy information" in events[0].text


@pytest.mark.anyio
async def test_rag_tool_llm_failure_is_a_safe_error_event():
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])
    mock_llm = MagicMock()

    async def bad_stream(messages):
        yield "partial"
        raise RuntimeError("llm secret error")

    mock_llm.stream = bad_stream
    tool = PolicyRAGTool(retriever=mock_retriever, llm_provider=mock_llm)

    events = [e async for e in tool.execute("How much leave?", _make_user(), _rag_decision())]

    assert events[-1].is_error is True
    assert "llm secret error" not in events[-1].text
    assert "generate an answer" in events[-1].text


@pytest.mark.anyio
async def test_personal_assistant_tool_execution():
    mock_llm = MagicMock()

    async def mock_stream(messages):
        yield "Your manager is "
        yield "John Smith."

    mock_llm.stream = mock_stream

    tool = PersonalAssistantTool(llm_provider=mock_llm)
    decision = RoutingDecision(
        route=Route.PERSONAL_INFO,
        reasoning="Personal query",
        extracted_keywords=("manager",),
        domain_category=DomainCategory.PERSONAL,
    )

    events = [e async for e in tool.execute(_make_user(), decision, question="Who is my manager?")]
    assert len(events) == 2
    assert "".join(e.text for e in events) == "Your manager is John Smith."


@pytest.mark.anyio
async def test_personal_assistant_reports_unknown_employee_without_raising():
    """A user absent from the sample_data tables must surface a message, not break the socket."""
    tool = PersonalAssistantTool()
    decision = RoutingDecision(
        route=Route.PERSONAL_INFO,
        reasoning="Personal query",
        extracted_keywords=("manager",),
        domain_category=DomainCategory.PERSONAL,
    )
    user = AuthenticatedUser(
        user_id="exec-003", email="exec@test.com", department="exec", level=3
    )

    events = [e async for e in tool.execute(user, decision, question="Who is my manager?")]

    assert len(events) == 1
    assert "couldn't retrieve your employee details" in events[0].text
    # The internal reason (and the workbook name) must not reach the client.
    assert "profile.xlsx" not in events[0].text


@pytest.mark.anyio
async def test_personal_assistant_parallel_fetch():
    """Three mock MCP calls with 1s delay each must complete in ~1s, not ~3s."""
    import time

    tool = PersonalAssistantTool()
    start = time.monotonic()
    context = await tool.get_employee_context("emp-001")
    elapsed = time.monotonic() - start

    # Values come from the app/sample_data tables (profile/manager/team info).
    assert context["name"] == "John Doe"
    assert context["grade"] == "senior"
    assert context["manager"] == "Dwayne Johnson"
    assert context["team_size"] == 8
    assert context["team_name"] == "Platform"
    assert elapsed < 2.0, f"Parallel fetch took {elapsed:.1f}s — expected ~1s"
