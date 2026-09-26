from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.auth.models import AuthenticatedUser
from app.orchestration.models import DomainCategory, Route, RoutingDecision
from app.orchestration.tools import PersonalAssistantTool, PolicyRAGTool


def _make_user() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="emp-001", email="emp@test.com", department="hr", level=1)


@pytest.mark.anyio
async def test_policy_rag_tool_execution():
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []
    mock_llm = MagicMock()

    async def mock_stream(messages):
        yield "Chunk 1"
        yield "Chunk 2"

    mock_llm.stream = mock_stream

    tool = PolicyRAGTool(retriever=mock_retriever, llm_provider=mock_llm)
    decision = RoutingDecision(
        route=Route.RAG,
        reasoning="Policy question",
        extracted_keywords=("leave",),
        domain_category=DomainCategory.HR,
    )

    events = [e async for e in tool.execute("How much leave?", _make_user(), decision)]
    assert len(events) == 2
    assert events[0].text == "Chunk 1"
    assert events[1].text == "Chunk 2"


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
async def test_personal_assistant_parallel_fetch():
    """Three mock MCP calls with 1s delay each must complete in ~1s, not ~3s."""
    import time

    tool = PersonalAssistantTool()
    start = time.monotonic()
    context = await tool.get_employee_context("emp-001")
    elapsed = time.monotonic() - start

    assert context["name"] == "Jane Doe"
    assert context["grade"] == "Senior"
    assert context["manager"] == "John Smith"
    assert context["team_size"] == 8
    assert context["team_name"] == "Platform"
    assert elapsed < 2.0, f"Parallel fetch took {elapsed:.1f}s — expected ~1s"
