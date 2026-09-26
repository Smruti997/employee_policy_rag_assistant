import pytest
from unittest.mock import MagicMock

from app.auth.models import AuthenticatedUser
from app.orchestration.chat import ChatOrchestrator
from app.orchestration.models import ChatEvent, DomainCategory, Route, RoutingDecision


_RAG_DECISION = RoutingDecision(
    route=Route.RAG, reasoning="test", extracted_keywords=(), domain_category=DomainCategory.HR,
)
_PERSONAL_DECISION = RoutingDecision(
    route=Route.PERSONAL_INFO, reasoning="test", extracted_keywords=(), domain_category=DomainCategory.PERSONAL,
)
_GREETING_DECISION = RoutingDecision(
    route=Route.GREETING, reasoning="test", extracted_keywords=(), domain_category=DomainCategory.GENERAL,
)
_CLOSURE_DECISION = RoutingDecision(
    route=Route.CLOSURE, reasoning="test", extracted_keywords=(), domain_category=DomainCategory.GENERAL,
)
_DIRECT_DECISION = RoutingDecision(
    route=Route.DIRECT_RESPONSE, reasoning="test", extracted_keywords=(), domain_category=DomainCategory.GENERAL,
)


class FakeRouter:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision
        self.messages = []

    async def route(self, message: str) -> RoutingDecision:
        self.messages.append(message)
        return self.decision


def _make_user() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="emp-001", email="emp@test.com", department="hr", level=1)


@pytest.mark.anyio
async def test_chat_orchestrator_rag_response() -> None:
    from app.data.qdrant_store import SearchResult

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [SearchResult(
        chunk_id="c1", text="Employees get 15 days of sick leave per calendar year.",
        source_file="hr_policy.pdf", department="hr", access_level=1, page=2, score=0.92,
    )]

    class FakeLLM:
        async def stream(self, messages):
            yield "You get "
            yield "15 days "
            yield "of sick leave."

    orchestrator = ChatOrchestrator(
        intent_router=FakeRouter(_RAG_DECISION), retriever=mock_retriever, llm_provider=FakeLLM(),
    )
    events = [e async for e in orchestrator.respond("How many days of sick leave can I take?", _make_user())]
    assert len(events) == 3
    assert "".join(e.text for e in events) == "You get 15 days of sick leave."


@pytest.mark.anyio
async def test_chat_orchestrator_rag_passes_history() -> None:
    from app.data.qdrant_store import SearchResult

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [SearchResult(
        chunk_id="c1", text="Some policy text.", source_file="p.pdf",
        department="hr", access_level=1, page=1, score=0.9,
    )]
    captured_messages = {}

    class FakeLLM:
        async def stream(self, messages):
            captured_messages["msgs"] = messages
            yield "answer"

    orchestrator = ChatOrchestrator(
        intent_router=FakeRouter(_RAG_DECISION), retriever=mock_retriever, llm_provider=FakeLLM(),
    )
    history = [{"role": "user", "content": "prev question"}, {"role": "assistant", "content": "prev answer"}]
    [e async for e in orchestrator.respond("follow up", _make_user(), history=history)]

    msgs = captured_messages["msgs"]
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "prev question"}
    assert msgs[2] == {"role": "assistant", "content": "prev answer"}
    assert msgs[3] == {"role": "user", "content": "follow up"}


@pytest.mark.anyio
async def test_chat_orchestrator_personal_response() -> None:
    mock_personal = MagicMock()

    async def mock_execute(user, decision, question="", history=None):
        yield ChatEvent(text="Your manager is John Smith.", decision=decision)

    mock_personal.execute = mock_execute
    orchestrator = ChatOrchestrator(intent_router=FakeRouter(_PERSONAL_DECISION), personal_tool=mock_personal)
    events = [e async for e in orchestrator.respond("Who is my direct manager?", _make_user())]
    assert len(events) == 1
    assert "John Smith" in events[0].text


@pytest.mark.anyio
async def test_chat_orchestrator_out_of_scope() -> None:
    orchestrator = ChatOrchestrator(intent_router=FakeRouter(_DIRECT_DECISION))
    events = [e async for e in orchestrator.respond("What is the capital of France?", _make_user())]
    assert len(events) == 1
    assert events[0].text == "I can help with company policies and your employee information."


@pytest.mark.anyio
async def test_chat_orchestrator_greeting() -> None:
    orchestrator = ChatOrchestrator(intent_router=FakeRouter(_GREETING_DECISION))
    events = [e async for e in orchestrator.respond("Hello!", _make_user())]
    assert len(events) == 1
    assert "How can I help" in events[0].text


@pytest.mark.anyio
async def test_chat_orchestrator_closure_response() -> None:
    orchestrator = ChatOrchestrator(intent_router=FakeRouter(_CLOSURE_DECISION))
    events = [e async for e in orchestrator.respond("That's all, thank you.", _make_user())]
    assert len(events) == 1
    assert "Thank you for connecting" in events[0].text
