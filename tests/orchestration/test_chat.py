import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from app.auth.models import AuthenticatedUser
from app.orchestration.chat import ChatOrchestrator
from app.orchestration.models import ChatEvent, DomainCategory, Route, RoutingDecision


_IST = timezone(timedelta(hours=5, minutes=30))

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


def _make_user() -> AuthenticatedUser:
    return AuthenticatedUser(user_id="emp-001", email="emp@test.com", department="hr", level=1)


@pytest.mark.anyio
async def test_chat_orchestrator_rag_response() -> None:
    from app.data.qdrant_store import SearchResult

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        SearchResult(
            chunk_id="c1",
            text="Employees get 15 days of sick leave per calendar year.",
            source_file="hr_policy.pdf",
            department="hr",
            access_level=1,
            page=2,
            score=0.92,
        )
    ]

    mock_llm = MagicMock()

    async def mock_stream(messages):
        yield "You get "
        yield "15 days "
        yield "of sick leave."

    mock_llm.stream = mock_stream

    mock_router = MagicMock()
    mock_router.route.return_value = _RAG_DECISION

    orchestrator = ChatOrchestrator(
        intent_router=mock_router, retriever=mock_retriever, llm_provider=mock_llm,
    )
    events = [e async for e in orchestrator.respond("How many days of sick leave can I take?", _make_user())]
    assert len(events) == 3
    assert "".join(e.text for e in events) == "You get 15 days of sick leave."


@pytest.mark.anyio
async def test_chat_orchestrator_rag_passes_history() -> None:
    from app.data.qdrant_store import SearchResult

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        SearchResult(
            chunk_id="c1", text="Some policy text.", source_file="p.pdf",
            department="hr", access_level=1, page=1, score=0.9,
        )
    ]
    captured_messages = {}

    mock_llm = MagicMock()

    async def mock_stream(messages):
        captured_messages["msgs"] = messages
        yield "answer"

    mock_llm.stream = mock_stream

    mock_router = MagicMock()
    mock_router.route.return_value = _RAG_DECISION

    orchestrator = ChatOrchestrator(
        intent_router=mock_router, retriever=mock_retriever, llm_provider=mock_llm,
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
    mock_router = MagicMock()
    mock_router.route.return_value = _PERSONAL_DECISION

    mock_personal = MagicMock()

    async def mock_execute(user, decision, question="", history=None):
        yield ChatEvent(text="Your manager is John Smith.", decision=decision)

    mock_personal.execute = mock_execute

    orchestrator = ChatOrchestrator(intent_router=mock_router, personal_tool=mock_personal)
    events = [e async for e in orchestrator.respond("Who is my direct manager?", _make_user())]
    assert len(events) == 1
    assert "John Smith" in events[0].text


@pytest.mark.anyio
async def test_chat_orchestrator_out_of_scope() -> None:
    mock_router = MagicMock()
    mock_router.route.return_value = _DIRECT_DECISION

    orchestrator = ChatOrchestrator(intent_router=mock_router)
    events = [e async for e in orchestrator.respond("What is the capital of France?", _make_user())]
    assert len(events) == 1
    assert events[0].text == "unable to provide answer"


@pytest.mark.anyio
async def test_chat_orchestrator_greeting_morning() -> None:
    mock_router = MagicMock()
    mock_router.route.return_value = _GREETING_DECISION

    morning = datetime(2026, 1, 1, 9, 0, 0, tzinfo=_IST)
    with patch("app.orchestration.chat.datetime") as mock_dt:
        mock_dt.now.return_value = morning
        orchestrator = ChatOrchestrator(intent_router=mock_router)
        events = [e async for e in orchestrator.respond("Hello!", _make_user())]
    assert len(events) == 1
    assert "Morning" in events[0].text


@pytest.mark.anyio
async def test_chat_orchestrator_greeting_afternoon() -> None:
    mock_router = MagicMock()
    mock_router.route.return_value = _GREETING_DECISION

    afternoon = datetime(2026, 1, 1, 14, 0, 0, tzinfo=_IST)
    with patch("app.orchestration.chat.datetime") as mock_dt:
        mock_dt.now.return_value = afternoon
        orchestrator = ChatOrchestrator(intent_router=mock_router)
        events = [e async for e in orchestrator.respond("Hi there", _make_user())]
    assert len(events) == 1
    assert "Afternoon" in events[0].text


@pytest.mark.anyio
async def test_chat_orchestrator_greeting_evening() -> None:
    mock_router = MagicMock()
    mock_router.route.return_value = _GREETING_DECISION

    evening = datetime(2026, 1, 1, 19, 0, 0, tzinfo=_IST)
    with patch("app.orchestration.chat.datetime") as mock_dt:
        mock_dt.now.return_value = evening
        orchestrator = ChatOrchestrator(intent_router=mock_router)
        events = [e async for e in orchestrator.respond("Good evening!", _make_user())]
    assert len(events) == 1
    assert "Evening" in events[0].text


@pytest.mark.anyio
async def test_chat_orchestrator_closure_response() -> None:
    mock_router = MagicMock()
    mock_router.route.return_value = _CLOSURE_DECISION

    orchestrator = ChatOrchestrator(intent_router=mock_router)
    events = [e async for e in orchestrator.respond("That's all, thank you.", _make_user())]
    assert len(events) == 1
    assert "Thank you for connecting" in events[0].text