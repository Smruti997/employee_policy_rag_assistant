import pytest
from unittest.mock import patch, MagicMock

from app.orchestration.intent import IntentRouter
from app.orchestration.models import DomainCategory, Route


def _mock_api_response(route: str, category: str):
    """Return a mock httpx response for the Drex API."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "answers": {
            "route": {"noul": route},
            "domain_category": {"noul": category},
        }
    }
    return resp


def test_rag_route():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", return_value=_mock_api_response("RAG", "hr")):
        decision = router.route("How many days of sick leave can I take?")
    assert decision.route is Route.RAG
    assert decision.domain_category is DomainCategory.HR


def test_personal_info_route():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", return_value=_mock_api_response("PERSONAL_INFO", "personal")):
        decision = router.route("Who is my direct manager?")
    assert decision.route is Route.PERSONAL_INFO
    assert decision.domain_category is DomainCategory.PERSONAL


def test_both_route():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", return_value=_mock_api_response("BOTH", "finance")):
        decision = router.route("Given my current grade, what is my per diem?")
    assert decision.route is Route.BOTH
    assert decision.domain_category is DomainCategory.FINANCE


def test_greeting_route():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", return_value=_mock_api_response("GREETING", "general")):
        decision = router.route("Hi, good morning!")
    assert decision.route is Route.GREETING
    assert decision.domain_category is DomainCategory.GENERAL


def test_closure_route():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", return_value=_mock_api_response("CLOSURE", "general")):
        decision = router.route("Goodbye, thanks!")
    assert decision.route is Route.CLOSURE


def test_direct_response_route():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", return_value=_mock_api_response("DIRECT_RESPONSE", "general")):
        decision = router.route("What is the capital of France?")
    assert decision.route is Route.DIRECT_RESPONSE


def test_fallback_to_rag_on_api_failure():
    router = IntentRouter(api_key="test-key")
    with patch("httpx.post", side_effect=Exception("network error")):
        decision = router.route("What is the expense policy?")
    assert decision.route is Route.RAG
    assert "unavailable" in decision.reasoning.lower()


def test_fallback_to_rag_when_no_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DREX_API_KEY", raising=False)
    monkeypatch.delenv("ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    router = IntentRouter(api_key=None)
    decision = router.route("Any question at all")
    assert decision.route is Route.RAG