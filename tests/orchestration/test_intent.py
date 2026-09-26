import json

import pytest

from app.core.exceptions import ChatServiceError
from app.orchestration.intent import IntentRouter
from app.orchestration.models import DomainCategory, Route


class FakeLLM:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.messages = None

    async def stream(self, messages):
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        yield self.response


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("route", "category"),
    [
        ("RAG", "hr"),
        ("PERSONAL_INFO", "personal"),
        ("BOTH", "finance"),
        ("GREETING", "general"),
        ("CLOSURE", "general"),
        ("DIRECT_RESPONSE", "general"),
    ],
)
async def test_routes_with_configured_llm(route: str, category: str) -> None:
    provider = FakeLLM(json.dumps({
        "route": route,
        "domain_category": category,
        "extracted_keywords": ["annual leave"],
    }))
    decision = await IntentRouter(llm_provider=provider).route("How does annual leave work?")

    assert decision.route is Route(route)
    assert decision.domain_category is DomainCategory(category)
    assert decision.extracted_keywords == ("annual leave",)
    assert provider.messages[0]["role"] == "system"
    assert provider.messages[1] == {"role": "user", "content": "How does annual leave work?"}


@pytest.mark.anyio
async def test_accepts_json_wrapped_in_markdown_fence() -> None:
    provider = FakeLLM('```json\n{"route":"RAG","domain_category":"hr","extracted_keywords":[]}\n```')
    decision = await IntentRouter(llm_provider=provider).route("What is the leave policy?")
    assert decision.route is Route.RAG


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"route":"UNKNOWN","domain_category":"hr","extracted_keywords":[]}',
        '{"route":"RAG","domain_category":"hr","extracted_keywords":"leave"}',
    ],
)
async def test_rejects_invalid_llm_classification(response: str) -> None:
    with pytest.raises(ChatServiceError, match="couldn't classify"):
        await IntentRouter(llm_provider=FakeLLM(response)).route("Policy question")


@pytest.mark.anyio
async def test_provider_failure_becomes_safe_chat_error() -> None:
    with pytest.raises(ChatServiceError, match="couldn't classify"):
        await IntentRouter(llm_provider=FakeLLM(RuntimeError("provider unavailable"))).route("Policy question")
