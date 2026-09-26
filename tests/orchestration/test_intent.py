from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.orchestration.intent import (
    DEFAULT_MAX_TOKENS,
    IntentRouter,
    _configured_router,
    parse_classification,
)
from app.orchestration.models import DomainCategory, Route


class FakeProvider:
    """Minimal LLMProvider that records the messages and max_tokens it was given."""

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.calls: list[dict[str, object]] = []

    async def stream(self, messages, max_tokens=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        if self.reply is None:
            raise RuntimeError("provider exploded")
        yield self.reply


def _router(reply: str | None = '{"route": "RAG", "domain_category": "hr"}') -> tuple[IntentRouter, FakeProvider]:
    provider = FakeProvider(reply)
    return IntentRouter(llm_provider=provider), provider


# --- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"route": "RAG"}',
        '  {"route": "RAG"}  ',
        '```json\n{"route": "RAG"}\n```',
        '```\n{"route": "RAG"}\n```',
        'Sure, here you go: {"route": "RAG"}',
    ],
)
def test_parse_accepts_realistic_model_replies(raw: str) -> None:
    assert parse_classification(raw) == {"route": "RAG"}


@pytest.mark.parametrize("raw", [None, "", "   ", "no object here", "[1, 2]", "{not json}"])
def test_parse_rejects_unusable_replies(raw: str | None) -> None:
    assert parse_classification(raw) is None


# --- routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "category", "expected_route", "expected_category"),
    [
        ("RAG", "hr", Route.RAG, DomainCategory.HR),
        ("PERSONAL_INFO", "personal", Route.PERSONAL_INFO, DomainCategory.PERSONAL),
        ("BOTH", "finance", Route.BOTH, DomainCategory.FINANCE),
        ("GREETING", "general", Route.GREETING, DomainCategory.GENERAL),
        ("CLOSURE", "general", Route.CLOSURE, DomainCategory.GENERAL),
        ("DIRECT_RESPONSE", "general", Route.DIRECT_RESPONSE, DomainCategory.GENERAL),
    ],
)
@pytest.mark.anyio
async def test_routes_every_intent(
    route: str, category: str, expected_route: Route, expected_category: DomainCategory
) -> None:
    router, _ = _router(f'{{"route": "{route}", "domain_category": "{category}"}}')

    decision = await router.route("Some message")

    assert decision.route is expected_route
    assert decision.domain_category is expected_category


@pytest.mark.anyio
async def test_sends_the_message_and_a_capped_max_tokens() -> None:
    router, provider = _router()

    await router.route("what long term incentives")

    call = provider.calls[0]
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    messages = call["messages"]
    assert messages[1] == {"role": "user", "content": "what long term incentives"}
    assert messages[0]["role"] == "system"
    # The terse-topic case that a decision model used to send to DIRECT_RESPONSE.
    assert "long term incentives" in messages[0]["content"]


@pytest.mark.anyio
async def test_carries_reasoning_and_keywords_through() -> None:
    router, _ = _router(
        '{"route": "RAG", "domain_category": "exec", "reasoning": "exec pay question",'
        ' "keywords": ["long term incentives", "RSU"]}'
    )

    decision = await router.route("what long term incentives")

    assert decision.reasoning == "exec pay question"
    assert decision.extracted_keywords == ("long term incentives", "RSU")


@pytest.mark.anyio
async def test_bare_topic_phrase_no_longer_falls_back_to_direct_response() -> None:
    """Regression: this used to route to DIRECT_RESPONSE and never reach retrieval."""
    router, _ = _router('{"route": "RAG", "domain_category": "exec"}')

    decision = await router.route("what long term incentives")

    assert decision.route is Route.RAG


@pytest.mark.anyio
async def test_tolerates_label_casing_and_spacing_drift() -> None:
    router, _ = _router('{"route": "personal info", "domain_category": "PERSONAL"}')

    decision = await router.route("who is my manager")

    assert decision.route is Route.PERSONAL_INFO
    assert decision.domain_category is DomainCategory.PERSONAL


@pytest.mark.anyio
async def test_unknown_route_returns_error_and_keeps_the_category() -> None:
    router, _ = _router('{"route": "SOMETHING_ELSE", "domain_category": "exec"}')

    decision = await router.route("???")

    assert decision.route is Route.ERROR
    assert decision.domain_category is DomainCategory.EXEC
    assert "unknown route" in decision.reasoning.lower()


@pytest.mark.anyio
async def test_unparsable_reply_returns_error() -> None:
    router, _ = _router("I am not going to answer that.")

    decision = await router.route("Hello")

    assert decision.route is Route.ERROR
    assert "not parsable" in decision.reasoning.lower()


@pytest.mark.anyio
async def test_provider_failure_returns_error() -> None:
    router, _ = _router(None)

    decision = await router.route("What is the expense policy?")

    assert decision.route is Route.ERROR
    assert "failed" in decision.reasoning.lower()


@pytest.mark.anyio
async def test_no_provider_configured_returns_error() -> None:
    router = IntentRouter.__new__(IntentRouter)
    router._llm = None
    router._max_tokens = DEFAULT_MAX_TOKENS

    decision = await router.route("Any question at all")

    assert decision.route is Route.ERROR
    assert "no llm provider" in decision.reasoning.lower()


def test_prompt_tells_the_model_that_department_is_personal_info() -> None:
    from app.orchestration.intent import _SYSTEM_PROMPT

    assert "which department they belong to" in _SYSTEM_PROMPT
    assert '"which department I belong to"          -> PERSONAL_INFO / personal' in _SYSTEM_PROMPT


def test_fallback_message_is_the_retry_wording() -> None:
    from app.orchestration.intent import FALLBACK_MESSAGE

    assert "unable to generate the response" in FALLBACK_MESSAGE
    assert "switching to another LLM" in FALLBACK_MESSAGE


def test_router_settings_are_read_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 40)
    monkeypatch.setenv("ROUTER_MODEL", "deepseek/deepseek-r1-0528")
    monkeypatch.setenv("ROUTER_MAX_TOKENS", "2048")

    model, max_tokens = _configured_router()

    assert model == "deepseek/deepseek-r1-0528"
    assert max_tokens == 2048


def test_router_settings_tolerate_a_broken_environment(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "too-short")
    monkeypatch.delenv("ROUTER_MODEL", raising=False)

    assert _configured_router() == (None, None)


def test_injected_provider_skips_environment_lookup() -> None:
    provider = AsyncMock()
    router = IntentRouter(llm_provider=provider, max_tokens=64)

    assert router._llm is provider
    assert router._max_tokens == 64
