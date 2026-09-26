"""Intent routing via a single LLM classification call.

One completion classifies the message into a route and a domain category, and
the JSON object is parsed back into a :class:`RoutingDecision`. The prompt is
deliberately explicit that a bare topic phrase ("long term incentives",
"strategic plan") is a policy question: an earlier decision-model router sent
those to ``DIRECT_RESPONSE`` and they never reached retrieval.

Classification never raises into the request path. If no provider is configured,
the call fails, or the reply is not parsable, the router returns ``Route.ERROR``;
the orchestrator turns that into a user-facing retry message
(:data:`FALLBACK_MESSAGE`). A question is never answered from a route it could
not classify.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.orchestration.llm import LLMProvider, complete
from app.orchestration.models import DomainCategory, Route, RoutingDecision

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

# A non-thinking instruct model answers a six-way classification in well under a
# second and never needs the reasoning budget a thinking model (like DeepSeek R1)
# spends before emitting content. Keep the cap bounded anyway: OpenRouter
# pre-authorises max_tokens against the account balance, and leaving it unset
# requests the model's full window (HTTP 402 on a low-balance key). If you switch
# ROUTER_MODEL to a reasoning model, raise this to ~4000 — a thinking model that
# is cut short returns empty content with no error, which the router can only
# treat as unparsable.
DEFAULT_MAX_TOKENS = 512

# Shown to the client when classification cannot produce a route. The request is
# not retried: the router must never answer a question it could not classify.
FALLBACK_MESSAGE = (
    "I am unable to generate the response at this moment, "
    "please try after sometime or try with switching to another LLM"
)

_SYSTEM_PROMPT = """\
You are the intent router for Acme Corp's internal policy chatbot.

Classify the user's message into exactly one route:

- RAG: any question about company policy or one of its terms, however terse.
  This covers HR (leave, sick leave, code of conduct, performance review),
  finance (expenses, receipts, per diem, travel, reimbursement) and executive
  (compensation, long term incentives, clawback, strategic plan) topics.
  A bare topic phrase is RAG: "long term incentives", "strategic plan" and
  "per diem" are all RAG, not DIRECT_RESPONSE.
- PERSONAL_INFO: a question about the user's own record — their name, grade,
  manager, team, or which department they belong to.
- BOTH: the user's own record together with a policy, for example a policy
  amount that depends on the user's grade.
- GREETING: hello, hi, hey, good morning/afternoon/evening.
- CLOSURE: bye, goodbye, thanks, that's all, no more questions.
- DIRECT_RESPONSE: only for messages clearly unrelated to company policy and the
  user's own record, such as geography, sport, weather, or small talk.

Also choose the primary domain: hr, finance, exec, personal, or general.

Examples:
"who is my manager"                     -> PERSONAL_INFO / personal
"which department I belong to"          -> PERSONAL_INFO / personal
"what long term incentives"             -> RAG / exec
"tell me about strategic plan"          -> RAG / exec
"whats the expense policy"              -> RAG / finance
"how much leave do I get given my grade"-> BOTH / hr
"hi there"                              -> GREETING / general
"what is the capital of France"         -> DIRECT_RESPONSE / general

Reply with ONLY a JSON object, no prose and no code fences:
{"route": "...", "domain_category": "...", "reasoning": "one short sentence", "keywords": ["..."]}
"""


def _normalise(label: Any) -> str:
    """Normalise a returned label so minor casing or spacing drift still matches."""
    return str(label or "").strip().upper().replace(" ", "_").replace("-", "_")


def _configured_router() -> tuple[str | None, int | None]:
    """Read router settings, tolerating an unconfigured or incomplete environment."""
    try:
        from app.core.config import Settings

        settings = Settings.from_environment()
    except Exception:
        return None, None
    return settings.router_model, settings.router_max_tokens


def parse_classification(raw: str | None) -> dict[str, Any] | None:
    """Extract the JSON object from a model reply, tolerating fences and prose."""
    if not raw:
        return None

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:].strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class IntentRouter:
    """Classify intent with one LLM call."""

    _ROUTE_MAP: dict[str, Route] = {route.value: route for route in Route}
    _CATEGORY_MAP: dict[str, DomainCategory] = {
        _normalise(category.value): category for category in DomainCategory
    }

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        configured_model, configured_tokens = _configured_router()
        self._max_tokens = max_tokens or configured_tokens or DEFAULT_MAX_TOKENS

        if llm_provider is None:
            from app.orchestration.llm import provider_from_env

            try:
                llm_provider = provider_from_env(
                    model=model or configured_model or DEFAULT_MODEL
                )
            except Exception as exc:
                logger.warning("No LLM provider available for intent routing: %s", exc)

        self._llm = llm_provider

    def _build_messages(self, message: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

    async def route(self, message: str) -> RoutingDecision:
        """Classify *message*, falling back to ``RAG`` if that is not possible."""
        if self._llm is None:
            return _fallback("No LLM provider is configured for intent routing.")

        try:
            raw = await complete(
                self._llm, self._build_messages(message), max_tokens=self._max_tokens
            )
        except Exception as exc:
            logger.warning("Intent classification call failed: %s: %s", type(exc).__name__, exc)
            return _fallback("Intent classification call failed; defaulting to policy retrieval.")

        parsed = parse_classification(raw)
        if parsed is None:
            logger.warning("Intent classification was not parsable: %r", (raw or "")[:200])
            return _fallback("Intent classification was not parsable; defaulting to policy retrieval.")

        category = self._CATEGORY_MAP.get(
            _normalise(parsed.get("domain_category")), DomainCategory.GENERAL
        )
        route = self._ROUTE_MAP.get(_normalise(parsed.get("route")))
        if route is None:
            logger.warning("Intent classification returned an unknown route: %r", parsed.get("route"))
            return _fallback(
                f"Intent classification returned an unknown route "
                f"{parsed.get('route')!r}; defaulting to policy retrieval.",
                category,
            )

        reasoning = parsed.get("reasoning")
        return RoutingDecision(
            route=route,
            reasoning=reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else f"Classified as {route.value}.",
            extracted_keywords=_keywords(parsed.get("keywords")),
            domain_category=category,
        )


def _keywords(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, (str, int)) and str(item).strip())


def _fallback(reason: str, category: DomainCategory = DomainCategory.GENERAL) -> RoutingDecision:
    return RoutingDecision(
        route=Route.ERROR,
        reasoning=reason,
        extracted_keywords=(),
        domain_category=category,
    )
