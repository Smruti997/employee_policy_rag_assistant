"""Asynchronous LLM intent classification for authenticated chat turns."""

from __future__ import annotations

import json
import logging
import re

from app.core.exceptions import ChatServiceError
from app.orchestration.models import DomainCategory, Route, RoutingDecision

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Classify the employee's latest message for a company policy assistant.
Return one JSON object only with keys: route, domain_category, extracted_keywords.
When route is PERSONAL_INFO, the application will invoke its sole employee-data
tool, get_employee_context, using the authenticated user ID from the session.
Allowed routes: RAG, PERSONAL_INFO, BOTH, GREETING, CLOSURE, DIRECT_RESPONSE.
Allowed categories: hr, finance, exec, personal, general.
RAG is an official HR, finance, or executive policy question. PERSONAL_INFO asks
about this authenticated employee's mock profile, grade, manager, or team.
BOTH explicitly combines employee-specific facts and a policy question. GREETING
is only a greeting; CLOSURE is a farewell or thanks without another question.
DIRECT_RESPONSE is unrelated or unsupported. A policy's "manager approval" is
RAG, not PERSONAL_INFO. Do not infer that every occurrence of "my" is personal.
Keywords should be short phrases present in the message. Never answer the user."""

_ROUTES = {route.value: route for route in Route}
_CATEGORIES = {category.value: category for category in DomainCategory}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class IntentRouter:
    """Use the configured LLM provider to classify intent, without blocking the loop."""

    def __init__(self, llm_provider=None) -> None:
        self._llm = llm_provider

    def _provider(self):
        if self._llm is None:
            from app.orchestration.llm import provider_from_env

            self._llm = provider_from_env()
        return self._llm

    async def route(self, message: str) -> RoutingDecision:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]
        try:
            answer = "".join([part async for part in self._provider().stream(messages)])
            data = _parse_json(answer)
            route = _ROUTES[str(data.get("route", "")).strip().upper()]
            category = _CATEGORIES[str(data.get("domain_category", "general")).strip().lower()]
            keywords = data.get("extracted_keywords", [])
            if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
                raise ValueError("invalid extracted_keywords")
            return RoutingDecision(
                route=route,
                reasoning="Classified by the configured LLM router.",
                extracted_keywords=tuple(keywords[:8]),
                domain_category=category,
            )
        except Exception as exc:
            logger.warning("LLM intent classification failed", exc_info=True)
            raise ChatServiceError("I couldn't classify that request right now. Please try again.") from exc


def _parse_json(text: str) -> dict[str, object]:
    candidate = _FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("router returned no JSON object")
        data = json.loads(candidate[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("router response must be a JSON object")
    return data
