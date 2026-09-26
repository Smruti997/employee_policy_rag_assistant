"""Models exchanged within the chat orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    RAG = "RAG"
    PERSONAL_INFO = "PERSONAL_INFO"
    BOTH = "BOTH"
    GREETING = "GREETING"
    CLOSURE = "CLOSURE"
    DIRECT_RESPONSE = "DIRECT_RESPONSE"
    ERROR = "ERROR"


class DomainCategory(str, Enum):
    HR = "hr"
    FINANCE = "finance"
    EXEC = "exec"
    PERSONAL = "personal"
    GENERAL = "general"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    route: Route
    reasoning: str
    extracted_keywords: tuple[str, ...]
    domain_category: DomainCategory

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-compatible schema used by a future LLM router."""
        return {
            "route": self.route.value,
            "reasoning": self.reasoning,
            "extracted_keywords": list(self.extracted_keywords),
            "domain_category": self.domain_category.value,
        }


@dataclass(frozen=True, slots=True)
class ChatEvent:
    """An outgoing event emitted by the orchestration layer."""

    text: str
    decision: RoutingDecision
    is_error: bool = False
