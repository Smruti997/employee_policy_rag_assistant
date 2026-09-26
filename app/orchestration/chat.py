"""Chat orchestrator (Supervisor): intent router → tool dispatching."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone, timedelta

from app.auth.models import AuthenticatedUser
from app.core.observability import bind_turn, elapsed_ms, log_event, new_turn_id
from app.orchestration.intent import FALLBACK_MESSAGE, IntentRouter
from app.orchestration.models import ChatEvent, Route
from app.orchestration.tools import PersonalAssistantTool, PolicyRAGTool


# IST is UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))


def _greeting_by_ist_time() -> str:
    hour = datetime.now(_IST).hour
    if 5 <= hour < 12:
        period = "Morning"
    elif 12 <= hour < 17:
        period = "Afternoon"
    else:
        period = "Evening"
    return f"Hello! Good {period}. How can I assist you today?"


class ChatOrchestrator:
    """Supervisor Agent: Intent Router → Tool Dispatcher (PolicyRAGTool & PersonalAssistantTool)."""

    def __init__(
        self,
        intent_router: IntentRouter | None = None,
        rag_tool: PolicyRAGTool | None = None,
        personal_tool: PersonalAssistantTool | None = None,
        llm_provider=None,
        retriever=None,
    ) -> None:
        self.router = intent_router or IntentRouter()
        self.rag_tool = rag_tool or PolicyRAGTool(
            retriever=retriever, llm_provider=llm_provider
        )
        self.personal_tool = personal_tool or PersonalAssistantTool()

    async def respond(
        self,
        text: str,
        user: AuthenticatedUser,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        bind_turn(new_turn_id())
        turn_started = time.monotonic()
        log_event(
            "turn.start",
            user_id=user.user_id,
            department=user.department,
            level=user.level,
            query=text,
            history_turns=len(history or []),
        )

        # Step 1: Supervisor Router checks intent
        router_started = time.monotonic()
        decision = await self.router.route(text)
        log_event(
            "router.result",
            route=decision.route.value,
            domain_category=decision.domain_category.value,
            reasoning=decision.reasoning,
            latency_ms=elapsed_ms(router_started),
        )

        try:
            # Step 2: Route dispatch
            if decision.route is Route.ERROR:
                log_event("tool.selected", tool="none")
                yield ChatEvent(text=FALLBACK_MESSAGE, decision=decision, is_error=True)
                return

            if decision.route is Route.GREETING:
                log_event("tool.selected", tool="greeting")
                yield ChatEvent(text=_greeting_by_ist_time(), decision=decision)
                return

            if decision.route is Route.CLOSURE:
                log_event("tool.selected", tool="closure")
                yield ChatEvent(
                    text="Thank you for connecting with us. Feel free to reach out any time.",
                    decision=decision,
                )
                return

            if decision.route is Route.PERSONAL_INFO:
                log_event("tool.selected", tool="personal_assistant")
                async for event in self.personal_tool.execute(
                    user, decision, question=text, history=history
                ):
                    yield event
                return

            if decision.route is Route.RAG:
                log_event("tool.selected", tool="policy_rag")
                async for event in self.rag_tool.execute(text, user, decision, history=history):
                    yield event
                return

            if decision.route is Route.BOTH:
                log_event("tool.selected", tool="personal_assistant+policy_rag")
                async for event in self.personal_tool.execute(
                    user, decision, question=text, history=history
                ):
                    yield event
                async for event in self.rag_tool.execute(text, user, decision, history=history):
                    yield event
                return

            # Direct response for non-relevant / unsupported topics
            log_event("tool.selected", tool="none")
            yield ChatEvent(text="unable to provide answer", decision=decision)
        finally:
            log_event(
                "turn.end",
                route=decision.route.value,
                latency_ms=elapsed_ms(turn_started),
            )
