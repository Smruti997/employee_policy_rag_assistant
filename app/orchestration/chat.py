"""LLM intent supervisor and policy/personal tool orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.auth.models import AuthenticatedUser
from app.orchestration.intent import IntentRouter
from app.orchestration.models import ChatEvent, Route
from app.orchestration.tools import PersonalAssistantTool, PolicyRAGTool


class ChatOrchestrator:
    """Classify each turn with an LLM, then run only the required tool(s)."""

    def __init__(
        self,
        intent_router: IntentRouter | None = None,
        rag_tool: PolicyRAGTool | None = None,
        personal_tool: PersonalAssistantTool | None = None,
        llm_provider=None,
        retriever=None,
    ) -> None:
        self.router = intent_router or IntentRouter(llm_provider=llm_provider)
        self.rag_tool = rag_tool or PolicyRAGTool(
            retriever=retriever, llm_provider=llm_provider
        )
        self.personal_tool = personal_tool or PersonalAssistantTool(llm_provider=llm_provider)

    async def respond(
        self,
        text: str,
        user: AuthenticatedUser,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        decision = await self.router.route(text)

        if decision.route is Route.GREETING:
            yield ChatEvent(text="Hello! How can I help with company policies or your employee information?", decision=decision)
            return
        if decision.route is Route.CLOSURE:
            yield ChatEvent(text="Thank you for connecting. Feel free to reach out any time.", decision=decision)
            return
        if decision.route is Route.DIRECT_RESPONSE:
            yield ChatEvent(text="I can help with company policies and your employee information.", decision=decision)
            return
        if decision.route is Route.PERSONAL_INFO:
            async for event in self.personal_tool.execute(user, decision, text, history):
                yield event
            return
        if decision.route is Route.RAG:
            async for event in self.rag_tool.execute(text, user, decision, history):
                yield event
            return
        if decision.route is Route.BOTH:
            # The MCP operations themselves run concurrently; policy retrieval runs
            # alongside them so the combined answer waits for neither in sequence.
            context, chunks = await asyncio.gather(
                self.personal_tool.get_employee_context(user.user_id),
                self.rag_tool.retrieve(text, user),
            )
            async for event in self.rag_tool.stream_answer(
                text, chunks, decision, history, employee_context=context
            ):
                yield event
