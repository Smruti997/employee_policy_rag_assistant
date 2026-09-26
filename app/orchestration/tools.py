"""Application tools used by the chat orchestrator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.auth.models import AuthenticatedUser
from app.core.exceptions import ChatServiceError
from app.data.qdrant_store import SearchResult
from app.orchestration.models import ChatEvent, RoutingDecision

logger = logging.getLogger(__name__)


class PolicyRAGTool:
    """Retrieve only authorized policy chunks and stream a grounded answer."""

    def __init__(self, retriever=None, llm_provider=None) -> None:
        self._retriever = retriever
        self._llm = llm_provider

    def _get_retriever(self):
        if self._retriever is None:
            from app.data.retriever import Retriever

            self._retriever = Retriever()
        return self._retriever

    def _get_llm(self):
        if self._llm is None:
            from app.orchestration.llm import provider_from_env

            self._llm = provider_from_env()
        return self._llm

    async def retrieve(self, question: str, user: AuthenticatedUser) -> list[SearchResult]:
        """Run CPU embedding and synchronous Qdrant calls outside the event loop."""
        try:
            def search() -> list[SearchResult]:
                return self._get_retriever().search(question, user.access_scope, 5)

            return await asyncio.to_thread(search)
        except Exception as exc:
            logger.exception("Policy retrieval failed")
            raise ChatServiceError("Policy search is temporarily unavailable. Please try again.") from exc

    async def stream_answer(
        self,
        question: str,
        chunks: list[SearchResult],
        decision: RoutingDecision,
        history: list[dict[str, str]] | None = None,
        employee_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        from app.orchestration.prompts import build_rag_messages

        messages = build_rag_messages(
            question, chunks, history=history, employee_context=employee_context
        )
        try:
            async for token in self._get_llm().stream(messages):
                yield ChatEvent(text=token, decision=decision)
        except Exception as exc:
            logger.exception("Policy answer generation failed")
            raise ChatServiceError("I couldn't generate an answer right now. Please try again.") from exc

    async def execute(
        self,
        question: str,
        user: AuthenticatedUser,
        decision: RoutingDecision,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        chunks = await self.retrieve(question, user)
        async for event in self.stream_answer(question, chunks, decision, history):
            yield event


class PersonalAssistantTool:
    """Fetch the authenticated employee's mock context through parallel operations."""

    def __init__(self, llm_provider=None) -> None:
        self._llm = llm_provider

    def _get_llm(self):
        if self._llm is None:
            from app.orchestration.llm import provider_from_env

            self._llm = provider_from_env()
        return self._llm

    async def get_employee_context(self, user_id: str) -> dict[str, Any]:
        from app.data.mock_mcp_client import get_employee_context

        try:
            return await get_employee_context(user_id)
        except Exception as exc:
            logger.exception("Employee context lookup failed")
            raise ChatServiceError("Employee information is temporarily unavailable. Please try again.") from exc

    async def stream_answer(
        self,
        question: str,
        context: dict[str, Any],
        decision: RoutingDecision,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        from app.orchestration.prompts import build_personal_messages

        try:
            async for token in self._get_llm().stream(
                build_personal_messages(question, context, history=history)
            ):
                yield ChatEvent(text=token, decision=decision)
        except Exception as exc:
            logger.exception("Personal answer generation failed")
            raise ChatServiceError("I couldn't generate an answer right now. Please try again.") from exc

    async def execute(
        self,
        user: AuthenticatedUser,
        decision: RoutingDecision,
        question: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        context = await self.get_employee_context(user.user_id)
        async for event in self.stream_answer(question, context, decision, history):
            yield event
