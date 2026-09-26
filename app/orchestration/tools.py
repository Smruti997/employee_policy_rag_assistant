"""Orchestration Tools: Policy RAG Tool and Personal Assistant Tool."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app.auth.models import AuthenticatedUser
from app.core.observability import elapsed_ms, log_event
from app.orchestration.models import ChatEvent, RoutingDecision

logger = logging.getLogger(__name__)

# Number of policy chunks requested per RAG query.
_RAG_LIMIT = 5

# User-facing error texts. The real reason is logged, never shown to the client.
_RETRIEVAL_ERROR = "I couldn't retrieve policy information right now. Please try again."
_LLM_ERROR = "I couldn't generate an answer right now. Please try again."


class PolicyRAGTool:
    """RAG Tool: vector semantic retrieval + LLM streaming response."""

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

    async def execute(
        self,
        question: str,
        user: AuthenticatedUser,
        decision: RoutingDecision,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        from app.orchestration.prompts import build_rag_messages

        scope = user.access_scope
        log_event(
            "rag.input",
            question=question,
            history_turns=len(history or []),
            departments=list(scope.departments),
            max_access_level=scope.maximum_access_level,
            limit=_RAG_LIMIT,
        )

        retrieve_started = time.monotonic()
        try:
            chunks = await self._get_retriever().search(question, scope, limit=_RAG_LIMIT)
        except Exception as exc:
            log_event(
                "rag.retrieve.error",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed_ms(retrieve_started),
            )
            logger.exception("Policy retrieval failed")
            yield ChatEvent(text=_RETRIEVAL_ERROR, decision=decision, is_error=True)
            return

        log_event(
            "rag.retrieved",
            chunk_count=len(chunks),
            latency_ms=elapsed_ms(retrieve_started),
            sources=[
                {"source_file": chunk.source_file, "page": chunk.page, "score": chunk.score}
                for chunk in chunks
            ],
        )

        messages = build_rag_messages(question, chunks, history=history)
        answer: list[str] = []
        llm_started = time.monotonic()
        try:
            async for token in self._get_llm().stream(messages):
                answer.append(token)
                yield ChatEvent(text=token, decision=decision)
        except Exception as exc:
            log_event(
                "rag.output.error",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed_ms(llm_started),
            )
            logger.exception("Policy answer generation failed")
            yield ChatEvent(text=_LLM_ERROR, decision=decision, is_error=True)
            return

        log_event(
            "rag.output",
            answer="".join(answer),
            chars=sum(len(part) for part in answer),
            latency_ms=elapsed_ms(llm_started),
        )


class PersonalAssistantTool:
    """Personal Assistant Tool: fetches employee context via get_employee_context(user_id).

    Internally calls three mock MCP operations in parallel (~1 s total)
    then asks the LLM to compose a natural-language answer.
    """

    def __init__(self, llm_provider=None) -> None:
        self._llm = llm_provider

    def _get_llm(self):
        if self._llm is None:
            from app.orchestration.llm import provider_from_env

            self._llm = provider_from_env()
        return self._llm

    async def get_employee_context(self, user_id: str) -> dict[str, Any]:
        """Fetch employee context (profile, manager, team) in parallel."""
        from app.data.mock_mcp_client import get_employee_context

        return await get_employee_context(user_id)

    async def execute(
        self,
        user: AuthenticatedUser,
        decision: RoutingDecision,
        question: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        from app.data.sample_data import TABLE_FILES
        from app.orchestration.prompts import build_personal_messages

        log_event(
            "employee_db.read.start",
            user_id=user.user_id,
            mode="parallel",
            tables=list(TABLE_FILES),
        )
        db_started = time.monotonic()
        try:
            context = await self.get_employee_context(user.user_id)
        except Exception as exc:
            log_event(
                "employee_db.read.error",
                user_id=user.user_id,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed_ms(db_started),
            )
            # Keep the internal reason in the logs; the client gets a safe message.
            logger.exception("Employee context lookup failed for user %s", user.user_id)
            yield ChatEvent(
                text="I couldn't retrieve your employee details right now. Please try again.",
                decision=decision,
                is_error=True,
            )
            return

        log_event(
            "employee_db.read.done",
            user_id=user.user_id,
            latency_ms=elapsed_ms(db_started),
            context=context,
        )

        # Department comes from the verified JWT, not the employee tables.
        context["department"] = user.department

        if not question:
            parts = [f"{k}: {v}" for k, v in context.items()]
            yield ChatEvent(text="\n".join(parts), decision=decision)
            return

        messages = build_personal_messages(question, context, history=history)
        answer: list[str] = []
        llm_started = time.monotonic()
        try:
            async for token in self._get_llm().stream(messages):
                answer.append(token)
                yield ChatEvent(text=token, decision=decision)
        except Exception as exc:
            log_event(
                "personal.output.error",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=elapsed_ms(llm_started),
            )
            logger.exception("Personal answer generation failed")
            yield ChatEvent(text=_LLM_ERROR, decision=decision, is_error=True)
            return

        log_event(
            "personal.output",
            answer="".join(answer),
            latency_ms=elapsed_ms(llm_started),
        )
