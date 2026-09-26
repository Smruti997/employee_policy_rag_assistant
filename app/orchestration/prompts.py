"""System prompt construction for RAG answers."""

from __future__ import annotations

from app.data.qdrant_store import SearchResult


_PERSONAL_SYSTEM = """\
You are a corporate assistant helping an employee with personal information.
Answer using ONLY the employee context provided below.
If the context does not contain the answer, say:
"I don't have that information in your employee profile."

Do NOT speculate or draw on outside knowledge.

---
EMPLOYEE CONTEXT:
{context}
---
"""


_RAG_SYSTEM = """\
You are a corporate assistant for internal policy questions.
Answer ONLY using the policy excerpts provided below.
If the answer is not contained in the excerpts, say:
"I don't have enough information in the available policy documents to answer that."

Do NOT speculate, invent, or draw on outside knowledge.
Cite the source document and page number at the end of the response when you reference a specific policy rule.

---
POLICY EXCERPTS:
{context}
---
"""


def build_personal_messages(
    question: str,
    context: dict[str, object],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Return an OpenAI-style messages list for a personal-context completion."""
    context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
    system_prompt = _PERSONAL_SYSTEM.format(context=context_str)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages


def build_rag_messages(
    question: str,
    chunks: list[SearchResult],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Return an OpenAI-style messages list for a RAG completion.

    *history* contains the prior turns of the current conversation as
    ``{"role": ..., "content": ...}`` dicts (user and assistant only — no
    system messages). They are inserted between the system prompt and the
    current question so the LLM can resolve follow-up references.
    """
    if not chunks:
        context = "(No matching policy excerpts found for your access level.)"
    else:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[{i}] Source: {chunk.source_file}, Page {chunk.page}\n{chunk.text}"
            )
        context = "\n\n".join(parts)

    system_prompt = _RAG_SYSTEM.format(context=context)
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages
