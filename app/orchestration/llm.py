"""Pluggable LLM streaming provider protocol and implementations.

Providers supported:
- OpenRouter (default — uses OPENROUTER_API_KEY + OPENROUTER_MODEL)
- Groq        (GROQ_API_KEY + GROQ_MODEL)
- Gemini      (GEMINI_API_KEY + GEMINI_MODEL)
- Ollama      (OLLAMA_BASE_URL + OLLAMA_MODEL, fully local)

All implement the same LLMProvider protocol:
    async def stream(messages, max_tokens=None) -> AsyncIterator[str]

``max_tokens`` is optional but important on metered providers: a request that
omits it asks for the model's full output window, and OpenRouter pre-authorises
that many tokens, which a low-balance key cannot cover (HTTP 402).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


async def complete(
    provider: LLMProvider,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
) -> str:
    """Return a full completion by collecting a provider's stream.

    One-shot callers (like the intent router) want the whole answer; collecting
    the stream keeps a single provider interface rather than a second one.
    """
    parts: list[str] = []
    async for token in provider.stream(messages, max_tokens=max_tokens):
        parts.append(token)
    return "".join(parts)


# ---------------------------------------------------------------------------
# OpenRouter (default)
# ---------------------------------------------------------------------------

class OpenRouterProvider:
    """Stream text tokens via OpenRouter's OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/miniragchatbot",
        }
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", self.BASE_URL, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data = json.loads(line)
                        token = data["choices"][0]["delta"].get("content") or ""
                        if token:
                            yield token
                    except (KeyError, json.JSONDecodeError):
                        continue


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

class GroqProvider:
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        import httpx

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload: dict[str, object] = {"model": self._model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", self.BASE_URL, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        token = json.loads(line)["choices"][0]["delta"].get("content") or ""
                        if token:
                            yield token
                    except (KeyError, json.JSONDecodeError):
                        continue


# ---------------------------------------------------------------------------
# Gemini (via REST)
# ---------------------------------------------------------------------------

class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        import httpx

        # Convert OpenAI-style messages to Gemini contents
        contents = [{"role": m["role"] if m["role"] != "system" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:streamGenerateContent?alt=sse&key={self._api_key}"
        )
        payload: dict[str, object] = {"contents": contents}
        if max_tokens is not None:
            payload["generationConfig"] = {"maxOutputTokens": max_tokens}
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            token = (
                                data["candidates"][0]["content"]["parts"][0]["text"]
                            )
                            if token:
                                yield token
                        except (KeyError, json.JSONDecodeError):
                            continue


# ---------------------------------------------------------------------------
# Ollama (fully local)
# ---------------------------------------------------------------------------

class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def stream(
        self, messages: list[dict[str, str]], max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        import httpx

        url = f"{self._base_url}/api/chat"
        payload: dict[str, object] = {"model": self._model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content") or ""
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue


# ---------------------------------------------------------------------------
# Factory: build provider from environment
# ---------------------------------------------------------------------------

def provider_from_env(model: str | None = None) -> LLMProvider:
    """Return the first configured LLM provider found in environment variables.

    *model* overrides the provider's default model, so the intent router can run
    on a different model than the answer path.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    if key := os.getenv("OPENROUTER_API_KEY"):
        model = model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        return OpenRouterProvider(api_key=key, model=model)

    if key := os.getenv("GROQ_API_KEY"):
        model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqProvider(api_key=key, model=model)

    if key := os.getenv("GEMINI_API_KEY"):
        model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiProvider(api_key=key, model=model)

    if base_url := os.getenv("OLLAMA_BASE_URL"):
        model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        return OllamaProvider(base_url=base_url, model=model)

    raise RuntimeError(
        "No LLM provider configured. Set one of: OPENROUTER_API_KEY, GROQ_API_KEY, "
        "GEMINI_API_KEY, or OLLAMA_BASE_URL in your .env file."
    )
