"""Pluggable LLM streaming provider protocol and implementations.

Providers supported:
- OpenRouter (default — uses OPENROUTER_API_KEY + OPENROUTER_MODEL)
- Groq        (GROQ_API_KEY + GROQ_MODEL)
- Gemini      (GEMINI_API_KEY + GEMINI_MODEL)
- Ollama      (OLLAMA_BASE_URL + OLLAMA_MODEL, fully local)

All implement the same LLMProvider protocol:
    async def stream(messages) -> AsyncIterator[str]
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


# ---------------------------------------------------------------------------
# OpenRouter (default)
# ---------------------------------------------------------------------------

class OpenRouterProvider:
    """Stream text tokens via OpenRouter's OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/miniragchatbot",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
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

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        import httpx

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model, "messages": messages, "stream": True}
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

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        import httpx

        # Convert OpenAI-style messages to Gemini contents
        contents = [{"role": m["role"] if m["role"] != "system" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:streamGenerateContent?alt=sse&key={self._api_key}"
        )
        payload = {"contents": contents}
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

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        import httpx

        url = f"{self._base_url}/api/chat"
        payload = {"model": self._model, "messages": messages, "stream": True}
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

def provider_from_env() -> LLMProvider:
    """Build the explicitly selected LLM provider from environment variables."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    selected = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    if selected == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY for LLM_PROVIDER=openrouter")
        return OpenRouterProvider(key, os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"))
    if selected == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Set GROQ_API_KEY for LLM_PROVIDER=groq")
        return GroqProvider(key, os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    if selected == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY for LLM_PROVIDER=gemini")
        return GeminiProvider(key, os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    if selected == "ollama":
        return OllamaProvider(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            os.getenv("OLLAMA_MODEL", "llama3.2"),
        )
    raise RuntimeError("LLM_PROVIDER must be openrouter, groq, gemini, or ollama")
