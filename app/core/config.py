"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    jwt_secret: str
    llm_provider: str = "openrouter"
    llm_api_key: str = ""
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_path: str = "./app/qdrant_db"
    qdrant_url: str = "http://localhost:6333"

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        secret = os.getenv("JWT_SECRET", "")
        if len(secret) < 32:
            raise ConfigurationError("JWT_SECRET must be at least 32 characters long")

        provider = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
        provider_settings = {
            "openrouter": (os.getenv("OPENROUTER_API_KEY", ""), os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"), "https://openrouter.ai/api/v1"),
            "groq": (os.getenv("GROQ_API_KEY", ""), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "https://api.groq.com/openai/v1"),
            "gemini": (os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), "https://generativelanguage.googleapis.com"),
            "ollama": ("", os.getenv("OLLAMA_MODEL", "llama3.2"), os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
        }
        llm_api_key, llm_model, llm_base_url = provider_settings.get(
            provider, ("", "", "")
        )
        qdrant_path = os.getenv("QDRANT_PATH", "./app/qdrant_db")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        return cls(
            jwt_secret=secret,
            llm_provider=provider,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
            qdrant_path=qdrant_path,
            qdrant_url=qdrant_url,
        )


