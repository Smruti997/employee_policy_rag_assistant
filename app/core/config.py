"""Runtime configuration loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.core.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class Settings:
    jwt_secret: str
    router_model: str = "qwen/qwen3-30b-a3b-instruct-2507"
    router_max_tokens: int = 512
    qdrant_path: str = "./app/qdrant_db"
    qdrant_url: str = "http://localhost:6333"

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv()
        secret = os.getenv("JWT_SECRET", "")
        if len(secret) < 32:
            raise ConfigurationError("JWT_SECRET must be at least 32 characters long")

        router_model = os.getenv("ROUTER_MODEL", "qwen/qwen3-30b-a3b-instruct-2507")
        router_max_tokens = _positive_int("ROUTER_MAX_TOKENS", 512)
        qdrant_path = os.getenv("QDRANT_PATH", "./app/qdrant_db")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        return cls(
            jwt_secret=secret,
            router_model=router_model,
            router_max_tokens=router_max_tokens,
            qdrant_path=qdrant_path,
            qdrant_url=qdrant_url,
        )


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")
    return value


