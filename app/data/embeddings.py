"""Embedding provider protocol and sentence-transformers implementation."""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Any object that can embed a batch of strings into float vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


class SentenceTransformerEmbedder:
    """CPU-based embedder using sentence-transformers/all-MiniLM-L6-v2.

    The model is downloaded once on first use (~90 MB) and cached by
    the sentence-transformers library in ~/.cache/huggingface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy load
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is None:
            # Guarded so several concurrent first calls load the model once.
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name, device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]


def default_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()
