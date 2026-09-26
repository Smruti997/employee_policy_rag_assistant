"""Retriever: embeds a question and fetches RBAC-filtered chunks from Qdrant."""

from __future__ import annotations

from app.auth.rbac import AccessScope
from app.data.embeddings import EmbeddingProvider, default_embedder
from app.data.qdrant_store import QdrantStore, SearchResult


class Retriever:
    """Encapsulates embedding + vector search. Requires an AccessScope for every query."""

    def __init__(
        self,
        store: QdrantStore | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._store = store or _default_store()
        self._embedder = embedder or default_embedder()

    def search(
        self,
        question: str,
        scope: AccessScope,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Embed *question*, apply RBAC filter, return top-*limit* chunks.

        The RBAC filter is passed directly to Qdrant — results are never
        filtered in Python after retrieval.
        """
        vectors = self._embedder.embed([question])
        query_vector = vectors[0]
        rbac_filter = scope.qdrant_filter()
        return self._store.search(query_vector, rbac_filter, limit=limit)


_STORE_INSTANCE: QdrantStore | None = None


def _default_store() -> QdrantStore:
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        import os
        from dotenv import load_dotenv

        load_dotenv()
        url = os.getenv("QDRANT_URL")
        path = os.getenv("QDRANT_PATH")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _STORE_INSTANCE = QdrantStore(url=url, api_key=api_key, path=path)
    return _STORE_INSTANCE


