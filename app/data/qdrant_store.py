"""Qdrant vector store: upsert chunks and RBAC-filtered semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4


COLLECTION_NAME = "policy_docs"


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    text: str
    source_file: str
    department: str
    access_level: int
    page: int
    score: float


class QdrantStore:
    """Thin wrapper around QdrantClient for policy-document storage and retrieval."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ) -> None:
        import os
        from qdrant_client import QdrantClient

        target_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        target_path = path or os.getenv("QDRANT_PATH", "./app/qdrant_db")

        # An explicit local path must not silently attach to a reachable server.
        if path:
            self._client = QdrantClient(path=path)
            self.location = path
            return

        # 1. Try server URL first if available
        try:
            client = QdrantClient(url=target_url, api_key=api_key or None, timeout=2.0)
            client.get_collections()
            self._client = client
            self.location = target_url
            return
        except Exception:
            pass

        # 2. Fall back to embedded disk database
        self._client = QdrantClient(path=target_path)
        self.location = target_path





    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def create_collection(self, dimension: int, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME in existing:
            if recreate:
                self._client.delete_collection(COLLECTION_NAME)
            else:
                return  # already exists — do nothing

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        # Create payload indexes for RBAC filtering at query time (not post-filter).
        self._client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="department",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="access_level",
            field_schema=PayloadSchemaType.INTEGER,
        )

    def upsert_chunks(
        self,
        chunks: list[Any],  # list[Chunk] from chunker
        embeddings: list[list[float]],
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "department": chunk.department,
                    "access_level": chunk.access_level,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        rbac_filter: dict[str, Any],
        limit: int = 5,
    ) -> list[SearchResult]:
        from qdrant_client.models import Filter

        res = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(**rbac_filter),
            limit=limit,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk_id=r.payload["chunk_id"],
                text=r.payload["text"],
                source_file=r.payload["source_file"],
                department=r.payload["department"],
                access_level=r.payload["access_level"],
                page=r.payload["page"],
                score=r.score,
            )
            for r in res.points
        ]

