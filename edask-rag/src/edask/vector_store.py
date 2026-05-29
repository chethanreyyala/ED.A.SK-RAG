"""Qdrant vector store wrapper — ED(A)SK component #2.

Stores embeddings + payload and exposes a cosine-similarity k-NN search. We
recreate the collection only when the dimension differs from the embedder.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from edask.config import get_settings
from edask.embeddings import get_embedder


@dataclass(slots=True)
class StoredChunk:
    text: str
    source: str
    index: int


@dataclass(slots=True)
class SearchHit:
    text: str
    source: str
    score: float
    index: int


class VectorStore:
    """Qdrant-backed store of (vector, payload) pairs."""

    def __init__(self, client: QdrantClient, collection: str, dim: int) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            info = self.client.get_collection(self.collection)
            current_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            if current_dim == self.dim:
                return
            # dimension drift — rebuild
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": ch.text, "source": ch.source, "index": ch.index},
            )
            for ch, vec in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        """k-NN cosine similarity search — ED(A)SK component #3."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for p in result.points:
            payload = p.payload or {}
            hits.append(
                SearchHit(
                    text=payload.get("text", ""),
                    source=payload.get("source", "unknown"),
                    score=float(p.score),
                    index=int(payload.get("index", -1)),
                )
            )
        return hits

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def reset(self) -> None:
        self.client.delete_collection(self.collection)
        self._ensure_collection()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    return VectorStore(
        client=client,
        collection=settings.qdrant_collection,
        dim=get_embedder().dimension,
    )
