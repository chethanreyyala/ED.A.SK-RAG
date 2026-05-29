"""Retriever — ED(A)SK component #3.

Takes a user query, embeds it, and returns the top-k chunks from the vector
store ranked by cosine similarity.
"""

from __future__ import annotations

from edask.config import get_settings
from edask.embeddings import get_embedder
from edask.vector_store import SearchHit, get_vector_store


def retrieve(query: str, *, top_k: int | None = None) -> list[SearchHit]:
    settings = get_settings()
    embedder = get_embedder()
    store = get_vector_store()

    query_vector = embedder.embed_query(query)
    return store.search(
        query_vector=query_vector,
        top_k=top_k or settings.top_k,
        score_threshold=settings.score_threshold,
    )
