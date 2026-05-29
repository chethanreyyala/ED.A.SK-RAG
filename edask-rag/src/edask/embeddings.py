"""Embedding model wrapper — ED(A)SK component #1.

Converts text into dense vectors that capture semantic meaning. Wraps FastEmbed
so we can swap models via config without touching call sites.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

from fastembed import TextEmbedding

from edask.config import get_settings


class Embedder:
    """Thin facade over FastEmbed's TextEmbedding."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # FastEmbed exposes dimension only via a probe embed; cache it lazily.
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            probe = next(self._model.embed(["dimension probe"]))
            self._dim = len(probe)
        return self._dim

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts; preserves input order."""
        # FastEmbed returns a generator of numpy arrays
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return process-wide cached embedder; model load is expensive."""
    settings = get_settings()
    return Embedder(model_name=settings.embedding_model)
