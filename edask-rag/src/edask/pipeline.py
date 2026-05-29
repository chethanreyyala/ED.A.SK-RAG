"""End-to-end RAG pipeline: retrieve then generate."""

from __future__ import annotations

from edask.generator import GenerationResult, generate
from edask.retriever import retrieve


def ask(query: str, *, top_k: int | None = None) -> GenerationResult:
    """Retrieve relevant context and produce a grounded answer."""
    hits = retrieve(query, top_k=top_k)
    return generate(query, hits)
