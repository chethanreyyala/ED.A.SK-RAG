"""Tests for the pipeline orchestration layer.

These tests stub out external services (Qdrant, the LLM) so they can run in CI
without any infra. They verify that the retriever output is correctly threaded
into the generator's prompt.
"""

from __future__ import annotations

from unittest.mock import patch

from edask import pipeline
from edask.generator import GenerationResult
from edask.vector_store import SearchHit


def _fake_hits() -> list[SearchHit]:
    return [
        SearchHit(text="A vector store persists embeddings.", source="docs.md", score=0.91, index=0),
        SearchHit(text="Cosine similarity ranks vectors by angle.", source="docs.md", score=0.83, index=2),
    ]


def test_ask_threads_retrieved_hits_into_generator() -> None:
    fake_hits = _fake_hits()
    captured = {}

    def fake_generate(query: str, hits: list[SearchHit]) -> GenerationResult:
        captured["query"] = query
        captured["hits"] = hits
        return GenerationResult(answer="stubbed answer", citations=hits)

    with (
        patch("edask.pipeline.retrieve", return_value=fake_hits),
        patch("edask.pipeline.generate", side_effect=fake_generate),
    ):
        result = pipeline.ask("What is a vector store?")

    assert result.answer == "stubbed answer"
    assert result.citations == fake_hits
    assert captured["query"] == "What is a vector store?"
    assert captured["hits"] == fake_hits


def test_ask_passes_top_k_through() -> None:
    with (
        patch("edask.pipeline.retrieve", return_value=[]) as mock_retrieve,
        patch("edask.pipeline.generate", return_value=GenerationResult(answer="", citations=[])),
    ):
        pipeline.ask("anything", top_k=7)

    mock_retrieve.assert_called_once_with("anything", top_k=7)
