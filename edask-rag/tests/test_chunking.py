"""Tests for the token-aware chunker."""

from __future__ import annotations

import pytest

from edask.chunking import chunk_text


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_text("", source="x", chunk_size=100, overlap=10) == []


def test_short_input_yields_single_chunk() -> None:
    chunks = chunk_text("hello world", source="x", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].source == "x"
    assert chunks[0].index == 0
    assert "hello" in chunks[0].text


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", source="x", chunk_size=10, overlap=10)


def test_long_input_is_split_with_overlap() -> None:
    # construct content that's clearly multi-window: ~2000 tokens worth
    text = ("ED(A)SK Retrieval Augmented Generation. " * 400).strip()
    chunks = chunk_text(text, source="big", chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(c.source == "big" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))

    # adjacent chunks should share content from the overlap region
    assert any(word in chunks[1].text for word in chunks[0].text.split()[-5:])
