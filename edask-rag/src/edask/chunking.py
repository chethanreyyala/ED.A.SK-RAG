"""Token-aware text chunking.

We use tiktoken's cl100k_base encoder as a stable token-counting proxy. The chunker
emits overlapping windows so semantic boundaries spanning chunk edges still get
indexed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


@dataclass(slots=True, frozen=True)
class Chunk:
    text: str
    index: int
    source: str


def _decode(tokens: list[int]) -> str:
    return _encoder().decode(tokens)


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Split ``text`` into overlapping token windows.

    Parameters
    ----------
    text: source string.
    source: identifier (e.g. filename) attached to every produced chunk.
    chunk_size: window size in tokens.
    overlap: number of tokens shared between adjacent windows.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _encoder().encode(text)
    if not tokens:
        return []

    stride = chunk_size - overlap
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(tokens), stride)):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(Chunk(text=_decode(window).strip(), index=i, source=source))
        if start + chunk_size >= len(tokens):
            break
    return chunks
