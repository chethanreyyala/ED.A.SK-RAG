"""Document ingestion: read files → chunk → embed → upsert to vector store."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from edask.chunking import chunk_text
from edask.config import get_settings
from edask.embeddings import get_embedder
from edask.vector_store import StoredChunk, get_vector_store

# Files we know how to read as plain text. PDFs etc. are out of scope for v1.
_TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in _TEXT_SUFFIXES:
                    yield child
        elif p.is_file():
            yield p


def ingest_paths(paths: Iterable[Path]) -> dict[str, int]:
    """Ingest the given files/directories. Returns a summary dict."""
    settings = get_settings()
    embedder = get_embedder()
    store = get_vector_store()

    all_chunks: list[StoredChunk] = []
    for path in _iter_files(paths):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(
            text,
            source=str(path),
            chunk_size=settings.chunk_size_tokens,
            overlap=settings.chunk_overlap_tokens,
        )
        all_chunks.extend(
            StoredChunk(text=c.text, source=c.source, index=c.index) for c in chunks
        )

    if not all_chunks:
        return {"files": 0, "chunks": 0, "upserted": 0}

    vectors = embedder.embed(ch.text for ch in all_chunks)
    upserted = store.upsert(all_chunks, vectors)

    file_count = len({ch.source for ch in all_chunks})
    return {"files": file_count, "chunks": len(all_chunks), "upserted": upserted}
