"""FastAPI app: /health, /ingest, /query."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from edask.ingestion import ingest_paths
from edask.pipeline import ask
from edask.vector_store import get_vector_store

app = FastAPI(
    title="ED(A)SK RAG",
    version="0.1.0",
    description="Embeddings + vector store + similarity search + grounded generation.",
)


class IngestRequest(BaseModel):
    paths: list[str] = Field(..., description="Files or directories to ingest.")


class IngestResponse(BaseModel):
    files: int
    chunks: int
    upserted: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class Citation(BaseModel):
    source: str
    index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "stored_chunks": get_vector_store().count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    paths = [Path(p) for p in req.paths]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Path(s) not found: {missing}")
    summary = ingest_paths(paths)
    return IngestResponse(**summary)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    result = ask(req.question, top_k=req.top_k)
    return QueryResponse(
        answer=result.answer,
        citations=[
            Citation(source=h.source, index=h.index, score=h.score, text=h.text)
            for h in result.citations
        ],
    )
