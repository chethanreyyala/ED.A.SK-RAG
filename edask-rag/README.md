# ED(A)SK  A Minimal, Modern RAG

A small but production-shaped Retrieval-Augmented Generation system built around the three components from the *ED(A)SK* deck:

1. **Embedding model**  converts text into dense semantic vectors.
2. **Vector store**  persists embeddings and supports fast similarity search.
3. **Retrieval via similarity search** — k-NN over cosine similarity to fetch the most relevant chunks for a query, then ground an LLM on them.

The goal is to keep each concept visible (one file per concept) rather than hide it behind a single framework call, while still using the best-in-class tools for each layer.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language / packaging | Python 3.12 + [`uv`](https://github.com/astral-sh/uv) | Fast resolver, single-file lockfile |
| Embeddings | [`fastembed`](https://github.com/qdrant/fastembed) (`BAAI/bge-small-en-v1.5`) | Local, no API key, ONNX-quantized, ~133MB |
| Vector DB | [`qdrant`](https://qdrant.tech/) | Open-source, gRPC, HNSW, payload filtering |
| LLM | Anthropic Claude (default) / OpenAI (optional) | Long context, strong grounded-answer quality |
| API | FastAPI + Uvicorn | Typed, async, OpenAPI for free |
| UI | Streamlit | Cheapest path to a demo |
| Config | `pydantic-settings` | Env-driven, typed |
| Tests | `pytest` | — |
| Lint/format | `ruff` | One tool, fast |
| Container | `docker compose` (Qdrant only) | App runs on host for fast iteration |

---

## Quickstart

```bash
# 1. install deps
uv sync

# 2. start qdrant
docker compose up -d

# 3. set your API key
cp .env.example .env
# then edit .env and add ANTHROPIC_API_KEY=...

# 4. ingest the sample documents
uv run python scripts/ingest.py data/sample/

# 5. ask a question from the CLI
uv run python scripts/query.py "What is a vector store?"

# 6. or start the API
uv run uvicorn edask.api:app --reload

# 7. or the UI
uv run streamlit run ui/streamlit_app.py
```

---

## Architecture

```
                      ┌────────────────┐
   raw docs ─► chunk ─►   embeddings   ├──► vector store (Qdrant)
                      │ (BGE / FastEmbed)              │
                      └────────────────┘               │
                                                       │ top-k cosine
                                                       ▼
                                                ┌────────────┐
                                  user query ──►│  retriever │
                                                └─────┬──────┘
                                                      │ context
                                                      ▼
                                                ┌────────────┐
                                                │ generator  │──► answer
                                                │  (Claude)  │
                                                └────────────┘
```

Each box maps to one file under `src/edask/`:

| Slide concept | Module |
|---|---|
| Embedding model | `embeddings.py` |
| Vector store | `vector_store.py` |
| Similarity search (k-NN / cosine) | `retriever.py` |
| Grounded generation | `generator.py` |
| End-to-end orchestration | `pipeline.py` |

---

## Project layout

```
edask-rag/
├── docker-compose.yml         # Qdrant
├── pyproject.toml             # uv-managed deps
├── .env.example
├── src/edask/
│   ├── config.py              # env-driven settings
│   ├── chunking.py            # token-aware splitter
│   ├── embeddings.py          # FastEmbed wrapper
│   ├── vector_store.py        # Qdrant client wrapper
│   ├── ingestion.py           # docs → chunks → vectors → store
│   ├── retriever.py           # query → top-k chunks
│   ├── generator.py           # context + query → grounded answer
│   ├── pipeline.py            # one-call RAG
│   └── api.py                 # FastAPI
├── scripts/
│   ├── ingest.py
│   └── query.py
├── ui/streamlit_app.py
├── tests/
│   ├── test_chunking.py
│   └── test_pipeline.py
└── data/sample/
    └── rag_concepts.md        # sourced from the ED(A)SK deck
```

---

## Extending

- **Swap embeddings**: change `EMBEDDING_MODEL` in `.env`. FastEmbed supports BGE, e5, jina, nomic, and others — see their model list.
- **Swap LLM**: set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY`.
- **Hybrid search**: Qdrant supports sparse vectors — add a BM25 sparse encoder alongside dense to do reciprocal rank fusion.
- **Reranking**: drop a cross-encoder (e.g. `BAAI/bge-reranker-base`) between `retriever` and `generator` to re-score the top-k.

---

## License

MIT.
