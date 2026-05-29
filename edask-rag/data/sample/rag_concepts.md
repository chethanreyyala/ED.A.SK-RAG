# ED(A)SK — RAG Concepts

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation (RAG) enhances language models by allowing them
to retrieve relevant documents from a knowledge base and generate responses
based on that information. It overcomes context-window limitations and improves
factual accuracy by grounding responses in external knowledge.

## Architecture Overview

A RAG pipeline has three core components plus a generator:

1. An **embedding model** that turns text into dense vectors.
2. A **vector store** that persists those vectors and serves similarity queries.
3. A **retriever** that, given a user query, fetches the most similar chunks
   using k-NN over a similarity metric (typically cosine).
4. A **generator** (the LLM) that produces an answer conditioned on the
   retrieved chunks.

## Embedding Model

An embedding model converts text, documents, or code into dense vectors —
numeric representations that capture semantic meaning. Texts with similar
meaning produce vectors that are close together in the embedding space, which
is what makes similarity comparison possible.

Popular choices include OpenAI's `text-embedding-3` family, BAAI's BGE models,
Cohere embed, and Voyage. For local, no-API workflows, BGE-small via FastEmbed
is a strong default.

## Vector Stores

Vector stores are specialized databases designed to store and manage the
numerical vectors (embeddings) used in similarity searches.

- **Storage**: they hold the embeddings of all documents, often alongside the
  original text and metadata as payload.
- **Efficient retrieval**: they are optimized for quickly retrieving the most
  similar vectors to a given query vector, typically using an Approximate
  Nearest Neighbor (ANN) index such as HNSW.

Examples include Qdrant, Weaviate, Pinecone, Milvus, Chroma, and pgvector.

## Retrieval via Similarity Search

Retrieval via similarity search means finding documents or information that
are similar to a given query based on their numerical representations.

- **Similarity search**: compare the embedding of the query against stored
  document embeddings, and return the most similar ones via k-Nearest
  Neighbors (k-NN).
- **Cosine similarity**: a common metric that measures the cosine of the angle
  between two vectors. A higher cosine similarity means the vectors are more
  similar in direction, regardless of magnitude.

## Common Extensions

- **Hybrid search** combines dense (semantic) and sparse (lexical, e.g. BM25)
  retrieval; results are merged via Reciprocal Rank Fusion.
- **Reranking** runs a cross-encoder over the top-k candidates from the
  retriever to produce a more accurate final ranking.
- **Chunking strategy** matters: token-aware chunking with overlap is a strong
  default for prose; structure-aware splitters work better for code or
  semi-structured documents.
