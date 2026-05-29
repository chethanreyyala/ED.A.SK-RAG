"""Centralized, typed configuration loaded from the environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Vector store ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "edask"

    # --- Retrieval ---
    top_k: int = Field(default=4, ge=1, le=50)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Chunking ---
    chunk_size_tokens: int = Field(default=400, ge=50)
    chunk_overlap_tokens: int = Field(default=60, ge=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
