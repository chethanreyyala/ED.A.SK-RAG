"""Grounded answer generation.

Given a query and a list of retrieved chunks, produce an answer that cites them.
Supports Anthropic Claude (default) and OpenAI via configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from edask.config import get_settings
from edask.vector_store import SearchHit

_SYSTEM_PROMPT = """You are a precise, grounded assistant. Answer the user's \
question using ONLY the provided context passages. If the answer is not in the \
context, say so plainly. Cite passages inline using their bracketed index, e.g. [1], [2]. \
Keep answers tight."""


@dataclass(slots=True)
class GenerationResult:
    answer: str
    citations: list[SearchHit]


def _format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"[{i}] ({h.source}, chunk {h.index}, score={h.score:.3f})\n{h.text}")
    return "\n\n".join(blocks)


def _build_user_message(query: str, hits: list[SearchHit]) -> str:
    if not hits:
        return f"Question: {query}\n\nContext: (none — answer that you do not know)"
    return f"Context passages:\n\n{_format_context(hits)}\n\nQuestion: {query}"


def _generate_anthropic(query: str, hits: list[SearchHit]) -> str:
    from anthropic import Anthropic

    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(query, hits)}],
    )
    # concatenate text blocks
    return "".join(block.text for block in msg.content if block.type == "text")


def _generate_openai(query: str, hits: list[SearchHit]) -> str:
    from openai import OpenAI

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(query, hits)},
        ],
        max_tokens=1024,
    )
    return resp.choices[0].message.content or ""


def generate(query: str, hits: list[SearchHit]) -> GenerationResult:
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        answer = _generate_anthropic(query, hits)
    else:
        answer = _generate_openai(query, hits)
    return GenerationResult(answer=answer, citations=hits)
