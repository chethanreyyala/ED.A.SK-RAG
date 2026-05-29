"""CLI: ask a question against the indexed corpus.

Usage:
    uv run python scripts/query.py "What is a vector store?"
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from edask.pipeline import ask

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Override retriever top-k"),
) -> None:
    result = ask(question, top_k=top_k)

    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))

    if result.citations:
        console.print("\n[bold]Citations[/bold]")
        for i, c in enumerate(result.citations, start=1):
            preview = (c.text[:140] + "…") if len(c.text) > 140 else c.text
            console.print(
                f"  [cyan][{i}][/cyan] {c.source} (chunk {c.index}, score {c.score:.3f})"
            )
            console.print(f"      {preview}")


if __name__ == "__main__":
    app()
