"""CLI: ingest files or directories into the vector store.

Usage:
    uv run python scripts/ingest.py data/sample/
    uv run python scripts/ingest.py doc1.md doc2.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from edask.ingestion import ingest_paths

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    paths: Annotated[
        list[Path],
        typer.Argument(exists=True, help="Files or directories"),
    ],
) -> None:
    summary = ingest_paths(paths)
    console.print(
        f"[green]ingested[/green] files=[bold]{summary['files']}[/bold] "
        f"chunks=[bold]{summary['chunks']}[/bold] "
        f"upserted=[bold]{summary['upserted']}[/bold]"
    )


if __name__ == "__main__":
    app()
