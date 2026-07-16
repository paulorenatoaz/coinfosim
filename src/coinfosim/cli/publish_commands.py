"""``coinfosim publish ...`` commands.

The ``gh-pages`` branch, remote, push, and dry-run controls are added in a
follow-up change alongside the rewritten publisher (see
``coinfosim.publish.publisher``); this module currently wraps the existing
index-generation entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from coinfosim.cli.errors import PackagingEnvironmentCLIError, fail

app = typer.Typer(help="Publish reports and datasets to GitHub Pages.")


@app.command("pages")
def publish_pages(
    ctx: typer.Context,
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Output directory containing reports/ and data/ (default: ~/coinfosim/output)."
    ),
    title: str = typer.Option("CoInfoSim Reports", "--title", help="Title for the generated index page."),
) -> None:
    """Regenerate the Pages index and publish via the configured publish script."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    from coinfosim.publish.publisher import publish_to_pages

    console.print("[bold cyan]Publishing to GitHub Pages[/bold cyan]")
    success = publish_to_pages(
        output_dir=str(output_dir) if output_dir is not None else None, title=title
    )
    if not success:
        fail(
            console,
            PackagingEnvironmentCLIError(
                "publishing failed; see the messages above for the underlying cause"
            ),
            debug=cli_ctx.debug,
        )
    console.print("[bold green]Published successfully.[/bold green]")
