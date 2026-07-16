"""``coinfosim dataset ...`` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from coinfosim.cli.errors import UsageCLIError, fail, translate_exception
from coinfosim.datasets.catalog import get_dataset, list_datasets

app = typer.Typer(help="Inspect, download, and verify built-in dataset mirrors.")


@app.command("list")
def dataset_list(ctx: typer.Context) -> None:
    """List the built-in datasets and their local availability."""

    cli_ctx = ctx.obj
    from coinfosim.datasets.integrity import verify_dataset
    from coinfosim.datasets.resolver import default_dataset_cache_root

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Slug", style="cyan")
    table.add_column("Display name")
    table.add_column("License")
    table.add_column("Files")
    table.add_column("Local")
    for dataset in list_datasets():
        cache_dir = default_dataset_cache_root() / dataset.local_directory
        local = "yes" if verify_dataset(cache_dir, dataset).is_valid else "no"
        table.add_row(
            dataset.slug,
            dataset.display_name,
            dataset.license.name,
            str(len(dataset.files)),
            local,
        )
    cli_ctx.console.print(table)


@app.command("show")
def dataset_show(
    ctx: typer.Context, dataset: str = typer.Argument(..., help="Dataset slug or alias.")
) -> None:
    """Show provenance, citation, files, hashes, and Pages URLs for one dataset."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    try:
        definition = get_dataset(dataset)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    console.print(f"[bold cyan]{definition.display_name}[/bold cyan] [dim]({definition.slug})[/dim]")
    console.print(f"Source: {definition.source_url}")
    console.print(f"Citation: {definition.citation}")
    console.print(f"License: {definition.license.name}")
    if definition.license.url:
        console.print(f"License URL: {definition.license.url}")
    console.print(f"Notice: {definition.license.notice}")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Filename")
    table.add_column("SHA-256")
    table.add_column("Size (bytes)", justify="right")
    table.add_column("URL")
    for file in definition.files:
        table.add_row(file.filename, file.sha256, str(file.size_bytes), file.url)
    console.print(table)


@app.command("status")
def dataset_status(
    ctx: typer.Context, dataset: str = typer.Argument(..., help="Dataset slug or alias.")
) -> None:
    """Show resolution candidates and per-file verification for one dataset."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    try:
        definition = get_dataset(dataset)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    from coinfosim.datasets.integrity import verify_dataset
    from coinfosim.datasets.resolver import default_dataset_cache_root

    cache_dir = default_dataset_cache_root() / definition.local_directory
    console.print(f"Platform cache: {cache_dir}")
    result = verify_dataset(cache_dir, definition)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Filename")
    table.add_column("Status")
    for file_result in result.files:
        table.add_row(file_result.filename, file_result.status.value)
    console.print(table)
    console.print(
        f"Overall: [bold {'green' if result.is_valid else 'yellow'}]"
        f"{'valid' if result.is_valid else 'invalid or incomplete'}[/]"
    )


@app.command("fetch")
def dataset_fetch(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., help="Dataset slug or alias."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing valid file."),
    destination: Optional[Path] = typer.Option(
        None, "--destination", help="Advanced/testing: download into an explicit directory."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Advanced/testing: override the CoInfoSim Pages base URL."
    ),
) -> None:
    """Download and hash-verify a dataset into the platform cache."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    try:
        definition = get_dataset(dataset)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    from coinfosim.datasets.catalog import with_base_url
    from coinfosim.datasets.download import fetch_dataset
    from coinfosim.datasets.resolver import default_dataset_cache_root

    if base_url:
        definition = with_base_url(definition, base_url)
    target = destination if destination is not None else (
        default_dataset_cache_root() / definition.local_directory
    )

    console.print(f"Fetching [cyan]{definition.display_name}[/cyan] into {target}")
    result = fetch_dataset(definition, target, force=force, require_https=base_url is None)
    for file_result in result.files:
        style = "green" if file_result.ok else "red"
        console.print(f"  [{style}]{file_result.status.value}[/{style}] {file_result.filename}")
        if file_result.error:
            console.print(f"    [red]{file_result.error}[/red]")
    if not result.success:
        fail(
            console,
            UsageCLIError(f"failed to fetch one or more files for {definition.slug!r}"),
            debug=cli_ctx.debug,
        )
    console.print(f"[bold green]Done.[/bold green] {target}")


@app.command("verify")
def dataset_verify(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., help="Dataset slug or alias."),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Explicit directory to verify (default: resolve normally)."
    ),
) -> None:
    """Validate an explicit or resolved directory without downloading."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    try:
        definition = get_dataset(dataset)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    from coinfosim.datasets.integrity import verify_dataset
    from coinfosim.datasets.resolver import resolve_dataset_directory

    if data_dir is not None:
        directory = data_dir
    else:
        try:
            directory = resolve_dataset_directory(
                definition,
                config=cli_ctx.config,
                allow_download=False,
            )
        except Exception as exc:  # noqa: BLE001
            fail(console, translate_exception(exc), debug=cli_ctx.debug)

    result = verify_dataset(directory, definition)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Filename")
    table.add_column("Status")
    for file_result in result.files:
        table.add_row(file_result.filename, file_result.status.value)
    console.print(table)
    if not result.is_valid:
        fail(
            console,
            UsageCLIError(f"{directory} is not a valid copy of {definition.slug!r}"),
            debug=cli_ctx.debug,
        )
    console.print(f"[bold green]Valid.[/bold green] {directory}")


@app.command("path")
def dataset_path(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., help="Dataset slug or alias."),
    no_download: bool = typer.Option(
        False, "--no-download", help="Do not download; fail if not already verified locally."
    ),
) -> None:
    """Print the resolved, verified dataset directory (one path on stdout)."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    try:
        definition = get_dataset(dataset)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    from coinfosim.datasets.resolver import resolve_dataset_directory

    try:
        directory = resolve_dataset_directory(
            definition,
            config=cli_ctx.config,
            allow_download=not no_download,
        )
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    typer.echo(str(directory))
