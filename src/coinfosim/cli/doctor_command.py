"""``coinfosim doctor``: environment and dataset-cache diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import sys

import typer
from rich.table import Table


def register(app: typer.Typer) -> None:
    app.command("doctor")(doctor)


def doctor(
    ctx: typer.Context,
    fetch_missing: bool = typer.Option(
        False, "--fetch-missing", help="Download any missing built-in dataset before reporting."
    ),
) -> None:
    """Diagnose the CoInfoSim installation, environment, and dataset cache."""

    cli_ctx = ctx.obj
    console = cli_ctx.console

    from importlib.metadata import PackageNotFoundError, version as pkg_version

    import coinfosim
    from coinfosim.datasets.catalog import list_datasets
    from coinfosim.datasets.integrity import verify_dataset
    from coinfosim.datasets.resolver import default_dataset_cache_root
    from coinfosim.simulation.execution import resolve_start_method

    table = Table(show_header=False)
    table.add_column("Check", style="cyan")
    table.add_column("Value")

    try:
        coinfosim_version = pkg_version("coinfosim")
    except PackageNotFoundError:
        coinfosim_version = "0.0.0+dev (not installed as a distribution)"
    table.add_row("CoInfoSim version", coinfosim_version)
    table.add_row("Python version", sys.version.split()[0])
    table.add_row("Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    table.add_row("Package location", str(os.path.dirname(coinfosim.__file__)))
    table.add_row("Logical CPUs", str(os.cpu_count()))
    table.add_row("Config source", str(cli_ctx.config.get("_source", "defaults")))

    for name, requested in (("spawn", "spawn"), ("forkserver", "forkserver")):
        try:
            resolved = resolve_start_method(requested, backend="process")
            table.add_row(f"Start method {name!r} available", "yes" if resolved == requested else "no")
        except ValueError:
            table.add_row(f"Start method {name!r} available", "no")

    user_data_dir = default_dataset_cache_root()
    writable = _is_writable(user_data_dir)
    table.add_row("Dataset cache directory", f"{user_data_dir} ({'writable' if writable else 'NOT writable'})")

    output_dir = cli_ctx.config.get("paths", {}).get("output_dir") or "./output"
    table.add_row("Default output directory", f"{output_dir} ({'writable' if _is_writable_parent(output_dir) else 'check permissions'})")

    for display_name, distribution_name in (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "scikit-learn"),
        ("pandas", "pandas"),
    ):
        try:
            from importlib.metadata import version as _v

            table.add_row(f"{display_name} version", _v(distribution_name))
        except PackageNotFoundError:
            table.add_row(f"{display_name} version", "[red]not installed[/red]")

    table.add_row("Git available", "yes" if shutil.which("git") else "no (informational only)")

    console.print(table)

    console.print("\n[bold]Dataset availability[/bold]")
    dataset_table = Table(show_header=True, header_style="bold magenta")
    dataset_table.add_column("Dataset")
    dataset_table.add_column("Status")
    for dataset in list_datasets():
        cache_dir = user_data_dir / dataset.local_directory
        if fetch_missing:
            from coinfosim.datasets.download import fetch_dataset

            fetch_dataset(dataset, cache_dir)
        result = verify_dataset(cache_dir, dataset)
        dataset_table.add_row(dataset.slug, "available" if result.is_valid else "missing or invalid")
    console.print(dataset_table)


def _is_writable(directory) -> bool:
    from pathlib import Path

    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".coinfosim_write_probe"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def _is_writable_parent(path) -> bool:
    from pathlib import Path

    path = Path(path)
    existing = path
    while not existing.exists():
        if existing.parent == existing:
            return False
        existing = existing.parent
    return os.access(existing, os.W_OK)
