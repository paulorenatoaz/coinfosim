"""``coinfosim scenario ...`` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from coinfosim.cli.errors import RegenerationCLIError, UsageCLIError, fail, translate_exception
from coinfosim.scenarios.catalog import get_scenario, list_scenarios
from coinfosim.simulation.config import VALID_MODES

app = typer.Typer(help="Manage and run the built-in dataset-anchored scenarios.")


@app.command("list")
def scenario_list(ctx: typer.Context) -> None:
    """List the built-in dataset-anchored scenarios."""

    cli_ctx = ctx.obj
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Slug", style="cyan")
    table.add_column("Display name")
    table.add_column("Dataset")
    table.add_column("Description")
    table.add_column("Modes")
    for scenario in list_scenarios():
        table.add_row(
            scenario.slug,
            scenario.display_name,
            scenario.dataset_slug,
            scenario.description,
            ", ".join(VALID_MODES),
        )
    cli_ctx.console.print(table)


@app.command("show")
def scenario_show(
    ctx: typer.Context,
    scenario: str = typer.Argument(..., help="Scenario slug or alias."),
) -> None:
    """Show scientific-protocol detail for one built-in scenario."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    try:
        definition = get_scenario(scenario)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    console.print(
        f"[bold cyan]{definition.display_name}[/bold cyan] "
        f"[dim]({definition.slug})[/dim]"
    )
    if definition.aliases:
        console.print(f"Aliases: {', '.join(definition.aliases)}")
    console.print(f"Dataset: {definition.dataset_slug}")
    console.print(f"\n{definition.description}\n")

    try:
        spec = definition.spec
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=cli_ctx.debug)

    console.print(f"Scientific question: {spec.question}")
    console.print(
        "Three-arm protocol: real-to-real, single-Gaussian-to-real, GMM-to-real"
    )
    console.print(f"Real training source: {spec.real_training_description}")
    console.print(f"Fixed test source: {spec.fixed_test_description}")
    console.print(
        f"\nSupports --dataset-report-only: {definition.supports_dataset_report_only}"
    )
    console.print(f"\nExample:\n  coinfosim scenario run {definition.slug} --mode smoke")


@app.command("run")
def scenario_run(
    ctx: typer.Context,
    scenario: str = typer.Argument(..., help="Scenario slug or alias."),
    mode: str = typer.Option(
        "smoke", "--mode", help=f"Execution mode: {', '.join(VALID_MODES)}."
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Explicit dataset directory (bypasses download once verified)."
    ),
    output_dir: Path = typer.Option(
        Path("output/reports"), "--output-dir", help="Report/registry output root."
    ),
    backend: str = typer.Option(
        "sequential", "--backend", help="Replication execution backend: sequential or process."
    ),
    workers: int = typer.Option(1, "--workers", help="Requested replication workers."),
    worker_threads: int = typer.Option(
        1, "--worker-threads", help="Numeric-library threads per worker."
    ),
    start_method: str = typer.Option(
        "auto", "--start-method", help="Multiprocessing start method: auto, spawn, forkserver, fork."
    ),
    no_download: bool = typer.Option(
        False, "--no-download", help="Never download missing dataset files."
    ),
    refresh_data: bool = typer.Option(
        False, "--refresh-data", help="Force re-download and re-verify the dataset."
    ),
    no_visualizations: bool = typer.Option(
        False, "--no-visualizations", help="Skip visualization panels and graphs."
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output."),
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks on failure."),
) -> None:
    """Run a built-in dataset-anchored scenario (downloads the dataset if needed)."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    effective_debug = debug or cli_ctx.debug

    if mode not in VALID_MODES:
        fail(
            console,
            UsageCLIError(f"unknown mode {mode!r}; valid modes: {list(VALID_MODES)}"),
            debug=effective_debug,
        )
    if no_download and refresh_data:
        fail(
            console,
            UsageCLIError("--no-download and --refresh-data are mutually exclusive"),
            debug=effective_debug,
        )
    if backend == "sequential" and workers > 1:
        fail(
            console,
            UsageCLIError(
                "--workers > 1 requires --backend process; the sequential backend "
                "always uses exactly one worker"
            ),
            debug=effective_debug,
        )

    from coinfosim.scenarios.service import run_registered_scenario
    from coinfosim.simulation.execution import ExecutionConfig, resolve_start_method
    from coinfosim.simulation.progress import CooperativeProgressReporter

    try:
        resolved_start_method = resolve_start_method(start_method, backend=backend)
        execution_config = ExecutionConfig(
            backend=backend,
            n_jobs=workers,
            worker_inner_threads=worker_threads,
            start_method=resolved_start_method,
        )
    except ValueError as exc:
        fail(console, UsageCLIError(str(exc)), debug=effective_debug)

    reporter = CooperativeProgressReporter(
        verbose=not quiet, no_color=not cli_ctx.color, console=console
    )

    if not quiet:
        console.print(f"[bold cyan]Running scenario[/bold cyan] {scenario} (mode={mode})")
        console.print(f"Output directory: {output_dir}")

    try:
        result = run_registered_scenario(
            scenario,
            mode=mode,
            data_dir=data_dir,
            output_dir=output_dir,
            execution_config=execution_config,
            reporter=reporter,
            visualize=not no_visualizations,
            allow_download=not no_download,
            force_download=refresh_data,
            cli_config=cli_ctx.config,
        )
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc), debug=effective_debug)

    if not quiet:
        console.print(
            f"[bold green]Done.[/bold green] scenario_run_id={result['scenario_run_id']}"
        )
        console.print(f"Scenario report: {result['scenario_report']}")


@app.command("regenerate")
def scenario_regenerate(
    ctx: typer.Context,
    scenario: str = typer.Argument(..., help="Scenario slug or alias."),
    run_id: int = typer.Option(
        ..., "--run-id", help="Scenario run id to regenerate reports for."
    ),
    output_dir: Path = typer.Option(
        Path("output/reports"), "--output-dir", help="Report/registry output root."
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output."),
) -> None:
    """Regenerate reports for a scenario run from persisted results (no Monte Carlo)."""

    cli_ctx = ctx.obj
    console = cli_ctx.console

    from coinfosim.scenarios.service import regenerate_registered_scenario
    from coinfosim.simulation.progress import CooperativeProgressReporter

    reporter = CooperativeProgressReporter(
        verbose=not quiet, no_color=not cli_ctx.color, console=console
    )
    try:
        result = regenerate_registered_scenario(
            scenario, scenario_run_id=run_id, output_dir=output_dir, reporter=reporter
        )
    except Exception as exc:  # noqa: BLE001
        fail(
            console,
            translate_exception(exc, execution_error_cls=RegenerationCLIError),
            debug=cli_ctx.debug,
        )

    if not quiet:
        console.print(f"[bold green]Regenerated.[/bold green] {result['scenario_report']}")
