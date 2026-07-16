"""``coinfosim runs ...`` commands: render the scenario/simulation run registries."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from coinfosim.cli.errors import UsageCLIError, fail

app = typer.Typer(help="Inspect scenario and simulation run registries.")

_DEFAULT_OUTPUT_DIR = Path("output/reports")


@app.command("scenarios")
def runs_scenarios(
    ctx: typer.Context,
    output_dir: Path = typer.Option(
        _DEFAULT_OUTPUT_DIR, "--output-dir", help="Report/registry output root."
    ),
) -> None:
    """List all recorded scenario runs."""

    cli_ctx = ctx.obj
    from coinfosim.runs.registry import ScenarioRunRegistry

    registry = ScenarioRunRegistry(base_output_dir=output_dir)
    table = Table(show_header=True, header_style="bold magenta")
    for column in ("ID", "Slug", "Family", "Mode", "Status", "Started", "Runtime (s)"):
        table.add_column(column)
    for run in registry.list_runs():
        table.add_row(
            str(run.scenario_run_id),
            run.scenario_slug,
            run.scenario_family,
            run.mode,
            run.status,
            run.started_at or "",
            f"{run.runtime_seconds:.2f}" if run.runtime_seconds is not None else "",
        )
    cli_ctx.console.print(table)


@app.command("simulations")
def runs_simulations(
    ctx: typer.Context,
    output_dir: Path = typer.Option(
        _DEFAULT_OUTPUT_DIR, "--output-dir", help="Report/registry output root."
    ),
) -> None:
    """List all recorded simulation runs."""

    cli_ctx = ctx.obj
    from coinfosim.runs.registry import SimulationRunRegistry

    registry = SimulationRunRegistry(base_output_dir=output_dir)
    table = Table(show_header=True, header_style="bold magenta")
    for column in ("ID", "Slug", "Family", "Mode", "Status", "Started", "Runtime (s)"):
        table.add_column(column)
    for run in registry.list_runs():
        table.add_row(
            str(run.simulation_run_id),
            run.simulation_slug,
            run.simulation_family,
            run.mode,
            run.status,
            run.started_at or "",
            f"{run.runtime_seconds:.2f}" if run.runtime_seconds is not None else "",
        )
    cli_ctx.console.print(table)


@app.command("scenario")
def runs_scenario_detail(
    ctx: typer.Context,
    run_id: int = typer.Argument(..., help="Scenario run id."),
    output_dir: Path = typer.Option(
        _DEFAULT_OUTPUT_DIR, "--output-dir", help="Report/registry output root."
    ),
) -> None:
    """Show detail for one scenario run."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    from coinfosim.runs.registry import ScenarioRunRegistry

    registry = ScenarioRunRegistry(base_output_dir=output_dir)
    record = registry.get_run(run_id)
    if record is None:
        fail(
            console,
            UsageCLIError(f"scenario run id {run_id} not found in {output_dir}"),
            debug=cli_ctx.debug,
        )
    console.print(f"[bold cyan]Scenario run {run_id}[/bold cyan]: {record.scenario_name}")
    console.print(f"Slug: {record.scenario_slug}  Family: {record.scenario_family}")
    console.print(f"Mode: {record.mode}  Status: {record.status}")
    console.print(f"Started: {record.started_at}  Finished: {record.finished_at}")
    console.print(f"Run directory: {record.run_dir}")
    if record.artifacts:
        console.print("Artifacts:")
        for name, path in record.artifacts.items():
            console.print(f"  {name}: {path}")
    if record.error:
        console.print(f"[red]Error: {record.error}[/red]")


@app.command("simulation")
def runs_simulation_detail(
    ctx: typer.Context,
    run_id: int = typer.Argument(..., help="Simulation run id."),
    output_dir: Path = typer.Option(
        _DEFAULT_OUTPUT_DIR, "--output-dir", help="Report/registry output root."
    ),
) -> None:
    """Show detail for one simulation run."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    from coinfosim.runs.registry import SimulationRunRegistry

    registry = SimulationRunRegistry(base_output_dir=output_dir)
    record = registry.get_run(run_id)
    if record is None:
        fail(
            console,
            UsageCLIError(f"simulation run id {run_id} not found in {output_dir}"),
            debug=cli_ctx.debug,
        )
    console.print(f"[bold cyan]Simulation run {run_id}[/bold cyan]: {record.simulation_slug}")
    console.print(f"Family: {record.simulation_family}  Mode: {record.mode}")
    console.print(f"Status: {record.status}")
    console.print(f"Started: {record.started_at}  Finished: {record.finished_at}")
    console.print(f"Run directory: {record.run_dir}")
    if record.scenario_run_id_origin is not None:
        console.print(f"Originating scenario run: {record.scenario_run_id_origin}")
    if record.reused_by_scenario_run_ids:
        console.print(f"Reused by scenario runs: {record.reused_by_scenario_run_ids}")
    if record.artifacts:
        console.print("Artifacts:")
        for name, path in record.artifacts.items():
            console.print(f"  {name}: {path}")
    if record.error:
        console.print(f"[red]Error: {record.error}[/red]")
