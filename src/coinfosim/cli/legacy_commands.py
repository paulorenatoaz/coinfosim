"""Legacy parameter-model commands, preserved unchanged for 0.2.0.

These predate the dataset-anchored scenario CLI (``coinfosim scenario ...``)
and operate on synthetic parameter grids rather than the tracked datasets.
They are kept for backward compatibility and will be removed in a future
major version.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import typer

_LEGACY_NOTICE = (
    "[dim]Note: this is a legacy parameter-model command, distinct from the "
    "dataset-anchored 'coinfosim scenario' commands.[/dim]"
)


def register(app: typer.Typer) -> None:
    app.command("run-simulation")(run_simulation)
    app.command("run-experiment")(run_experiment)
    app.command("make-report")(make_report)
    app.command("cleanup-logs")(cleanup_logs)


def _get_next_scenario_id(reports_dir: Path, test_mode: bool) -> int:
    if not reports_dir.exists():
        return 1
    test_suffix = "[test]" if test_mode else ""
    pattern = re.compile(rf"scenario_(\d+)_report{re.escape(test_suffix)}\.html")
    max_id = 0
    for file in reports_dir.glob("scenario_*_report*.html"):
        match = pattern.match(file.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def run_simulation(
    ctx: typer.Context,
    params: str = typer.Option(
        ...,
        "--params",
        "-p",
        help="Model parameters as JSON array, e.g., '[1,4,0.6]' for 2D or '[1,1,2,0,0,0]' for 3D",
    ),
    test_mode: bool = typer.Option(
        False, "--test-mode", "-t", help="Run in test mode (10x faster, reduced precision)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Show per-iteration debug logs (verbose output)"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress console output during simulation (summary only)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: from config or ./output)"
    ),
) -> None:
    """[Legacy] Run a single simulation with specified parameters."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    console.print(_LEGACY_NOTICE)
    from coinfosim.logging_config import get_logger

    logger = get_logger("coinfosim.cli.run_simulation")
    config = dict(cli_ctx.config)
    verbose_mode = not quiet

    if output_dir:
        config["paths"] = {**config.get("paths", {}), "output_dir": output_dir}
        from coinfosim.utils import init_report_service_conf, report_service_conf

        report_service_conf.update(init_report_service_conf(config))
        from coinfosim.logging_config import setup_logging_from_config

        setup_logging_from_config(config, force_reconfigure=True)

    if verbose_mode:
        console.print("\n[bold cyan]Running simulation[/bold cyan]")
        console.print(f"Parameters: [yellow]{params}[/yellow]")
        console.print(f"Test mode: [yellow]{test_mode}[/yellow]")
        if debug:
            console.print("Debug mode: [yellow]ON[/yellow]")
        console.print(f"Output: [cyan]{config['paths']['output_dir']}[/cyan]\n")

    try:
        import json

        param_list = json.loads(params)
        logger.info(f"Starting simulation with params: {param_list}")

        from coinfosim import Model, Simulator

        if verbose_mode:
            console.print("Creating model...")
        model = Model(param_list)
        logger.debug(f"Model created: {model}")

        if verbose_mode:
            console.print("Creating simulator...")
        simulator = Simulator(model, test_mode=test_mode, verbose=verbose_mode, debug=debug)

        if verbose_mode:
            console.print("Running simulation (this may take a while)...")
        with console.status("[bold green]Computing...[/bold green]"):
            simulator.run()

        logger.info("Simulation completed successfully")

        if verbose_mode:
            console.print("Saving results...")

        simulator.report.print_N_star_matrix_between_all_dims()
        simulator.report.save_graphs_png_images_files()
        simulator.report.create_report_tables()
        simulator.report.write_to_json()
        simulator.report.create_html_report()

        if verbose_mode:
            console.print("\n[bold green]OK[/bold green] All results saved successfully!\n")

    except json.JSONDecodeError as exc:
        console.print(f"[red]Error:[/red] Invalid parameters format: {exc}", style="bold red")
        logger.error(f"Parameter parsing failed: {exc}")
        raise typer.Exit(1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Simulation failed: {exc}", style="bold red")
        logger.error(f"Simulation error: {exc}", exc_info=True)
        raise typer.Exit(1)


def run_experiment(
    ctx: typer.Context,
    scenarios: Optional[str] = typer.Option(
        None,
        "--scenarios",
        "-s",
        help="Comma-separated scenario numbers (e.g., '1,2,3'). Runs all predefined scenarios if not specified.",
    ),
    custom_params: Optional[str] = typer.Option(
        None, "--custom-params", "-c", help="Custom parameter sets as JSON array"
    ),
    params_file: Optional[str] = typer.Option(
        None, "--params-file", "-f", help="Load parameter sets from JSON file"
    ),
    test_mode: bool = typer.Option(
        False, "--test-mode", "-t", help="Run in test mode (10x faster, reduced precision)"
    ),
    debug: bool = typer.Option(False, "--debug", help="Show per-iteration debug logs (verbose output)"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress console output during simulations (summary only)"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip-existing", help="Skip simulations already in simulation_reports.json"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory (default: from config or user home directory)"
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        help="Organize this experiment under ~/coinfosim/experiments/{tag}/ (ignored if --output-dir is set)",
    ),
) -> None:
    """[Legacy] Run multiple simulations from predefined or custom scenarios."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    console.print(_LEGACY_NOTICE)
    from coinfosim.config import get_reports_dir
    from coinfosim.logging_config import get_logger

    logger = get_logger("coinfosim.cli.run_experiment")
    config = dict(cli_ctx.config)

    if tag and not output_dir:
        if not re.match(r"^[a-zA-Z0-9_-]+$", tag):
            console.print(
                f"[red]Error:[/red] Invalid tag '{tag}': must contain only letters, numbers, underscore, or dash",
                style="bold red",
            )
            raise typer.Exit(1)
        output_dir = str(Path.home() / "coinfosim" / "experiments" / tag)
        console.print(f"[dim]-> Using tagged experiment folder: {output_dir}[/dim]")

    if output_dir:
        config["paths"] = {**config.get("paths", {}), "output_dir": output_dir}
        from coinfosim.utils import init_report_service_conf, report_service_conf

        report_service_conf.update(init_report_service_conf(config))
        from coinfosim.logging_config import setup_logging_from_config

        setup_logging_from_config(config, force_reconfigure=True)

    console.print("\n[bold cyan]Running experiment[/bold cyan]")
    console.print(f"Output directory: [cyan]{config['paths']['output_dir']}[/cyan]\n")

    try:
        from coinfosim import Model, Simulator
        from coinfosim.utils import is_param_in_simulation_reports
        import json

        mode_count = sum([scenarios is not None, custom_params is not None, params_file is not None])
        if mode_count > 1:
            console.print(
                "[red]ERROR:[/red] Cannot combine --scenarios, --custom-params, and --params-file",
                style="bold red",
            )
            console.print("Choose one: predefined scenarios, custom inline params, or params from file")
            raise typer.Exit(1)

        all_param_sets: List = []
        mode_description = ""
        scenario_list: List = []

        if custom_params:
            try:
                loaded_data = json.loads(custom_params)
                if not isinstance(loaded_data, list):
                    raise ValueError("custom-params must be a JSON array")
                if loaded_data and isinstance(loaded_data[0], list):
                    if loaded_data[0] and isinstance(loaded_data[0][0], list):
                        console.print(f"Running [yellow]{len(loaded_data)}[/yellow] custom scenarios\n")
                        for idx, scenario in enumerate(loaded_data, 1):
                            console.print(f"  Scenario {idx}: {len(scenario)} parameter sets")
                            scenario_list.append((None, scenario))
                            all_param_sets.extend(scenario)
                        mode_description = "custom scenarios"
                    else:
                        all_param_sets = loaded_data
                        scenario_list = [(None, all_param_sets)]
                        console.print(f"Running [yellow]{len(all_param_sets)}[/yellow] custom simulations\n")
                        mode_description = "custom inline parameters"
                else:
                    all_param_sets = loaded_data
                    scenario_list = [(None, all_param_sets)]
                    console.print(f"Running [yellow]{len(all_param_sets)}[/yellow] custom simulations\n")
                    mode_description = "custom inline parameters"
            except json.JSONDecodeError as exc:
                console.print(f"[red]ERROR:[/red] Invalid JSON in --custom-params: {exc}", style="bold red")
                raise typer.Exit(1)

        elif params_file:
            try:
                file_path = Path(params_file)
                if not file_path.exists():
                    console.print(f"[red]ERROR:[/red] File not found: {params_file}", style="bold red")
                    raise typer.Exit(1)
                with open(file_path, "r") as handle:
                    loaded_data = json.load(handle)
                if not isinstance(loaded_data, list):
                    raise ValueError("params file must contain a JSON array")
                if loaded_data and isinstance(loaded_data[0], list):
                    if loaded_data[0] and isinstance(loaded_data[0][0], list):
                        console.print(
                            f"Loaded [yellow]{len(loaded_data)}[/yellow] scenarios from [cyan]{params_file}[/cyan]"
                        )
                        for idx, scenario in enumerate(loaded_data, 1):
                            console.print(f"  Scenario {idx}: {len(scenario)} parameter sets")
                            scenario_list.append((None, scenario))
                            all_param_sets.extend(scenario)
                        mode_description = f"scenarios from {params_file}"
                    else:
                        all_param_sets = loaded_data
                        scenario_list = [(None, all_param_sets)]
                        console.print(
                            f"Loaded [yellow]{len(all_param_sets)}[/yellow] simulations from [cyan]{params_file}[/cyan]\n"
                        )
                        mode_description = f"parameters from {params_file}"
                else:
                    all_param_sets = loaded_data
                    scenario_list = [(None, all_param_sets)]
                    console.print(
                        f"Loaded [yellow]{len(all_param_sets)}[/yellow] simulations from [cyan]{params_file}[/cyan]\n"
                    )
                    mode_description = f"parameters from {params_file}"
            except json.JSONDecodeError as exc:
                console.print(f"[red]ERROR:[/red] Invalid JSON in {params_file}: {exc}", style="bold red")
                raise typer.Exit(1)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]ERROR:[/red] Failed to load {params_file}: {exc}", style="bold red")
                raise typer.Exit(1)

        else:
            from coinfosim.demo import SCENARIOS

            scenario_filter = None
            if scenarios:
                scenario_filter = [int(s.strip()) - 1 for s in scenarios.split(",")]
                console.print(f"Running predefined scenarios: [yellow]{scenarios}[/yellow]\n")
            else:
                console.print(f"Running [yellow]all {len(SCENARIOS)}[/yellow] predefined scenarios\n")

            for idx, scenario in enumerate(SCENARIOS):
                if scenario_filter and idx not in scenario_filter:
                    continue
                scenario_list.append((idx + 1, scenario))
                all_param_sets.extend(scenario)
            mode_description = "predefined scenarios"

        total_simulations = len(all_param_sets)
        console.print(f"Total simulations to process: [yellow]{total_simulations}[/yellow]\n")

        completed = 0
        skipped = 0

        for param in all_param_sets:
            if skip_existing and is_param_in_simulation_reports(param, test_mode=test_mode):
                console.print(f"  [dim]SKIP: {param} (already exists)[/dim]")
                skipped += 1
                continue

            console.print(f"  [cyan]>>>[/cyan] Running {param}...")
            logger.info(f"Simulation {completed + 1}/{total_simulations}: {param}")

            try:
                model = Model(param)
                verbose_mode = not quiet
                simulator = Simulator(
                    model, test_mode=test_mode, full_n_range=True, verbose=verbose_mode, debug=debug
                )
                with console.status(f"[bold green]Computing {param}...[/bold green]"):
                    simulator.run()

                simulator.report.print_N_star_matrix_between_all_dims()
                simulator.report.save_graphs_png_images_files()
                simulator.report.create_report_tables()
                simulator.report.write_to_json()
                simulator.report.create_html_report()

                console.print(f"  [green]OK:[/green] Completed {param}")
                completed += 1

            except Exception as exc:  # noqa: BLE001
                console.print(f"  [red]ERROR:[/red] Failed {param}: {exc}")
                logger.error(f"Simulation failed for {param}: {exc}", exc_info=True)

        console.print("\n[bold green]SUCCESS:[/bold green] Experiment complete!")
        console.print(f"  Mode: [cyan]{mode_description}[/cyan]")
        console.print(f"  Completed: [green]{completed}[/green]")
        console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
        console.print(f"  Total: {completed + skipped}/{total_simulations}\n")

        if completed > 0 and scenario_list:
            console.print("[bold cyan]Generating scenario reports...[/bold cyan]\n")
            from coinfosim.reporting.report_utils import create_scenario_report

            reports_dir = Path(get_reports_dir(config))
            next_id = _get_next_scenario_id(reports_dir, test_mode)
            updated_scenario_list = []
            for scenario_id, scenario_params in scenario_list:
                if scenario_id is None:
                    scenario_id = next_id
                    next_id += 1
                updated_scenario_list.append((scenario_id, scenario_params))

            for scenario_id, scenario_params in updated_scenario_list:
                try:
                    console.print(f"  Creating report for scenario: [yellow]{scenario_id}[/yellow]...")
                    create_scenario_report(scenario_params, scenario_id, test_mode=test_mode)
                    console.print(f"  [green]OK:[/green] Scenario {scenario_id} report created")
                except Exception as exc:  # noqa: BLE001
                    console.print(f"  [red]ERROR:[/red] Failed to create scenario {scenario_id} report: {exc}")
                    logger.error(f"Scenario report creation failed: {exc}", exc_info=True)

            console.print("\n[bold green]SUCCESS:[/bold green] All scenario reports generated!\n")

    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Experiment failed: {exc}", style="bold red")
        logger.error(f"Experiment error: {exc}", exc_info=True)
        raise typer.Exit(1)


def make_report(
    ctx: typer.Context,
    scenario: Optional[int] = typer.Option(None, "--scenario", "-s", help="Scenario number to generate report for"),
    params: Optional[str] = typer.Option(None, "--params", "-p", help="Specific parameters as JSON array"),
    test_mode: bool = typer.Option(False, "--test-mode", "-t", help="Read from the test-mode JSON data file"),
) -> None:
    """[Legacy] Generate HTML reports from existing simulation data."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    console.print(_LEGACY_NOTICE)
    from coinfosim.logging_config import get_logger

    logger = get_logger("coinfosim.cli.make_report")

    if not scenario and not params:
        console.print("[red]Error:[/red] Must specify either --scenario or --params", style="bold red")
        raise typer.Exit(1)

    console.print("\n[bold cyan]Generating report[/bold cyan]\n")

    try:
        if scenario:
            console.print(f"Scenario: [yellow]{scenario}[/yellow]")
            from coinfosim.demo import SCENARIOS
            from coinfosim.reporting.report_utils import create_scenario_report

            if scenario < 1 or scenario > len(SCENARIOS):
                console.print(
                    f"[red]Error:[/red] Invalid scenario number. Must be 1-{len(SCENARIOS)}", style="bold red"
                )
                raise typer.Exit(1)

            scenario_params = SCENARIOS[scenario - 1]
            console.print(f"Parameters: {len(scenario_params)} simulations\n")

            with console.status("[bold green]Generating report...[/bold green]"):
                create_scenario_report(scenario_params, scenario, test_mode=test_mode)

            console.print(f"[bold green]OK[/bold green] Report generated: scenario_{scenario}_report.html\n")

        else:
            import json

            param_list = json.loads(params)
            console.print(f"Parameters: [yellow]{param_list}[/yellow]\n")

            from coinfosim.reporting.report import Report

            with console.status("[bold green]Regenerating individual report from JSON...[/bold green]"):
                report = Report(params=param_list)
                report.create_html_report()

            console.print(
                f"[bold green]OK[/bold green] Report regenerated: {report.export_path_html_report}\n"
            )

    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Report generation failed: {exc}", style="bold red")
        logger.error(f"Report error: {exc}", exc_info=True)
        raise typer.Exit(1)


def cleanup_logs(
    ctx: typer.Context,
    older_than_days: int = typer.Option(30, "--older-than", "-d", help="Delete log files older than N days"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deleted without actually deleting"),
) -> None:
    """Clean up old log files."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    from coinfosim.config import get_output_dir
    from coinfosim.logging_config import get_logger

    logger = get_logger("coinfosim.cli.cleanup_logs")
    config = cli_ctx.config

    from datetime import datetime, timedelta

    console.print("\n[bold cyan]Cleaning up log files[/bold cyan]\n")

    try:
        output_dir = get_output_dir(config)
        logs_dir = output_dir / "logs"

        if not logs_dir.exists():
            console.print(f"[yellow]No logs directory found at:[/yellow] {logs_dir}\n")
            return

        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0
        deleted_size = 0

        console.print(f"Scanning: [cyan]{logs_dir}[/cyan]")
        console.print(f"Cutoff date: [yellow]{cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}[/yellow]\n")

        log_files = list(logs_dir.glob("*.log*"))

        if not log_files:
            console.print("[yellow]No log files found[/yellow]\n")
            return

        for log_file in log_files:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff_date:
                size = log_file.stat().st_size
                size_mb = size / (1024 * 1024)
                if dry_run:
                    console.print(
                        f"  [dim]Would delete:[/dim] {log_file.name} ({size_mb:.2f} MB) - {mtime.strftime('%Y-%m-%d')}"
                    )
                else:
                    console.print(
                        f"  [red]Deleting:[/red] {log_file.name} ({size_mb:.2f} MB) - {mtime.strftime('%Y-%m-%d')}"
                    )
                    log_file.unlink()
                    logger.info(f"Deleted log file: {log_file}")
                deleted_count += 1
                deleted_size += size

        if deleted_count > 0:
            size_mb = deleted_size / (1024 * 1024)
            if dry_run:
                console.print(f"\n[yellow]Would delete {deleted_count} file(s), {size_mb:.2f} MB total[/yellow]\n")
            else:
                console.print(f"\n[green]Deleted {deleted_count} file(s), {size_mb:.2f} MB total[/green]\n")
        else:
            console.print(f"\n[green]No log files older than {older_than_days} days[/green]\n")

    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] Cleanup failed: {exc}", style="bold red")
        logger.error(f"Log cleanup error: {exc}", exc_info=True)
        raise typer.Exit(1)
