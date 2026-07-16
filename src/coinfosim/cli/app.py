"""Root CoInfoSim Typer application.

Keeps startup cheap: only ``typer``/``rich`` and lightweight modules are
imported at module level here and in the registered subcommand modules.
Anything that pulls in matplotlib, scikit-learn, dataset loaders, or report
generators is imported inside a command body, so ``coinfosim --help`` and
``coinfosim scenario list`` stay fast.
"""

from __future__ import annotations

from typing import Optional, Sequence

import typer
from rich.console import Console

from coinfosim.cli.context import CLIContext

app = typer.Typer(
    name="coinfosim",
    help="CoInfoSim - A Simulator for Cooperative Classification from Multiple Information Channels",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if not value:
        return
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        pkg_version = _pkg_version("coinfosim")
    except PackageNotFoundError:
        pkg_version = "0.0.0+dev"
    typer.echo(f"coinfosim {pkg_version}")
    raise typer.Exit()


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Log file path (default: from config or no file logging)",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress console output (logs only to file)"
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output"),
    debug: bool = typer.Option(
        False, "--debug", help="Show full tracebacks for unexpected errors"
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (default: ./coinfosim.toml or ~/.config/coinfosim/config.toml)",
    ),
) -> None:
    """Global options for CoInfoSim CLI."""

    from coinfosim.config import ConfigError, load_config

    console = Console(no_color=no_color)

    try:
        config = load_config(config_file=config_file) if config_file else load_config()
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=2)

    try:
        from coinfosim.logging_config import get_logger, setup_logging_from_config

        cli_overrides = {
            "log_level": log_level,
            "log_file": log_file,
            "quiet": quiet,
            "no_color": no_color,
        }
        setup_logging_from_config(config, cli_overrides)
        get_logger("coinfosim.cli").debug(
            f"CLI started with command: {ctx.invoked_subcommand}"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Failed to set up logging:[/bold red] {exc}")
        raise typer.Exit(code=2)

    ctx.obj = CLIContext(
        config=config,
        console=console,
        quiet=quiet,
        color=not no_color,
        debug=debug,
        log_level=log_level,
    )


def _register_subcommands() -> None:
    from coinfosim.cli import (
        config_commands,
        dataset_commands,
        doctor_command,
        legacy_commands,
        publish_commands,
        run_commands,
        scenario_commands,
    )

    app.add_typer(scenario_commands.app, name="scenario")
    app.add_typer(dataset_commands.app, name="dataset")
    app.add_typer(run_commands.app, name="runs")
    app.add_typer(config_commands.app, name="config")
    app.add_typer(publish_commands.app, name="publish")
    doctor_command.register(app)
    legacy_commands.register(app)


_register_subcommands()


def main(args: Optional[Sequence[str]] = None) -> Optional[int]:
    """Entry point for the CLI: returns an exit code without calling sys.exit."""

    try:
        result = app(
            args=list(args) if args is not None else None, standalone_mode=False
        )
    except typer.Abort:
        return 130
    except Exception as exc:  # noqa: BLE001
        if hasattr(exc, "exit_code") and callable(getattr(exc, "show", None)):
            exc.show()
            return exc.exit_code
        raise
    return result if isinstance(result, int) else 0
