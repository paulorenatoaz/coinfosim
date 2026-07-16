"""``coinfosim config ...`` commands."""

from __future__ import annotations

from typing import Optional

import typer
from rich.table import Table

from coinfosim.cli.errors import PackagingEnvironmentCLIError, fail

app = typer.Typer(help="Inspect and manage CoInfoSim configuration files.")


@app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show the currently loaded, merged configuration."""

    cli_ctx = ctx.obj
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Value", style="green")
    for section, values in cli_ctx.config.items():
        if section.startswith("_"):
            continue
        if isinstance(values, dict):
            for key, value in values.items():
                table.add_row(section, key, str(value))
        else:
            table.add_row("", section, str(values))
    cli_ctx.console.print(table)


@app.command("init")
def config_init(
    ctx: typer.Context,
    project: bool = typer.Option(
        True,
        "--project/--user",
        help="Create project config (./coinfosim.toml) or user config (~/.config/coinfosim/config.toml).",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing config file."),
) -> None:
    """Create a configuration file template."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    from coinfosim.config import ConfigError, init_project_config, init_user_config

    try:
        path = init_project_config(force=force) if project else init_user_config(force=force)
    except ConfigError as exc:
        fail(console, PackagingEnvironmentCLIError(str(exc)), debug=cli_ctx.debug)
    console.print(f"[bold green]Created:[/bold green] {path}")


@app.command("validate")
def config_validate(
    ctx: typer.Context,
    config_file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Config file to validate (default: currently resolved config)."
    ),
) -> None:
    """Validate a configuration file."""

    cli_ctx = ctx.obj
    console = cli_ctx.console
    from coinfosim.config import ConfigError, load_config, validate_config

    try:
        config = load_config(config_file=config_file) if config_file else cli_ctx.config
        validate_config(config)
    except ConfigError as exc:
        fail(console, PackagingEnvironmentCLIError(str(exc)), debug=cli_ctx.debug)
    console.print("[bold green]Configuration is valid.[/bold green]")
