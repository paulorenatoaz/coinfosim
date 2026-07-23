"""``coinfosim publish ...`` commands."""

from __future__ import annotations

from pathlib import Path

import typer

from coinfosim.cli.errors import PackagingEnvironmentCLIError, fail, translate_exception

app = typer.Typer(help="Publish reports and datasets to GitHub Pages.")


@app.command("pages")
def publish_pages(
    ctx: typer.Context,
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        help="Directory containing reports/ and data/ (typically the scenario run output root).",
    ),
    branch: str = typer.Option("gh-pages", "--branch", help="Target GitHub Pages branch."),
    remote: str = typer.Option("origin", "--remote", help="Git remote to fetch from and push to."),
    push: bool = typer.Option(
        False, "--push/--no-push", help="Push a new commit when there are changes."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Regenerate the site and report changes without committing."
    ),
    init_branch_if_missing: bool = typer.Option(
        False,
        "--init-branch",
        help="Create the target branch as a new orphan branch if it does not exist.",
    ),
    mirror: bool = typer.Option(
        False,
        "--mirror",
        help=(
            "Remove the branch's existing reports/data before copying, so the "
            "published site exactly mirrors --output-dir. Only use when "
            "--output-dir contains every scenario that should remain published."
        ),
    ),
) -> None:
    """Regenerate and publish the CoInfoSim reports/datasets Pages site."""

    cli_ctx = ctx.obj
    console = cli_ctx.console

    from coinfosim.publish.publisher import PublishError, publish_pages as _publish_pages

    console.print(f"[bold cyan]Publishing to {branch}[/bold cyan] (remote={remote})")
    try:
        result = _publish_pages(
            output_dir,
            branch=branch,
            remote=remote,
            push=push,
            dry_run=dry_run,
            init_branch_if_missing=init_branch_if_missing,
            mirror_reports=mirror,
        )
    except PublishError as exc:
        fail(console, PackagingEnvironmentCLIError(str(exc)), debug=cli_ctx.debug)
    except Exception as exc:  # noqa: BLE001
        fail(console, translate_exception(exc, execution_error_cls=PackagingEnvironmentCLIError), debug=cli_ctx.debug)

    console.print(
        f"Scenarios: {result.scenario_count}  Dataset files: {result.dataset_file_count}"
    )
    if not result.changed:
        console.print("[green]No changes to publish.[/green]")
        return
    if result.dry_run:
        console.print(f"[yellow]Dry run.[/yellow] {len(result.changed_paths)} path(s) would change:")
        for path in result.changed_paths:
            console.print(f"  {path}")
        return
    console.print(f"[bold green]Committed.[/bold green] {len(result.changed_paths)} path(s) changed.")
    if result.pushed:
        console.print(f"[bold green]Pushed to {remote}/{branch}.[/bold green]")
    else:
        console.print(f"[yellow]Not pushed.[/yellow] Run with --push, or: git push {remote} {branch}")
