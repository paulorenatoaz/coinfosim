"""
Console progress reporting for the cooperative Monte Carlo engine.

This module provides :class:`CooperativeProgressReporter`, a reusable
console-output layer for the new CoInfoSim simulation architecture. It is
inspired by the UX of the legacy ``coinfosim.progress.ProgressTracker`` but
uses the CoInfoSim vocabulary instead of the legacy dimensionality/Bayes-risk
terminology:

- scenarios and experiment arms (real-data arm, Gaussian-anchored arm);
- channel subsets and classifiers;
- empirical test loss;
- sample sizes (``n_per_class``) and replications;
- replication batches and CI half-width;
- convergence and max-budget stopping.

The reporter is intentionally decoupled from the scientific loop: the
:class:`~coinfosim.simulation.monte_carlo.CooperativeMonteCarloSimulator`
accepts one optionally and stays completely silent when none is provided,
which keeps tests and programmatic use quiet by default.

Rich is used when available (it ships with ``typer[all]``), but the reporter
degrades gracefully to plain ``print`` output so it stays readable in
PyCharm, plain terminals, redirected logs, and CI. No live/animated display
is used, so output remains robust for non-interactive consoles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:  # pragma: no cover - exercised indirectly
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False


# Plain-text symbols kept ASCII-friendly for redirected logs and CI.
_SYM_START = ">>"
_SYM_STEP = "->"
_SYM_OK = "OK"
_SYM_WARN = "!!"
_SYM_ERR = "XX"
_SYM_INFO = "--"


def _fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


class CooperativeProgressReporter:
    """Beautiful, robust console output for the cooperative Monte Carlo engine.

    Parameters
    ----------
    verbose:
        When ``False`` every method is a no-op. This makes it safe to always
        construct a reporter and hand it around without producing output.
    no_color:
        Disable ANSI colors / Rich styling. Useful for redirected logs and CI.
    console:
        Optional pre-built Rich :class:`~rich.console.Console`. Ignored when
        Rich is unavailable.
    """

    def __init__(
        self,
        verbose: bool = True,
        no_color: bool = False,
        console: Optional[Any] = None,
    ) -> None:
        self.verbose = bool(verbose)
        self.no_color = bool(no_color)
        self._use_rich = _RICH_AVAILABLE and not no_color
        if self._use_rich:
            self._console = console or Console(highlight=False)
        else:
            self._console = None

    # -- low-level emit helpers -------------------------------------------
    def _emit(self, plain: str, rich: Optional[str] = None) -> None:
        if not self.verbose:
            return
        if self._use_rich and self._console is not None:
            self._console.print(rich if rich is not None else plain)
        else:
            print(plain, flush=True)

    def _emit_panel(self, title: str, body_lines: Sequence[str], style: str) -> None:
        if not self.verbose:
            return
        if self._use_rich and self._console is not None:
            self._console.print(
                Panel(
                    "\n".join(body_lines),
                    title=title,
                    border_style=style,
                    expand=False,
                )
            )
        else:
            print(f"\n=== {title} ===", flush=True)
            for line in body_lines:
                print(f"  {line}", flush=True)
            print("", flush=True)

    # -- scenario level ---------------------------------------------------
    def scenario_start(
        self,
        *,
        scenario_name: str,
        mode: str,
        raw_dir: Any,
        output_dir: Any,
        config: Any,
    ) -> None:
        """Announce the start of a full scenario run and its configuration."""
        if not self.verbose:
            return
        body = [
            f"Scenario     : {scenario_name}",
            f"Mode         : {mode}",
            f"Raw data     : {raw_dir}",
            f"Output dir   : {output_dir}",
            "",
            f"Sample sizes : {list(config.sample_sizes)}",
            f"Replications : min={config.min_replications} "
            f"max={config.max_replications} batch={config.replication_batch_size}",
            f"CI target    : {config.ci_half_width_target} (95% half-width)",
            f"Test / class : {config.test_samples_per_class}",
            f"Base seed    : {config.base_seed}",
        ]
        self._emit_panel(
            f"{_SYM_START} CoInfoSim scenario start", body, "cyan"
        )

    def scenario_step_start(self, label: str, detail: Optional[str] = None) -> None:
        """Announce the start of a macro scenario step."""
        suffix = f" ({detail})" if detail else ""
        self._emit(
            f"{_SYM_STEP} {label}{suffix} ...",
            f"[bold cyan]{_SYM_STEP}[/bold cyan] {label}{suffix} ...",
        )

    def scenario_step_finish(
        self,
        label: str,
        elapsed: Optional[float] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Announce the completion of a macro scenario step."""
        parts = [label, "done"]
        if detail:
            parts.append(detail)
        if elapsed is not None:
            parts.append(f"[{_fmt_seconds(elapsed)}]")
        text = " ".join(parts)
        self._emit(
            f"{_SYM_OK} {text}",
            f"[bold green]{_SYM_OK}[/bold green] {text}",
        )

    def scenario_finish(
        self,
        *,
        runtime: float,
        outputs: Mapping[str, Any],
    ) -> None:
        """Print the final success summary with runtime and output paths."""
        if not self.verbose:
            return
        if self._use_rich and self._console is not None and Table is not None:
            table = Table(show_header=True, header_style="bold magenta", expand=False)
            table.add_column("Output", style="cyan")
            table.add_column("Path", style="green")
            for name, path in outputs.items():
                table.add_row(name, str(path))
            self._console.print(
                Panel(
                    table,
                    title=f"{_SYM_OK} Scenario complete in {_fmt_seconds(runtime)}",
                    border_style="green",
                    expand=False,
                )
            )
        else:
            print(
                f"\n=== Scenario complete in {_fmt_seconds(runtime)} ===",
                flush=True,
            )
            for name, path in outputs.items():
                print(f"  {name}: {path}", flush=True)
            print("", flush=True)

    # -- simulation (one experiment arm) ----------------------------------
    def simulation_start(
        self,
        *,
        arm: str,
        n_sample_sizes: int,
        n_subsets: int,
        n_classifiers: int,
        n_cells: int,
        fixed_test_size: int,
        sample_sizes: Sequence[int],
        execution: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Announce the start of one experiment-arm Monte Carlo simulation."""
        if not self.verbose:
            return
        body = [
            f"Experiment arm         : {arm}",
            f"Sample sizes           : {n_sample_sizes}  {list(sample_sizes)}",
            f"Channel subsets        : {n_subsets}",
            f"Classifiers            : {n_classifiers}",
            f"Subset x classifier    : {n_cells} cells",
            f"Fixed test size        : {fixed_test_size}",
            "Metric                 : empirical test loss",
        ]
        if execution:
            cache_bytes = execution.get(
                "fixed_test_cache_bytes_per_worker", "unknown"
            )
            body.extend(
                [
                    "Execution backend       : "
                    f"{execution.get('backend', 'unknown')}",
                    "Workers requested/effective: "
                    f"{execution.get('requested_workers', 'unknown')} / "
                    f"{execution.get('effective_workers', 'unknown')}",
                    "Numeric threads / worker: "
                    f"{execution.get('worker_inner_threads', 'unknown')}",
                    "Multiprocessing start   : "
                    f"{execution.get('start_method', 'unknown')}",
                    "Detected logical CPUs   : "
                    f"{execution.get('logical_cpus', 'unknown')}",
                    "Fixed-test cache / worker: "
                    f"{cache_bytes} bytes",
                ]
            )
        self._emit_panel(
            f"{_SYM_START} Monte Carlo simulation start", body, "blue"
        )

    def sample_size_start(self, n_per_class: int, index: int, total: int) -> None:
        """Announce the start of one ``n_per_class`` sweep."""
        self._emit(
            f"{_SYM_STEP} n_per_class={n_per_class} "
            f"({index}/{total}) accumulating replications ...",
            f"[cyan]{_SYM_STEP}[/cyan] n_per_class="
            f"[bold]{n_per_class}[/bold] ([dim]{index}/{total}[/dim]) "
            f"accumulating replications ...",
        )

    def batch_finish(
        self,
        n_per_class: int,
        replications: int,
        max_ci_half_width: float,
    ) -> None:
        """Report progress at a replication batch boundary."""
        self._emit(
            f"   {_SYM_INFO} n_per_class={n_per_class} "
            f"replications={replications} "
            f"max_ci_half_width={max_ci_half_width:.4f}",
            f"   [dim]{_SYM_INFO} n_per_class={n_per_class} "
            f"replications={replications} "
            f"max_ci_half_width={max_ci_half_width:.4f}[/dim]",
        )

    def sample_size_finish(
        self,
        n_per_class: int,
        replications: int,
        reason: str,
        max_ci_half_width: float,
        elapsed: Optional[float] = None,
    ) -> None:
        """Report the stopping status when one ``n_per_class`` finishes."""
        if reason == "converged":
            tag_plain, tag_rich = "converged", "[green]converged[/green]"
            sym_rich = f"[bold green]{_SYM_OK}[/bold green]"
        else:
            tag_plain, tag_rich = "max_budget", "[yellow]max_budget[/yellow]"
            sym_rich = f"[bold yellow]{_SYM_WARN}[/bold yellow]"
        tail = f" [{_fmt_seconds(elapsed)}]" if elapsed is not None else ""
        self._emit(
            f"{_SYM_OK} n_per_class={n_per_class} {tag_plain} "
            f"replications={replications} "
            f"max_ci_half_width={max_ci_half_width:.4f}{tail}",
            f"{sym_rich} n_per_class=[bold]{n_per_class}[/bold] {tag_rich} "
            f"replications={replications} "
            f"max_ci_half_width={max_ci_half_width:.4f}{tail}",
        )

    def simulation_finish(self, runtime: float) -> None:
        """Report the total runtime for one experiment-arm simulation."""
        self._emit(
            f"{_SYM_OK} Simulation finished in {_fmt_seconds(runtime)}",
            f"[bold green]{_SYM_OK}[/bold green] Simulation finished in "
            f"{_fmt_seconds(runtime)}",
        )

    # -- generic messages -------------------------------------------------
    def info(self, message: str) -> None:
        self._emit(
            f"{_SYM_INFO} {message}",
            f"[dim]{_SYM_INFO} {message}[/dim]",
        )

    def warning(self, message: str) -> None:
        self._emit(
            f"{_SYM_WARN} {message}",
            f"[bold yellow]{_SYM_WARN}[/bold yellow] {message}",
        )

    def error(self, message: str, exc: Optional[BaseException] = None) -> None:
        """Print a clear, actionable error summary."""
        detail = f": {exc}" if exc is not None else ""
        # Errors are printed even in non-verbose mode so failures are visible.
        text_plain = f"{_SYM_ERR} {message}{detail}"
        text_rich = (
            f"[bold red]{_SYM_ERR}[/bold red] [red]{message}{detail}[/red]"
        )
        if self._use_rich and self._console is not None:
            self._console.print(text_rich)
        else:
            print(text_plain, flush=True)
