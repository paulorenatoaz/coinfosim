"""Shared CLI context object.

One :class:`CLIContext` is built once in the root callback and threaded
through every subcommand via ``typer.Context.obj``. Commands that need a
different output directory, dataset path, or similar override must copy
the loaded configuration mapping rather than mutating ``ctx.config`` in
place, since the same context instance is shared across the whole
invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rich.console import Console


@dataclass
class CLIContext:
    """Immutable-by-convention state shared by every CLI command."""

    config: Mapping[str, Any]
    console: Console
    quiet: bool = False
    color: bool = True
    debug: bool = False
    log_level: str = "INFO"
