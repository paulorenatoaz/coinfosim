"""CLI-domain error types, exit codes, and exception translation.

Library/service code raises its own typed exceptions (``UnknownScenarioError``,
``DatasetIntegrityError``, ...); this module is the single place that maps
those into a concise, actionable CLI message and a stable exit code. Command
bodies should catch broadly at their boundary and call :func:`fail`.
"""

from __future__ import annotations

import sys
from typing import NoReturn, Optional

import typer
from rich.console import Console

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_DATASET_MISSING = 3
EXIT_DATASET_INTEGRITY = 4
EXIT_SCENARIO_EXECUTION = 5
EXIT_REGENERATION = 6
EXIT_PACKAGING_ENVIRONMENT = 7


class CLIError(Exception):
    """Base class for CLI-domain errors with a fixed exit code."""

    exit_code: int = 1

    def __init__(self, message: str, *, next_command: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.next_command = next_command


class UsageCLIError(CLIError):
    exit_code = EXIT_USAGE


class DatasetMissingCLIError(CLIError):
    exit_code = EXIT_DATASET_MISSING


class DatasetIntegrityCLIError(CLIError):
    exit_code = EXIT_DATASET_INTEGRITY


class ScenarioExecutionCLIError(CLIError):
    exit_code = EXIT_SCENARIO_EXECUTION


class RegenerationCLIError(CLIError):
    exit_code = EXIT_REGENERATION


class PackagingEnvironmentCLIError(CLIError):
    exit_code = EXIT_PACKAGING_ENVIRONMENT


def translate_exception(
    exc: BaseException, *, execution_error_cls: type[CLIError] = ScenarioExecutionCLIError
) -> CLIError:
    """Map a lower-layer exception to a typed :class:`CLIError`.

    ``execution_error_cls`` lets a specific command (e.g. ``scenario
    regenerate``) select a more precise fallback exit code
    (:class:`RegenerationCLIError`) than the generic default.
    """

    if isinstance(exc, CLIError):
        return exc

    # Imported lazily: these modules are already loaded by the time a
    # command body is running, but importing them at module level here
    # would couple this lightweight error-mapping module to the dataset
    # and scenario catalogs unnecessarily.
    from coinfosim.datasets.catalog import DatasetCatalogError, UnknownDatasetError
    from coinfosim.datasets.integrity import DatasetIntegrityError
    from coinfosim.datasets.resolver import DatasetResolutionError
    from coinfosim.scenarios.catalog import UnknownScenarioError

    if isinstance(exc, UnknownScenarioError):
        return UsageCLIError(str(exc), next_command="coinfosim scenario list")
    if isinstance(exc, UnknownDatasetError):
        return UsageCLIError(str(exc), next_command="coinfosim dataset list")
    if isinstance(exc, DatasetIntegrityError):
        return DatasetIntegrityCLIError(
            str(exc), next_command="coinfosim dataset verify <dataset>"
        )
    if isinstance(exc, DatasetResolutionError):
        return DatasetMissingCLIError(
            str(exc), next_command="coinfosim dataset fetch <dataset>"
        )
    if isinstance(exc, DatasetCatalogError):
        return PackagingEnvironmentCLIError(str(exc))
    return execution_error_cls(str(exc))


def fail(console: Console, error: CLIError, *, debug: bool) -> NoReturn:
    """Print ``error`` (or a full traceback if ``debug`` and one is active) and exit."""

    if debug and sys.exc_info()[0] is not None:
        console.print_exception(show_locals=False)
    console.print(f"[bold red]Error:[/bold red] {error.message}")
    if error.next_command:
        console.print(f"[dim]Next:[/dim] {error.next_command}")
    raise typer.Exit(code=error.exit_code)
