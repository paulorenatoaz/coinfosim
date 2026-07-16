"""Orchestration layer over the built-in scenario catalog and dataset resolver.

Thin service functions that resolve a scenario and its dataset, then
delegate to the existing dataset-anchored Monte Carlo execution engine
(:func:`run_dataset_anchored_scenario` / :func:`regenerate_dataset_anchored_scenario`).
This module never reimplements the scientific protocol; it only wires the
CLI-facing scenario catalog and dataset resolver into the existing runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from coinfosim.datasets.catalog import get_dataset
from coinfosim.datasets.resolver import resolve_dataset_directory
from coinfosim.scenarios.catalog import get_scenario
from coinfosim.scenarios.dataset_anchored_runner import (
    regenerate_dataset_anchored_scenario,
    run_dataset_anchored_scenario,
)
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import ExecutionConfig
from coinfosim.simulation.progress import CooperativeProgressReporter


def run_registered_scenario(
    scenario_name: str,
    *,
    mode: str = "smoke",
    data_dir: Optional[Path] = None,
    output_dir: Path,
    execution_config: Optional[ExecutionConfig] = None,
    reporter: Optional[CooperativeProgressReporter] = None,
    visualize: bool = True,
    allow_download: bool = True,
    force_download: bool = False,
    config: Optional[MonteCarloConfig] = None,
    cli_config: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Resolve ``scenario_name`` and its dataset, then run the three-arm protocol.

    Parameters
    ----------
    scenario_name:
        Canonical scenario slug or accepted alias.
    cli_config:
        The loaded CoInfoSim runtime configuration (as returned by
        :func:`coinfosim.config.load_config`), consulted for a
        dataset-specific path override before falling back to
        ``COINFOSIM_DATA_DIR`` and the platform cache. May be omitted.

    Notes
    -----
    Never reimplements the Monte Carlo protocol: delegates directly to
    :func:`run_dataset_anchored_scenario` after resolving and verifying the
    dataset directory.
    """

    scenario = get_scenario(scenario_name)
    dataset = get_dataset(scenario.dataset_slug)
    resolved_dir = resolve_dataset_directory(
        dataset,
        explicit_data_dir=Path(data_dir) if data_dir is not None else None,
        config=cli_config,
        allow_download=allow_download,
        force_download=force_download,
    )

    classifier_configuration = None
    if scenario.default_classifier_configuration is not None:
        classifier_configuration = scenario.default_classifier_configuration.resolve()

    return run_dataset_anchored_scenario(
        scenario.spec,
        mode=mode,
        raw_dir=str(resolved_dir),
        output_dir=str(output_dir),
        reporter=reporter,
        config=config,
        execution_config=execution_config,
        visualize=visualize,
        classifier_configuration=classifier_configuration,
    )


def regenerate_registered_scenario(
    scenario_name: str,
    *,
    scenario_run_id: int,
    output_dir: Path,
    reporter: Optional[CooperativeProgressReporter] = None,
) -> Mapping[str, Any]:
    """Regenerate reports for a previously completed scenario run.

    Uses persisted results only; never reruns Monte Carlo.
    """

    scenario = get_scenario(scenario_name)
    return regenerate_dataset_anchored_scenario(
        scenario.spec,
        scenario_run_id,
        output_dir=str(output_dir),
        reporter=reporter,
    )
