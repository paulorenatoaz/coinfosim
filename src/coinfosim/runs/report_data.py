"""
Report-ready data snapshots for persisted scenario/simulation runs.

These helpers turn a :class:`~coinfosim.simulation.monte_carlo.SimulationResult`
into JSON-serializable dictionaries that are embedded into ``simulation.json``
and ``scenario.json``. They make those files self-contained: a reader can
inspect the key summary/ranking/threshold tables without decompressing the full
persisted result payload, while the complete result remains available in the
sibling ``result_data_*.json.gz`` file for exact report regeneration.

The snapshots reuse the exact same analysis functions the HTML reports use
(:func:`summary_dataframe`, :func:`best_subset_rankings`,
:func:`standard_threshold_comparisons`), so no statistical logic is duplicated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from coinfosim.results.analysis import (
    best_subset_rankings,
    standard_threshold_comparisons,
)
from coinfosim.results.summary import summary_dataframe
from coinfosim.results.structural import (
    scenario_structural_fidelity,
    simulation_structural_dynamics,
)
from coinfosim.simulation.monte_carlo import SimulationResult


def _clean(value: Any) -> Any:
    """Convert tuples/numpy/NaN into JSON-native, strict-JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):
            return value
    return value


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_clean(row) for row in df.to_dict(orient="records")]


def stopping_info_snapshot(result: SimulationResult) -> Dict[str, Any]:
    return {
        str(n): {
            "n_per_class": int(info.n_per_class),
            "replications": int(info.replications),
            "reason": info.reason,
            "max_ci_half_width": float(info.max_ci_half_width),
        }
        for n, info in result.stopping_info.items()
    }


def simulation_summary_snapshot(result: SimulationResult) -> Dict[str, Any]:
    """Compact control/summary snapshot for one simulation run."""
    return {
        "mode": result.config.mode,
        "sample_sizes": [int(n) for n in result.sample_sizes],
        "number_of_subsets": len(result.subsets),
        "number_of_classifiers": len(result.classifier_names),
        "classifier_names": list(result.classifier_names),
        "subsets": [list(s) for s in result.subsets],
        "fixed_test_size": result.metadata.get("fixed_test_size"),
        "execution": _clean(result.metadata.get("execution", {})),
        "runtime_seconds": float(result.runtime_seconds),
        "stopping_info": stopping_info_snapshot(result),
    }


def simulation_report_data(result: SimulationResult) -> Dict[str, Any]:
    """Report-ready tables sufficient to understand the simulation report."""
    summary_df = summary_dataframe(
        result.accumulator,
        result.sample_sizes,
        result.subsets,
        result.classifier_names,
    )
    rankings_df = best_subset_rankings(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )
    thresholds_df = standard_threshold_comparisons(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )
    return {
        "metric": "empirical_test_loss",
        "summary_table": _records(summary_df),
        "best_subset_rankings": _records(rankings_df),
        "threshold_comparisons": _records(thresholds_df),
        "structural_dynamics": _clean(simulation_structural_dynamics(result)),
    }


def scenario_report_data(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    channel_names: Sequence[str],
    gmm_model_selection: Any = None,
) -> Dict[str, Any]:
    """Scenario-level report-ready tables and per-arm summary snapshots.

    The three main Occupancy arms are ``real_to_real`` (real training pool, real
    evaluation split), ``single_gaussian_to_real`` (single-Gaussian synthetic
    training, real evaluation split) and ``gmm_to_real`` (class-conditional GMM
    synthetic training, real evaluation split). Each arm records its train/test
    semantics alongside its summary and report-ready tables.
    """
    gmm_arm: Dict[str, Any] = {
        "arm_id": "gmm_to_real",
        "train_source": "gmm_synthetic",
        "test_source": "real_occupancy_evaluation_split",
        "summary": simulation_summary_snapshot(gmm_result),
        "report_data": simulation_report_data(gmm_result),
    }
    if gmm_model_selection is not None:
        gmm_arm["gmm_model_selection"] = _clean(gmm_model_selection)
    arm_results = {
        "real_to_real": real_result,
        "single_gaussian_to_real": gaussian_result,
        "gmm_to_real": gmm_result,
    }
    arm_labels = {
        "real_to_real": "Real → Real",
        "single_gaussian_to_real": "Single Gaussian → Real",
        "gmm_to_real": "GMM → Real",
    }
    return {
        "channel_names": [str(c) for c in channel_names],
        "sample_sizes": [int(n) for n in real_result.sample_sizes],
        "arms": {
            "real_to_real": {
                "arm_id": "real_to_real",
                "train_source": "real_occupancy_training_pool",
                "test_source": "real_occupancy_evaluation_split",
                "summary": simulation_summary_snapshot(real_result),
                "report_data": simulation_report_data(real_result),
            },
            "single_gaussian_to_real": {
                "arm_id": "single_gaussian_to_real",
                "train_source": "single_gaussian_synthetic",
                "test_source": "real_occupancy_evaluation_split",
                "summary": simulation_summary_snapshot(gaussian_result),
                "report_data": simulation_report_data(gaussian_result),
            },
            "gmm_to_real": gmm_arm,
        },
        "structural_fidelity": generic_scenario_structural_report_data(
            arm_results, "real_to_real", arm_labels
        ),
    }


def generic_scenario_structural_report_data(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
    arm_labels: Mapping[str, str],
) -> Dict[str, Any]:
    """Return strict-JSON-safe structural data for an arbitrary arm mapping."""

    return _clean(
        scenario_structural_fidelity(arm_results, reference_arm, arm_labels)
    )
