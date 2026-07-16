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
        "classifier_selection": _clean(
            result.metadata.get("classifier_selection", {})
        ),
        "classifier_configurations": _clean(
            result.metadata.get("classifier_configurations", {})
        ),
        "subsets": [list(s) for s in result.subsets],
        "fixed_test_size": result.metadata.get("fixed_test_size"),
        "execution": _clean(result.metadata.get("execution", {})),
        "runtime_seconds": float(result.runtime_seconds),
        "stopping_info": stopping_info_snapshot(result),
    }


def simulation_report_data(
    result: SimulationResult, *, include_structural_dynamics: bool = True
) -> Dict[str, Any]:
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
    payload = {
        "metric": "empirical_test_loss",
        "classifier_selection": _clean(
            result.metadata.get("classifier_selection", {})
        ),
        "classifier_configurations": _clean(
            result.metadata.get("classifier_configurations", {})
        ),
        "summary_table": _records(summary_df),
        "best_subset_rankings": _records(rankings_df),
        "threshold_comparisons": _records(thresholds_df),
    }
    if include_structural_dynamics:
        payload["structural_dynamics"] = _clean(
            simulation_structural_dynamics(result)
        )
    return payload


def scenario_report_data(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    channel_names: Sequence[str],
    gmm_model_selection: Any = None,
    include_structural_snapshots: bool = True,
) -> Dict[str, Any]:
    """Compatibility wrapper for Occupancy scenario report data."""

    return dataset_anchored_scenario_report_data(
        real_result,
        gaussian_result,
        gmm_result,
        channel_names=channel_names,
        real_arm_id="real_to_real",
        gaussian_arm_id="single_gaussian_to_real",
        gmm_arm_id="gmm_to_real",
        real_train_source="real_occupancy_training_pool",
        gaussian_train_source="single_gaussian_synthetic",
        gmm_train_source="gmm_synthetic",
        real_test_source="real_occupancy_evaluation_split",
        gaussian_test_source="real_occupancy_evaluation_split",
        gmm_test_source="real_occupancy_evaluation_split",
        gmm_model_selection=gmm_model_selection,
        include_structural_snapshots=include_structural_snapshots,
    )


def dataset_anchored_scenario_report_data(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    *,
    channel_names: Sequence[str],
    real_arm_id: str,
    gaussian_arm_id: str,
    gmm_arm_id: str,
    real_train_source: str,
    gaussian_train_source: str,
    gmm_train_source: str,
    real_test_source: str,
    gaussian_test_source: str,
    gmm_test_source: str,
    dataset_metadata: Mapping[str, Any] | None = None,
    target_metadata: Mapping[str, Any] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
    preprocessing_metadata: Mapping[str, Any] | None = None,
    exclusion_metadata: Mapping[str, Any] | None = None,
    scenario_metadata: Mapping[str, Any] | None = None,
    gmm_model_selection: Any = None,
    include_structural_snapshots: bool = True,
) -> Dict[str, Any]:
    """Generic report-ready snapshot for the standard dataset three-arm protocol."""

    results = {
        real_arm_id: real_result,
        gaussian_arm_id: gaussian_result,
        gmm_arm_id: gmm_result,
    }
    sources = {
        real_arm_id: (real_train_source, real_test_source),
        gaussian_arm_id: (gaussian_train_source, gaussian_test_source),
        gmm_arm_id: (gmm_train_source, gmm_test_source),
    }
    arms: Dict[str, Dict[str, Any]] = {}
    for arm_id, result in results.items():
        train_source, test_source = sources[arm_id]
        arms[arm_id] = {
            "arm_id": arm_id,
            "train_source": train_source,
            "test_source": test_source,
            "summary": simulation_summary_snapshot(result),
            "report_data": simulation_report_data(
                result,
                include_structural_dynamics=include_structural_snapshots,
            ),
        }
    if gmm_model_selection is not None:
        arms[gmm_arm_id]["gmm_model_selection"] = _clean(gmm_model_selection)

    arm_labels = {
        real_arm_id: "Real → Real",
        gaussian_arm_id: "Single Gaussian → Real",
        gmm_arm_id: "GMM → Real",
    }
    payload: Dict[str, Any] = {
        "channel_names": [str(channel) for channel in channel_names],
        "sample_sizes": [int(n) for n in real_result.sample_sizes],
        "arms": arms,
        "structural_snapshot_policy": (
            "embedded" if include_structural_snapshots else "regenerate_from_result_data"
        ),
        "classifier_selection": _clean(
            real_result.metadata.get("classifier_selection", {})
        ),
        "classifier_configurations": _clean(
            real_result.metadata.get("classifier_configurations", {})
        ),
    }
    if include_structural_snapshots:
        payload["structural_fidelity"] = generic_scenario_structural_report_data(
            results, real_arm_id, arm_labels
        )
    if dataset_metadata is not None:
        payload["dataset"] = _clean(dict(dataset_metadata))
    if target_metadata is not None:
        payload["target"] = _clean(dict(target_metadata))
    if split_metadata is not None:
        payload["split"] = _clean(dict(split_metadata))
    if preprocessing_metadata is not None:
        payload["preprocessing"] = _clean(dict(preprocessing_metadata))
    if exclusion_metadata is not None:
        payload["exclusions"] = _clean(dict(exclusion_metadata))
    if scenario_metadata is not None:
        payload["scenario"] = _clean(dict(scenario_metadata))
    return payload


def generic_scenario_structural_report_data(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
    arm_labels: Mapping[str, str],
) -> Dict[str, Any]:
    """Return strict-JSON-safe structural data for an arbitrary arm mapping."""

    return _clean(
        scenario_structural_fidelity(arm_results, reference_arm, arm_labels)
    )
