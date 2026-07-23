"""Built-in Occupancy Detection dataset-anchored scenario definition.

Moved from ``scripts/run_occupancy_scenario.py`` without semantic change:
same scenario/simulation slugs, same report callbacks, same execution spec.
"""

from __future__ import annotations

from typing import Any, Dict

from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_dataset import generate_occupancy_dataset_report
from coinfosim.reports.occupancy_monte_carlo import (
    generate_occupancy_gmm_to_real_monte_carlo_report,
    generate_occupancy_real_monte_carlo_report,
    generate_occupancy_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.scenarios.dataset_anchored_runner import DatasetAnchoredExecutionSpec
from coinfosim.scenarios.occupancy import (
    OCCUPANCY_SCENARIO_QUESTION,
    build_gaussian_anchored_occupancy_model,
    build_gmm_anchored_occupancy_model,
)

SCENARIO_SLUG = "occupancy_baseline"
SCENARIO_NAME = "Occupancy Detection Baseline"
SCENARIO_FAMILY = "dataset"
REAL_SLUG = "occupancy_real_data"
REAL_FAMILY = "real_dataset"
SG2R_SLUG = "occupancy_single_gaussian_to_real"
SG2R_FAMILY = "single_gaussian_to_real"
GMM2R_SLUG = "occupancy_gmm_to_real"
GMM2R_FAMILY = "gmm_to_real"


def _real_report(result, channel_names, output_dir, **kwargs):
    return generate_occupancy_real_monte_carlo_report(result, output_dir, **kwargs)


def _synthetic_gaussian_report(result, channel_names, output_dir, **kwargs):
    return generate_occupancy_single_gaussian_to_real_monte_carlo_report(
        result, channel_names, output_dir, **kwargs
    )


def _gmm_report(result, channel_names, output_dir, **kwargs):
    return generate_occupancy_gmm_to_real_monte_carlo_report(
        result, channel_names, output_dir, **kwargs
    )


def _preprocessing_metadata(data) -> Dict[str, Any]:
    return {
        "method": "zscore",
        "fit_scope": "training_reservoir_only",
        "ddof": 0,
        "channel_order": list(data.channel_names),
        "means": {
            name: float(value) for name, value in data.standardization.means.items()
        },
        "standard_deviations": {
            name: float(value) for name, value in data.standardization.stds.items()
        },
    }


def _report_context(data) -> Dict[str, Dict[str, Any]]:
    return {
        "dataset": {
            "slug": "occupancy",
            "name": "Occupancy Detection",
            "file_hashes": dict(data.file_hashes),
        },
        "target": {"name": data.target_name, "class_labels": list(data.class_labels)},
        "split": {
            "strategy": "original UCI training and test files",
            "training_files": ["datatraining.txt"],
            "test_files": ["datatest.txt", "datatest2.txt"],
        },
        "preprocessing": _preprocessing_metadata(data),
    }


OCCUPANCY_SPEC = DatasetAnchoredExecutionSpec(
    scenario_slug=SCENARIO_SLUG,
    scenario_name=SCENARIO_NAME,
    scenario_family=SCENARIO_FAMILY,
    question=OCCUPANCY_SCENARIO_QUESTION,
    dataset_slug="occupancy",
    dataset_name="Occupancy Detection",
    real_simulation_slug=REAL_SLUG,
    real_simulation_family=REAL_FAMILY,
    gaussian_simulation_slug=SG2R_SLUG,
    gaussian_simulation_family=SG2R_FAMILY,
    gmm_simulation_slug=GMM2R_SLUG,
    gmm_simulation_family=GMM2R_FAMILY,
    real_training_source_id="real_occupancy_training_pool",
    real_test_source_id="real_occupancy_evaluation_split",
    real_training_description="standardized datatraining.txt",
    fixed_test_description="standardized datatest.txt + datatest2.txt",
    visualization_real_label="Real training sample",
    visualization_gaussian_label="Single Gaussian synthetic training sample",
    visualization_gmm_label="GMM synthetic training sample",
    visualization_real_source="standardized Occupancy training pool",
    visualization_gaussian_source="single Gaussian model",
    visualization_gmm_source="class-conditional Gaussian mixture models",
    dataset_report_prefix="occupancy_dataset_report",
    scenario_report_prefix="occupancy_baseline_scenario_report",
    loader=load_occupancy_data,
    gaussian_builder=build_gaussian_anchored_occupancy_model,
    gmm_builder=build_gmm_anchored_occupancy_model,
    dataset_report_callback=generate_occupancy_dataset_report,
    real_report_callback=_real_report,
    gaussian_report_callback=_synthetic_gaussian_report,
    gmm_report_callback=_gmm_report,
    scenario_report_callback=generate_occupancy_scenario_report,
    report_context_callback=_report_context,
    real_experiment_arm="real_data",
)
