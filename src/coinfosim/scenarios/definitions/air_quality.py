"""Built-in UCI Air Quality dataset-anchored scenario definition.

Moved from ``scripts/run_air_quality_scenario.py`` without semantic change.
"""

from __future__ import annotations

from typing import Any, Dict

from coinfosim.datasets.air_quality import load_air_quality_data
from coinfosim.reports.air_quality_dataset import generate_air_quality_dataset_report
from coinfosim.reports.air_quality_monte_carlo import (
    generate_air_quality_gmm_to_real_monte_carlo_report,
    generate_air_quality_real_monte_carlo_report,
    generate_air_quality_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.reports.air_quality_scenario import generate_air_quality_scenario_report
from coinfosim.scenarios.air_quality import (
    AIR_QUALITY_SCENARIO_QUESTION,
    build_gaussian_anchored_air_quality_model,
    build_gmm_anchored_air_quality_model,
)
from coinfosim.scenarios.dataset_anchored_runner import DatasetAnchoredExecutionSpec

SCENARIO_SLUG = "air_quality_baseline"
SCENARIO_NAME = "UCI Air Quality Baseline"
SCENARIO_FAMILY = "dataset"
REAL_SLUG = "air_quality_real_data"
REAL_FAMILY = "real_dataset"
SG2R_SLUG = "air_quality_single_gaussian_to_real"
SG2R_FAMILY = "single_gaussian_to_real"
GMM2R_SLUG = "air_quality_gmm_to_real"
GMM2R_FAMILY = "gmm_to_real"


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
            "slug": "air_quality",
            "name": "UCI Air Quality",
            "doi": "10.24432/C59K5F",
            "source_files": list(data.source_files),
            "file_hashes": dict(data.file_hashes),
        },
        "target": {
            "name": data.target_name,
            "reference": data.reference_name,
            "threshold_type": "training_only_quantile",
            "threshold_quantile": data.threshold_quantile,
            "threshold_value": data.threshold_value,
            "positive_rule": "C6H6(GT) >= threshold",
            "class_labels": list(data.class_labels),
        },
        "split": {
            "strategy": "chronological_first_80_percent_train_last_20_percent_test",
            "split_index": data.split_index,
            "train_fraction": data.train_fraction,
            "train_first_timestamp": str(data.train_first_timestamp),
            "train_last_timestamp": str(data.train_last_timestamp),
            "test_first_timestamp": str(data.test_first_timestamp),
            "test_last_timestamp": str(data.test_last_timestamp),
            "row_counts": data.row_counts(),
            "class_counts": data.class_counts(),
        },
        "preprocessing": _preprocessing_metadata(data),
    }


AIR_QUALITY_SPEC = DatasetAnchoredExecutionSpec(
    scenario_slug=SCENARIO_SLUG,
    scenario_name=SCENARIO_NAME,
    scenario_family=SCENARIO_FAMILY,
    question=AIR_QUALITY_SCENARIO_QUESTION,
    dataset_slug="air_quality",
    dataset_name="UCI Air Quality",
    real_simulation_slug=REAL_SLUG,
    real_simulation_family=REAL_FAMILY,
    gaussian_simulation_slug=SG2R_SLUG,
    gaussian_simulation_family=SG2R_FAMILY,
    gmm_simulation_slug=GMM2R_SLUG,
    gmm_simulation_family=GMM2R_FAMILY,
    real_training_source_id="real_air_quality_training_reservoir",
    real_test_source_id="real_air_quality_future_test",
    real_training_description="standardized chronological training reservoir",
    fixed_test_description="standardized fixed future real evaluation set",
    visualization_real_label="Real chronological training sample",
    visualization_gaussian_label="Single Gaussian synthetic training sample",
    visualization_gmm_label="GMM synthetic training sample",
    visualization_real_source="standardized Air Quality training reservoir",
    visualization_gaussian_source="single Gaussian model",
    visualization_gmm_source="class-conditional Gaussian mixture models",
    dataset_report_prefix="air_quality_dataset_report",
    scenario_report_prefix="air_quality_scenario_report",
    loader=load_air_quality_data,
    gaussian_builder=build_gaussian_anchored_air_quality_model,
    gmm_builder=build_gmm_anchored_air_quality_model,
    dataset_report_callback=generate_air_quality_dataset_report,
    real_report_callback=generate_air_quality_real_monte_carlo_report,
    gaussian_report_callback=generate_air_quality_single_gaussian_to_real_monte_carlo_report,
    gmm_report_callback=generate_air_quality_gmm_to_real_monte_carlo_report,
    scenario_report_callback=generate_air_quality_scenario_report,
    report_context_callback=_report_context,
    real_experiment_arm="real_to_real",
)
