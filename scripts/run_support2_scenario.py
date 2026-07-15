#!/usr/bin/env python3
"""Run or regenerate the tracked SUPPORT2 180-day mortality scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from coinfosim.datasets.support2 import (
    SUPPORT2_EXCLUDED_PREDICTORS,
    SUPPORT2_TARGET_HORIZON_DAYS,
    Support2Data,
    load_support2_data,
)
from coinfosim.reports.support2_dataset import generate_support2_dataset_report
from coinfosim.reports.support2_monte_carlo import (
    generate_support2_gmm_to_real_monte_carlo_report,
    generate_support2_real_monte_carlo_report,
    generate_support2_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.reports.support2_scenario import generate_support2_scenario_report
from coinfosim.scenarios.dataset_anchored_runner import (
    DatasetAnchoredExecutionSpec,
    regenerate_dataset_anchored_scenario,
    run_dataset_anchored_scenario,
)
from coinfosim.scenarios.support2 import (
    SUPPORT2_SCENARIO_QUESTION,
    build_gaussian_anchored_support2_model,
    build_gmm_anchored_support2_model,
)
from coinfosim.simulation.config import MonteCarloConfig, VALID_MODES
from coinfosim.simulation.execution import ExecutionConfig
from coinfosim.simulation.progress import CooperativeProgressReporter

SCENARIO_SLUG = "support2_baseline"
SCENARIO_NAME = "SUPPORT2 180-Day Mortality Baseline"
SCENARIO_FAMILY = "dataset"
REAL_SLUG = "support2_real_data"
REAL_FAMILY = "real_dataset"
SG2R_SLUG = "support2_single_gaussian_to_real"
SG2R_FAMILY = "single_gaussian_to_real"
GMM2R_SLUG = "support2_gmm_to_real"
GMM2R_FAMILY = "gmm_to_real"


def _target_metadata(data: Support2Data) -> Dict[str, Any]:
    return {
        "name": "death_180d",
        "description": "Death within 180 days after SUPPORT2 study entry",
        "type": "derived_binary_fixed_horizon",
        "source_event_column": "death",
        "source_time_column": "d.time",
        "horizon_days": SUPPORT2_TARGET_HORIZON_DAYS,
        "positive_rule": "death == 1 and d.time <= 180",
        "negative_rule": "death == 0 or d.time > 180",
        "derived_before_split": True,
        "class_labels": list(data.class_labels),
        "raw_class_counts": {str(k): v for k, v in data.class_counts()["raw"].items()},
        "cohort_class_counts": {
            str(k): v for k, v in data.class_counts()["cohort"].items()
        },
    }


def _preprocessing_metadata(data: Support2Data) -> Dict[str, Any]:
    return {
        "method": "zscore",
        "fit_scope": "training_reservoir_only",
        "ddof": 0,
        "imputation": "none",
        "transformation": "none",
        "clipping": "none",
        "zero_policy": "preserve",
        "channel_order": list(data.channel_names),
        "means": {name: float(value) for name, value in data.standardization.means.items()},
        "standard_deviations": {
            name: float(value) for name, value in data.standardization.stds.items()
        },
    }


def _split_metadata(data: Support2Data) -> Dict[str, Any]:
    return {
        "strategy": "fixed_joint_stratified_80_20",
        "train_fraction": data.train_fraction,
        "test_fraction": 1.0 - data.train_fraction,
        "random_state": data.split_seed,
        "stratification_variables": ["death_180d", "dzgroup"],
        "sort_after_split": "ascending_id",
        "row_counts": data.row_counts(),
        "class_counts": data.class_counts(),
        "joint_stratum_counts": data.joint_stratum_counts(),
        "id_fingerprints": data.id_fingerprints(),
        "split_manifest_artifact": "split_manifest.json",
    }


def _report_context(data: Support2Data) -> Dict[str, Dict[str, Any]]:
    return {
        "dataset": {
            "slug": "support2",
            "name": "SUPPORT2",
            "doi": "10.3886/ICPSR02957.v2",
            "source_files": list(data.source_files),
            "file_hashes": dict(data.file_hashes),
            "raw_rows": len(data.raw_frame),
            "complete_case_rows": len(data.cohort_frame),
            "raw_header_fields": 47,
            "raw_record_fields": 48,
        },
        "target": _target_metadata(data),
        "split": _split_metadata(data),
        "preprocessing": _preprocessing_metadata(data),
        "exclusions": {
            "predictor_exclusions": list(SUPPORT2_EXCLUDED_PREDICTORS),
            "target_related": ["death", "d.time", "hospdead"],
            "dzgroup_role": "stratification_and_reporting_only",
        },
    }


def _dataset_artifacts(data: Support2Data, scenario_dir: Path):
    payloads = {
        "split_manifest": data.split_manifest(),
        "target_metadata": _target_metadata(data),
        "preprocessing_metadata": _preprocessing_metadata(data),
    }
    artifacts = {}
    for artifact_name, payload in payloads.items():
        path = scenario_dir / f"{artifact_name}.json"
        path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        artifacts[artifact_name] = path
    return artifacts


SUPPORT2_SPEC = DatasetAnchoredExecutionSpec(
    scenario_slug=SCENARIO_SLUG,
    scenario_name=SCENARIO_NAME,
    scenario_family=SCENARIO_FAMILY,
    question=SUPPORT2_SCENARIO_QUESTION,
    dataset_slug="support2",
    dataset_name="SUPPORT2",
    real_simulation_slug=REAL_SLUG,
    real_simulation_family=REAL_FAMILY,
    gaussian_simulation_slug=SG2R_SLUG,
    gaussian_simulation_family=SG2R_FAMILY,
    gmm_simulation_slug=GMM2R_SLUG,
    gmm_simulation_family=GMM2R_FAMILY,
    real_training_source_id="real_support2_training_reservoir",
    real_test_source_id="fixed_real_support2_test_set",
    real_training_description="standardized fixed real SUPPORT2 training reservoir",
    fixed_test_description="same fixed standardized real SUPPORT2 test set",
    visualization_real_label="Real SUPPORT2 training sample",
    visualization_gaussian_label="Single Gaussian synthetic training sample",
    visualization_gmm_label="GMM synthetic training sample",
    visualization_real_source="standardized fixed SUPPORT2 training reservoir",
    visualization_gaussian_source="training-only single Gaussian model",
    visualization_gmm_source="training-only class-conditional Gaussian mixtures",
    dataset_report_prefix="support2_dataset_report",
    scenario_report_prefix="support2_scenario_report",
    loader=load_support2_data,
    gaussian_builder=build_gaussian_anchored_support2_model,
    gmm_builder=build_gmm_anchored_support2_model,
    dataset_report_callback=generate_support2_dataset_report,
    real_report_callback=generate_support2_real_monte_carlo_report,
    gaussian_report_callback=generate_support2_single_gaussian_to_real_monte_carlo_report,
    gmm_report_callback=generate_support2_gmm_to_real_monte_carlo_report,
    scenario_report_callback=generate_support2_scenario_report,
    report_context_callback=_report_context,
    real_experiment_arm="real_to_real",
    dataset_artifacts_callback=_dataset_artifacts,
)


def run_scenario(
    mode: str = "smoke",
    raw_dir: str = "data/raw/support2",
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
    config: Optional[MonteCarloConfig] = None,
    execution_config: Optional[ExecutionConfig] = None,
    visualize: bool = True,
) -> Dict[str, Any]:
    return run_dataset_anchored_scenario(
        SUPPORT2_SPEC,
        mode=mode,
        raw_dir=raw_dir,
        output_dir=output_dir,
        reporter=reporter,
        config=config,
        execution_config=execution_config,
        visualize=visualize,
    )


def regenerate_from_scenario_run(
    scenario_run_id: int,
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
) -> Dict[str, Any]:
    return regenerate_dataset_anchored_scenario(
        SUPPORT2_SPEC,
        scenario_run_id,
        output_dir=output_dir,
        reporter=reporter,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, default="smoke")
    parser.add_argument("--raw-dir", default="data/raw/support2")
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument("--execution-backend", choices=("sequential", "process"), default="sequential")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--worker-inner-threads", type=int, default=1)
    parser.add_argument("--multiprocessing-start-method", choices=("forkserver", "fork"), default="forkserver")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--report-from-scenario-run", type=int, default=None)
    args = parser.parse_args()
    reporter = CooperativeProgressReporter(verbose=not args.quiet, no_color=args.no_color)
    try:
        if args.report_from_scenario_run is not None:
            regenerate_from_scenario_run(
                args.report_from_scenario_run,
                output_dir=args.output_dir,
                reporter=reporter,
            )
        else:
            run_scenario(
                mode=args.mode,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                reporter=reporter,
                execution_config=ExecutionConfig(
                    backend=args.execution_backend,
                    n_jobs=args.n_jobs,
                    worker_inner_threads=args.worker_inner_threads,
                    start_method=args.multiprocessing_start_method,
                ),
            )
    except Exception as exc:  # noqa: BLE001
        reporter.error("SUPPORT2 scenario command failed", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
