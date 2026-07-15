#!/usr/bin/env python3
"""Run or regenerate the tracked Occupancy three-arm scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_dataset import generate_occupancy_dataset_report
from coinfosim.reports.occupancy_monte_carlo import (
    generate_occupancy_gmm_to_real_monte_carlo_report,
    generate_occupancy_real_monte_carlo_report,
    generate_occupancy_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.scenarios.dataset_anchored_runner import (
    VIZ_PER_CLASS,
    VIZ_SEED,
    DatasetAnchoredExecutionSpec,
    list_scenario_runs,
    list_simulation_runs,
    regenerate_dataset_anchored_scenario,
    run_dataset_anchored_scenario,
)
from coinfosim.scenarios.occupancy import (
    OCCUPANCY_SCENARIO_QUESTION,
    build_gaussian_anchored_occupancy_model,
    build_gmm_anchored_occupancy_model,
)
from coinfosim.simulation.config import MonteCarloConfig, VALID_MODES
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator
from coinfosim.simulation.progress import CooperativeProgressReporter

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


def run_scenario(
    mode: str = "smoke",
    raw_dir: str = "data/raw/occupancy",
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
    config: Optional[MonteCarloConfig] = None,
    visualize: bool = True,
) -> Dict[str, Any]:
    return run_dataset_anchored_scenario(
        OCCUPANCY_SPEC,
        mode=mode,
        raw_dir=raw_dir,
        output_dir=output_dir,
        reporter=reporter,
        config=config,
        visualize=visualize,
    )


def regenerate_from_scenario_run(
    scenario_run_id: int,
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
) -> Dict[str, Any]:
    return regenerate_dataset_anchored_scenario(
        OCCUPANCY_SPEC,
        scenario_run_id,
        output_dir=output_dir,
        reporter=reporter,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, default="smoke")
    parser.add_argument("--raw-dir", default="data/raw/occupancy")
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--report-from-scenario-run", type=int, default=None)
    parser.add_argument("--list-scenario-runs", action="store_true")
    parser.add_argument("--list-simulation-runs", action="store_true")
    parser.add_argument("--dataset-report-only", action="store_true")
    args = parser.parse_args()
    if args.list_scenario_runs:
        list_scenario_runs(args.output_dir)
        return 0
    if args.list_simulation_runs:
        list_simulation_runs(args.output_dir)
        return 0
    reporter = CooperativeProgressReporter(
        verbose=not args.quiet, no_color=args.no_color
    )
    try:
        if args.dataset_report_only:
            data = load_occupancy_data(args.raw_dir)
            report = generate_occupancy_dataset_report(
                data, Path(args.output_dir), filename="occupancy_dataset_report.html"
            )
            reporter.info(f"Dataset report written: {report}")
        elif args.report_from_scenario_run is not None:
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
            )
    except Exception as exc:  # noqa: BLE001
        reporter.error("Occupancy scenario command failed", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
