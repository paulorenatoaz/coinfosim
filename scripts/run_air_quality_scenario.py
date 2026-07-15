#!/usr/bin/env python3
"""Run or regenerate the tracked UCI Air Quality three-arm scenario."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Dict, Optional

from coinfosim.datasets.air_quality import load_air_quality_data
from coinfosim.scenarios.air_quality import (
    AIR_QUALITY_SCENARIO_QUESTION,
    build_gaussian_anchored_air_quality_model,
    build_gmm_anchored_air_quality_model,
)
from coinfosim.scenarios.dataset_anchored_runner import (
    DatasetAnchoredExecutionSpec,
    regenerate_dataset_anchored_scenario,
    run_dataset_anchored_scenario,
)
from coinfosim.simulation.config import MonteCarloConfig, VALID_MODES
from coinfosim.simulation.progress import CooperativeProgressReporter

SCENARIO_SLUG = "air_quality_baseline"
SCENARIO_NAME = "UCI Air Quality Baseline"
SCENARIO_FAMILY = "dataset"
REAL_SLUG = "air_quality_real_data"
REAL_FAMILY = "real_dataset"
SG2R_SLUG = "air_quality_single_gaussian_to_real"
SG2R_FAMILY = "single_gaussian_to_real"
GMM2R_SLUG = "air_quality_gmm_to_real"
GMM2R_FAMILY = "gmm_to_real"


def _write_temporary_report(output_dir: Path | str, filename: str, title: str, body: str) -> Path:
    """Write the minimal callback report used until Blocks 4 and 5."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>\n",
        encoding="utf-8",
    )
    return path


def _dataset_report(data, output_dir, *, filename):
    counts = data.row_counts()
    body = (
        "<p>Temporary Air Quality dataset report callback for runner validation.</p>"
        f"<p>Complete-case cohort: {counts['complete_case_cohort']}; "
        f"threshold: {data.threshold_value:g}.</p>"
    )
    return _write_temporary_report(
        output_dir, filename, "UCI Air Quality Dataset Report", body
    )


def _arm_report(title):
    def callback(result, channel_names, output_dir, *, filename, nstar_selection_result):
        body = (
            "<p>Temporary structured-report callback for runner validation.</p>"
            f"<p>Channels: {html.escape(', '.join(channel_names))}; "
            f"subsets: {len(result.subsets)}; fixed future real test.</p>"
        )
        return _write_temporary_report(output_dir, filename, title, body)

    return callback


def _scenario_report(
    real_result,
    gaussian_result,
    gmm_result,
    *,
    output_dir,
    dataset_report,
    real_report,
    sg_real_report,
    gmm_real_report,
    filename,
    channel_names,
    visualization,
    scenario_meta,
    graph_suffix,
    graphs_out,
    generate_graphs,
):
    links = (dataset_report, real_report, sg_real_report, gmm_real_report)
    body = (
        "<p>Temporary scenario-report callback for runner validation.</p>"
        "<p>Real → Real; Single Gaussian → Real; GMM → Real; fixed future real test.</p>"
        + "".join(
            f"<p><a href='{html.escape(link)}'>{html.escape(link)}</a></p>"
            for link in links
        )
    )
    return _write_temporary_report(output_dir, filename, SCENARIO_NAME, body)


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
    dataset_report_callback=_dataset_report,
    real_report_callback=_arm_report("Air Quality Real → Real"),
    gaussian_report_callback=_arm_report("Air Quality Single Gaussian → Real"),
    gmm_report_callback=_arm_report("Air Quality GMM → Real"),
    scenario_report_callback=_scenario_report,
    report_context_callback=_report_context,
    real_experiment_arm="real_to_real",
)


def run_scenario(
    mode: str = "smoke",
    raw_dir: str = "data/raw/air_quality",
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
    config: Optional[MonteCarloConfig] = None,
    visualize: bool = True,
) -> Dict[str, Any]:
    return run_dataset_anchored_scenario(
        AIR_QUALITY_SPEC,
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
        AIR_QUALITY_SPEC,
        scenario_run_id,
        output_dir=output_dir,
        reporter=reporter,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, default="smoke")
    parser.add_argument("--raw-dir", default="data/raw/air_quality")
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--report-from-scenario-run", type=int, default=None)
    args = parser.parse_args()
    reporter = CooperativeProgressReporter(
        verbose=not args.quiet, no_color=args.no_color
    )
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
            )
    except Exception as exc:  # noqa: BLE001
        reporter.error("Air Quality scenario command failed", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
