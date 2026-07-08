#!/usr/bin/env python3
"""Run the Sprint 2 Occupancy Detection scenario."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_dataset import generate_occupancy_dataset_report
from coinfosim.reports.occupancy_monte_carlo import (
    generate_occupancy_gaussian_anchored_monte_carlo_report,
    generate_occupancy_real_monte_carlo_report,
)
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.scenarios.occupancy import build_gaussian_anchored_occupancy_model
from coinfosim.simulation.config import VALID_MODES, get_mode_config
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator
from coinfosim.simulation.progress import CooperativeProgressReporter


def run_scenario(
    mode: str = "smoke",
    raw_dir: str = "data/raw/occupancy",
    output_dir: str = "output/reports",
    reporter: CooperativeProgressReporter | None = None,
) -> dict:
    """Run the Occupancy scenario end to end.

    Returns a dict of the generated output paths. ``reporter`` controls console
    output; when ``None`` the run is silent (useful for tests).
    """
    if reporter is None:
        reporter = CooperativeProgressReporter(verbose=False)

    start = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = get_mode_config(mode)
    reporter.scenario_start(
        scenario_name="Occupancy Detection",
        mode=config.mode,
        raw_dir=raw_dir,
        output_dir=out_dir,
        config=config,
    )

    reporter.scenario_step_start("Loading Occupancy dataset", detail=raw_dir)
    step = time.time()
    data = load_occupancy_data(raw_dir)
    reporter.scenario_step_finish(
        "Loading Occupancy dataset", elapsed=time.time() - step
    )

    reporter.scenario_step_start("Generating dataset report")
    step = time.time()
    dataset_report = generate_occupancy_dataset_report(data, out_dir)
    reporter.scenario_step_finish(
        "Generating dataset report",
        elapsed=time.time() - step,
        detail=str(dataset_report),
    )

    reporter.scenario_step_start("Real-data Monte Carlo (real-data arm)")
    step = time.time()
    real_sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=config.base_seed,
        channel_names=data.channel_names,
        name="occupancy_real_data",
    )
    real_result = CooperativeMonteCarloSimulator(
        real_sampler.model,
        config,
        sampler=real_sampler,
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "real_data",
            "channel_names": list(data.channel_names),
            "standardization": "train_pool_only",
        },
        progress=reporter,
    ).run()
    real_report = generate_occupancy_real_monte_carlo_report(real_result, out_dir)
    reporter.scenario_step_finish(
        "Real-data Monte Carlo (real-data arm)",
        elapsed=time.time() - step,
        detail=str(real_report),
    )

    reporter.scenario_step_start("Building Gaussian-anchored model")
    step = time.time()
    anchored = build_gaussian_anchored_occupancy_model(data)
    reporter.scenario_step_finish(
        "Building Gaussian-anchored model", elapsed=time.time() - step
    )

    reporter.scenario_step_start(
        "Gaussian-anchored Monte Carlo (Gaussian-anchored arm)"
    )
    step = time.time()
    gaussian_sampler = GaussianClassConditionalSampler(
        anchored.model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    gaussian_result = CooperativeMonteCarloSimulator(
        anchored.model,
        config,
        sampler=gaussian_sampler,
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "gaussian_anchored",
            "channel_names": list(data.channel_names),
            "standardization": "train_pool_only",
            "gaussian_ridge_by_class": dict(anchored.ridge_by_class),
        },
        progress=reporter,
    ).run()
    gaussian_report = generate_occupancy_gaussian_anchored_monte_carlo_report(
        gaussian_result,
        data.channel_names,
        out_dir,
    )
    reporter.scenario_step_finish(
        "Gaussian-anchored Monte Carlo (Gaussian-anchored arm)",
        elapsed=time.time() - step,
        detail=str(gaussian_report),
    )

    reporter.scenario_step_start("Generating scenario report")
    step = time.time()
    scenario_report = generate_occupancy_scenario_report(
        real_result,
        gaussian_result,
        output_dir=out_dir,
        channel_names=data.channel_names,
    )
    reporter.scenario_step_finish(
        "Generating scenario report",
        elapsed=time.time() - step,
        detail=str(scenario_report),
    )

    reporter.scenario_step_start("Writing JSON summaries")
    step = time.time()
    real_summary = _result_summary(real_result, real_report)
    gaussian_summary = _result_summary(gaussian_result, gaussian_report)
    real_summary_path = out_dir / "occupancy_real_monte_carlo_summary.json"
    gaussian_summary_path = (
        out_dir / "occupancy_gaussian_anchored_monte_carlo_summary.json"
    )
    real_summary_path.write_text(json.dumps(real_summary, indent=2), encoding="utf-8")
    gaussian_summary_path.write_text(
        json.dumps(gaussian_summary, indent=2), encoding="utf-8"
    )
    reporter.scenario_step_finish(
        "Writing JSON summaries", elapsed=time.time() - step
    )

    runtime = time.time() - start
    outputs = {
        "dataset_report": dataset_report,
        "scenario_report": scenario_report,
        "real_report": real_report,
        "gaussian_report": gaussian_report,
        "real_summary": real_summary_path,
        "gaussian_summary": gaussian_summary_path,
    }
    reporter.scenario_finish(runtime=runtime, outputs=outputs)

    return {
        "runtime_seconds": runtime,
        "config": config,
        "real_summary": real_summary,
        "gaussian_summary": gaussian_summary,
        **{k: str(v) for k, v in outputs.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, default="smoke")
    parser.add_argument("--raw-dir", default="data/raw/occupancy")
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (errors are still shown).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output (useful for redirected logs and CI).",
    )
    args = parser.parse_args()

    reporter = CooperativeProgressReporter(
        verbose=not args.quiet,
        no_color=args.no_color,
    )

    try:
        run_scenario(
            mode=args.mode,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            reporter=reporter,
        )
    except FileNotFoundError as exc:
        reporter.error(
            "Occupancy raw data not found. Expected files under "
            f"{args.raw_dir!r} (datatraining.txt, datatest.txt, datatest2.txt)",
            exc,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any failure clearly
        reporter.error("Occupancy scenario run failed", exc)
        return 1
    return 0


def _result_summary(result, report_path: Path) -> dict:
    return {
        "report_path": str(report_path),
        "mode": result.config.mode,
        "sample_sizes": list(result.sample_sizes),
        "number_of_subsets": len(result.subsets),
        "number_of_classifiers": len(result.classifier_names),
        "classifier_names": list(result.classifier_names),
        "fixed_test_size": result.metadata.get("fixed_test_size"),
        "metadata": result.metadata,
        "stopping_info": {
            str(n): {
                "replications": result.stopping_info[n].replications,
                "reason": result.stopping_info[n].reason,
                "max_ci_half_width": result.stopping_info[n].max_ci_half_width,
            }
            for n in result.sample_sizes
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
