#!/usr/bin/env python3
"""Run the Occupancy Detection scenario with full run tracking.

This script executes the Occupancy Detection *scenario*: a single scenario run
composed of three simulation runs. The three main arms are:

    real_to_real            train on the real Occupancy training pool,
                            test on the fixed real Occupancy evaluation split.
    single_gaussian_to_real train on synthetic samples from a single
                            class-conditional Gaussian model estimated from the
                            Occupancy training pool, test on the same fixed real
                            Occupancy evaluation split.
    gmm_to_real             train on synthetic samples from class-conditional
                            Gaussian mixture models fitted to the Occupancy
                            training pool, test on the same fixed real Occupancy
                            evaluation split.

Both main arms are evaluated on the fixed real Occupancy evaluation split. Every
execution is tracked in two global registries and written to dedicated,
id-stamped folders so that re-running never overwrites a previous execution:

    output/reports/
      scenario_runs.json
      simulation_runs.json
      scenarios/000000_occupancy_baseline_smoke/
      simulations/000000_occupancy_real_data_smoke/
      simulations/000001_occupancy_single_gaussian_to_real_smoke/
      simulations/000002_occupancy_gmm_to_real_smoke/

Reports can also be regenerated from persisted run data without rerunning Monte
Carlo (``--report-from-scenario-run ID``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_dataset import generate_occupancy_dataset_report
from coinfosim.reports.occupancy_monte_carlo import (
    generate_occupancy_gmm_to_real_monte_carlo_report,
    generate_occupancy_real_monte_carlo_report,
    generate_occupancy_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.reports.scenario_visualization import (
    build_balanced_sample,
    generate_scenario_visualizations,
)
from coinfosim.results.persistence import (
    load_simulation_result,
    save_simulation_result,
)
from coinfosim.runs.registry import ScenarioRunRegistry, SimulationRunRegistry
from coinfosim.runs.report_data import (
    scenario_report_data,
    simulation_report_data,
    simulation_summary_snapshot,
)
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios.occupancy import (
    OCCUPANCY_SCENARIO_QUESTION,
    build_gaussian_anchored_occupancy_model,
    build_gmm_anchored_occupancy_model,
)
from coinfosim.simulation.config import MonteCarloConfig, VALID_MODES, get_mode_config
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

# Deterministic seed for the (separate) data-visualization sample.
VIZ_SEED = 20240501
VIZ_PER_CLASS = 512


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _config_dict(config: MonteCarloConfig) -> Dict[str, Any]:
    return {
        "mode": config.mode,
        "sample_sizes": list(config.sample_sizes),
        "min_replications": config.min_replications,
        "max_replications": config.max_replications,
        "replication_batch_size": config.replication_batch_size,
        "test_samples_per_class": config.test_samples_per_class,
        "ci_half_width_target": config.ci_half_width_target,
        "base_seed": config.base_seed,
    }


def _relpath(target: Path | str, start: Path | str) -> str:
    return os.path.relpath(str(target), str(start))


def _write_json(path: Path | str, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _sim_report_filename(slug: str, mode: str, rid: int) -> str:
    return f"{slug}_monte_carlo_report_{mode}_{rid:06d}.html"


def _scenario_report_filename(mode: str, sid: int) -> str:
    return f"{SCENARIO_SLUG}_scenario_report_{mode}_{sid:06d}.html"


def _dataset_report_filename(mode: str, sid: int) -> str:
    return f"occupancy_dataset_report_{mode}_{sid:06d}.html"


def _build_visualizations(
    data, gaussian_model, gmm_model, scenario_dir: Path, mode: str, sid: int
) -> Dict[str, Any]:
    """Render the nine data-visualization panels and return their descriptor.

    A separate, deterministic visualization sample is used (documented in the
    returned metadata): a balanced draw from the standardized real training
    pool, an equally sized synthetic draw from the single Gaussian model, and an
    equally sized synthetic draw from the class-conditional GMM.
    """
    real_X, real_y, per_class = build_balanced_sample(
        data.train_dataset.X, data.train_dataset.y, VIZ_PER_CLASS, seed=VIZ_SEED
    )
    gaussian_sampler = GaussianClassConditionalSampler(
        gaussian_model, base_seed=VIZ_SEED, test_samples_per_class=per_class
    )
    gaussian_sample = gaussian_sampler.sample_test()
    gmm_sampler = GMMClassConditionalSampler(
        gmm_model, base_seed=VIZ_SEED, test_samples_per_class=per_class
    )
    gmm_sample = gmm_sampler.sample_test()

    # Build per-class component list for the GMM visualization.
    gmm_components = {
        label: [
            (
                float(gmm_model.component_weights(label)[j]),
                gmm_model.component_means(label)[j],
                gmm_model.component_covariances(label)[j],
            )
            for j in range(gmm_model.selected_components(label))
        ]
        for label in gmm_model.class_labels
    }

    images = generate_scenario_visualizations(
        {
            "real": (real_X, real_y, "Real training sample", None),
            "gaussian": (
                gaussian_sample.X,
                gaussian_sample.y,
                "Single Gaussian synthetic training sample",
                None,
            ),
            "gmm": (
                gmm_sample.X,
                gmm_sample.y,
                "GMM synthetic training sample",
                gmm_components,
            ),
        },
        scenario_dir,
        filename_suffix=f"{mode}_{sid:06d}",
    )
    metadata = {
        "visualization_sample_size": int(real_X.shape[0]),
        "per_class": int(per_class),
        "class_balance": "balanced (equal samples per class)",
        "real_data_source": "standardized Occupancy training pool",
        "single_gaussian_source": "single Gaussian model",
        "gmm_source": "class-conditional Gaussian mixture models",
        "visualization_seed": VIZ_SEED,
    }
    return {"images": images, "metadata": metadata}


def _persist_simulation(
    sim_registry: SimulationRunRegistry,
    record,
    result,
    report_fn: Callable[[Path, str], Path],
    model_metadata: Dict[str, Any],
    sampler_metadata: Dict[str, Any],
):
    """Generate the report, persist full result + summary, and finalize the run."""
    sim_dir = Path(record.run_dir)
    rid = record.simulation_run_id
    mode = record.mode

    report_filename = _sim_report_filename(record.simulation_slug, mode, rid)
    report_path = report_fn(sim_dir, report_filename)

    result_gz = sim_dir / f"result_data_{mode}_{rid:06d}.json.gz"
    save_simulation_result(result, result_gz)

    summary_snapshot = simulation_summary_snapshot(result)
    summary_path = sim_dir / f"summary_{mode}_{rid:06d}.json"
    _write_json(summary_path, summary_snapshot)

    report_data = simulation_report_data(result)
    artifacts = {
        "monte_carlo_report": str(report_path),
        "result_data": str(result_gz),
        "summary": str(summary_path),
        "simulation_json": record.simulation_json_path,
    }
    completed = sim_registry.complete_run(
        rid,
        runtime_seconds=result.runtime_seconds,
        model_metadata=model_metadata,
        sampler_metadata=sampler_metadata,
        summary_data=summary_snapshot,
        result_data=report_data,
        artifacts=artifacts,
    )
    # simulation.json is the self-contained run record (control metadata +
    # report-ready data + pointer to the full persisted result payload).
    _write_json(record.simulation_json_path, completed.to_dict())
    return completed, report_path, result_gz, summary_path


def _fail_simulation(sim_registry: SimulationRunRegistry, record, error: str) -> None:
    """Mark a started simulation run failed and persist a failed simulation.json."""
    try:
        failed = sim_registry.fail_run(record.simulation_run_id, error=error)
        _write_json(record.simulation_json_path, failed.to_dict())
    except Exception:  # noqa: BLE001 - never mask the original failure
        pass


# --------------------------------------------------------------------------- #
# Scenario execution
# --------------------------------------------------------------------------- #
def run_scenario(
    mode: str = "smoke",
    raw_dir: str = "data/raw/occupancy",
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
    config: Optional[MonteCarloConfig] = None,
    visualize: bool = True,
) -> Dict[str, Any]:
    """Run the Occupancy baseline scenario with full run tracking.

    Creates one scenario run and two simulation runs (real-data and
    Gaussian-anchored). ``config`` overrides the mode preset (used by tests to
    keep runs tiny). ``visualize`` toggles the data-visualization panels
    (disabled by tests that do not need them). Returns a dict with the
    allocated ids and output paths.
    """
    if reporter is None:
        reporter = CooperativeProgressReporter(verbose=False)
    if config is None:
        config = get_mode_config(mode)
    mode = config.mode

    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    scenario_registry = ScenarioRunRegistry(base_output_dir=base_dir)
    sim_registry = SimulationRunRegistry(base_output_dir=base_dir)
    scenario_registry.ensure_registry()
    sim_registry.ensure_registry()

    reporter.scenario_start(
        scenario_name=SCENARIO_NAME,
        mode=mode,
        raw_dir=raw_dir,
        output_dir=base_dir,
        config=config,
    )

    scenario_run = scenario_registry.start_run(
        scenario_slug=SCENARIO_SLUG,
        scenario_name=SCENARIO_NAME,
        scenario_family=SCENARIO_FAMILY,
        question=OCCUPANCY_SCENARIO_QUESTION,
        mode=mode,
        config=_config_dict(config),
    )
    sid = scenario_run.scenario_run_id
    scenario_dir = Path(scenario_run.run_dir)

    reporter.info(f"scenario_run_id={sid}")
    reporter.info(f"scenario run dir: {scenario_dir}")
    reporter.info(f"scenario registry: {scenario_registry.registry_path}")
    reporter.info(f"simulation registry: {sim_registry.registry_path}")

    start = time.time()
    real_record = None
    gaussian_record = None
    gmm_record = None

    try:
        # -- dataset (scenario-level artifact) ---------------------------- #
        reporter.scenario_step_start("Loading Occupancy dataset", detail=raw_dir)
        step = time.time()
        data = load_occupancy_data(raw_dir)
        reporter.scenario_step_finish(
            "Loading Occupancy dataset", elapsed=time.time() - step
        )

        # The dataset report describes the shared raw dataset, so it is a
        # scenario-level artifact stored in the scenario folder.
        reporter.scenario_step_start("Generating dataset report")
        step = time.time()
        dataset_report = generate_occupancy_dataset_report(
            data, scenario_dir, filename=_dataset_report_filename(mode, sid)
        )
        reporter.scenario_step_finish(
            "Generating dataset report",
            elapsed=time.time() - step,
            detail=str(dataset_report),
        )

        # -- real-data arm ------------------------------------------------- #
        reporter.scenario_step_start("Real-data Monte Carlo (real-data arm)")
        step = time.time()
        real_record = sim_registry.start_run(
            simulation_slug=REAL_SLUG,
            simulation_family=REAL_FAMILY,
            mode=mode,
            scenario_run_id_origin=sid,
            config=_config_dict(config),
        )
        reporter.info(
            f"simulation_run_id={real_record.simulation_run_id} "
            f"({REAL_SLUG}) dir: {real_record.run_dir}"
        )
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
                "scenario_name": SCENARIO_NAME,
                "experiment_arm": "real_data",
                "channel_names": list(data.channel_names),
                "standardization": "train_pool_only",
            },
            progress=reporter,
        ).run()
        real_completed, real_report, real_result_gz, real_summary_path = (
            _persist_simulation(
                sim_registry,
                real_record,
                real_result,
                report_fn=lambda d, f: generate_occupancy_real_monte_carlo_report(
                    real_result, d, filename=f, nstar_selection_result=real_result
                ),
                model_metadata=dict(real_result.metadata),
                sampler_metadata={
                    "type": "RealDatasetSampler",
                    "base_seed": config.base_seed,
                    "fixed_test_description": (
                        "standardized datatest.txt + datatest2.txt"
                    ),
                },
            )
        )
        reporter.scenario_step_finish(
            "Real-data Monte Carlo (real-data arm)",
            elapsed=time.time() - step,
            detail=str(real_report),
        )

        # -- single Gaussian -> real arm ---------------------------------- #
        reporter.scenario_step_start("Building single Gaussian model")
        step = time.time()
        anchored = build_gaussian_anchored_occupancy_model(data)
        reporter.scenario_step_finish(
            "Building single Gaussian model", elapsed=time.time() - step
        )

        reporter.scenario_step_start("Building class-conditional GMM model")
        step = time.time()
        gmm_anchored = build_gmm_anchored_occupancy_model(data)
        reporter.scenario_step_finish(
            "Building class-conditional GMM model", elapsed=time.time() - step
        )

        # Data-visualization panels (separate deterministic sample; documented
        # in the visualization metadata). Real sample is drawn from the
        # standardized training pool; synthetic samples from the single Gaussian
        # and GMM models.
        visualization = None
        if visualize:
            reporter.scenario_step_start("Rendering data visualization panels")
            step = time.time()
            visualization = _build_visualizations(
                data, anchored.model, gmm_anchored.model, scenario_dir, mode, sid
            )
            reporter.scenario_step_finish(
                "Rendering data visualization panels",
                elapsed=time.time() - step,
            )

        reporter.scenario_step_start(
            "Single Gaussian \u2192 Real Monte Carlo (single_gaussian_to_real arm)"
        )
        step = time.time()
        gaussian_record = sim_registry.start_run(
            simulation_slug=SG2R_SLUG,
            simulation_family=SG2R_FAMILY,
            mode=mode,
            scenario_run_id_origin=sid,
            config=_config_dict(config),
        )
        reporter.info(
            f"simulation_run_id={gaussian_record.simulation_run_id} "
            f"({SG2R_SLUG}) dir: {gaussian_record.run_dir}"
        )
        # Training samples are synthetic (single Gaussian model); evaluation uses
        # the fixed real Occupancy evaluation split (same split as the real arm).
        gaussian_train_sampler = GaussianClassConditionalSampler(
            anchored.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        )
        gaussian_sampler = SyntheticTrainRealTestSampler(
            gaussian_train_sampler,
            data.test_dataset,
            name="occupancy_single_gaussian_to_real",
        )
        gaussian_result = CooperativeMonteCarloSimulator(
            anchored.model,
            config,
            sampler=gaussian_sampler,
            metadata={
                "scenario_name": SCENARIO_NAME,
                "experiment_arm": "single_gaussian_to_real",
                "train_source": "single_gaussian_synthetic",
                "test_source": "real_occupancy_evaluation_split",
                "channel_names": list(data.channel_names),
                "standardization": "train_pool_only",
                "gaussian_ridge_by_class": dict(anchored.ridge_by_class),
            },
            progress=reporter,
        ).run()
        (
            gaussian_completed,
            gaussian_report,
            gaussian_result_gz,
            gaussian_summary_path,
        ) = _persist_simulation(
            sim_registry,
            gaussian_record,
            gaussian_result,
            report_fn=lambda d, f: (
                generate_occupancy_single_gaussian_to_real_monte_carlo_report(
                    gaussian_result,
                    data.channel_names,
                    d,
                    filename=f,
                    nstar_selection_result=real_result,
                )
            ),
            model_metadata=dict(gaussian_result.metadata),
            sampler_metadata={
                "type": "SyntheticTrainRealTestSampler",
                "train_sampler": "GaussianClassConditionalSampler",
                "base_seed": config.base_seed,
                "train_source": "single_gaussian_synthetic",
                "test_source": "real_occupancy_evaluation_split",
                "fixed_test_description": (
                    "standardized datatest.txt + datatest2.txt"
                ),
            },
        )
        reporter.scenario_step_finish(
            "Single Gaussian \u2192 Real Monte Carlo (single_gaussian_to_real arm)",
            elapsed=time.time() - step,
            detail=str(gaussian_report),
        )

        # -- GMM -> real arm ---------------------------------------------- #
        reporter.scenario_step_start(
            "GMM \u2192 Real Monte Carlo (gmm_to_real arm)"
        )
        step = time.time()
        gmm_record = sim_registry.start_run(
            simulation_slug=GMM2R_SLUG,
            simulation_family=GMM2R_FAMILY,
            mode=mode,
            scenario_run_id_origin=sid,
            config=_config_dict(config),
        )
        reporter.info(
            f"simulation_run_id={gmm_record.simulation_run_id} "
            f"({GMM2R_SLUG}) dir: {gmm_record.run_dir}"
        )
        # Training samples are synthetic (class-conditional GMMs); evaluation
        # uses the fixed real Occupancy evaluation split (same split as above).
        gmm_train_sampler = GMMClassConditionalSampler(
            gmm_anchored.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        )
        gmm_sampler = SyntheticTrainRealTestSampler(
            gmm_train_sampler,
            data.test_dataset,
            name="occupancy_gmm_to_real",
        )
        gmm_result = CooperativeMonteCarloSimulator(
            gmm_anchored.model,
            config,
            sampler=gmm_sampler,
            metadata={
                "scenario_name": SCENARIO_NAME,
                "experiment_arm": "gmm_to_real",
                "train_source": "gmm_synthetic",
                "test_source": "real_occupancy_evaluation_split",
                "channel_names": list(data.channel_names),
                "standardization": "train_pool_only",
                "gmm_model_selection": gmm_anchored.model_selection,
            },
            progress=reporter,
        ).run()
        (
            gmm_completed,
            gmm_report,
            gmm_result_gz,
            gmm_summary_path,
        ) = _persist_simulation(
            sim_registry,
            gmm_record,
            gmm_result,
            report_fn=lambda d, f: (
                generate_occupancy_gmm_to_real_monte_carlo_report(
                    gmm_result,
                    data.channel_names,
                    d,
                    filename=f,
                    nstar_selection_result=real_result,
                )
            ),
            model_metadata=dict(gmm_result.metadata),
            sampler_metadata={
                "type": "SyntheticTrainRealTestSampler",
                "train_sampler": "GMMClassConditionalSampler",
                "base_seed": config.base_seed,
                "train_source": "gmm_synthetic",
                "test_source": "real_occupancy_evaluation_split",
                "fixed_test_description": (
                    "standardized datatest.txt + datatest2.txt"
                ),
            },
        )
        reporter.scenario_step_finish(
            "GMM \u2192 Real Monte Carlo (gmm_to_real arm)",
            elapsed=time.time() - step,
            detail=str(gmm_report),
        )

        # -- scenario report ----------------------------------------------- #
        reporter.scenario_step_start("Generating scenario report")
        step = time.time()
        scenario_meta = {
            "scenario_run_id": sid,
            "scenario_family": SCENARIO_FAMILY,
            "mode": mode,
            "dataset": "Occupancy Detection",
        }
        graphs: Dict[str, Any] = {}
        scenario_report = generate_occupancy_scenario_report(
            real_result,
            gaussian_result,
            gmm_result,
            output_dir=scenario_dir,
            dataset_report=_relpath(dataset_report, scenario_dir),
            real_report=_relpath(real_report, scenario_dir),
            sg_real_report=_relpath(gaussian_report, scenario_dir),
            gmm_real_report=_relpath(gmm_report, scenario_dir),
            filename=_scenario_report_filename(mode, sid),
            channel_names=data.channel_names,
            visualization=visualization,
            scenario_meta=scenario_meta,
            graph_suffix=f"{mode}_{sid:06d}",
            graphs_out=graphs,
            generate_graphs=visualize,
        )
        reporter.scenario_step_finish(
            "Generating scenario report",
            elapsed=time.time() - step,
            detail=str(scenario_report),
        )

        runtime = time.time() - start

        simulation_refs = {
            REAL_SLUG: {
                "simulation_run_id": real_completed.simulation_run_id,
                "simulation_family": REAL_FAMILY,
                "run_dir": real_completed.run_dir,
                "simulation_json_path": real_completed.simulation_json_path,
                "report_path": str(real_report),
                "result_data_path": str(real_result_gz),
                "summary_data": real_completed.summary_data,
            },
            SG2R_SLUG: {
                "simulation_run_id": gaussian_completed.simulation_run_id,
                "simulation_family": SG2R_FAMILY,
                "run_dir": gaussian_completed.run_dir,
                "simulation_json_path": gaussian_completed.simulation_json_path,
                "report_path": str(gaussian_report),
                "result_data_path": str(gaussian_result_gz),
                "summary_data": gaussian_completed.summary_data,
            },
            GMM2R_SLUG: {
                "simulation_run_id": gmm_completed.simulation_run_id,
                "simulation_family": GMM2R_FAMILY,
                "run_dir": gmm_completed.run_dir,
                "simulation_json_path": gmm_completed.simulation_json_path,
                "report_path": str(gmm_report),
                "result_data_path": str(gmm_result_gz),
                "summary_data": gmm_completed.summary_data,
            },
        }
        artifacts = {
            "scenario_report": str(scenario_report),
            "dataset_report": str(dataset_report),
            "scenario_json": scenario_run.scenario_json_path,
        }
        if visualization:
            artifacts["visualization_images"] = {
                key: str(scenario_dir / fname)
                for key, fname in visualization["images"].items()
            }
        if graphs:
            artifacts["graph_images"] = {
                key: str(scenario_dir / fname) for key, fname in graphs.items()
            }
        report_data = scenario_report_data(
            real_result,
            gaussian_result,
            gmm_result,
            data.channel_names,
            gmm_model_selection=gmm_anchored.model_selection,
        )
        if visualization:
            report_data["visualization"] = visualization
        if graphs:
            report_data["graphs"] = graphs

        completed_scenario = scenario_registry.complete_run(
            sid,
            runtime_seconds=runtime,
            simulation_run_ids=[
                real_completed.simulation_run_id,
                gaussian_completed.simulation_run_id,
                gmm_completed.simulation_run_id,
            ],
            simulation_refs=simulation_refs,
            artifacts=artifacts,
            report_data=report_data,
        )
        _write_json(scenario_run.scenario_json_path, completed_scenario.to_dict())

        outputs = {
            "scenario_report": scenario_report,
            "dataset_report": dataset_report,
            "real_report": real_report,
            "single_gaussian_to_real_report": gaussian_report,
            "gmm_to_real_report": gmm_report,
            "scenario_json": Path(scenario_run.scenario_json_path),
            "real_simulation_json": Path(real_completed.simulation_json_path),
            "single_gaussian_to_real_simulation_json": Path(
                gaussian_completed.simulation_json_path
            ),
            "gmm_to_real_simulation_json": Path(
                gmm_completed.simulation_json_path
            ),
            "real_result_data": real_result_gz,
            "single_gaussian_to_real_result_data": gaussian_result_gz,
            "gmm_to_real_result_data": gmm_result_gz,
            "scenario_runs.json": scenario_registry.registry_path,
            "simulation_runs.json": sim_registry.registry_path,
        }
        reporter.scenario_finish(runtime=runtime, outputs=outputs)

        return {
            "scenario_run_id": sid,
            "real_simulation_run_id": real_completed.simulation_run_id,
            "single_gaussian_to_real_simulation_run_id": (
                gaussian_completed.simulation_run_id
            ),
            "gmm_to_real_simulation_run_id": gmm_completed.simulation_run_id,
            "scenario_run_dir": str(scenario_dir),
            "real_run_dir": real_completed.run_dir,
            "single_gaussian_to_real_run_dir": gaussian_completed.run_dir,
            "gmm_to_real_run_dir": gmm_completed.run_dir,
            "scenario_registry": str(scenario_registry.registry_path),
            "simulation_registry": str(sim_registry.registry_path),
            "runtime_seconds": runtime,
            **{k: str(v) for k, v in outputs.items()},
        }

    except Exception as exc:  # noqa: BLE001 - surface failure, mark runs failed
        error = f"{type(exc).__name__}: {exc}"
        if real_record is not None:
            current = sim_registry.get_run(real_record.simulation_run_id)
            if current is not None and current.status == "running":
                _fail_simulation(sim_registry, real_record, error)
        if gaussian_record is not None:
            current = sim_registry.get_run(gaussian_record.simulation_run_id)
            if current is not None and current.status == "running":
                _fail_simulation(sim_registry, gaussian_record, error)
        if gmm_record is not None:
            current = sim_registry.get_run(gmm_record.simulation_run_id)
            if current is not None and current.status == "running":
                _fail_simulation(sim_registry, gmm_record, error)
        try:
            failed_scenario = scenario_registry.fail_run(sid, error=error)
            _write_json(scenario_run.scenario_json_path, failed_scenario.to_dict())
        except Exception:  # noqa: BLE001
            pass
        reporter.error(
            f"Occupancy scenario run failed (scenario_run_id={sid}, "
            f"run_dir={scenario_dir})",
            exc,
        )
        raise


# --------------------------------------------------------------------------- #
# Report regeneration (no Monte Carlo rerun)
# --------------------------------------------------------------------------- #
def regenerate_from_scenario_run(
    scenario_run_id: int,
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
) -> Dict[str, Any]:
    """Regenerate Occupancy reports from a persisted scenario run.

    Loads the persisted simulation results and re-renders the simulation and
    scenario reports in place. Monte Carlo is *not* rerun. The dataset report is
    a scenario artifact that depends on the raw data pool (not persisted with a
    result), so the existing dataset report file is reused as-is for links.
    """
    if reporter is None:
        reporter = CooperativeProgressReporter(verbose=True)

    scenario_registry = ScenarioRunRegistry(base_output_dir=output_dir)
    record = scenario_registry.get_run(scenario_run_id)
    if record is None:
        raise KeyError(
            f"scenario_run_id {scenario_run_id} not found in "
            f"{scenario_registry.registry_path}"
        )

    scenario_json = json.loads(
        Path(record.scenario_json_path).read_text(encoding="utf-8")
    )
    refs = scenario_json.get("simulation_refs", {})
    scenario_dir = Path(record.run_dir)
    channel_names = scenario_json.get("report_data", {}).get("channel_names", [])

    reporter.info(f"Regenerating reports for scenario_run_id={scenario_run_id}")
    reporter.info(f"scenario run dir: {scenario_dir}")

    regenerated: Dict[str, Any] = {}

    # Real-data arm.
    real_ref = refs[REAL_SLUG]
    real_result = load_simulation_result(real_ref["result_data_path"])
    real_dir = Path(real_ref["run_dir"])
    real_report = generate_occupancy_real_monte_carlo_report(
        real_result,
        real_dir,
        filename=Path(real_ref["report_path"]).name,
        nstar_selection_result=real_result,
    )
    regenerated["real_report"] = str(real_report)
    reporter.scenario_step_finish(
        "Regenerated real-data report", detail=str(real_report)
    )

    # Single Gaussian -> Real arm.
    gaussian_ref = refs[SG2R_SLUG]
    gaussian_result = load_simulation_result(gaussian_ref["result_data_path"])
    gaussian_dir = Path(gaussian_ref["run_dir"])
    gaussian_report = generate_occupancy_single_gaussian_to_real_monte_carlo_report(
        gaussian_result,
        channel_names or gaussian_result.metadata.get("channel_names", []),
        gaussian_dir,
        filename=Path(gaussian_ref["report_path"]).name,
        nstar_selection_result=real_result,
    )
    regenerated["single_gaussian_to_real_report"] = str(gaussian_report)
    reporter.scenario_step_finish(
        "Regenerated Single Gaussian → Real report", detail=str(gaussian_report)
    )

    # GMM -> Real arm.
    gmm_ref = refs[GMM2R_SLUG]
    gmm_result = load_simulation_result(gmm_ref["result_data_path"])
    gmm_dir = Path(gmm_ref["run_dir"])
    gmm_report = generate_occupancy_gmm_to_real_monte_carlo_report(
        gmm_result,
        channel_names or gmm_result.metadata.get("channel_names", []),
        gmm_dir,
        filename=Path(gmm_ref["report_path"]).name,
        nstar_selection_result=real_result,
    )
    regenerated["gmm_to_real_report"] = str(gmm_report)
    reporter.scenario_step_finish(
        "Regenerated GMM → Real report", detail=str(gmm_report)
    )

    # Scenario report (dataset report and visualization panels reused as-is).
    dataset_report = record.artifacts.get("dataset_report", "")
    visualization = scenario_json.get("report_data", {}).get("visualization")
    scenario_meta = {
        "scenario_run_id": scenario_run_id,
        "scenario_family": record.scenario_family,
        "mode": record.mode,
        "dataset": "Occupancy Detection",
    }
    scenario_report = generate_occupancy_scenario_report(
        real_result,
        gaussian_result,
        gmm_result,
        output_dir=scenario_dir,
        dataset_report=(
            _relpath(dataset_report, scenario_dir)
            if dataset_report
            else "occupancy_dataset_report.html"
        ),
        real_report=_relpath(real_report, scenario_dir),
        sg_real_report=_relpath(gaussian_report, scenario_dir),
        gmm_real_report=_relpath(gmm_report, scenario_dir),
        filename=Path(
            record.artifacts.get(
                "scenario_report",
                _scenario_report_filename(record.mode, scenario_run_id),
            )
        ).name,
        channel_names=channel_names or real_result.metadata.get("channel_names", []),
        visualization=visualization,
        scenario_meta=scenario_meta,
        graph_suffix=f"{record.mode}_{scenario_run_id:06d}",
    )
    regenerated["scenario_report"] = str(scenario_report)
    reporter.scenario_step_finish(
        "Regenerated scenario report", detail=str(scenario_report)
    )
    reporter.info("Regeneration complete (Monte Carlo was not rerun).")
    return regenerated


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def list_scenario_runs(output_dir: str = "output/reports") -> None:
    registry = ScenarioRunRegistry(base_output_dir=output_dir)
    runs = registry.list_runs()
    print(f"Scenario runs registry: {registry.registry_path}")
    if not runs:
        print("  (no scenario runs)")
        return
    for r in runs:
        report = r.artifacts.get("scenario_report", "-")
        print(
            f"  id={r.scenario_run_id} slug={r.scenario_slug} "
            f"family={r.scenario_family} mode={r.mode} status={r.status} "
            f"started={r.started_at} runtime={r.runtime_seconds} report={report}"
        )


def list_simulation_runs(output_dir: str = "output/reports") -> None:
    registry = SimulationRunRegistry(base_output_dir=output_dir)
    runs = registry.list_runs()
    print(f"Simulation runs registry: {registry.registry_path}")
    if not runs:
        print("  (no simulation runs)")
        return
    for r in runs:
        report = r.artifacts.get("monte_carlo_report", "-")
        print(
            f"  id={r.simulation_run_id} slug={r.simulation_slug} "
            f"family={r.simulation_family} mode={r.mode} status={r.status} "
            f"started={r.started_at} runtime={r.runtime_seconds} report={report}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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
    parser.add_argument(
        "--report-from-scenario-run",
        type=int,
        default=None,
        metavar="ID",
        help="Regenerate reports from an existing scenario run id (no rerun).",
    )
    parser.add_argument(
        "--list-scenario-runs",
        action="store_true",
        help="List tracked scenario runs and exit.",
    )
    parser.add_argument(
        "--list-simulation-runs",
        action="store_true",
        help="List tracked simulation runs and exit.",
    )
    parser.add_argument(
        "--dataset-report-only",
        action="store_true",
        help=(
            "Generate (or regenerate) only the Occupancy dataset report "
            "from the raw data files. No Monte Carlo simulation is run."
        ),
    )
    args = parser.parse_args()

    if args.list_scenario_runs:
        list_scenario_runs(args.output_dir)
        return 0
    if args.list_simulation_runs:
        list_simulation_runs(args.output_dir)
        return 0

    reporter = CooperativeProgressReporter(
        verbose=not args.quiet,
        no_color=args.no_color,
    )

    if args.dataset_report_only:
        try:
            data = load_occupancy_data(args.raw_dir)
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report = generate_occupancy_dataset_report(
                data, out_dir, filename="occupancy_dataset_report.html"
            )
            reporter.info(f"Dataset report written: {report}")
        except FileNotFoundError as exc:
            reporter.error(
                "Occupancy raw data not found. Expected files under "
                f"{args.raw_dir!r} (datatraining.txt, datatest.txt, datatest2.txt)",
                exc,
            )
            return 1
        except Exception as exc:  # noqa: BLE001
            reporter.error("Dataset report generation failed", exc)
            return 1
        return 0

    if args.report_from_scenario_run is not None:
        try:
            regenerate_from_scenario_run(
                args.report_from_scenario_run,
                output_dir=args.output_dir,
                reporter=reporter,
            )
        except Exception as exc:  # noqa: BLE001
            reporter.error("Report regeneration failed", exc)
            return 1
        return 0

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
    except Exception:  # noqa: BLE001 - already reported inside run_scenario
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
