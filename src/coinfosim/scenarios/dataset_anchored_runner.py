"""Generic execution and persistence for three-arm dataset scenarios."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

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
    dataset_anchored_scenario_report_data,
    simulation_report_data,
    simulation_summary_snapshot,
)
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.simulation.config import MonteCarloConfig, get_mode_config
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator
from coinfosim.simulation.progress import CooperativeProgressReporter

REAL_ARM_ID = "real_to_real"
GAUSSIAN_ARM_ID = "single_gaussian_to_real"
GMM_ARM_ID = "gmm_to_real"
GAUSSIAN_TRAIN_SOURCE = "single_gaussian_synthetic"
GMM_TRAIN_SOURCE = "gmm_synthetic"
VIZ_SEED = 20240501
VIZ_PER_CLASS = 512


@dataclass(frozen=True)
class DatasetAnchoredExecutionSpec:
    """Typed configuration for the standard dataset-anchored three-arm run."""

    scenario_slug: str
    scenario_name: str
    scenario_family: str
    question: str
    dataset_slug: str
    dataset_name: str
    real_simulation_slug: str
    real_simulation_family: str
    gaussian_simulation_slug: str
    gaussian_simulation_family: str
    gmm_simulation_slug: str
    gmm_simulation_family: str
    real_training_source_id: str
    real_test_source_id: str
    real_training_description: str
    fixed_test_description: str
    visualization_real_label: str
    visualization_gaussian_label: str
    visualization_gmm_label: str
    visualization_real_source: str
    visualization_gaussian_source: str
    visualization_gmm_source: str
    dataset_report_prefix: str
    scenario_report_prefix: str
    loader: Callable[[str], Any]
    gaussian_builder: Callable[[Any], Any]
    gmm_builder: Callable[[Any], Any]
    dataset_report_callback: Callable[..., Path]
    real_report_callback: Callable[..., Path]
    gaussian_report_callback: Callable[..., Path]
    gmm_report_callback: Callable[..., Path]
    scenario_report_callback: Callable[..., Path]
    report_context_callback: Callable[[Any], Mapping[str, Mapping[str, Any]]]
    real_experiment_arm: str = "real_data"


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
    """Write strict JSON atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return path


def _simulation_report_filename(slug: str, mode: str, run_id: int) -> str:
    return f"{slug}_monte_carlo_report_{mode}_{run_id:06d}.html"


def _scenario_report_filename(
    spec: DatasetAnchoredExecutionSpec, mode: str, run_id: int
) -> str:
    return f"{spec.scenario_report_prefix}_{mode}_{run_id:06d}.html"


def _dataset_report_filename(
    spec: DatasetAnchoredExecutionSpec, mode: str, run_id: int
) -> str:
    return f"{spec.dataset_report_prefix}_{mode}_{run_id:06d}.html"


def _validate_sample_sizes(data: Any, config: MonteCarloConfig) -> None:
    labels, counts = np.unique(data.train_dataset.y, return_counts=True)
    if len(labels) < 2:
        raise ValueError("training reservoir must contain at least two classes")
    minority_count = int(counts.min())
    infeasible = [int(n) for n in config.sample_sizes if int(n) > minority_count]
    if infeasible:
        raise ValueError(
            "requested n_per_class values exceed the real training reservoir "
            f"minority-class count {minority_count}: {infeasible}"
        )


def _assert_shared_fixed_test(data: Any, *samplers: Any) -> None:
    expected = data.test_dataset
    for sampler in samplers:
        observed = sampler.sample_test()
        if observed is not expected:
            raise RuntimeError("all scenario arms must reuse the exact fixed test Dataset")
        if not (
            np.array_equal(observed.X, expected.X)
            and np.array_equal(observed.y, expected.y)
        ):
            raise RuntimeError("scenario arms do not have identical fixed test content")


def _build_visualizations(
    spec: DatasetAnchoredExecutionSpec,
    data: Any,
    gaussian_model: Any,
    gmm_model: Any,
    scenario_dir: Path,
    mode: str,
    scenario_run_id: int,
) -> Dict[str, Any]:
    real_X, real_y, per_class = build_balanced_sample(
        data.train_dataset.X,
        data.train_dataset.y,
        VIZ_PER_CLASS,
        seed=VIZ_SEED,
    )
    gaussian_sample = GaussianClassConditionalSampler(
        gaussian_model, base_seed=VIZ_SEED, test_samples_per_class=per_class
    ).sample_test()
    gmm_sample = GMMClassConditionalSampler(
        gmm_model, base_seed=VIZ_SEED, test_samples_per_class=per_class
    ).sample_test()
    gmm_components = {
        label: [
            (
                float(gmm_model.component_weights(label)[index]),
                gmm_model.component_means(label)[index],
                gmm_model.component_covariances(label)[index],
            )
            for index in range(gmm_model.selected_components(label))
        ]
        for label in gmm_model.class_labels
    }
    images = generate_scenario_visualizations(
        {
            "real": (real_X, real_y, spec.visualization_real_label, None),
            "gaussian": (
                gaussian_sample.X,
                gaussian_sample.y,
                spec.visualization_gaussian_label,
                None,
            ),
            "gmm": (
                gmm_sample.X,
                gmm_sample.y,
                spec.visualization_gmm_label,
                gmm_components,
            ),
        },
        scenario_dir,
        filename_suffix=f"{mode}_{scenario_run_id:06d}",
    )
    return {
        "images": images,
        "metadata": {
            "visualization_sample_size": int(real_X.shape[0]),
            "per_class": int(per_class),
            "class_balance": "balanced (equal samples per class)",
            "real_data_source": spec.visualization_real_source,
            "single_gaussian_source": spec.visualization_gaussian_source,
            "gmm_source": spec.visualization_gmm_source,
            "visualization_seed": VIZ_SEED,
        },
    }


def _persist_simulation(
    registry: SimulationRunRegistry,
    record: Any,
    result: Any,
    report_callback: Callable[[Path, str], Path],
    model_metadata: Dict[str, Any],
    sampler_metadata: Dict[str, Any],
):
    run_dir = Path(record.run_dir)
    run_id = record.simulation_run_id
    mode = record.mode
    report_path = report_callback(
        run_dir,
        _simulation_report_filename(record.simulation_slug, mode, run_id),
    )
    result_path = run_dir / f"result_data_{mode}_{run_id:06d}.json.gz"
    save_simulation_result(result, result_path)
    summary = simulation_summary_snapshot(result)
    summary_path = run_dir / f"summary_{mode}_{run_id:06d}.json"
    _write_json(summary_path, summary)
    completed = registry.complete_run(
        run_id,
        runtime_seconds=result.runtime_seconds,
        model_metadata=model_metadata,
        sampler_metadata=sampler_metadata,
        summary_data=summary,
        result_data=simulation_report_data(result),
        artifacts={
            "monte_carlo_report": str(report_path),
            "result_data": str(result_path),
            "summary": str(summary_path),
            "simulation_json": record.simulation_json_path,
        },
    )
    _write_json(record.simulation_json_path, completed.to_dict())
    return completed, report_path, result_path


def _fail_simulation(
    registry: SimulationRunRegistry, record: Any, error: str
) -> None:
    try:
        failed = registry.fail_run(record.simulation_run_id, error=error)
        _write_json(record.simulation_json_path, failed.to_dict())
    except Exception:  # noqa: BLE001
        pass


def run_dataset_anchored_scenario(
    spec: DatasetAnchoredExecutionSpec,
    *,
    mode: str = "smoke",
    raw_dir: str,
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
    config: Optional[MonteCarloConfig] = None,
    visualize: bool = True,
) -> Dict[str, Any]:
    """Execute the standard Real/Gaussian/GMM-to-fixed-real protocol."""

    if reporter is None:
        reporter = CooperativeProgressReporter(verbose=False)
    if config is None:
        config = get_mode_config(mode)
    mode = config.mode
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    scenario_registry = ScenarioRunRegistry(base_output_dir=base_dir)
    simulation_registry = SimulationRunRegistry(base_output_dir=base_dir)
    scenario_registry.ensure_registry()
    simulation_registry.ensure_registry()
    reporter.scenario_start(
        scenario_name=spec.scenario_name,
        mode=mode,
        raw_dir=raw_dir,
        output_dir=base_dir,
        config=config,
    )
    scenario_run = scenario_registry.start_run(
        scenario_slug=spec.scenario_slug,
        scenario_name=spec.scenario_name,
        scenario_family=spec.scenario_family,
        question=spec.question,
        mode=mode,
        config=_config_dict(config),
    )
    scenario_run_id = scenario_run.scenario_run_id
    scenario_dir = Path(scenario_run.run_dir)
    reporter.info(f"scenario_run_id={scenario_run_id}")
    reporter.info(f"scenario run dir: {scenario_dir}")
    start = time.time()
    records: Dict[str, Any] = {}

    try:
        reporter.scenario_step_start(f"Loading {spec.dataset_name} dataset", detail=raw_dir)
        step = time.time()
        data = spec.loader(raw_dir)
        _validate_sample_sizes(data, config)
        reporter.scenario_step_finish(
            f"Loading {spec.dataset_name} dataset", elapsed=time.time() - step
        )

        reporter.scenario_step_start("Generating dataset report")
        dataset_report = spec.dataset_report_callback(
            data,
            scenario_dir,
            filename=_dataset_report_filename(spec, mode, scenario_run_id),
        )
        reporter.scenario_step_finish("Generating dataset report", detail=str(dataset_report))

        reporter.scenario_step_start("Building single Gaussian model")
        gaussian_anchored = spec.gaussian_builder(data)
        reporter.scenario_step_finish("Building single Gaussian model")
        reporter.scenario_step_start("Building class-conditional GMM model")
        gmm_anchored = spec.gmm_builder(data)
        reporter.scenario_step_finish("Building class-conditional GMM model")

        visualization = None
        if visualize:
            reporter.scenario_step_start("Rendering data visualization panels")
            visualization = _build_visualizations(
                spec,
                data,
                gaussian_anchored.model,
                gmm_anchored.model,
                scenario_dir,
                mode,
                scenario_run_id,
            )
            reporter.scenario_step_finish("Rendering data visualization panels")

        real_sampler = RealDatasetSampler(
            data.train_dataset,
            data.test_dataset,
            base_seed=config.base_seed,
            channel_names=data.channel_names,
            name=spec.real_simulation_slug,
        )
        gaussian_sampler = SyntheticTrainRealTestSampler(
            GaussianClassConditionalSampler(
                gaussian_anchored.model,
                base_seed=config.base_seed,
                test_samples_per_class=config.test_samples_per_class,
            ),
            data.test_dataset,
            name=spec.gaussian_simulation_slug,
        )
        gmm_sampler = SyntheticTrainRealTestSampler(
            GMMClassConditionalSampler(
                gmm_anchored.model,
                base_seed=config.base_seed,
                test_samples_per_class=config.test_samples_per_class,
            ),
            data.test_dataset,
            name=spec.gmm_simulation_slug,
        )
        _assert_shared_fixed_test(data, real_sampler, gaussian_sampler, gmm_sampler)

        common_metadata = {
            "scenario_name": spec.scenario_name,
            "channel_names": list(data.channel_names),
            "standardization": "train_pool_only",
        }
        records[REAL_ARM_ID] = simulation_registry.start_run(
            simulation_slug=spec.real_simulation_slug,
            simulation_family=spec.real_simulation_family,
            mode=mode,
            scenario_run_id_origin=scenario_run_id,
            config=_config_dict(config),
        )
        real_result = CooperativeMonteCarloSimulator(
            real_sampler.model,
            config,
            sampler=real_sampler,
            metadata={
                **common_metadata,
                "experiment_arm": spec.real_experiment_arm,
                "arm_id": REAL_ARM_ID,
                "train_source": spec.real_training_source_id,
                "test_source": spec.real_test_source_id,
            },
            progress=reporter,
        ).run()
        real_completed, real_report, real_result_path = _persist_simulation(
            simulation_registry,
            records[REAL_ARM_ID],
            real_result,
            lambda directory, filename: spec.real_report_callback(
                real_result,
                data.channel_names,
                directory,
                filename=filename,
                nstar_selection_result=real_result,
            ),
            dict(real_result.metadata),
            {
                "type": "RealDatasetSampler",
                "base_seed": config.base_seed,
                "train_source": spec.real_training_source_id,
                "test_source": spec.real_test_source_id,
                "fixed_test_description": spec.fixed_test_description,
            },
        )

        records[GAUSSIAN_ARM_ID] = simulation_registry.start_run(
            simulation_slug=spec.gaussian_simulation_slug,
            simulation_family=spec.gaussian_simulation_family,
            mode=mode,
            scenario_run_id_origin=scenario_run_id,
            config=_config_dict(config),
        )
        gaussian_result = CooperativeMonteCarloSimulator(
            gaussian_anchored.model,
            config,
            sampler=gaussian_sampler,
            metadata={
                **common_metadata,
                "experiment_arm": GAUSSIAN_ARM_ID,
                "train_source": GAUSSIAN_TRAIN_SOURCE,
                "test_source": spec.real_test_source_id,
                "gaussian_ridge_by_class": dict(gaussian_anchored.ridge_by_class),
            },
            progress=reporter,
        ).run()
        gaussian_completed, gaussian_report, gaussian_result_path = _persist_simulation(
            simulation_registry,
            records[GAUSSIAN_ARM_ID],
            gaussian_result,
            lambda directory, filename: spec.gaussian_report_callback(
                gaussian_result,
                data.channel_names,
                directory,
                filename=filename,
                nstar_selection_result=real_result,
            ),
            dict(gaussian_result.metadata),
            {
                "type": "SyntheticTrainRealTestSampler",
                "train_sampler": "GaussianClassConditionalSampler",
                "base_seed": config.base_seed,
                "train_source": GAUSSIAN_TRAIN_SOURCE,
                "test_source": spec.real_test_source_id,
                "fixed_test_description": spec.fixed_test_description,
            },
        )

        records[GMM_ARM_ID] = simulation_registry.start_run(
            simulation_slug=spec.gmm_simulation_slug,
            simulation_family=spec.gmm_simulation_family,
            mode=mode,
            scenario_run_id_origin=scenario_run_id,
            config=_config_dict(config),
        )
        gmm_result = CooperativeMonteCarloSimulator(
            gmm_anchored.model,
            config,
            sampler=gmm_sampler,
            metadata={
                **common_metadata,
                "experiment_arm": GMM_ARM_ID,
                "train_source": GMM_TRAIN_SOURCE,
                "test_source": spec.real_test_source_id,
                "gmm_model_selection": gmm_anchored.model_selection,
            },
            progress=reporter,
        ).run()
        gmm_completed, gmm_report, gmm_result_path = _persist_simulation(
            simulation_registry,
            records[GMM_ARM_ID],
            gmm_result,
            lambda directory, filename: spec.gmm_report_callback(
                gmm_result,
                data.channel_names,
                directory,
                filename=filename,
                nstar_selection_result=real_result,
            ),
            dict(gmm_result.metadata),
            {
                "type": "SyntheticTrainRealTestSampler",
                "train_sampler": "GMMClassConditionalSampler",
                "base_seed": config.base_seed,
                "train_source": GMM_TRAIN_SOURCE,
                "test_source": spec.real_test_source_id,
                "fixed_test_description": spec.fixed_test_description,
            },
        )

        scenario_meta = {
            "scenario_run_id": scenario_run_id,
            "scenario_family": spec.scenario_family,
            "mode": mode,
            "dataset": spec.dataset_name,
        }
        graphs: Dict[str, Any] = {}
        scenario_report = spec.scenario_report_callback(
            real_result,
            gaussian_result,
            gmm_result,
            output_dir=scenario_dir,
            dataset_report=_relpath(dataset_report, scenario_dir),
            real_report=_relpath(real_report, scenario_dir),
            sg_real_report=_relpath(gaussian_report, scenario_dir),
            gmm_real_report=_relpath(gmm_report, scenario_dir),
            filename=_scenario_report_filename(spec, mode, scenario_run_id),
            channel_names=data.channel_names,
            visualization=visualization,
            scenario_meta=scenario_meta,
            graph_suffix=f"{mode}_{scenario_run_id:06d}",
            graphs_out=graphs,
            generate_graphs=visualize,
        )
        runtime = time.time() - start
        completed_by_arm = {
            REAL_ARM_ID: real_completed,
            GAUSSIAN_ARM_ID: gaussian_completed,
            GMM_ARM_ID: gmm_completed,
        }
        reports_by_arm = {
            REAL_ARM_ID: real_report,
            GAUSSIAN_ARM_ID: gaussian_report,
            GMM_ARM_ID: gmm_report,
        }
        paths_by_arm = {
            REAL_ARM_ID: real_result_path,
            GAUSSIAN_ARM_ID: gaussian_result_path,
            GMM_ARM_ID: gmm_result_path,
        }
        slug_by_arm = {
            REAL_ARM_ID: spec.real_simulation_slug,
            GAUSSIAN_ARM_ID: spec.gaussian_simulation_slug,
            GMM_ARM_ID: spec.gmm_simulation_slug,
        }
        family_by_arm = {
            REAL_ARM_ID: spec.real_simulation_family,
            GAUSSIAN_ARM_ID: spec.gaussian_simulation_family,
            GMM_ARM_ID: spec.gmm_simulation_family,
        }
        simulation_refs = {
            slug_by_arm[arm]: {
                "simulation_run_id": completed.simulation_run_id,
                "simulation_family": family_by_arm[arm],
                "run_dir": completed.run_dir,
                "simulation_json_path": completed.simulation_json_path,
                "report_path": str(reports_by_arm[arm]),
                "result_data_path": str(paths_by_arm[arm]),
                "summary_data": completed.summary_data,
            }
            for arm, completed in completed_by_arm.items()
        }
        artifacts: Dict[str, Any] = {
            "scenario_report": str(scenario_report),
            "dataset_report": str(dataset_report),
            "scenario_json": scenario_run.scenario_json_path,
        }
        if visualization:
            artifacts["visualization_images"] = {
                key: str(scenario_dir / filename)
                for key, filename in visualization["images"].items()
            }
        if graphs:
            artifacts["graph_images"] = {
                key: str(scenario_dir / filename) for key, filename in graphs.items()
            }
        context = dict(spec.report_context_callback(data))
        report_data = dataset_anchored_scenario_report_data(
            real_result,
            gaussian_result,
            gmm_result,
            channel_names=data.channel_names,
            real_arm_id=REAL_ARM_ID,
            gaussian_arm_id=GAUSSIAN_ARM_ID,
            gmm_arm_id=GMM_ARM_ID,
            real_train_source=spec.real_training_source_id,
            gaussian_train_source=GAUSSIAN_TRAIN_SOURCE,
            gmm_train_source=GMM_TRAIN_SOURCE,
            real_test_source=spec.real_test_source_id,
            gaussian_test_source=spec.real_test_source_id,
            gmm_test_source=spec.real_test_source_id,
            dataset_metadata=context.get("dataset"),
            target_metadata=context.get("target"),
            split_metadata=context.get("split"),
            gmm_model_selection=gmm_anchored.model_selection,
        )
        if visualization:
            report_data["visualization"] = visualization
        if graphs:
            report_data["graphs"] = graphs
        completed_scenario = scenario_registry.complete_run(
            scenario_run_id,
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
            "gmm_to_real_simulation_json": Path(gmm_completed.simulation_json_path),
            "real_result_data": real_result_path,
            "single_gaussian_to_real_result_data": gaussian_result_path,
            "gmm_to_real_result_data": gmm_result_path,
            "scenario_runs.json": scenario_registry.registry_path,
            "simulation_runs.json": simulation_registry.registry_path,
        }
        reporter.scenario_finish(runtime=runtime, outputs=outputs)
        return {
            "scenario_run_id": scenario_run_id,
            "real_simulation_run_id": real_completed.simulation_run_id,
            "single_gaussian_to_real_simulation_run_id": gaussian_completed.simulation_run_id,
            "gmm_to_real_simulation_run_id": gmm_completed.simulation_run_id,
            "scenario_run_dir": str(scenario_dir),
            "real_run_dir": real_completed.run_dir,
            "single_gaussian_to_real_run_dir": gaussian_completed.run_dir,
            "gmm_to_real_run_dir": gmm_completed.run_dir,
            "scenario_registry": str(scenario_registry.registry_path),
            "simulation_registry": str(simulation_registry.registry_path),
            "runtime_seconds": runtime,
            **{key: str(value) for key, value in outputs.items()},
        }
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        for record in records.values():
            current = simulation_registry.get_run(record.simulation_run_id)
            if current is not None and current.status == "running":
                _fail_simulation(simulation_registry, record, error)
        try:
            failed = scenario_registry.fail_run(scenario_run_id, error=error)
            _write_json(scenario_run.scenario_json_path, failed.to_dict())
        except Exception:  # noqa: BLE001
            pass
        reporter.error(
            f"{spec.dataset_name} scenario run failed "
            f"(scenario_run_id={scenario_run_id}, run_dir={scenario_dir})",
            exc,
        )
        raise


def regenerate_dataset_anchored_scenario(
    spec: DatasetAnchoredExecutionSpec,
    scenario_run_id: int,
    *,
    output_dir: str = "output/reports",
    reporter: Optional[CooperativeProgressReporter] = None,
) -> Dict[str, Any]:
    """Regenerate reports from persisted results without rerunning Monte Carlo."""

    if reporter is None:
        reporter = CooperativeProgressReporter(verbose=True)
    scenario_registry = ScenarioRunRegistry(base_output_dir=output_dir)
    simulation_registry = SimulationRunRegistry(base_output_dir=output_dir)
    record = scenario_registry.get_run(scenario_run_id)
    if record is None:
        raise KeyError(
            f"scenario_run_id {scenario_run_id} not found in {scenario_registry.registry_path}"
        )
    if record.scenario_slug != spec.scenario_slug:
        raise ValueError(
            f"scenario_run_id {scenario_run_id} belongs to {record.scenario_slug!r}, "
            f"not {spec.scenario_slug!r}"
        )
    scenario_json = json.loads(Path(record.scenario_json_path).read_text(encoding="utf-8"))
    refs = scenario_json["simulation_refs"]
    scenario_dir = Path(record.run_dir)
    channel_names = scenario_json.get("report_data", {}).get("channel_names", [])
    real_ref = refs[spec.real_simulation_slug]
    gaussian_ref = refs[spec.gaussian_simulation_slug]
    gmm_ref = refs[spec.gmm_simulation_slug]
    real_result = load_simulation_result(real_ref["result_data_path"])
    gaussian_result = load_simulation_result(gaussian_ref["result_data_path"])
    gmm_result = load_simulation_result(gmm_ref["result_data_path"])
    real_report = spec.real_report_callback(
        real_result,
        channel_names,
        Path(real_ref["run_dir"]),
        filename=Path(real_ref["report_path"]).name,
        nstar_selection_result=real_result,
    )
    gaussian_report = spec.gaussian_report_callback(
        gaussian_result,
        channel_names,
        Path(gaussian_ref["run_dir"]),
        filename=Path(gaussian_ref["report_path"]).name,
        nstar_selection_result=real_result,
    )
    gmm_report = spec.gmm_report_callback(
        gmm_result,
        channel_names,
        Path(gmm_ref["run_dir"]),
        filename=Path(gmm_ref["report_path"]).name,
        nstar_selection_result=real_result,
    )
    old_report_data = scenario_json.get("report_data", {})
    visualization = old_report_data.get("visualization")
    graphs: Dict[str, str] = {}
    dataset_report = record.artifacts.get("dataset_report", "")
    scenario_report = spec.scenario_report_callback(
        real_result,
        gaussian_result,
        gmm_result,
        output_dir=scenario_dir,
        dataset_report=(
            _relpath(dataset_report, scenario_dir)
            if dataset_report
            else f"{spec.dataset_report_prefix}.html"
        ),
        real_report=_relpath(real_report, scenario_dir),
        sg_real_report=_relpath(gaussian_report, scenario_dir),
        gmm_real_report=_relpath(gmm_report, scenario_dir),
        filename=Path(
            record.artifacts.get(
                "scenario_report",
                _scenario_report_filename(spec, record.mode, scenario_run_id),
            )
        ).name,
        channel_names=channel_names or real_result.metadata.get("channel_names", []),
        visualization=visualization,
        scenario_meta={
            "scenario_run_id": scenario_run_id,
            "scenario_family": record.scenario_family,
            "mode": record.mode,
            "dataset": spec.dataset_name,
        },
        graph_suffix=f"{record.mode}_{scenario_run_id:06d}",
        graphs_out=graphs,
        generate_graphs=True,
    )
    simulation_updates = [
        (real_ref, real_result),
        (gaussian_ref, gaussian_result),
        (gmm_ref, gmm_result),
    ]
    old_gmm_selection = (
        old_report_data.get("arms", {}).get(GMM_ARM_ID, {}).get("gmm_model_selection")
    )
    prepared_report_data = dataset_anchored_scenario_report_data(
        real_result,
        gaussian_result,
        gmm_result,
        channel_names=channel_names or real_result.metadata.get("channel_names", []),
        real_arm_id=REAL_ARM_ID,
        gaussian_arm_id=GAUSSIAN_ARM_ID,
        gmm_arm_id=GMM_ARM_ID,
        real_train_source=spec.real_training_source_id,
        gaussian_train_source=GAUSSIAN_TRAIN_SOURCE,
        gmm_train_source=GMM_TRAIN_SOURCE,
        real_test_source=spec.real_test_source_id,
        gaussian_test_source=spec.real_test_source_id,
        gmm_test_source=spec.real_test_source_id,
        dataset_metadata=old_report_data.get("dataset"),
        target_metadata=old_report_data.get("target"),
        split_metadata=old_report_data.get("split"),
        gmm_model_selection=old_gmm_selection,
    )
    if visualization:
        prepared_report_data["visualization"] = visualization
    prepared_report_data["graphs"] = graphs
    artifacts = dict(record.artifacts)
    artifacts["scenario_report"] = str(scenario_report)
    artifacts["graph_images"] = {
        key: str(scenario_dir / filename) for key, filename in graphs.items()
    }
    simulation_jsons = []
    for ref, result in simulation_updates:
        updated = simulation_registry.update_run(
            int(ref["simulation_run_id"]), result_data=simulation_report_data(result)
        )
        _write_json(ref["simulation_json_path"], updated.to_dict())
        simulation_jsons.append(str(ref["simulation_json_path"]))
    updated_scenario = scenario_registry.update_run(
        scenario_run_id,
        artifacts=artifacts,
        report_data=prepared_report_data,
    )
    _write_json(record.scenario_json_path, updated_scenario.to_dict())
    reporter.info("Regeneration complete (Monte Carlo was not rerun).")
    return {
        "scenario_report": str(scenario_report),
        "real_report": str(real_report),
        "single_gaussian_to_real_report": str(gaussian_report),
        "gmm_to_real_report": str(gmm_report),
        "scenario_json": str(record.scenario_json_path),
        "simulation_jsons": simulation_jsons,
    }


def list_scenario_runs(output_dir: str = "output/reports") -> None:
    registry = ScenarioRunRegistry(base_output_dir=output_dir)
    runs = registry.list_runs()
    print(f"Scenario runs registry: {registry.registry_path}")
    if not runs:
        print("  (no scenario runs)")
        return
    for record in runs:
        print(
            f"  id={record.scenario_run_id} slug={record.scenario_slug} "
            f"family={record.scenario_family} mode={record.mode} "
            f"status={record.status} started={record.started_at} "
            f"runtime={record.runtime_seconds} "
            f"report={record.artifacts.get('scenario_report', '-')}"
        )


def list_simulation_runs(output_dir: str = "output/reports") -> None:
    registry = SimulationRunRegistry(base_output_dir=output_dir)
    runs = registry.list_runs()
    print(f"Simulation runs registry: {registry.registry_path}")
    if not runs:
        print("  (no simulation runs)")
        return
    for record in runs:
        print(
            f"  id={record.simulation_run_id} slug={record.simulation_slug} "
            f"family={record.simulation_family} mode={record.mode} "
            f"status={record.status} started={record.started_at} "
            f"runtime={record.runtime_seconds} "
            f"report={record.artifacts.get('monte_carlo_report', '-')}"
        )
