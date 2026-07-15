import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coinfosim.results.persistence import load_simulation_result
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios import dataset_anchored_runner as runner
from coinfosim.scenarios.air_quality import build_gmm_anchored_air_quality_model
from coinfosim.simulation.config import MonteCarloConfig, get_mode_config
from coinfosim.simulation.execution import ExecutionConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "air_quality"


def _load_script_module():
    script = REPO_ROOT / "scripts" / "run_air_quality_scenario.py"
    spec = importlib.util.spec_from_file_location("run_air_quality_scenario", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_air_quality_scenario"] = module
    spec.loader.exec_module(module)
    return module


def _tiny_config(sample_sizes=(2, 4)):
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=sample_sizes,
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=3,
    )


def _fast_spec(module):
    return replace(
        module.AIR_QUALITY_SPEC,
        gmm_builder=lambda data: build_gmm_anchored_air_quality_model(
            data,
            max_components=1,
            min_points_per_component=50,
            n_init=1,
            random_state=0,
        ),
    )


def _strict_json(path: Path):
    def reject_constant(value):
        raise AssertionError(f"non-strict JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def _assert_same_losses(first, second):
    assert first.subsets == second.subsets
    assert first.classifier_names == second.classifier_names
    assert first.sample_sizes == second.sample_sizes
    for n_per_class in first.sample_sizes:
        for subset in first.subsets:
            for classifier in first.classifier_names:
                assert np.array_equal(
                    first.accumulator.losses(n_per_class, subset, classifier),
                    second.accumulator.losses(n_per_class, subset, classifier),
                )


def test_air_quality_runner_tracks_three_arms_and_shared_fixed_test(
    tmp_path, monkeypatch
):
    module = _load_script_module()
    monkeypatch.setattr(module, "AIR_QUALITY_SPEC", _fast_spec(module))
    captured_samplers = []
    captured_execution_configs = []
    original_simulator = runner.CooperativeMonteCarloSimulator

    class CapturingSimulator(original_simulator):
        def __init__(self, *args, **kwargs):
            captured_samplers.append(kwargs["sampler"])
            captured_execution_configs.append(kwargs["execution_config"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner, "CooperativeMonteCarloSimulator", CapturingSimulator)
    execution_config = ExecutionConfig()
    output = module.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        execution_config=execution_config,
        visualize=False,
    )

    assert output["scenario_run_id"] == 0
    assert output["real_simulation_run_id"] == 0
    assert output["single_gaussian_to_real_simulation_run_id"] == 1
    assert output["gmm_to_real_simulation_run_id"] == 2
    scenario = _strict_json(Path(output["scenario_json"]))
    assert scenario["scenario_slug"] == "air_quality_baseline"
    assert scenario["scenario_family"] == "dataset"
    assert scenario["status"] == "completed"
    assert scenario["simulation_run_ids"] == [0, 1, 2]
    assert set(scenario["simulation_refs"]) == {
        "air_quality_real_data",
        "air_quality_single_gaussian_to_real",
        "air_quality_gmm_to_real",
    }
    families = {
        ref["simulation_family"] for ref in scenario["simulation_refs"].values()
    }
    assert families == {"real_dataset", "single_gaussian_to_real", "gmm_to_real"}

    assert len(captured_samplers) == 3
    assert captured_execution_configs == [execution_config] * 3
    assert isinstance(captured_samplers[0], RealDatasetSampler)
    assert all(
        isinstance(sampler, SyntheticTrainRealTestSampler)
        for sampler in captured_samplers[1:]
    )
    fixed_tests = [sampler.sample_test() for sampler in captured_samplers]
    assert fixed_tests[0] is fixed_tests[1] is fixed_tests[2]
    for observed in fixed_tests[1:]:
        assert np.array_equal(observed.X, fixed_tests[0].X)
        assert np.array_equal(observed.y, fixed_tests[0].y)

    real_train = captured_samplers[0].sample_train(4, replication_id=0)
    pool_rows = {tuple(row) for row in captured_samplers[0].train_pool.X}
    assert all(tuple(row) in pool_rows for row in real_train.X)
    for sampler in captured_samplers[1:]:
        synthetic_train = sampler.sample_train(4, replication_id=0)
        labels, counts = np.unique(synthetic_train.y, return_counts=True)
        assert tuple(labels) == (0, 1)
        assert tuple(counts) == (4, 4)

    loaded_results = [
        load_simulation_result(
            scenario["simulation_refs"][slug]["result_data_path"]
        )
        for slug in (
            "air_quality_real_data",
            "air_quality_single_gaussian_to_real",
            "air_quality_gmm_to_real",
        )
    ]
    assert all(len(result.subsets) == 31 for result in loaded_results)
    assert all(result.sample_sizes == [2, 4] for result in loaded_results)
    assert loaded_results[0].classifier_names == loaded_results[1].classifier_names
    assert loaded_results[1].classifier_names == loaded_results[2].classifier_names
    assert all(result.metadata["fixed_test_size"] == 1799 for result in loaded_results)
    assert [
        result.metadata["experiment_arm"] for result in loaded_results
    ] == ["real_to_real", "single_gaussian_to_real", "gmm_to_real"]

    assert set(scenario["report_data"]["arms"]) == {
        "real_to_real",
        "single_gaussian_to_real",
        "gmm_to_real",
    }
    for arm in scenario["report_data"]["arms"].values():
        assert arm["test_source"] == "real_air_quality_future_test"
    assert scenario["report_data"]["dataset"]["doi"] == "10.24432/C59K5F"
    assert scenario["report_data"]["target"]["reference"] == "C6H6(GT)"
    assert scenario["report_data"]["split"]["split_index"] == 7192

    for path in tmp_path.rglob("*.json"):
        _strict_json(path)


def test_full_scale_resolves_once_and_persists_shared_config(
    tmp_path, monkeypatch
):
    module = _load_script_module()
    monkeypatch.setattr(module, "AIR_QUALITY_SPEC", _fast_spec(module))
    config = replace(
        get_mode_config("full-scale"),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
    )
    original_resolver = runner.resolve_sample_sizes_for_training_capacity
    resolver_calls = []

    def resolving_spy(observed_config, minority_class_count):
        assert observed_config.mode == "full-scale"
        assert minority_class_count == 1818
        resolver_calls.append((observed_config, minority_class_count))
        resolved = original_resolver(observed_config, minority_class_count)
        return replace(resolved, sample_sizes=(2, 4))

    captured_configs = []
    original_simulator = runner.CooperativeMonteCarloSimulator

    class CapturingSimulator(original_simulator):
        def __init__(self, *args, **kwargs):
            captured_configs.append(args[1])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        runner, "resolve_sample_sizes_for_training_capacity", resolving_spy
    )
    monkeypatch.setattr(
        runner, "CooperativeMonteCarloSimulator", CapturingSimulator
    )

    output = module.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=config,
        visualize=False,
    )

    assert len(resolver_calls) == 1
    assert len(captured_configs) == 3
    assert all(item.mode == "full-scale" for item in captured_configs)
    assert all(item.sample_sizes == (2, 4) for item in captured_configs)
    expected_requested = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    json_paths = [
        Path(output["scenario_json"]),
        Path(output["real_simulation_json"]),
        Path(output["single_gaussian_to_real_simulation_json"]),
        Path(output["gmm_to_real_simulation_json"]),
    ]
    for path in json_paths:
        persisted = _strict_json(path)["config"]
        assert persisted["mode"] == "full-scale"
        assert persisted["sample_sizes"] == [2, 4]
        assert persisted["requested_sample_sizes"] == expected_requested
        assert persisted["training_class_counts"] == {
            "0": 5374,
            "1": 1818,
        }
        assert persisted["training_minority_class_count"] == 1818
        assert persisted["resolved_max_n_per_class"] == 4
        assert persisted["sample_size_strategy"] == (
            "powers_of_two_up_to_training_minority"
        )


def test_same_seed_reproduces_losses_and_rerun_never_overwrites(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module, "AIR_QUALITY_SPEC", _fast_spec(module))
    first = module.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )
    second = module.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )

    assert first["scenario_run_id"] == 0
    assert second["scenario_run_id"] == 1
    assert first["scenario_run_dir"] != second["scenario_run_dir"]
    assert Path(first["scenario_json"]).exists()
    assert Path(second["scenario_json"]).exists()
    for key in (
        "real_result_data",
        "single_gaussian_to_real_result_data",
        "gmm_to_real_result_data",
    ):
        first_result = load_simulation_result(first[key])
        second_result = load_simulation_result(second[key])
        _assert_same_losses(first_result, second_result)
        assert first_result.metadata == second_result.metadata


def test_persistence_roundtrip_and_regeneration_do_not_run_monte_carlo(
    tmp_path, monkeypatch
):
    module = _load_script_module()
    monkeypatch.setattr(module, "AIR_QUALITY_SPEC", _fast_spec(module))
    output = module.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )
    original = load_simulation_result(output["gmm_to_real_result_data"])
    reloaded = load_simulation_result(output["gmm_to_real_result_data"])
    _assert_same_losses(original, reloaded)
    assert original.metadata == reloaded.metadata

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Monte Carlo must not run during regeneration")

    monkeypatch.setattr(runner, "CooperativeMonteCarloSimulator", fail_if_called)
    regenerated = module.regenerate_from_scenario_run(0, output_dir=str(tmp_path))

    assert Path(regenerated["scenario_report"]).exists()
    assert Path(regenerated["real_report"]).exists()
    assert Path(regenerated["single_gaussian_to_real_report"]).exists()
    assert Path(regenerated["gmm_to_real_report"]).exists()
    scenario_registry = _strict_json(tmp_path / "scenario_runs.json")
    simulation_registry = _strict_json(tmp_path / "simulation_runs.json")
    assert scenario_registry["next_scenario_run_id"] == 1
    assert simulation_registry["next_simulation_run_id"] == 3


def test_infeasible_real_sample_size_fails_before_reports_models_or_simulations(
    tmp_path, monkeypatch
):
    module = _load_script_module()

    def expensive_step(*args, **kwargs):
        raise AssertionError("expensive execution started before feasibility check")

    spec = replace(
        _fast_spec(module),
        dataset_report_callback=expensive_step,
        gaussian_builder=expensive_step,
        gmm_builder=expensive_step,
    )
    monkeypatch.setattr(module, "AIR_QUALITY_SPEC", spec)
    with pytest.raises(ValueError, match="minority-class count 1818.*1819"):
        module.run_scenario(
            raw_dir=str(RAW_DIR),
            output_dir=str(tmp_path),
            config=_tiny_config(sample_sizes=(1819,)),
            visualize=False,
        )

    scenario_registry = _strict_json(tmp_path / "scenario_runs.json")
    simulation_registry = _strict_json(tmp_path / "simulation_runs.json")
    assert scenario_registry["runs"][0]["status"] == "failed"
    assert simulation_registry["next_simulation_run_id"] == 0
