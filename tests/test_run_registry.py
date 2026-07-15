"""Tests for the scenario/simulation run registries and result persistence."""

import json

import numpy as np

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.results.persistence import (
    load_simulation_result,
    save_simulation_result,
)
from coinfosim.runs.registry import ScenarioRunRegistry, SimulationRunRegistry
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


# --- Scenario registry --------------------------------------------------------
def test_scenario_registry_created_and_ids_increment(tmp_path):
    reg = ScenarioRunRegistry(base_output_dir=tmp_path)
    assert not reg.registry_path.exists()

    first = reg.start_run(
        scenario_slug="occupancy_baseline",
        scenario_name="Occupancy Detection Baseline",
        scenario_family="dataset",
        question="Q?",
        mode="smoke",
    )
    assert reg.registry_path.exists()
    assert first.scenario_run_id == 0

    second = reg.start_run(
        scenario_slug="occupancy_baseline",
        scenario_name="Occupancy Detection Baseline",
        scenario_family="dataset",
        question="Q?",
        mode="smoke",
    )
    assert second.scenario_run_id == 1
    # Distinct run directories; nothing overwritten.
    assert first.run_dir != second.run_dir
    assert len(reg.list_runs()) == 2


def test_simulation_registry_created_and_ids_increment(tmp_path):
    reg = SimulationRunRegistry(base_output_dir=tmp_path)
    assert not reg.registry_path.exists()

    first = reg.start_run(
        simulation_slug="occupancy_real_data",
        simulation_family="real_dataset",
        mode="smoke",
        scenario_run_id_origin=0,
    )
    second = reg.start_run(
        simulation_slug="occupancy_gaussian_anchored",
        simulation_family="gaussian_anchored",
        mode="smoke",
        scenario_run_id_origin=0,
    )
    assert first.simulation_run_id == 0
    assert second.simulation_run_id == 1
    assert first.run_dir != second.run_dir
    assert first.reused_by_scenario_run_ids == [0]


def test_registry_appends_and_json_valid_after_updates(tmp_path):
    reg = ScenarioRunRegistry(base_output_dir=tmp_path)
    r0 = reg.start_run(
        scenario_slug="s",
        scenario_name="n",
        scenario_family="dataset",
        question="Q",
        mode="smoke",
    )
    reg.update_run(r0.scenario_run_id, artifacts={"a": "1"})
    reg.complete_run(r0.scenario_run_id, runtime_seconds=1.5)

    r1 = reg.start_run(
        scenario_slug="s",
        scenario_name="n",
        scenario_family="dataset",
        question="Q",
        mode="smoke",
    )
    reg.fail_run(r1.scenario_run_id, error="boom")

    # JSON remains valid and holds both runs after repeated updates.
    data = json.loads(reg.registry_path.read_text(encoding="utf-8"))
    assert data["next_scenario_run_id"] == 2
    assert len(data["runs"]) == 2

    completed = reg.get_run(0)
    failed = reg.get_run(1)
    assert completed.status == "completed"
    assert completed.runtime_seconds == 1.5
    assert completed.artifacts == {"a": "1"}
    assert failed.status == "failed"
    assert failed.error == "boom"


def test_simulation_fail_run_marks_failed(tmp_path):
    reg = SimulationRunRegistry(base_output_dir=tmp_path)
    r = reg.start_run(
        simulation_slug="s", simulation_family="real_dataset", mode="smoke"
    )
    reg.fail_run(r.simulation_run_id, error="explode")
    got = reg.get_run(r.simulation_run_id)
    assert got.status == "failed"
    assert got.error == "explode"


# --- Result persistence -------------------------------------------------------
def _tiny_gaussian_result():
    model = GaussianSimulationModel(
        means={0: [-0.6, -0.3], 1: [0.6, 0.3]},
        covariances={0: [[1.0, 0.2], [0.2, 1.0]], 1: [[1.0, 0.2], [0.2, 1.0]]},
    )
    config = MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4, 8),
        min_replications=2,
        max_replications=4,
        replication_batch_size=2,
        test_samples_per_class=40,
        ci_half_width_target=0.05,
        base_seed=5,
    )
    sampler = GaussianClassConditionalSampler(
        model, base_seed=config.base_seed, test_samples_per_class=40
    )
    return CooperativeMonteCarloSimulator(
        model, config, sampler=sampler, metadata={"experiment_arm": "gaussian_anchored"}
    ).run()


def test_save_and_load_simulation_result_roundtrip(tmp_path):
    result = _tiny_gaussian_result()
    path = save_simulation_result(result, tmp_path / "result.json.gz")
    assert path.exists()

    loaded = load_simulation_result(path)

    assert loaded.sample_sizes == result.sample_sizes
    assert loaded.subsets == result.subsets
    assert loaded.classifier_names == result.classifier_names
    assert loaded.metadata["experiment_arm"] == "gaussian_anchored"
    assert loaded.metadata["execution"] == result.metadata["execution"]
    assert set(loaded.stopping_info.keys()) == set(result.stopping_info.keys())

    for n in result.sample_sizes:
        for subset in result.subsets:
            for clf in result.classifier_names:
                original = result.accumulator.losses(n, subset, clf)
                restored = loaded.accumulator.losses(n, subset, clf)
                assert np.array_equal(original, restored)

    # Gaussian model reconstructed with identical parameters.
    assert loaded.model.d == result.model.d
    for label in result.model.class_labels:
        assert np.allclose(loaded.model.mean(label), result.model.mean(label))
        assert np.allclose(
            loaded.model.covariance(label), result.model.covariance(label)
        )
