"""Tests for Monte Carlo execution configuration and sequential execution."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from coinfosim.samplers.dataset import Dataset
from coinfosim.scenarios.synthetic import make_synthetic_scenario_1
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import (
    ExecutionConfig,
    SequentialReplicationExecutor,
)
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


class _CountingSampler:
    def __init__(self, training_dataset):
        self.training_dataset = training_dataset
        self.calls = []

    def sample_train(self, n_per_class, replication_id):
        self.calls.append((n_per_class, replication_id))
        return self.training_dataset


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2,),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=3,
    )


def test_execution_config_defaults_are_sequential_and_immutable():
    config = ExecutionConfig()

    assert config.backend == "sequential"
    assert config.n_jobs == 1
    assert config.start_method == "forkserver"
    assert config.worker_inner_threads == 1
    with pytest.raises(FrozenInstanceError):
        config.n_jobs = 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "threads"}, "unknown execution backend"),
        ({"n_jobs": 0}, "n_jobs must be positive"),
        ({"worker_inner_threads": 0}, "worker_inner_threads must be positive"),
    ],
)
def test_execution_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExecutionConfig(**kwargs)


def test_process_config_rejects_unavailable_start_method(monkeypatch):
    monkeypatch.setattr(
        "coinfosim.simulation.execution.mp.get_all_start_methods",
        lambda: ["spawn"],
    )

    with pytest.raises(ValueError, match="is not available"):
        ExecutionConfig(backend="process", start_method="forkserver")


def test_sequential_config_does_not_require_a_multiprocessing_start_method():
    config = ExecutionConfig(backend="sequential", start_method="not-installed")

    assert config.start_method == "not-installed"


def test_sequential_executor_evaluates_requested_ids_in_order():
    training_dataset = Dataset(
        [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]],
        [0, 0, 1, 1],
    )
    test_dataset = Dataset(
        [[-1.5, -1.5], [1.5, 1.5]],
        [0, 1],
    )
    cells = [((0,), "gaussian_nb"), ((1,), "gaussian_nb")]
    sampler = _CountingSampler(training_dataset)
    executor = SequentialReplicationExecutor(
        sampler=sampler,
        cells=cells,
        test_by_subset={
            subset: test_dataset.select_channels(subset) for subset, _ in cells
        },
    )

    results = executor.run_batch(n_per_class=2, replication_ids=[2, 0])

    assert sampler.calls == [(2, 2), (2, 0)]
    assert [result.replication_id for result in results] == [2, 0]
    assert all(len(result.losses) == len(cells) for result in results)
    assert all(np.all(np.isfinite(result.losses)) for result in results)


def test_simulator_default_matches_explicit_sequential_execution_exactly():
    scenario = make_synthetic_scenario_1()
    kwargs = {
        "model": scenario.model,
        "config": _tiny_config(),
        "subsets": [(0,)],
        "classifier_names": ["gaussian_nb"],
    }

    default_result = CooperativeMonteCarloSimulator(**kwargs).run()
    explicit_result = CooperativeMonteCarloSimulator(
        **kwargs,
        execution_config=ExecutionConfig(backend="sequential"),
    ).run()

    assert np.array_equal(
        default_result.accumulator.losses(2, (0,), "gaussian_nb"),
        explicit_result.accumulator.losses(2, (0,), "gaussian_nb"),
    )
    assert default_result.stopping_info == explicit_result.stopping_info
    assert default_result.metadata == explicit_result.metadata
