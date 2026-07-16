"""Exact sequential/process equivalence tests across sampler families."""

import pickle

import numpy as np
import pytest

from coinfosim.classifiers.registry import default_classifiers
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.models.gmm import GMMSimulationModel
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import ExecutionConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator

SAMPLER_KINDS = (
    "gaussian",
    "real",
    "gmm",
    "gaussian_to_real",
    "gmm_to_real",
)
SUBSETS = [(0,), (1,), (0, 1)]


def _gaussian_model():
    return GaussianSimulationModel(
        means={0: [-3.0, -2.5], 1: [3.0, 2.5]},
        covariances={
            0: [[0.2, 0.02], [0.02, 0.2]],
            1: [[0.2, 0.02], [0.02, 0.2]],
        },
    )


def _gmm_model():
    return GMMSimulationModel(
        weights={0: [0.6, 0.4], 1: [0.4, 0.6]},
        means={
            0: [[-3.2, -2.7], [-2.6, -2.2]],
            1: [[2.6, 2.2], [3.2, 2.7]],
        },
        covariances={
            0: [
                [[0.15, 0.01], [0.01, 0.15]],
                [[0.12, 0.01], [0.01, 0.12]],
            ],
            1: [
                [[0.12, 0.01], [0.01, 0.12]],
                [[0.15, 0.01], [0.01, 0.15]],
            ],
        },
        channel_names=("A", "B"),
        name="tiny_gmm",
    )


def _real_datasets():
    class_zero = np.column_stack(
        (
            np.linspace(-3.5, -2.0, 8),
            np.linspace(-2.0, -3.5, 8),
        )
    )
    class_one = -class_zero
    train = Dataset(
        np.vstack((class_zero, class_one)),
        np.array([0] * len(class_zero) + [1] * len(class_one)),
    )
    test = Dataset(
        [
            [-3.2, -2.2],
            [-2.8, -2.8],
            [-2.2, -3.2],
            [3.2, 2.2],
            [2.8, 2.8],
            [2.2, 3.2],
        ],
        [0, 0, 0, 1, 1, 1],
    )
    return train, test


def _make_sampler(kind):
    if kind == "gaussian":
        return GaussianClassConditionalSampler(
            _gaussian_model(),
            base_seed=13,
            test_samples_per_class=10,
        )
    if kind == "gmm":
        return GMMClassConditionalSampler(
            _gmm_model(),
            base_seed=13,
            test_samples_per_class=10,
        )

    train, test = _real_datasets()
    if kind == "real":
        return RealDatasetSampler(
            train,
            test,
            base_seed=13,
            channel_names=("A", "B"),
            name="tiny_real",
        )
    if kind == "gaussian_to_real":
        return SyntheticTrainRealTestSampler(
            GaussianClassConditionalSampler(
                _gaussian_model(),
                base_seed=13,
                test_samples_per_class=10,
            ),
            test,
            name="tiny_gaussian_to_real",
        )
    if kind == "gmm_to_real":
        return SyntheticTrainRealTestSampler(
            GMMClassConditionalSampler(
                _gmm_model(),
                base_seed=13,
                test_samples_per_class=10,
            ),
            test,
            name="tiny_gmm_to_real",
        )
    raise AssertionError(f"unknown test sampler kind: {kind}")


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2,),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=10,
        ci_half_width_target=0.05,
        base_seed=13,
    )


def _run(kind, execution_config):
    sampler = _make_sampler(kind)
    return CooperativeMonteCarloSimulator(
        sampler.model,
        _tiny_config(),
        subsets=SUBSETS,
        classifier_names=default_classifiers(),
        sampler=sampler,
        execution_config=execution_config,
    ).run()


@pytest.mark.parametrize("kind", SAMPLER_KINDS)
def test_sampler_sequential_and_process_results_are_exactly_equal(kind):
    sequential = _run(kind, ExecutionConfig())
    process = _run(
        kind,
        ExecutionConfig(
            backend="process",
            n_jobs=2,
            start_method="forkserver",
            worker_inner_threads=1,
        ),
    )

    assert sequential.sample_sizes == process.sample_sizes
    assert sequential.subsets == process.subsets
    assert sequential.classifier_names == process.classifier_names
    assert sequential.stopping_info == process.stopping_info
    assert {
        key: value
        for key, value in sequential.metadata.items()
        if key != "execution"
    } == {
        key: value for key, value in process.metadata.items() if key != "execution"
    }
    for n_per_class in sequential.sample_sizes:
        for subset in sequential.subsets:
            for classifier_name in sequential.classifier_names:
                assert np.array_equal(
                    sequential.accumulator.losses(
                        n_per_class, subset, classifier_name
                    ),
                    process.accumulator.losses(
                        n_per_class, subset, classifier_name
                    ),
                )


@pytest.mark.parametrize("kind", SAMPLER_KINDS)
def test_worker_sampler_pickle_roundtrip_preserves_samples(kind):
    sampler = _make_sampler(kind)
    restored = pickle.loads(pickle.dumps(sampler))

    expected_train = sampler.sample_train(n_per_class=2, replication_id=1)
    restored_train = restored.sample_train(n_per_class=2, replication_id=1)
    expected_test = sampler.sample_test()
    restored_test = restored.sample_test()

    assert type(restored) is type(sampler)
    assert np.array_equal(expected_train.X, restored_train.X)
    assert np.array_equal(expected_train.y, restored_train.y)
    assert np.array_equal(expected_test.X, restored_test.X)
    assert np.array_equal(expected_test.y, restored_test.y)


def test_gmm_sampling_reuses_precomputed_weights_and_cholesky(monkeypatch):
    model = _gmm_model()
    calls = {"cumsum": 0, "cholesky": 0}
    original_cumsum = np.cumsum
    original_cholesky = np.linalg.cholesky

    def counting_cumsum(*args, **kwargs):
        calls["cumsum"] += 1
        return original_cumsum(*args, **kwargs)

    def counting_cholesky(*args, **kwargs):
        calls["cholesky"] += 1
        return original_cholesky(*args, **kwargs)

    monkeypatch.setattr(np, "cumsum", counting_cumsum)
    monkeypatch.setattr(np.linalg, "cholesky", counting_cholesky)

    sampler = GMMClassConditionalSampler(
        model,
        base_seed=13,
        test_samples_per_class=5,
    )
    construction_calls = dict(calls)
    sampler.sample_train(n_per_class=2, replication_id=0)
    sampler.sample_train(n_per_class=4, replication_id=1)
    sampler.sample_test()

    assert construction_calls["cumsum"] == len(model.class_labels)
    assert construction_calls["cholesky"] > 0
    assert calls == construction_calls
