"""Unit tests for :class:`SyntheticTrainRealTestSampler`.

These do not require the Occupancy raw files: they build a tiny two-class
Gaussian training sampler and a small synthetic "real" test dataset directly.
"""

import numpy as np
import pytest

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler


def _model():
    means = {0: np.zeros(2), 1: np.ones(2)}
    covariances = {0: np.eye(2), 1: np.eye(2)}
    return GaussianSimulationModel(means=means, covariances=covariances)


def _train_sampler():
    return GaussianClassConditionalSampler(
        _model(), base_seed=7, test_samples_per_class=5
    )


def test_transfer_sampler_trains_synthetic_tests_real():
    train_sampler = _train_sampler()
    real_test = Dataset(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), np.array([0, 1, 1])
    )
    sampler = SyntheticTrainRealTestSampler(train_sampler, real_test)

    # The exposed model is the synthetic training model.
    assert sampler.model is train_sampler.model

    # sample_test returns the fixed real evaluation dataset unchanged.
    test = sampler.sample_test()
    assert test is real_test

    # sample_train delegates to the synthetic Gaussian sampler (balanced draws).
    train = sampler.sample_train(n_per_class=3, replication_id=0)
    assert train.n_samples == 6
    direct = train_sampler.sample_train(n_per_class=3, replication_id=0)
    assert (train.X == direct.X).all()
    assert (train.y == direct.y).all()


def test_transfer_sampler_dimension_mismatch_raises():
    train_sampler = _train_sampler()  # d = 2
    bad_test = Dataset(np.zeros((3, 3)), np.array([0, 1, 1]))  # d = 3
    with pytest.raises(ValueError):
        SyntheticTrainRealTestSampler(train_sampler, bad_test)


def test_transfer_sampler_preserves_synthetic_to_synthetic_capability():
    # The wrapped Gaussian sampler still supports its own synthetic test set,
    # so purely synthetic (train+test) scenarios remain available.
    train_sampler = _train_sampler()
    synthetic_test = train_sampler.sample_test()
    assert synthetic_test.n_samples == 2 * 5
