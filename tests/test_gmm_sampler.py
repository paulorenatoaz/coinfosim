"""Unit tests for the GMM model and sampler (no raw Occupancy files needed)."""

import numpy as np
import pytest

from coinfosim.models.gmm import GMMSimulationModel
from coinfosim.samplers.gmm import GMMClassConditionalSampler


def _model():
    weights = {0: np.array([0.6, 0.4]), 1: np.array([1.0])}
    means = {
        0: np.array([[0.0, 0.0], [4.0, 4.0]]),
        1: np.array([[1.0, -1.0]]),
    }
    covariances = {
        0: np.stack([np.eye(2), np.eye(2)]),
        1: np.eye(2)[None],
    }
    selection = {
        0: {"class_label": 0, "selected_components": 2, "criterion": "bic",
            "candidate_components": [1, 2], "scores": {"1": {"bic": 10.0, "aic": 9.0},
                                                        "2": {"bic": 8.0, "aic": 7.0}}},
        1: {"class_label": 1, "selected_components": 1, "criterion": "bic",
            "candidate_components": [1], "scores": {"1": {"bic": 5.0, "aic": 4.0}}},
    }
    return GMMSimulationModel(
        weights, means, covariances, model_selection=selection,
        channel_names=("A", "B"),
    )


def test_gmm_model_interface():
    model = _model()
    assert model.d == 2
    assert model.num_channels == 2
    assert model.K == 2
    assert model.num_classes == 2
    assert model.class_labels == (0, 1)
    assert model.selected_components(0) == 2
    assert model.selected_components(1) == 1
    assert model.component_weights(0).shape == (2,)
    assert model.component_means(0).shape == (2, 2)
    assert model.component_covariances(0).shape == (2, 2, 2)
    assert model.model_selection(0)["criterion"] == "bic"


def test_gmm_sampler_balanced_and_deterministic():
    model = _model()
    s = GMMClassConditionalSampler(model, base_seed=11, test_samples_per_class=7)
    a = s.sample_train(n_per_class=10, replication_id=0)
    assert a.n_samples == 20
    # Deterministic in (base_seed, class, replication).
    b = s.sample_train(n_per_class=10, replication_id=0)
    assert (a.X == b.X).all()
    # Different replication ids differ.
    c = s.sample_train(n_per_class=10, replication_id=1)
    assert not np.allclose(a.X, c.X)
    # Fixed test set has requested class counts.
    test = s.sample_test()
    assert test.n_samples == 14


def test_gmm_sampler_prefix_nested_per_class():
    model = _model()
    s = GMMClassConditionalSampler(model, base_seed=3, test_samples_per_class=5)
    big = s.sample_train(n_per_class=32, replication_id=2)
    small = s.sample_train(n_per_class=16, replication_id=2)
    for label in model.class_labels:
        big_rows = big.X[big.y == label]
        small_rows = small.X[small.y == label]
        assert np.allclose(big_rows[:16], small_rows)


def test_gmm_sampler_rejects_bad_test_count():
    with pytest.raises(ValueError):
        GMMClassConditionalSampler(_model(), base_seed=1, test_samples_per_class=0)
