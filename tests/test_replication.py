"""Focused tests for complete Monte Carlo replication evaluation."""

import numpy as np
import pytest

from coinfosim.classifiers.registry import ClassifierExecutionPlan, make_classifier
import coinfosim.simulation.replication as replication_module
from coinfosim.samplers.dataset import Dataset
from coinfosim.simulation.metrics import empirical_test_loss
from coinfosim.simulation.replication import (
    ReplicationTask,
    evaluate_replication,
)


class _CountingSampler:
    def __init__(self, training_dataset: Dataset) -> None:
        self.training_dataset = training_dataset
        self.calls = []

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        self.calls.append((n_per_class, replication_id))
        return self.training_dataset


def test_complete_replication_uses_one_sample_and_preserves_cell_order():
    training_dataset = Dataset(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.5, -1.5],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.5, 1.5],
        ],
        [0, 0, 0, 1, 1, 1],
    )
    test_dataset = Dataset(
        [
            [-1.25, -1.75],
            [-1.75, -1.25],
            [1.25, 1.75],
            [1.75, 1.25],
        ],
        [0, 0, 1, 1],
    )
    cells = [
        ((0,), "linear_svm"),
        ((0,), "gaussian_nb"),
        ((1,), "linear_svm"),
        ((1,), "gaussian_nb"),
    ]
    test_by_subset = {
        subset: test_dataset.select_channels(subset) for subset, _ in cells
    }
    sampler = _CountingSampler(training_dataset)
    task = ReplicationTask(n_per_class=3, replication_id=7)

    result = evaluate_replication(sampler, cells, test_by_subset, task)

    expected_losses = []
    for subset, classifier_name in cells:
        train_subset = training_dataset.select_channels(subset)
        estimator = make_classifier(classifier_name)
        estimator.fit(train_subset.X, train_subset.y)
        expected_losses.append(empirical_test_loss(estimator, test_by_subset[subset]))

    assert sampler.calls == [(3, 7)]
    assert result.n_per_class == 3
    assert result.replication_id == 7
    assert len(result.losses) == len(cells)
    assert result.losses == tuple(expected_losses)
    assert np.all(np.isfinite(result.losses))
    assert np.all(
        (np.asarray(result.losses) >= 0.0)
        & (np.asarray(result.losses) <= 1.0)
    )


def test_one_class_training_sample_fails_without_fallback():
    training = Dataset([[0.0], [1.0]], [0, 0])
    test = Dataset([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="training sample must contain at least two classes"):
        evaluate_replication(
            _CountingSampler(training),
            [((0,), "linear_svm")],
            {(0,): test},
            ReplicationTask(n_per_class=1, replication_id=0),
        )


def test_random_forest_uses_one_seed_within_replication(monkeypatch):
    training = Dataset([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]], [0, 0, 1, 1])
    test = Dataset([[-1.0, -1.0], [1.0, 1.0]], [0, 1])
    observed = []
    original = replication_module.make_classifier

    def capturing_factory(*args, **kwargs):
        observed.append(kwargs.get("random_state"))
        return original(*args, **kwargs)

    monkeypatch.setattr(replication_module, "make_classifier", capturing_factory)
    plan = ClassifierExecutionPlan(
        names=("random_forest",),
        parameters={"random_forest": {"n_estimators": 2, "max_depth": 2, "min_samples_leaf": 1, "max_features": "sqrt"}},
        provenance={},
    )
    evaluate_replication(
        _CountingSampler(training),
        [((0,), "random_forest"), ((1,), "random_forest")],
        {(0,): test.select_channels((0,)), (1,): test.select_channels((1,))},
        ReplicationTask(n_per_class=2, replication_id=9),
        classifier_plan=plan,
        base_seed=12,
    )
    assert len(observed) == 2
    assert observed[0] == observed[1]
