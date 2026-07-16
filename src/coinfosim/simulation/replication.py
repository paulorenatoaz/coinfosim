"""Complete-replication evaluation for cooperative Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from coinfosim.classifiers.registry import (
    ClassifierExecutionPlan,
    default_execution_plan,
    derive_classifier_seed,
    make_classifier,
)
from coinfosim.samplers.dataset import Dataset
from coinfosim.simulation.metrics import empirical_test_loss

Cell = tuple[tuple[int, ...], str]


class ReplicationSampler(Protocol):
    """Sampler behavior required to evaluate one replication."""

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        ...


@dataclass(frozen=True)
class ReplicationTask:
    """One complete Monte Carlo replication for one training sample size."""

    n_per_class: int
    replication_id: int


@dataclass(frozen=True)
class ReplicationResult:
    """Ordered empirical test losses for every canonical simulation cell."""

    n_per_class: int
    replication_id: int
    losses: tuple[float, ...]


def evaluate_replication(
    sampler: ReplicationSampler,
    cells: Sequence[Cell],
    test_by_subset: Mapping[tuple[int, ...], Dataset],
    task: ReplicationTask,
    classifier_plan: ClassifierExecutionPlan | None = None,
    base_seed: int = 0,
) -> ReplicationResult:
    """Evaluate every canonical cell using one shared training sample."""

    train = sampler.sample_train(
        n_per_class=task.n_per_class,
        replication_id=task.replication_id,
    )
    if np.unique(train.y).size < 2:
        raise ValueError("training sample must contain at least two classes")
    if classifier_plan is None:
        classifier_plan = default_execution_plan(
            tuple(dict.fromkeys(name for _, name in cells))
        )
    estimator_seeds = {
        classifier_name: derive_classifier_seed(
            base_seed, classifier_name, task.replication_id
        )
        for classifier_name in classifier_plan.names
        if classifier_name == "random_forest"
    }
    train_by_subset: dict[tuple[int, ...], Dataset] = {}
    losses = []

    for subset, classifier_name in cells:
        if subset not in train_by_subset:
            train_by_subset[subset] = train.select_channels(subset)
        estimator = make_classifier(
            classifier_name,
            parameters=classifier_plan.parameters.get(classifier_name),
            random_state=estimator_seeds.get(classifier_name),
        )
        train_subset = train_by_subset[subset]
        estimator.fit(train_subset.X, train_subset.y)
        losses.append(empirical_test_loss(estimator, test_by_subset[subset]))

    return ReplicationResult(
        n_per_class=task.n_per_class,
        replication_id=task.replication_id,
        losses=tuple(losses),
    )
