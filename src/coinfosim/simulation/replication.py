"""Complete-replication evaluation for cooperative Monte Carlo simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from coinfosim.classifiers.registry import make_classifier
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
) -> ReplicationResult:
    """Evaluate every canonical cell using one shared training sample."""

    train = sampler.sample_train(
        n_per_class=task.n_per_class,
        replication_id=task.replication_id,
    )
    train_by_subset: dict[tuple[int, ...], Dataset] = {}
    losses = []

    for subset, classifier_name in cells:
        if subset not in train_by_subset:
            train_by_subset[subset] = train.select_channels(subset)
        estimator = make_classifier(classifier_name)
        train_subset = train_by_subset[subset]
        estimator.fit(train_subset.X, train_subset.y)
        losses.append(empirical_test_loss(estimator, test_by_subset[subset]))

    return ReplicationResult(
        n_per_class=task.n_per_class,
        replication_id=task.replication_id,
        losses=tuple(losses),
    )
