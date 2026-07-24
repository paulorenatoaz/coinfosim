"""Finite real-dataset sampler for CoInfoSim Monte Carlo experiments.

``RealDatasetSampler`` draws balanced training samples from a finite training
pool and reuses a fixed test dataset. Sampling is deterministic in
``(base_seed, class_label, replication_id)`` and prefix-nested across
``n_per_class`` for each class and replication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import derive_seed


@dataclass(frozen=True)
class RealDatasetModel:
    """Minimal model-like metadata exposed by :class:`RealDatasetSampler`."""

    d: int
    class_labels: Tuple[int, ...]
    channel_names: Tuple[str, ...] = ()
    name: str = "real dataset"

    @property
    def K(self) -> int:
        return len(self.class_labels)

    @property
    def num_channels(self) -> int:
        return self.d

    @property
    def num_classes(self) -> int:
        return self.K


class RealDatasetSampler:
    """Balanced deterministic sampler from a finite real training pool.

    Parameters
    ----------
    train_pool:
        Finite pool from which training samples are drawn.
    test_dataset:
        Fixed test set returned by :meth:`sample_test`.
    base_seed:
        Base random seed controlling deterministic permutations.
    channel_names:
        Optional attribute names for model metadata.
    name:
        Optional model/scenario label for metadata.
    """

    def __init__(
        self,
        train_pool: Dataset,
        test_dataset: Dataset,
        base_seed: int = 0,
        channel_names: Optional[Sequence[str]] = None,
        name: str = "real dataset",
    ) -> None:
        if train_pool.d != test_dataset.d:
            raise ValueError(
                f"train_pool and test_dataset must have the same d; "
                f"got {train_pool.d} and {test_dataset.d}"
            )

        class_labels = tuple(sorted(int(v) for v in np.unique(train_pool.y)))
        test_labels = tuple(sorted(int(v) for v in np.unique(test_dataset.y)))
        if class_labels != test_labels:
            raise ValueError(
                f"train and test class labels must match; "
                f"got train={class_labels} test={test_labels}"
            )
        if len(class_labels) < 2:
            raise ValueError("at least two classes are required")

        if channel_names is None:
            channel_names = tuple(f"X{i + 1}" for i in range(train_pool.d))
        else:
            channel_names = tuple(channel_names)
            if len(channel_names) != train_pool.d:
                raise ValueError(
                    f"channel_names length must equal d={train_pool.d}; "
                    f"got {len(channel_names)}"
                )

        self._train_pool = train_pool
        self._test_dataset = test_dataset
        self._base_seed = int(base_seed)
        self._model = RealDatasetModel(
            d=train_pool.d,
            class_labels=class_labels,
            channel_names=tuple(channel_names),
            name=name,
        )
        self._indices_by_class: Dict[int, np.ndarray] = {
            label: np.flatnonzero(train_pool.y == label) for label in class_labels
        }

    @property
    def model(self) -> RealDatasetModel:
        return self._model

    @property
    def base_seed(self) -> int:
        return self._base_seed

    @property
    def train_pool(self) -> Dataset:
        return self._train_pool

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    @property
    def min_class_count(self) -> int:
        return min(len(indices) for indices in self._indices_by_class.values())

    def class_count(self, class_label: int) -> int:
        return int(len(self._indices_by_class[int(class_label)]))

    def _permutation_for_class(self, class_label: int, replication_id: int) -> np.ndarray:
        seed = derive_seed(
            self._base_seed,
            int(class_label),
            split="train",
            replication_id=replication_id,
        )
        rng = np.random.default_rng(seed)
        indices = self._indices_by_class[int(class_label)]
        return indices[rng.permutation(len(indices))]

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        """Return a balanced, prefix-nested training sample."""

        if n_per_class <= 0:
            raise ValueError("n_per_class must be a positive integer")
        if n_per_class > self.min_class_count:
            raise ValueError(
                f"n_per_class={n_per_class} exceeds the minority-class count "
                f"{self.min_class_count}"
            )

        blocks_X = []
        blocks_y = []
        for label in self._model.class_labels:
            selected = self._permutation_for_class(label, replication_id)[:n_per_class]
            blocks_X.append(self._train_pool.X[selected])
            blocks_y.append(self._train_pool.y[selected])

        return Dataset(np.vstack(blocks_X), np.concatenate(blocks_y))

    def sample_test(self) -> Dataset:
        """Return the fixed real test dataset."""

        return self._test_dataset
