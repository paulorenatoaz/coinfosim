"""
Gaussian class-conditional sampler for CoInfoSim Sprint 1.

:class:`GaussianClassConditionalSampler` draws balanced, deterministic,
prefix-nested training datasets and a single fixed test dataset from a
:class:`~coinfosim.models.gaussian.GaussianSimulationModel`.

Key Sprint 1 semantics
----------------------
- ``n_per_class`` is the number of *training* samples per class.
- Training output is balanced: every class contributes ``n_per_class`` rows.
- Training generation is deterministic in ``(class, replication_id)``.
- Training samples are *prefix-nested*: for a fixed class and replication,
  the first ``m`` rows of ``sample_train(n_per_class=N, ...)`` are exactly the
  rows of ``sample_train(n_per_class=m, ...)`` whenever ``m <= N``.
- The test set is generated once and reused (fixed) across sample sizes,
  replications, subsets, and classifiers.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.samplers.dataset import Dataset

# Stable integer codes for the split component of the seed derivation.
_SPLIT_CODES = {"train": 1, "test": 2}


def derive_seed(
    base_seed: int,
    class_label: int,
    split: str,
    replication_id: Optional[int] = None,
) -> int:
    """Derive a deterministic 64-bit seed from explicit components.

    The seed is a pure function of its inputs so that runs are reproducible.

    Parameters
    ----------
    base_seed:
        Base seed for the whole simulation.
    class_label:
        Class label being sampled.
    split:
        Either ``"train"`` or ``"test"``.
    replication_id:
        Replication index for training samples. Ignored for the test split
        (the test set is fixed and replication-independent).
    """
    if split not in _SPLIT_CODES:
        raise ValueError(f"split must be one of {sorted(_SPLIT_CODES)}, got {split!r}")

    split_code = _SPLIT_CODES[split]
    rep_component = 0 if replication_id is None else int(replication_id) + 1

    entropy = (
        int(base_seed),
        split_code,
        int(class_label),
        rep_component,
    )
    seed_seq = np.random.SeedSequence(entropy)
    return int(seed_seq.generate_state(1, dtype=np.uint64)[0])


class GaussianClassConditionalSampler:
    """Sampler producing balanced train sets and one fixed test set.

    Parameters
    ----------
    model:
        The :class:`GaussianSimulationModel` to sample from.
    base_seed:
        Base random seed controlling all deterministic derivations.
    test_samples_per_class:
        Number of test samples generated per class for the fixed test set.
        Used for every class unless ``test_samples_per_class_by_label`` is given.
    test_samples_per_class_by_label:
        Optional per-class override mapping ``class_label -> count``. When
        provided, the fixed test set uses these exact class counts instead of a
        single uniform ``test_samples_per_class``. This supports scenarios (such
        as the Occupancy Gaussian-anchored arm) whose synthetic evaluation set
        must match the class counts of a real evaluation split. Generic
        synthetic Gaussian simulations may ignore this parameter and keep the
        uniform behavior.
    """

    def __init__(
        self,
        model: GaussianSimulationModel,
        base_seed: int = 0,
        test_samples_per_class: int = 1000,
        test_samples_per_class_by_label: Optional[Mapping[int, int]] = None,
    ) -> None:
        if test_samples_per_class <= 0:
            raise ValueError("test_samples_per_class must be a positive integer")

        self._model = model
        self._base_seed = int(base_seed)
        self._test_samples_per_class = int(test_samples_per_class)
        self._test_dataset: Optional[Dataset] = None

        if test_samples_per_class_by_label is None:
            self._test_counts_by_label: Optional[dict] = None
        else:
            counts = {
                int(label): int(count)
                for label, count in test_samples_per_class_by_label.items()
            }
            missing = [
                label for label in model.class_labels if label not in counts
            ]
            if missing:
                raise ValueError(
                    f"test_samples_per_class_by_label is missing class labels {missing}"
                )
            if any(count <= 0 for count in counts.values()):
                raise ValueError(
                    "test_samples_per_class_by_label counts must be positive"
                )
            self._test_counts_by_label = counts

    def _test_count(self, label: int) -> int:
        if self._test_counts_by_label is not None:
            return self._test_counts_by_label[int(label)]
        return self._test_samples_per_class

    @property
    def model(self) -> GaussianSimulationModel:
        return self._model

    @property
    def base_seed(self) -> int:
        return self._base_seed

    @property
    def test_samples_per_class(self) -> int:
        return self._test_samples_per_class

    def _draw(self, label: int, n: int, seed: int) -> np.ndarray:
        """Draw ``n`` samples for class ``label`` using a fresh seeded RNG.

        NumPy's ``Generator.multivariate_normal`` produces rows independently
        from consecutive blocks of standard normals, so drawing ``n`` rows
        from a fixed seed is prefix-nested: the first ``m`` rows match a draw
        of ``m`` rows from the same seed.
        """
        rng = np.random.default_rng(seed)
        mean = self._model.mean(label)
        cov = self._model.covariance(label)
        return rng.multivariate_normal(mean, cov, size=n)

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        """Return a balanced training :class:`Dataset` with ``n_per_class`` per class.

        Rows are ordered class-by-class in ascending class-label order.
        """
        if n_per_class <= 0:
            raise ValueError("n_per_class must be a positive integer")

        feature_blocks = []
        label_blocks = []
        for label in self._model.class_labels:
            seed = derive_seed(
                self._base_seed, label, split="train", replication_id=replication_id
            )
            samples = self._draw(label, n_per_class, seed)
            feature_blocks.append(samples)
            label_blocks.append(np.full(n_per_class, label))

        X = np.vstack(feature_blocks)
        y = np.concatenate(label_blocks)
        return Dataset(X, y)

    def sample_test(self) -> Dataset:
        """Return the fixed test :class:`Dataset`, generating it once and caching it."""
        if self._test_dataset is None:
            feature_blocks = []
            label_blocks = []
            for label in self._model.class_labels:
                seed = derive_seed(self._base_seed, label, split="test")
                count = self._test_count(label)
                samples = self._draw(label, count, seed)
                feature_blocks.append(samples)
                label_blocks.append(np.full(count, label))

            X = np.vstack(feature_blocks)
            y = np.concatenate(label_blocks)
            self._test_dataset = Dataset(X, y)
        return self._test_dataset
