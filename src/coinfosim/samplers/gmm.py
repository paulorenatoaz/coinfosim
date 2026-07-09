"""
Class-conditional Gaussian Mixture sampler for CoInfoSim.

:class:`GMMClassConditionalSampler` draws balanced, deterministic,
prefix-nested training datasets and a single fixed test dataset from a
:class:`~coinfosim.models.gmm.GMMSimulationModel`.

It mirrors the semantics of
:class:`~coinfosim.samplers.gaussian.GaussianClassConditionalSampler`:

- ``n_per_class`` is the number of *training* samples per class;
- training output is balanced (every class contributes ``n_per_class`` rows);
- generation is deterministic in ``(base_seed, class_label, replication_id)``;
- training samples are *prefix-nested*: the first ``m`` rows of an ``N``-row
  draw are exactly the rows of an ``m``-row draw for the same class/replication.

Prefix-nesting is guaranteed by sampling row-by-row from a single seeded RNG:
each row consumes, in order, one uniform (to pick the mixture component via the
cumulative weights) followed by ``d`` standard normals (transformed by the
component Cholesky factor). Consuming a fixed, ordered amount of randomness per
row makes prefixes identical across ``n``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from coinfosim.models.gmm import GMMSimulationModel
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import derive_seed


class GMMClassConditionalSampler:
    """Sampler producing balanced train sets and one fixed test set from a GMM.

    Parameters
    ----------
    model:
        The :class:`GMMSimulationModel` to sample from.
    base_seed:
        Base random seed controlling all deterministic derivations.
    test_samples_per_class:
        Number of test samples generated per class for the fixed test set.
    """

    def __init__(
        self,
        model: GMMSimulationModel,
        base_seed: int = 0,
        test_samples_per_class: int = 1000,
    ) -> None:
        if test_samples_per_class <= 0:
            raise ValueError("test_samples_per_class must be a positive integer")

        self._model = model
        self._base_seed = int(base_seed)
        self._test_samples_per_class = int(test_samples_per_class)
        self._test_dataset: Optional[Dataset] = None

        # Precompute per-class cumulative weights and component Cholesky factors.
        self._cumweights: Dict[int, np.ndarray] = {}
        self._chols: Dict[int, List[np.ndarray]] = {}
        self._means: Dict[int, np.ndarray] = {}
        for label in model.class_labels:
            weights = model.component_weights(label)
            self._cumweights[label] = np.cumsum(weights)
            self._means[label] = model.component_means(label)
            covs = model.component_covariances(label)
            self._chols[label] = [
                np.linalg.cholesky(self._jitter(cov)) for cov in covs
            ]

    @staticmethod
    def _jitter(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """Return a positive-definite version of ``cov`` for Cholesky."""
        cov = np.asarray(cov, dtype=float)
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            return cov + eps * np.eye(cov.shape[0])

    @property
    def model(self) -> GMMSimulationModel:
        return self._model

    @property
    def base_seed(self) -> int:
        return self._base_seed

    @property
    def test_samples_per_class(self) -> int:
        return self._test_samples_per_class

    def _draw(self, label: int, n: int, seed: int) -> np.ndarray:
        """Draw ``n`` rows for class ``label`` in a prefix-nested manner.

        Each row consumes one uniform (component pick) then ``d`` standard
        normals, in order, from a single seeded generator.
        """
        rng = np.random.default_rng(seed)
        cum = self._cumweights[label]
        means = self._means[label]
        chols = self._chols[label]
        d = self._model.d

        rows = np.empty((n, d), dtype=float)
        for i in range(n):
            u = rng.random()
            comp = int(np.searchsorted(cum, u, side="right"))
            if comp >= len(chols):  # numerical guard (u == 1.0)
                comp = len(chols) - 1
            z = rng.standard_normal(d)
            rows[i] = means[comp] + chols[comp] @ z
        return rows

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        """Return a balanced training :class:`Dataset` with ``n_per_class`` per class."""
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
                samples = self._draw(label, self._test_samples_per_class, seed)
                feature_blocks.append(samples)
                label_blocks.append(
                    np.full(self._test_samples_per_class, label)
                )

            X = np.vstack(feature_blocks)
            y = np.concatenate(label_blocks)
            self._test_dataset = Dataset(X, y)
        return self._test_dataset
