"""
Class-conditional Gaussian Mixture Model for CoInfoSim.

:class:`GMMSimulationModel` represents one Gaussian mixture per class in the
full attribute space. It is a distinct model type from
:class:`~coinfosim.models.gaussian.GaussianSimulationModel`: instead of a single
mean/covariance per class it stores, for each class, a set of mixture weights,
component means and component covariances, plus optional per-class model
selection metadata (selected number of components, BIC/AIC scores, etc.).

The Monte Carlo simulator only needs dimensionality and class labels from the
model; synthetic samples are produced by
:class:`~coinfosim.samplers.gmm.GMMClassConditionalSampler`. The rich mixture
parameters are used by the detailed GMM Monte Carlo report and are persisted for
scientific inspection.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


class GMMSimulationModel:
    """A class-conditional Gaussian mixture model in the full attribute space.

    Parameters
    ----------
    weights:
        Mapping ``class_label -> array`` of mixture weights, shape ``(K_c,)``.
    means:
        Mapping ``class_label -> array`` of component means, shape ``(K_c, d)``.
    covariances:
        Mapping ``class_label -> array`` of component covariance matrices,
        shape ``(K_c, d, d)``.
    model_selection:
        Optional mapping ``class_label -> dict`` with per-class model-selection
        metadata (candidate/selected component counts, BIC/AIC scores, fit
        configuration).
    channel_names:
        Optional attribute names.
    name:
        Optional model label.
    """

    def __init__(
        self,
        weights: Mapping[int, Sequence[float]],
        means: Mapping[int, Sequence[Sequence[float]]],
        covariances: Mapping[int, Sequence[Sequence[Sequence[float]]]],
        model_selection: Optional[Mapping[int, Dict[str, Any]]] = None,
        channel_names: Optional[Sequence[str]] = None,
        name: str = "gmm",
    ) -> None:
        if not (isinstance(weights, Mapping) and isinstance(means, Mapping)
                and isinstance(covariances, Mapping)):
            raise ValueError("weights, means and covariances must be mappings")

        labels = sorted(int(label) for label in weights.keys())
        if labels != sorted(int(label) for label in means.keys()) or labels != sorted(
            int(label) for label in covariances.keys()
        ):
            raise ValueError(
                "weights, means and covariances must define the same class labels"
            )
        if len(labels) < 2:
            raise ValueError("at least two classes are required")

        stored_weights: Dict[int, np.ndarray] = {}
        stored_means: Dict[int, np.ndarray] = {}
        stored_covs: Dict[int, np.ndarray] = {}

        d: Optional[int] = None
        for label in labels:
            w = np.asarray(weights[label], dtype=float)
            m = np.asarray(means[label], dtype=float)
            c = np.asarray(covariances[label], dtype=float)

            if w.ndim != 1 or w.shape[0] < 1:
                raise ValueError(f"weights for class {label} must be 1-D and non-empty")
            k = int(w.shape[0])
            if m.ndim != 2 or m.shape[0] != k:
                raise ValueError(
                    f"means for class {label} must have shape (K_c, d) with K_c={k}"
                )
            if d is None:
                d = int(m.shape[1])
            if m.shape[1] != d:
                raise ValueError(
                    f"means for class {label} have d={m.shape[1]}, expected {d}"
                )
            if c.ndim != 3 or c.shape != (k, d, d):
                raise ValueError(
                    f"covariances for class {label} must have shape ({k}, {d}, {d}), "
                    f"got {c.shape}"
                )
            if not np.allclose(w.sum(), 1.0, atol=1e-6):
                w = w / w.sum()

            stored_weights[label] = w
            stored_means[label] = m
            stored_covs[label] = c

        assert d is not None
        self._class_labels: Tuple[int, ...] = tuple(labels)
        self._d = int(d)
        self._weights = stored_weights
        self._means = stored_means
        self._covariances = stored_covs
        self._model_selection: Dict[int, Dict[str, Any]] = {
            int(label): dict(info)
            for label, info in (model_selection or {}).items()
        }
        self._channel_names: Tuple[str, ...] = (
            tuple(str(c) for c in channel_names)
            if channel_names is not None
            else tuple(f"X{i + 1}" for i in range(self._d))
        )
        self.name = str(name)

    # -- minimal simulator/sampler interface -------------------------------- #
    @property
    def d(self) -> int:
        return self._d

    @property
    def num_channels(self) -> int:
        return self._d

    @property
    def K(self) -> int:
        """Number of classes."""
        return len(self._class_labels)

    @property
    def num_classes(self) -> int:
        return len(self._class_labels)

    @property
    def class_labels(self) -> Tuple[int, ...]:
        return self._class_labels

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return self._channel_names

    # -- GMM-specific accessors --------------------------------------------- #
    def component_weights(self, label: int) -> np.ndarray:
        return self._weights[int(label)].copy()

    def component_means(self, label: int) -> np.ndarray:
        return self._means[int(label)].copy()

    def component_covariances(self, label: int) -> np.ndarray:
        return self._covariances[int(label)].copy()

    def selected_components(self, label: int) -> int:
        return int(self._weights[int(label)].shape[0])

    def model_selection(self, label: int) -> Dict[str, Any]:
        return dict(self._model_selection.get(int(label), {}))

    @property
    def model_selection_by_class(self) -> Dict[int, Dict[str, Any]]:
        return {label: dict(info) for label, info in self._model_selection.items()}
