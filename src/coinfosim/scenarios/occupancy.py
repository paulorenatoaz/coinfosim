"""Occupancy Detection scenario helpers for Sprint 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from coinfosim.datasets.occupancy import OccupancyData
from coinfosim.models.gaussian import GaussianSimulationModel

OCCUPANCY_SCENARIO_NAME = "Occupancy Detection — Synthetic-to-Real Transfer Scenario"
OCCUPANCY_SCENARIO_QUESTION = (
    "Does training on single-Gaussian synthetic data preserve the cooperative "
    "advantages observed when classifiers are evaluated on real Occupancy data? "
    "Equivalently: which training distribution best preserves the cooperative "
    "structure observed under real-data evaluation in the Occupancy Detection "
    "dataset?"
)


@dataclass(frozen=True)
class GaussianAnchoredOccupancyScenario:
    """Gaussian model estimated from standardized occupancy training data."""

    name: str
    question: str
    model: GaussianSimulationModel
    channel_names: Tuple[str, ...]
    means: Dict[int, np.ndarray]
    covariances: Dict[int, np.ndarray]
    ridge_by_class: Dict[int, float]
    source: str = "standardized datatraining.txt"

    @property
    def d(self) -> int:
        return self.model.d


def build_gaussian_anchored_occupancy_model(
    data: OccupancyData,
    initial_ridge: float = 1e-10,
    max_ridge: float = 1e-3,
) -> GaussianAnchoredOccupancyScenario:
    """Estimate a Gaussian simulation model from standardized train data.

    Means and covariance matrices are estimated separately for each occupancy
    class using the standardized training pool. If a covariance matrix is not
    positive definite, progressively larger diagonal ridge values are tried and
    the applied value is recorded.
    """

    if initial_ridge <= 0:
        raise ValueError("initial_ridge must be positive")
    if max_ridge < initial_ridge:
        raise ValueError("max_ridge must be >= initial_ridge")

    X = data.train_dataset.X
    y = data.train_dataset.y
    means: Dict[int, np.ndarray] = {}
    covariances: Dict[int, np.ndarray] = {}
    ridge_by_class: Dict[int, float] = {}

    for label in data.class_labels:
        class_X = X[y == label]
        if class_X.shape[0] < 2:
            raise ValueError(f"class {label} needs at least two rows")
        means[label] = class_X.mean(axis=0)
        sample_cov = np.cov(class_X, rowvar=False, ddof=1)
        covariances[label], ridge_by_class[label] = _make_positive_definite(
            sample_cov,
            initial_ridge=initial_ridge,
            max_ridge=max_ridge,
        )

    model = GaussianSimulationModel(means=means, covariances=covariances)
    return GaussianAnchoredOccupancyScenario(
        name=OCCUPANCY_SCENARIO_NAME,
        question=OCCUPANCY_SCENARIO_QUESTION,
        model=model,
        channel_names=tuple(data.channel_names),
        means={label: value.copy() for label, value in means.items()},
        covariances={label: value.copy() for label, value in covariances.items()},
        ridge_by_class=dict(ridge_by_class),
    )


def _make_positive_definite(
    covariance: np.ndarray,
    initial_ridge: float,
    max_ridge: float,
) -> Tuple[np.ndarray, float]:
    cov = np.asarray(covariance, dtype=float)
    cov = (cov + cov.T) / 2.0
    if _is_positive_definite(cov):
        return cov, 0.0

    ridge = float(initial_ridge)
    eye = np.eye(cov.shape[0])
    while ridge <= max_ridge:
        candidate = cov + ridge * eye
        if _is_positive_definite(candidate):
            return candidate, ridge
        ridge *= 10.0

    raise ValueError(
        f"covariance could not be made positive definite with ridge <= {max_ridge}"
    )


def _is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
