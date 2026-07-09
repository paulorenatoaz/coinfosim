"""Occupancy Detection scenario helpers for Sprint 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from coinfosim.datasets.occupancy import OccupancyData
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.models.gmm import GMMSimulationModel

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


@dataclass(frozen=True)
class GMMAnchoredOccupancyScenario:
    """Class-conditional GMM model estimated from standardized occupancy data."""

    name: str
    question: str
    model: GMMSimulationModel
    channel_names: Tuple[str, ...]
    selected_components: Dict[int, int]
    model_selection: Dict[int, Dict[str, Any]]
    source: str = "standardized datatraining.txt"

    @property
    def d(self) -> int:
        return self.model.d


def build_gmm_anchored_occupancy_model(
    data: OccupancyData,
    max_components: int = 5,
    min_points_per_component: int = 50,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    criterion: str = "bic",
    random_state: int = 0,
) -> GMMAnchoredOccupancyScenario:
    """Fit one class-conditional Gaussian mixture per class from train data.

    For each class the number of mixture components is selected by BIC (default)
    over a conservative candidate range ``K = 1..Kmax_class`` where
    ``Kmax_class = max(1, min(max_components, n_class // min_points_per_component))``.
    Mixtures are fitted in the full channel space (no per-subset fitting), so a
    single joint generative distribution per class is preserved; the existing
    subset machinery restricts features downstream.
    """
    from sklearn.mixture import GaussianMixture

    if criterion not in ("bic", "aic"):
        raise ValueError("criterion must be 'bic' or 'aic'")

    X = data.train_dataset.X
    y = data.train_dataset.y

    weights: Dict[int, np.ndarray] = {}
    means: Dict[int, np.ndarray] = {}
    covariances: Dict[int, np.ndarray] = {}
    selected_components: Dict[int, int] = {}
    model_selection: Dict[int, Dict[str, Any]] = {}

    for label in data.class_labels:
        class_X = X[y == label]
        n_class = int(class_X.shape[0])
        if n_class < 2:
            raise ValueError(f"class {label} needs at least two rows")

        kmax = max(1, min(max_components, n_class // min_points_per_component))
        candidate_components = list(range(1, kmax + 1))

        scores: Dict[str, Dict[str, float]] = {}
        best_k = 1
        best_estimator = None
        best_score = np.inf
        for k in candidate_components:
            estimator = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                n_init=n_init,
                random_state=random_state,
            )
            estimator.fit(class_X)
            bic = float(estimator.bic(class_X))
            aic = float(estimator.aic(class_X))
            scores[str(k)] = {"bic": bic, "aic": aic}
            score = bic if criterion == "bic" else aic
            if score < best_score:
                best_score = score
                best_k = k
                best_estimator = estimator

        assert best_estimator is not None
        full_covs = _gmm_full_covariances(best_estimator, covariance_type)

        weights[label] = np.asarray(best_estimator.weights_, dtype=float)
        means[label] = np.asarray(best_estimator.means_, dtype=float)
        covariances[label] = full_covs
        selected_components[label] = int(best_k)
        model_selection[label] = {
            "class_label": int(label),
            "n_samples": n_class,
            "candidate_components": candidate_components,
            "selected_components": int(best_k),
            "criterion": criterion,
            "scores": scores,
            "covariance_type": covariance_type,
            "reg_covar": reg_covar,
            "n_init": n_init,
        }

    model = GMMSimulationModel(
        weights=weights,
        means=means,
        covariances=covariances,
        model_selection=model_selection,
        channel_names=tuple(data.channel_names),
        name=OCCUPANCY_SCENARIO_NAME,
    )
    return GMMAnchoredOccupancyScenario(
        name=OCCUPANCY_SCENARIO_NAME,
        question=OCCUPANCY_SCENARIO_QUESTION,
        model=model,
        channel_names=tuple(data.channel_names),
        selected_components=dict(selected_components),
        model_selection=model_selection,
    )


def _gmm_full_covariances(estimator, covariance_type: str) -> np.ndarray:
    """Return component covariances as a full ``(K, d, d)`` array.

    ``sklearn`` stores covariances differently per ``covariance_type``; this
    normalizes them into explicit per-component ``(d, d)`` matrices.
    """
    covs = np.asarray(estimator.covariances_, dtype=float)
    k = int(estimator.n_components)
    d = int(estimator.means_.shape[1])
    if covariance_type == "full":
        return covs
    if covariance_type == "tied":
        return np.repeat(covs[np.newaxis, :, :], k, axis=0)
    if covariance_type == "diag":
        return np.stack([np.diag(covs[j]) for j in range(k)], axis=0)
    if covariance_type == "spherical":
        return np.stack([covs[j] * np.eye(d) for j in range(k)], axis=0)
    raise ValueError(f"unsupported covariance_type: {covariance_type!r}")


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
