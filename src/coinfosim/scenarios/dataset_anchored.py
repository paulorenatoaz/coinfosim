"""Generic model fitting for dataset-anchored CoInfoSim scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from coinfosim.datasets.common import DatasetAnchoredData
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.models.gmm import GMMSimulationModel


@dataclass(frozen=True)
class GaussianAnchoredScenario:
    """Single-Gaussian model estimated from standardized training data."""

    name: str
    question: str
    model: GaussianSimulationModel
    channel_names: Tuple[str, ...]
    means: Dict[int, np.ndarray]
    covariances: Dict[int, np.ndarray]
    ridge_by_class: Dict[int, float]
    source: str

    @property
    def d(self) -> int:
        return self.model.d


@dataclass(frozen=True)
class GMMAnchoredScenario:
    """Class-conditional GMM estimated from standardized training data."""

    name: str
    question: str
    model: GMMSimulationModel
    channel_names: Tuple[str, ...]
    selected_components: Dict[int, int]
    model_selection: Dict[int, Dict[str, Any]]
    source: str

    @property
    def d(self) -> int:
        return self.model.d


def build_gaussian_anchored_model(
    data: DatasetAnchoredData,
    *,
    name: str,
    question: str,
    source: str,
    initial_ridge: float = 1e-10,
    max_ridge: float = 1e-3,
) -> GaussianAnchoredScenario:
    """Fit one sample Gaussian per class using only the training reservoir."""

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
        sample_covariance = np.cov(class_X, rowvar=False, ddof=1)
        covariances[label], ridge_by_class[label] = _make_positive_definite(
            sample_covariance,
            initial_ridge=initial_ridge,
            max_ridge=max_ridge,
        )

    model = GaussianSimulationModel(means=means, covariances=covariances)
    return GaussianAnchoredScenario(
        name=name,
        question=question,
        model=model,
        channel_names=tuple(data.channel_names),
        means={label: value.copy() for label, value in means.items()},
        covariances={label: value.copy() for label, value in covariances.items()},
        ridge_by_class=dict(ridge_by_class),
        source=source,
    )


def build_gmm_anchored_model(
    data: DatasetAnchoredData,
    *,
    name: str,
    question: str,
    source: str,
    max_components: int = 5,
    min_points_per_component: int = 50,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    criterion: str = "bic",
    random_state: int = 0,
) -> GMMAnchoredScenario:
    """Fit one full-space class-conditional GMM per training class."""

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
        weights[label] = np.asarray(best_estimator.weights_, dtype=float)
        means[label] = np.asarray(best_estimator.means_, dtype=float)
        covariances[label] = _gmm_full_covariances(
            best_estimator, covariance_type
        )
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
        name=name,
    )
    return GMMAnchoredScenario(
        name=name,
        question=question,
        model=model,
        channel_names=tuple(data.channel_names),
        selected_components=dict(selected_components),
        model_selection=model_selection,
        source=source,
    )


def _gmm_full_covariances(estimator, covariance_type: str) -> np.ndarray:
    """Normalize sklearn covariance storage to explicit ``(K, d, d)``."""

    covariances = np.asarray(estimator.covariances_, dtype=float)
    k = int(estimator.n_components)
    d = int(estimator.means_.shape[1])
    if covariance_type == "full":
        return covariances
    if covariance_type == "tied":
        return np.repeat(covariances[np.newaxis, :, :], k, axis=0)
    if covariance_type == "diag":
        return np.stack([np.diag(covariances[j]) for j in range(k)], axis=0)
    if covariance_type == "spherical":
        return np.stack(
            [covariances[j] * np.eye(d) for j in range(k)], axis=0
        )
    raise ValueError(f"unsupported covariance_type: {covariance_type!r}")


def _make_positive_definite(
    covariance: np.ndarray,
    initial_ridge: float,
    max_ridge: float,
) -> Tuple[np.ndarray, float]:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    if _is_positive_definite(covariance):
        return covariance, 0.0

    ridge = float(initial_ridge)
    identity = np.eye(covariance.shape[0])
    while ridge <= max_ridge:
        candidate = covariance + ridge * identity
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
