"""Occupancy-specific wrappers for dataset-anchored model fitting."""

from __future__ import annotations

from coinfosim.datasets.occupancy import OccupancyData
from coinfosim.scenarios.dataset_anchored import (
    GMMAnchoredScenario,
    GaussianAnchoredScenario,
    build_gaussian_anchored_model,
    build_gmm_anchored_model,
)

OCCUPANCY_SCENARIO_NAME = "Occupancy Detection — Synthetic-to-Real Transfer Scenario"
OCCUPANCY_SCENARIO_QUESTION = (
    "Does training on single-Gaussian synthetic data preserve the cooperative "
    "advantages observed when classifiers are evaluated on real Occupancy data? "
    "Equivalently: which training distribution best preserves the cooperative "
    "structure observed under real-data evaluation in the Occupancy Detection "
    "dataset?"
)
OCCUPANCY_TRAINING_SOURCE = "standardized datatraining.txt"

GaussianAnchoredOccupancyScenario = GaussianAnchoredScenario
GMMAnchoredOccupancyScenario = GMMAnchoredScenario


def build_gaussian_anchored_occupancy_model(
    data: OccupancyData,
    initial_ridge: float = 1e-10,
    max_ridge: float = 1e-3,
) -> GaussianAnchoredOccupancyScenario:
    """Fit the Occupancy single-Gaussian model through the generic builder."""

    return build_gaussian_anchored_model(
        data,
        name=OCCUPANCY_SCENARIO_NAME,
        question=OCCUPANCY_SCENARIO_QUESTION,
        source=OCCUPANCY_TRAINING_SOURCE,
        initial_ridge=initial_ridge,
        max_ridge=max_ridge,
    )


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
    """Fit the Occupancy GMM model through the generic builder."""

    return build_gmm_anchored_model(
        data,
        name=OCCUPANCY_SCENARIO_NAME,
        question=OCCUPANCY_SCENARIO_QUESTION,
        source=OCCUPANCY_TRAINING_SOURCE,
        max_components=max_components,
        min_points_per_component=min_points_per_component,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        criterion=criterion,
        random_state=random_state,
    )
