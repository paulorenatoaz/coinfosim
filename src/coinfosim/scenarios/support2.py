"""SUPPORT2 wrappers for generic dataset-anchored model fitting."""

from __future__ import annotations

from coinfosim.datasets.support2 import Support2Data
from coinfosim.scenarios.dataset_anchored import (
    GMMAnchoredScenario,
    GaussianAnchoredScenario,
    build_gaussian_anchored_model,
    build_gmm_anchored_model,
)

SUPPORT2_SCENARIO_NAME = "SUPPORT2 180-Day Mortality Baseline"
SUPPORT2_SCENARIO_QUESTION = (
    "To what extent do class-conditional Single Gaussian and GMM synthetic "
    "training distributions preserve the cooperative attribute-subset structure "
    "observed for 180-day mortality prediction on a fixed real SUPPORT2 test set?"
)
SUPPORT2_TRAINING_SOURCE = (
    "standardized fixed SUPPORT2 training reservoir with death_180d labels"
)

GaussianAnchoredSupport2Scenario = GaussianAnchoredScenario
GMMAnchoredSupport2Scenario = GMMAnchoredScenario


def build_gaussian_anchored_support2_model(
    data: Support2Data,
    initial_ridge: float = 1e-10,
    max_ridge: float = 1e-3,
) -> GaussianAnchoredSupport2Scenario:
    return build_gaussian_anchored_model(
        data,
        name=SUPPORT2_SCENARIO_NAME,
        question=SUPPORT2_SCENARIO_QUESTION,
        source=SUPPORT2_TRAINING_SOURCE,
        initial_ridge=initial_ridge,
        max_ridge=max_ridge,
    )


def build_gmm_anchored_support2_model(
    data: Support2Data,
    max_components: int = 5,
    min_points_per_component: int = 50,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    criterion: str = "bic",
    random_state: int = 0,
) -> GMMAnchoredSupport2Scenario:
    return build_gmm_anchored_model(
        data,
        name=SUPPORT2_SCENARIO_NAME,
        question=SUPPORT2_SCENARIO_QUESTION,
        source=SUPPORT2_TRAINING_SOURCE,
        max_components=max_components,
        min_points_per_component=min_points_per_component,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        criterion=criterion,
        random_state=random_state,
    )
