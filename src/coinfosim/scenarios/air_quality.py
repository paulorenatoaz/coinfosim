"""Air Quality wrappers for dataset-anchored model fitting."""

from __future__ import annotations

from coinfosim.datasets.air_quality import AirQualityData
from coinfosim.scenarios.dataset_anchored import (
    GMMAnchoredScenario,
    GaussianAnchoredScenario,
    build_gaussian_anchored_model,
    build_gmm_anchored_model,
)

AIR_QUALITY_SCENARIO_NAME = (
    "Air Quality — Dataset-Anchored Synthetic-to-Real Transfer Scenario"
)
AIR_QUALITY_SCENARIO_QUESTION = (
    "To what extent do single-Gaussian and class-conditional GMM synthetic "
    "training distributions preserve the cooperative attribute-subset structure observed "
    "when classifiers detect elevated benzene concentration on a fixed future "
    "real-data evaluation period?"
)
AIR_QUALITY_TRAINING_SOURCE = (
    "standardized chronological training reservoir from AirQualityUCI.csv"
)

GaussianAnchoredAirQualityScenario = GaussianAnchoredScenario
GMMAnchoredAirQualityScenario = GMMAnchoredScenario


def build_gaussian_anchored_air_quality_model(
    data: AirQualityData,
    initial_ridge: float = 1e-10,
    max_ridge: float = 1e-3,
) -> GaussianAnchoredAirQualityScenario:
    return build_gaussian_anchored_model(
        data,
        name=AIR_QUALITY_SCENARIO_NAME,
        question=AIR_QUALITY_SCENARIO_QUESTION,
        source=AIR_QUALITY_TRAINING_SOURCE,
        initial_ridge=initial_ridge,
        max_ridge=max_ridge,
    )


def build_gmm_anchored_air_quality_model(
    data: AirQualityData,
    max_components: int = 5,
    min_points_per_component: int = 50,
    covariance_type: str = "full",
    reg_covar: float = 1e-6,
    n_init: int = 5,
    criterion: str = "bic",
    random_state: int = 0,
) -> GMMAnchoredAirQualityScenario:
    return build_gmm_anchored_model(
        data,
        name=AIR_QUALITY_SCENARIO_NAME,
        question=AIR_QUALITY_SCENARIO_QUESTION,
        source=AIR_QUALITY_TRAINING_SOURCE,
        max_components=max_components,
        min_points_per_component=min_points_per_component,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        n_init=n_init,
        criterion=criterion,
        random_state=random_state,
    )
