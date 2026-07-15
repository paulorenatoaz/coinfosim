from dataclasses import replace

import numpy as np
import pytest

from coinfosim.datasets.air_quality import AIR_QUALITY_CHANNELS, load_air_quality_data
from coinfosim.samplers.dataset import Dataset
from coinfosim.scenarios.air_quality import (
    build_gaussian_anchored_air_quality_model,
    build_gmm_anchored_air_quality_model,
)


@pytest.fixture(scope="module")
def air_quality_data():
    return load_air_quality_data("data/raw/air_quality")


@pytest.fixture(scope="module")
def gaussian_scenario(air_quality_data):
    return build_gaussian_anchored_air_quality_model(air_quality_data)


@pytest.fixture(scope="module")
def gmm_scenario(air_quality_data):
    return build_gmm_anchored_air_quality_model(
        air_quality_data,
        max_components=2,
        min_points_per_component=10_000,
        n_init=1,
        random_state=17,
    )


def test_air_quality_wrappers_fit_five_dimensional_training_models(
    air_quality_data, gaussian_scenario, gmm_scenario
):
    assert gaussian_scenario.d == gmm_scenario.d == 5
    assert gaussian_scenario.channel_names == AIR_QUALITY_CHANNELS
    assert gmm_scenario.channel_names == AIR_QUALITY_CHANNELS
    assert gaussian_scenario.model.class_labels == (0, 1)
    assert gmm_scenario.model.class_labels == (0, 1)

    for label in (0, 1):
        class_X = air_quality_data.train_dataset.X[
            air_quality_data.train_dataset.y == label
        ]
        assert np.allclose(gaussian_scenario.means[label], class_X.mean(axis=0))
        assert np.allclose(
            gaussian_scenario.covariances[label],
            np.cov(class_X, rowvar=False, ddof=1)
            + gaussian_scenario.ridge_by_class[label] * np.eye(5),
        )
        assert gmm_scenario.selected_components[label] == 1
        assert gmm_scenario.model.component_means(label).shape == (1, 5)
        assert gmm_scenario.model.component_covariances(label).shape == (1, 5, 5)
        assert np.allclose(
            gmm_scenario.model.component_means(label)[0], class_X.mean(axis=0)
        )


def test_air_quality_models_are_unchanged_when_only_test_data_changes(
    air_quality_data, gaussian_scenario, gmm_scenario
):
    changed_test = Dataset(
        air_quality_data.test_dataset.X + 1_000_000.0,
        air_quality_data.test_dataset.y[::-1],
    )
    changed_data = replace(air_quality_data, test_dataset=changed_test)
    gaussian_changed = build_gaussian_anchored_air_quality_model(changed_data)
    gmm_changed = build_gmm_anchored_air_quality_model(
        changed_data,
        max_components=2,
        min_points_per_component=10_000,
        n_init=1,
        random_state=17,
    )

    for label in (0, 1):
        assert np.array_equal(
            gaussian_scenario.means[label], gaussian_changed.means[label]
        )
        assert np.array_equal(
            gaussian_scenario.covariances[label], gaussian_changed.covariances[label]
        )
        assert np.array_equal(
            gmm_scenario.model.component_weights(label),
            gmm_changed.model.component_weights(label),
        )
        assert np.array_equal(
            gmm_scenario.model.component_means(label),
            gmm_changed.model.component_means(label),
        )
        assert np.array_equal(
            gmm_scenario.model.component_covariances(label),
            gmm_changed.model.component_covariances(label),
        )


def test_air_quality_wrapper_metadata_contains_no_occupancy_names(
    gaussian_scenario, gmm_scenario
):
    for scenario in (gaussian_scenario, gmm_scenario):
        metadata = " ".join((scenario.name, scenario.question, scenario.source)).lower()
        assert "air quality" in metadata
        assert "benzene" in metadata
        assert "occupancy" not in metadata
        assert "datatraining.txt" not in metadata
