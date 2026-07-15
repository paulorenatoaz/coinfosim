from dataclasses import replace

import numpy as np
import pytest

from coinfosim.datasets.support2 import SUPPORT2_CHANNELS, load_support2_data
from coinfosim.samplers.dataset import Dataset
from coinfosim.scenarios.support2 import (
    build_gaussian_anchored_support2_model,
    build_gmm_anchored_support2_model,
)


@pytest.fixture(scope="module")
def support2_data():
    return load_support2_data("data/raw/support2")


@pytest.fixture(scope="module")
def anchored_models(support2_data):
    gaussian = build_gaussian_anchored_support2_model(support2_data)
    gmm = build_gmm_anchored_support2_model(
        support2_data,
        max_components=2,
        min_points_per_component=10_000,
        n_init=1,
        random_state=17,
    )
    return gaussian, gmm


def test_support2_wrappers_fit_training_only_seven_dimensional_models(
    support2_data, anchored_models
):
    gaussian, gmm = anchored_models
    assert gaussian.d == gmm.d == 7
    assert gaussian.channel_names == gmm.channel_names == SUPPORT2_CHANNELS
    assert gaussian.model.class_labels == gmm.model.class_labels == (0, 1)
    assert np.bincount(support2_data.train_dataset.y).tolist() == [3_768, 3_330]
    for label, expected_size in ((0, 3_768), (1, 3_330)):
        class_X = support2_data.train_dataset.X[support2_data.train_dataset.y == label]
        assert len(class_X) == expected_size
        assert np.allclose(gaussian.means[label], class_X.mean(axis=0))
        assert np.allclose(
            gaussian.covariances[label],
            np.cov(class_X, rowvar=False, ddof=1)
            + gaussian.ridge_by_class[label] * np.eye(7),
        )
        assert gmm.selected_components[label] == 1
        assert gmm.model.component_means(label).shape == (1, 7)
        assert gmm.model.component_covariances(label).shape == (1, 7, 7)


def test_support2_models_ignore_test_mutation_and_excluded_fields(
    support2_data, anchored_models
):
    gaussian, gmm = anchored_models
    changed_data = replace(
        support2_data,
        test_dataset=Dataset(
            support2_data.test_dataset.X + 1_000_000,
            support2_data.test_dataset.y[::-1],
        ),
    )
    changed_gaussian = build_gaussian_anchored_support2_model(changed_data)
    changed_gmm = build_gmm_anchored_support2_model(
        changed_data,
        max_components=2,
        min_points_per_component=10_000,
        n_init=1,
        random_state=17,
    )
    for label in (0, 1):
        assert np.array_equal(gaussian.means[label], changed_gaussian.means[label])
        assert np.array_equal(
            gaussian.covariances[label], changed_gaussian.covariances[label]
        )
        assert np.array_equal(
            gmm.model.component_means(label), changed_gmm.model.component_means(label)
        )
        assert np.array_equal(
            gmm.model.component_covariances(label),
            changed_gmm.model.component_covariances(label),
        )
    assert support2_data.train_dataset.X.shape[1] == len(SUPPORT2_CHANNELS)
    assert {"death", "d.time", "hospdead"}.isdisjoint(SUPPORT2_CHANNELS)
