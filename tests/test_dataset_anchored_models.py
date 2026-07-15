from dataclasses import dataclass

import numpy as np
import pandas as pd

from coinfosim.datasets.common import StandardizationParameters
from coinfosim.samplers.dataset import Dataset
from coinfosim.scenarios.dataset_anchored import (
    build_gaussian_anchored_model,
    build_gmm_anchored_model,
)


@dataclass(frozen=True)
class TinyAnchoredData:
    train_dataset: Dataset
    test_dataset: Dataset
    channel_names: tuple[str, ...]
    class_labels: tuple[int, ...]
    standardization: StandardizationParameters


def _tiny_data(test_offset: float = 0.0) -> TinyAnchoredData:
    class_zero = np.array(
        [[-2.0, -0.5], [-1.2, -1.8], [-0.4, -0.9], [-1.7, -1.1]]
    )
    class_one = np.array(
        [[0.4, 1.2], [1.3, 0.5], [2.0, 1.8], [0.8, 2.1]]
    )
    train_X = np.vstack([class_zero, class_one])
    train_y = np.array([0] * len(class_zero) + [1] * len(class_one))
    test_X = np.array([[-1.0, -1.0], [1.0, 1.0]]) + test_offset
    params = StandardizationParameters(
        means=pd.Series([0.0, 0.0], index=["sensor_a", "sensor_b"]),
        stds=pd.Series([1.0, 1.0], index=["sensor_a", "sensor_b"]),
    )
    return TinyAnchoredData(
        train_dataset=Dataset(train_X, train_y),
        test_dataset=Dataset(test_X, np.array([0, 1])),
        channel_names=("sensor_a", "sensor_b"),
        class_labels=(0, 1),
        standardization=params,
    )


def _gaussian(data: TinyAnchoredData):
    return build_gaussian_anchored_model(
        data,
        name="Tiny anchored scenario",
        question="Does structure transfer?",
        source="tiny standardized training reservoir",
    )


def _gmm(data: TinyAnchoredData):
    return build_gmm_anchored_model(
        data,
        name="Tiny anchored scenario",
        question="Does structure transfer?",
        source="tiny standardized training reservoir",
        max_components=2,
        min_points_per_component=2,
        n_init=1,
        random_state=11,
    )


def test_generic_gaussian_matches_direct_training_calculations():
    data = _tiny_data()
    scenario = _gaussian(data)

    assert scenario.name == "Tiny anchored scenario"
    assert scenario.question == "Does structure transfer?"
    assert scenario.source == "tiny standardized training reservoir"
    assert scenario.channel_names == data.channel_names
    assert scenario.d == 2
    for label in data.class_labels:
        class_X = data.train_dataset.X[data.train_dataset.y == label]
        assert np.allclose(scenario.means[label], class_X.mean(axis=0))
        assert np.allclose(
            scenario.covariances[label],
            np.cov(class_X, rowvar=False, ddof=1)
            + scenario.ridge_by_class[label] * np.eye(2),
        )


def test_generic_builders_do_not_use_test_data():
    original = _tiny_data(test_offset=0.0)
    changed_test = _tiny_data(test_offset=10_000.0)
    gaussian_original = _gaussian(original)
    gaussian_changed = _gaussian(changed_test)
    gmm_original = _gmm(original)
    gmm_changed = _gmm(changed_test)

    for label in original.class_labels:
        assert np.array_equal(
            gaussian_original.means[label], gaussian_changed.means[label]
        )
        assert np.array_equal(
            gaussian_original.covariances[label],
            gaussian_changed.covariances[label],
        )
        assert np.array_equal(
            gmm_original.model.component_weights(label),
            gmm_changed.model.component_weights(label),
        )
        assert np.array_equal(
            gmm_original.model.component_means(label),
            gmm_changed.model.component_means(label),
        )
        assert np.array_equal(
            gmm_original.model.component_covariances(label),
            gmm_changed.model.component_covariances(label),
        )
        assert gmm_original.model_selection[label] == gmm_changed.model_selection[label]


def test_generic_gaussian_applies_and_records_ridge_for_singular_covariance():
    class_zero = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    class_one = class_zero + 4.0
    base = _tiny_data()
    data = TinyAnchoredData(
        train_dataset=Dataset(
            np.vstack([class_zero, class_one]), np.array([0, 0, 0, 1, 1, 1])
        ),
        test_dataset=base.test_dataset,
        channel_names=base.channel_names,
        class_labels=base.class_labels,
        standardization=base.standardization,
    )

    scenario = _gaussian(data)

    for label in data.class_labels:
        assert scenario.ridge_by_class[label] > 0.0
        np.linalg.cholesky(scenario.covariances[label])


def test_generic_gmm_model_selection_and_full_space_metadata_are_complete():
    data = _tiny_data()
    scenario = _gmm(data)

    assert scenario.d == 2
    assert scenario.model.class_labels == (0, 1)
    assert scenario.model.channel_names == data.channel_names
    assert scenario.channel_names == data.channel_names
    assert scenario.name == scenario.model.name
    for label in data.class_labels:
        selection = scenario.model_selection[label]
        assert selection["class_label"] == label
        assert selection["n_samples"] == 4
        assert selection["candidate_components"] == [1, 2]
        assert selection["selected_components"] == scenario.selected_components[label]
        assert selection["criterion"] == "bic"
        assert selection["covariance_type"] == "full"
        assert selection["reg_covar"] == 1e-6
        assert selection["n_init"] == 1
        assert set(selection["scores"]) == {"1", "2"}
        for scores in selection["scores"].values():
            assert set(scores) == {"bic", "aic"}
        assert scenario.model.component_means(label).shape[1] == 2
        assert scenario.model.component_covariances(label).shape[1:] == (2, 2)
