import inspect
import json
from importlib.metadata import version as package_version

import numpy as np
import pytest

from coinfosim.datasets.support2 import load_support2_data
from coinfosim.samplers.dataset import Dataset
from coinfosim.scenarios.support2_rf_calibration import (
    REQUIRED_ARTIFACT_FIELDS,
    RandomForestCalibrationSpec,
    calibrate_support2_random_forest,
    load_and_validate_calibration_artifact,
    select_one_standard_error_candidate,
    support2_training_provenance,
    write_calibration_artifact,
)


def _tiny_training():
    rng = np.random.default_rng(4)
    return Dataset(
        np.vstack((rng.normal(-1, 1, (30, 2)), rng.normal(1, 1, (30, 2)))),
        np.asarray([0] * 30 + [1] * 30),
    )


def _tiny_provenance(dataset):
    return {
        "dataset_slug": "support2",
        "target_name": "death_180d",
        "channel_names": ["a", "b"],
        "raw_file_sha256": "raw",
        "training_partition_fingerprint": "train",
        "split_seed": 0,
        "training_rows": dataset.n_samples,
        "training_class_counts": {"0": 30, "1": 30},
    }


def _reduced_spec():
    return RandomForestCalibrationSpec(
        n_estimators=(5, 8),
        max_depth=(3,),
        min_samples_leaf=(1,),
        max_features=("sqrt",),
        representative_sample_sizes=(4,),
        n_splits=2,
        test_size=0.2,
    )


def test_calibration_engine_is_structurally_isolated_from_fixed_test_set():
    parameters = inspect.signature(calibrate_support2_random_forest).parameters
    assert tuple(parameters) == (
        "training_dataset",
        "training_provenance",
        "calibration_spec",
        "calibration_seed",
    )
    assert "test" not in " ".join(parameters)


def test_reduced_calibration_is_reproducible_and_records_every_evaluation():
    dataset = _tiny_training()
    first = calibrate_support2_random_forest(
        dataset, _tiny_provenance(dataset), _reduced_spec(), 7
    )
    second = calibrate_support2_random_forest(
        dataset, _tiny_provenance(dataset), _reduced_spec(), 7
    )
    assert first["selected_parameters"] == second["selected_parameters"]
    assert first["candidate_results"] == second["candidate_results"]
    assert len(first["candidate_results"]) == 2
    assert len(first["evaluation_results"]) == 4
    json.dumps(first, allow_nan=False)


def test_one_standard_error_rule_tie_breaking_and_candidate_order_invariance():
    candidates = [
        {
            "parameters": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.5},
            "mean_loss": 0.10,
            "standard_error": 0.02,
        },
        {
            "parameters": {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
            "mean_loss": 0.119,
            "standard_error": 0.01,
        },
    ]
    selected, details = select_one_standard_error_candidate(candidates)
    reversed_selected, _ = select_one_standard_error_candidate(list(reversed(candidates)))
    assert selected == reversed_selected == candidates[1]["parameters"]
    assert details["eligibility_threshold"] == pytest.approx(0.12)


@pytest.fixture(scope="module")
def support2_data():
    return load_support2_data("data/raw/support2")


def _valid_artifact(data):
    artifact = {field: "value" for field in REQUIRED_ARTIFACT_FIELDS}
    artifact.update(
        {
            "schema_version": 1,
            "classifier_key": "random_forest",
            "estimator_class": "sklearn.ensemble.RandomForestClassifier",
            "selected_parameters": {
                "n_estimators": 100,
                "max_depth": 12,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
            },
            "enforced_parameters": {"criterion": "gini", "bootstrap": True, "n_jobs": 1},
            "candidate_results": [],
            "evaluation_results": [],
            "scikit_learn_version": package_version("scikit-learn"),
            **support2_training_provenance(data),
        }
    )
    return artifact


def test_strict_atomic_artifact_write_and_missing_artifact(tmp_path, support2_data):
    path = tmp_path / "calibration.json"
    artifact = _valid_artifact(support2_data)
    write_calibration_artifact(artifact, path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        write_calibration_artifact(artifact, path)
    write_calibration_artifact(artifact, path, force=True)
    with pytest.raises(FileNotFoundError):
        load_and_validate_calibration_artifact(tmp_path / "missing.json", support2_data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda artifact: artifact.update(raw_file_sha256="stale"), "stale raw-file"),
        (lambda artifact: artifact.update(training_partition_fingerprint="stale"), "stale training"),
        (lambda artifact: artifact["enforced_parameters"].update(n_jobs=2), "n_jobs must be 1"),
        (lambda artifact: artifact.update(scikit_learn_version="999.0.0"), "major version"),
    ],
)
def test_artifact_validator_rejects_stale_or_incompatible_artifacts(
    tmp_path, support2_data, mutation, message
):
    artifact = _valid_artifact(support2_data)
    mutation(artifact)
    path = tmp_path / "invalid.json"
    write_calibration_artifact(artifact, path)
    with pytest.raises(ValueError, match=message):
        load_and_validate_calibration_artifact(path, support2_data)
