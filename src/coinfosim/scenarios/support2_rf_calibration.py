"""One-time Random Forest calibration and artifact validation for SUPPORT2."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from coinfosim.classifiers.registry import (
    CLASSIFIER_SEED_POLICY_VERSION,
    ClassifierExecutionPlan,
    classifier_seed_policy_metadata,
    make_classifier,
)
from coinfosim.datasets.support2 import (
    SUPPORT2_RAW_FILENAME,
    SUPPORT2_TARGET,
    SUPPORT2_TRAIN_FINGERPRINT,
    Support2Data,
)
from coinfosim.samplers.dataset import Dataset
from coinfosim.simulation.metrics import empirical_test_loss

SCHEMA_VERSION = 1
CANONICAL_ARTIFACT_PATH = Path("config/calibration/support2_random_forest.json")
CANONICAL_CALIBRATION_COMMAND = (
    ".venv/bin/python scripts/calibrate_support2_random_forest.py "
    "--raw-dir data/raw/support2 "
    "--output config/calibration/support2_random_forest.json "
    "--calibration-seed 0"
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "classifier_key",
    "estimator_class",
    "selected_parameters",
    "enforced_parameters",
    "search_space",
    "representative_sample_sizes",
    "validation_splitter",
    "selection_metric",
    "aggregation_rule",
    "selection_rule",
    "tie_breaking_rule",
    "candidate_results",
    "evaluation_results",
    "calibration_seed",
    "seed_policy",
    "dataset_slug",
    "target_name",
    "channel_names",
    "raw_file_sha256",
    "training_partition_fingerprint",
    "split_seed",
    "training_rows",
    "training_class_counts",
    "git_commit",
    "python_version",
    "numpy_version",
    "scikit_learn_version",
    "created_at",
    "calibration_command",
)


@dataclass(frozen=True)
class RandomForestCalibrationSpec:
    """Approved grid, representative sizes, and internal validation design."""

    n_estimators: tuple[int, ...] = (100, 200)
    max_depth: tuple[int | None, ...] = (12, None)
    min_samples_leaf: tuple[int, ...] = (1, 5)
    max_features: tuple[str | float, ...] = ("sqrt", 0.5)
    representative_sample_sizes: tuple[int, ...] = (32, 256, 1024)
    n_splits: int = 3
    test_size: float = 0.20

    def candidates(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            for n_estimators, max_depth, min_samples_leaf, max_features in itertools.product(
                self.n_estimators,
                self.max_depth,
                self.min_samples_leaf,
                self.max_features,
            )
        )

    def search_space(self) -> dict[str, list[object]]:
        return {
            "n_estimators": list(self.n_estimators),
            "max_depth": list(self.max_depth),
            "min_samples_leaf": list(self.min_samples_leaf),
            "max_features": list(self.max_features),
        }


# Short alias used by callers/tests that prefer the generic name.
CalibrationSpecification = RandomForestCalibrationSpec


@dataclass(frozen=True)
class LoadedRandomForestCalibration:
    path: str
    sha256: str
    artifact: Mapping[str, Any]
    compatibility_warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def selected_parameters(self) -> Mapping[str, object]:
        return self.artifact["selected_parameters"]


def support2_training_provenance(data: Support2Data) -> dict[str, object]:
    """Extract calibration provenance without reading the fixed test dataset."""

    labels, counts = np.unique(data.train_dataset.y, return_counts=True)
    return {
        "dataset_slug": "support2",
        "target_name": data.target_name,
        "channel_names": list(data.channel_names),
        "raw_file_sha256": data.file_hashes[SUPPORT2_RAW_FILENAME],
        "training_partition_fingerprint": SUPPORT2_TRAIN_FINGERPRINT,
        "split_seed": int(data.split_seed),
        "training_rows": int(data.train_dataset.n_samples),
        "training_class_counts": {
            str(label): int(count) for label, count in zip(labels, counts)
        },
    }


def _balanced_development_sample(
    dataset: Dataset,
    n_per_class: int,
    *,
    calibration_seed: int,
    split_index: int,
) -> Dataset:
    labels = np.unique(dataset.y)
    if labels.size < 2:
        raise ValueError("training sample must contain at least two classes")
    indices = []
    for label_index, label in enumerate(labels):
        candidates = np.flatnonzero(dataset.y == label)
        if candidates.size < n_per_class:
            raise ValueError(
                f"representative sample size {n_per_class} exceeds development "
                f"class {label!r} capacity {candidates.size}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(calibration_seed), 0x53555050, split_index, n_per_class, label_index]
            )
        )
        indices.extend(rng.choice(candidates, size=n_per_class, replace=False).tolist())
    selected = np.asarray(indices, dtype=int)
    return Dataset(dataset.X[selected], dataset.y[selected])


def _parameter_key(parameters: Mapping[str, object]) -> tuple[object, ...]:
    depth = parameters["max_depth"]
    features = parameters["max_features"]
    return (
        int(parameters["n_estimators"]),
        depth is None,
        float("inf") if depth is None else int(depth),
        -int(parameters["min_samples_leaf"]),
        0 if features == "sqrt" else 1,
    )


def select_one_standard_error_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, object], dict[str, float]]:
    """Select a candidate using the approved order-invariant 1-SE rule."""

    if not candidate_results:
        raise ValueError("candidate_results must not be empty")
    best = min(
        candidate_results,
        key=lambda row: (float(row["mean_loss"]), _parameter_key(row["parameters"])),
    )
    threshold = float(best["mean_loss"]) + float(best["standard_error"])
    eligible = [
        row for row in candidate_results if float(row["mean_loss"]) <= threshold
    ]
    selected = min(eligible, key=lambda row: _parameter_key(row["parameters"]))
    return dict(selected["parameters"]), {
        "best_mean_loss": float(best["mean_loss"]),
        "best_standard_error": float(best["standard_error"]),
        "eligibility_threshold": threshold,
        "eligible_candidate_count": len(eligible),
    }


def _candidate_id(parameters: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))
    return "rf_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _environment_metadata() -> dict[str, str]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    return {
        "git_commit": git_commit,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": package_version("scikit-learn"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def calibrate_support2_random_forest(
    training_dataset: Dataset,
    training_provenance: Mapping[str, object],
    calibration_spec: RandomForestCalibrationSpec | None = None,
    calibration_seed: int = 0,
) -> dict[str, Any]:
    """Calibrate exclusively from a prepared SUPPORT2 training reservoir."""

    spec = calibration_spec or RandomForestCalibrationSpec()
    missing_provenance = {
        "dataset_slug",
        "target_name",
        "channel_names",
        "raw_file_sha256",
        "training_partition_fingerprint",
        "split_seed",
        "training_rows",
        "training_class_counts",
    } - set(training_provenance)
    if missing_provenance:
        raise ValueError(
            f"training provenance is missing fields: {sorted(missing_provenance)}"
        )
    if np.unique(training_dataset.y).size < 2:
        raise ValueError("training sample must contain at least two classes")
    if int(training_provenance["training_rows"]) != training_dataset.n_samples:
        raise ValueError("training provenance row count differs from training Dataset")
    if len(training_provenance["channel_names"]) != training_dataset.d:
        raise ValueError("training provenance attributes differ from training Dataset")
    labels, counts = np.unique(training_dataset.y, return_counts=True)
    observed_counts = {
        str(label): int(count) for label, count in zip(labels, counts)
    }
    if dict(training_provenance["training_class_counts"]) != observed_counts:
        raise ValueError("training provenance class counts differ from training Dataset")

    splitter = StratifiedShuffleSplit(
        n_splits=spec.n_splits,
        test_size=spec.test_size,
        random_state=int(calibration_seed),
    )
    prepared = []
    for split_index, (development_indices, validation_indices) in enumerate(
        splitter.split(training_dataset.X, training_dataset.y)
    ):
        development = Dataset(
            training_dataset.X[development_indices],
            training_dataset.y[development_indices],
        )
        validation = Dataset(
            training_dataset.X[validation_indices],
            training_dataset.y[validation_indices],
        )
        for n_per_class in spec.representative_sample_sizes:
            sample = _balanced_development_sample(
                development,
                n_per_class,
                calibration_seed=int(calibration_seed),
                split_index=split_index,
            )
            prepared.append((split_index, int(n_per_class), sample, validation))

    candidate_results = []
    evaluation_results = []
    for parameters in spec.candidates():
        candidate_id = _candidate_id(parameters)
        evaluations = []
        for split_index, n_per_class, sample, validation in prepared:
            fit_seed = int(
                np.random.SeedSequence(
                    [int(calibration_seed), 0x43414C52, split_index, n_per_class]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            estimator = make_classifier(
                "random_forest", parameters=parameters, random_state=fit_seed
            )
            estimator.fit(sample.X, sample.y)
            loss = empirical_test_loss(estimator, validation)
            balanced_accuracy = balanced_accuracy_score(
                validation.y, estimator.predict(validation.X)
            )
            row = {
                "candidate_id": candidate_id,
                "split_index": split_index,
                "n_per_class": n_per_class,
                "development_sample_rows": int(sample.n_samples),
                "validation_rows": int(validation.n_samples),
                "misclassification_rate": float(loss),
                "balanced_accuracy": float(balanced_accuracy),
                "estimator_random_state": fit_seed,
            }
            evaluations.append(row)
            evaluation_results.append(row)
        losses = np.asarray(
            [row["misclassification_rate"] for row in evaluations], dtype=float
        )
        by_size = {
            str(n): float(
                np.mean(
                    [
                        row["misclassification_rate"]
                        for row in evaluations
                        if row["n_per_class"] == n
                    ]
                )
            )
            for n in spec.representative_sample_sizes
        }
        standard_deviation = float(np.std(losses, ddof=1)) if losses.size > 1 else 0.0
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "parameters": parameters,
                "mean_loss": float(np.mean(losses)),
                "standard_deviation": standard_deviation,
                "standard_error": standard_deviation / float(np.sqrt(losses.size)),
                "mean_loss_by_sample_size": by_size,
                "evaluation_count": int(losses.size),
            }
        )

    selected_parameters, selection_details = select_one_standard_error_candidate(
        candidate_results
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "classifier_key": "random_forest",
        "estimator_class": "sklearn.ensemble.RandomForestClassifier",
        "selected_parameters": selected_parameters,
        "enforced_parameters": {
            "criterion": "gini",
            "bootstrap": True,
            "n_jobs": 1,
        },
        "search_space": spec.search_space(),
        "representative_sample_sizes": list(spec.representative_sample_sizes),
        "validation_splitter": {
            "class": "sklearn.model_selection.StratifiedShuffleSplit",
            "n_splits": spec.n_splits,
            "test_size": spec.test_size,
            "random_state": int(calibration_seed),
        },
        "selection_metric": "misclassification_rate",
        "aggregation_rule": "unweighted_mean_over_sample_sizes_and_splits",
        "selection_rule": {
            "name": "one_standard_error",
            **selection_details,
        },
        "tie_breaking_rule": [
            "fewer_n_estimators",
            "finite_depth_before_none",
            "smaller_finite_max_depth",
            "larger_min_samples_leaf",
            "sqrt_before_0.5",
        ],
        "candidate_results": candidate_results,
        "evaluation_results": evaluation_results,
        "calibration_seed": int(calibration_seed),
        "seed_policy": {
            "calibration": "SeedSequence(calibration_seed, split_index, n_per_class)",
            "scenario_execution": classifier_seed_policy_metadata(),
        },
        **dict(training_provenance),
        **_environment_metadata(),
        "calibration_command": CANONICAL_CALIBRATION_COMMAND,
    }
    # Exercise strict serialization before returning to prevent delayed NaN
    # failures at write time.
    json.dumps(artifact, allow_nan=False)
    return artifact


# Generic alias retained for a compact public calibration API.
calibrate_random_forest = calibrate_support2_random_forest


def write_calibration_artifact(
    artifact: Mapping[str, Any],
    path: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Write strict JSON atomically; require ``force`` for replacement."""

    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"calibration artifact already exists: {path}; use --force to overwrite"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                artifact,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return path


def _version_tuple(value: str) -> tuple[int, int, int]:
    numeric = value.split("+", 1)[0].split(".")
    try:
        parts = [int(part.split("rc", 1)[0]) for part in numeric[:3]]
    except ValueError as exc:
        raise ValueError(f"invalid scikit-learn version in artifact: {value!r}") from exc
    return tuple((parts + [0, 0, 0])[:3])


def _validate_estimator_parameters(artifact: Mapping[str, Any]) -> None:
    selected = artifact["selected_parameters"]
    if not isinstance(selected, Mapping):
        raise ValueError("selected_parameters must be an object")
    required = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    if set(selected) != required:
        raise ValueError(
            f"selected_parameters must contain exactly {sorted(required)}"
        )
    if not isinstance(selected["n_estimators"], int) or selected["n_estimators"] <= 0:
        raise ValueError("invalid random_forest n_estimators")
    depth = selected["max_depth"]
    if depth is not None and (not isinstance(depth, int) or depth <= 0):
        raise ValueError("invalid random_forest max_depth")
    if not isinstance(selected["min_samples_leaf"], int) or selected["min_samples_leaf"] <= 0:
        raise ValueError("invalid random_forest min_samples_leaf")
    if selected["max_features"] not in ("sqrt", 0.5):
        raise ValueError("invalid random_forest max_features")
    enforced = artifact["enforced_parameters"]
    if not isinstance(enforced, Mapping):
        raise ValueError("enforced_parameters must be an object")
    if enforced.get("criterion") != "gini" or enforced.get("bootstrap") is not True:
        raise ValueError("invalid enforced random_forest parameters")
    if enforced.get("n_jobs") != 1:
        raise ValueError("random_forest n_jobs must be 1")
    make_classifier("random_forest", parameters=selected, random_state=0)


def load_and_validate_calibration_artifact(
    path: Path | str,
    data: Support2Data,
) -> LoadedRandomForestCalibration:
    """Load and strictly validate a frozen artifact against loaded SUPPORT2."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Random Forest calibration artifact not found: {path}")
    raw = path.read_bytes()
    try:
        artifact = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Random Forest calibration JSON: {path}") from exc
    if not isinstance(artifact, dict):
        raise ValueError("Random Forest calibration artifact must be a JSON object")
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"calibration artifact is missing fields: {missing}")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported calibration schema {artifact['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if artifact["classifier_key"] != "random_forest":
        raise ValueError("calibration artifact classifier_key must be 'random_forest'")
    expected = support2_training_provenance(data)
    comparisons = {
        "raw_file_sha256": "stale raw-file SHA-256",
        "training_partition_fingerprint": "stale training partition fingerprint",
        "target_name": "different target",
        "channel_names": "different attributes or attribute order",
        "split_seed": "different split seed",
        "dataset_slug": "different dataset",
        "training_rows": "different training row count",
        "training_class_counts": "different training class counts",
    }
    for field_name, message in comparisons.items():
        if artifact[field_name] != expected[field_name]:
            raise ValueError(
                f"{message}: artifact={artifact[field_name]!r}, "
                f"current={expected[field_name]!r}"
            )
    _validate_estimator_parameters(artifact)

    artifact_version = _version_tuple(str(artifact["scikit_learn_version"]))
    current_version_text = package_version("scikit-learn")
    current_version = _version_tuple(current_version_text)
    compatibility_warnings = []
    if artifact_version[0] != current_version[0]:
        raise ValueError(
            "calibration artifact scikit-learn major version differs: "
            f"artifact={artifact['scikit_learn_version']}, current={current_version_text}"
        )
    if artifact_version[1:] != current_version[1:]:
        message = (
            "calibration artifact scikit-learn minor/patch version differs: "
            f"artifact={artifact['scikit_learn_version']}, current={current_version_text}"
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        compatibility_warnings.append(message)

    return LoadedRandomForestCalibration(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        artifact=artifact,
        compatibility_warnings=tuple(compatibility_warnings),
    )


def classifier_execution_plan_from_calibration(
    loaded: LoadedRandomForestCalibration,
) -> ClassifierExecutionPlan:
    """Create the fixed SUPPORT2 plan and persistence-ready provenance."""

    parameters = dict(loaded.selected_parameters)
    parameters["n_jobs"] = 1
    artifact = loaded.artifact
    return ClassifierExecutionPlan(
        names=("linear_svm", "random_forest"),
        parameters={"random_forest": parameters},
        provenance={
            "classifier_selection": {
                "source": "scenario_spec",
                "ordered_keys": ["linear_svm", "random_forest"],
            },
            "classifier_configurations": {
                "linear_svm": {
                    "estimator": "sklearn.svm.SVC",
                    "parameters": {"kernel": "linear", "random_state": 0},
                    "seed_policy": {"kind": "fixed", "value": 0},
                },
                "random_forest": {
                    "estimator": "sklearn.ensemble.RandomForestClassifier",
                    "parameters": parameters,
                    "seed_policy": {
                        "kind": "per_replication",
                        "version": CLASSIFIER_SEED_POLICY_VERSION,
                    },
                    "calibration": {
                        "artifact_path": loaded.path,
                        "artifact_sha256": loaded.sha256,
                        "schema_version": artifact["schema_version"],
                        "training_partition_fingerprint": artifact[
                            "training_partition_fingerprint"
                        ],
                    },
                    "compatibility_warnings": list(loaded.compatibility_warnings),
                },
            },
        },
    )


# Concise loader alias for programmatic callers.
load_calibration_artifact = load_and_validate_calibration_artifact
