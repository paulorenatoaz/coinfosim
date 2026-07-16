"""Stable classifier registry, execution plans, and scientific seed policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

REGISTERED_CLASSIFIER_KEYS = (
    "linear_svm",
    "logistic_regression",
    "gaussian_nb",
    "random_forest",
)

DEFAULT_CLASSIFIER_KEYS = (
    "linear_svm",
    "logistic_regression",
    "gaussian_nb",
)

# Backward-compatible public ordering used by report helpers.  It now denotes
# the complete registry; callers that need scenario defaults must use
# ``default_classifiers``.
CLASSIFIER_KEYS = list(REGISTERED_CLASSIFIER_KEYS)

DISPLAY_LABELS = {
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes",
    "random_forest": "Random Forest",
}

CLASSIFIER_SEED_POLICY_VERSION = "classifier_seed_v1"
_CLASSIFIER_SEED_NAMESPACES = {
    # Literal integers make the namespace stable across Python versions and
    # avoid Python's deliberately process-randomized hash().
    "random_forest": (0x434F494E, 0x52464F52, 0x45535431),
}


@dataclass(frozen=True)
class ClassifierExecutionPlan:
    """Resolved classifiers and their frozen parameters/provenance.

    Plain dictionaries are intentionally accepted as mappings: unlike
    ``MappingProxyType`` they remain pickle-safe for process workers.
    """

    names: tuple[str, ...]
    parameters: Mapping[str, Mapping[str, object]]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        resolved = resolve_classifier_names(self.names)
        unknown_parameters = set(self.parameters) - set(resolved)
        if unknown_parameters:
            raise ValueError(
                "classifier parameters supplied for unselected classifiers: "
                f"{sorted(unknown_parameters)}"
            )
        object.__setattr__(self, "names", resolved)
        object.__setattr__(
            self,
            "parameters",
            {
                name: dict(self.parameters.get(name, {}))
                for name in resolved
            },
        )
        object.__setattr__(self, "provenance", dict(self.provenance))


def available_classifiers() -> list[str]:
    """Return all registered classifier keys in stable order."""

    return list(REGISTERED_CLASSIFIER_KEYS)


def default_classifiers() -> list[str]:
    """Return the historical classifier trio used by generic simulations."""

    return list(DEFAULT_CLASSIFIER_KEYS)


def resolve_classifier_names(names: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and freeze an ordered classifier selection."""

    resolved = DEFAULT_CLASSIFIER_KEYS if names is None else tuple(names)
    if not resolved:
        raise ValueError("classifier selection must not be empty")
    unknown = [name for name in resolved if name not in REGISTERED_CLASSIFIER_KEYS]
    if unknown:
        raise ValueError(
            f"unknown classifier keys {unknown}; registered: "
            f"{list(REGISTERED_CLASSIFIER_KEYS)}"
        )
    duplicates = []
    seen = set()
    for name in resolved:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(f"duplicate classifier keys: {duplicates}")
    return tuple(resolved)


def default_execution_plan(
    names: Sequence[str] | None = None,
) -> ClassifierExecutionPlan:
    """Build a parameter-free plan for historical/default classifiers."""

    resolved = resolve_classifier_names(names)
    return ClassifierExecutionPlan(
        names=resolved,
        parameters={},
        provenance={
            "classifier_selection": {
                "source": "simulator_default" if names is None else "explicit",
                "ordered_keys": list(resolved),
            }
        },
    )


def derive_classifier_seed(
    base_seed: int,
    classifier_name: str,
    replication_id: int,
) -> int:
    """Derive the versioned per-replication estimator seed.

    The seed deliberately excludes sample size, subset, arm, backend, process,
    worker identity, and completion order.
    """

    if classifier_name not in _CLASSIFIER_SEED_NAMESPACES:
        raise ValueError(
            f"classifier {classifier_name!r} has no {CLASSIFIER_SEED_POLICY_VERSION} namespace"
        )
    if int(replication_id) < 0:
        raise ValueError("replication_id must be non-negative")
    namespace = _CLASSIFIER_SEED_NAMESPACES[classifier_name]
    sequence = np.random.SeedSequence(
        [int(base_seed), *namespace, int(replication_id)]
    )
    # sklearn accepts non-negative 32-bit integers as random_state values.
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def classifier_seed_policy_metadata() -> dict[str, object]:
    """Return the machine-readable scientific seed policy description."""

    return {
        "version": CLASSIFIER_SEED_POLICY_VERSION,
        "generator": "numpy.random.SeedSequence",
        "inputs": ["base_seed", "stable_classifier_namespace", "replication_id"],
        "excluded_inputs": [
            "feature_subset",
            "n_per_class",
            "scenario_arm",
            "worker",
            "pid",
            "execution_backend",
            "completion_order",
        ],
    }


def make_classifier(
    key: str,
    *,
    parameters: Mapping[str, object] | None = None,
    random_state: int | None = None,
):
    """Return a fresh, unfitted estimator for ``key``."""

    if key not in REGISTERED_CLASSIFIER_KEYS:
        raise KeyError(
            f"unknown classifier {key!r}; available: {available_classifiers()}"
        )
    supplied = dict(parameters or {})
    if key == "linear_svm":
        defaults = {"kernel": "linear", "random_state": 0}
        defaults.update(supplied)
        # Preserve the historical fixed setting even if generic callers pass a
        # replication seed.
        defaults["random_state"] = 0
        return SVC(**defaults)
    if key == "logistic_regression":
        defaults = {"max_iter": 1000, "random_state": 0}
        defaults.update(supplied)
        return LogisticRegression(**defaults)
    if key == "gaussian_nb":
        return GaussianNB(**supplied)

    required = {
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    }
    missing = required - set(supplied)
    if missing:
        raise ValueError(
            f"random_forest parameters are missing required keys: {sorted(missing)}"
        )
    unknown = set(supplied) - required - {"criterion", "bootstrap", "n_jobs"}
    if unknown:
        raise ValueError(f"unsupported random_forest parameters: {sorted(unknown)}")
    if supplied.get("criterion", "gini") != "gini":
        raise ValueError("random_forest criterion must be 'gini'")
    if supplied.get("bootstrap", True) is not True:
        raise ValueError("random_forest bootstrap must be true")
    if int(supplied.get("n_jobs", 1)) != 1:
        raise ValueError("random_forest n_jobs must be 1")
    if random_state is None:
        raise ValueError("random_forest requires a derived random_state")
    return RandomForestClassifier(
        n_estimators=supplied["n_estimators"],
        max_depth=supplied["max_depth"],
        min_samples_leaf=supplied["min_samples_leaf"],
        max_features=supplied["max_features"],
        criterion="gini",
        bootstrap=True,
        random_state=int(random_state),
        n_jobs=1,
    )


def classifier_label(key: str) -> str:
    """Return the stable display label for a classifier key."""

    if key not in DISPLAY_LABELS:
        raise KeyError(
            f"unknown classifier {key!r}; available: {available_classifiers()}"
        )
    return DISPLAY_LABELS[key]
