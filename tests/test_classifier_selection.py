import pickle

import pytest

from coinfosim.classifiers.registry import (
    CLASSIFIER_SEED_POLICY_VERSION,
    DEFAULT_CLASSIFIER_KEYS,
    REGISTERED_CLASSIFIER_KEYS,
    ClassifierExecutionPlan,
    available_classifiers,
    default_classifiers,
    derive_classifier_seed,
    make_classifier,
    resolve_classifier_names,
)


RF_PARAMETERS = {
    "n_estimators": 10,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}


def test_registered_classifiers_and_historical_defaults_are_separate():
    assert tuple(available_classifiers()) == REGISTERED_CLASSIFIER_KEYS
    assert tuple(default_classifiers()) == DEFAULT_CLASSIFIER_KEYS
    assert "random_forest" in available_classifiers()
    assert "random_forest" not in default_classifiers()


def test_classifier_selection_preserves_order_and_defaults():
    assert resolve_classifier_names(None) == DEFAULT_CLASSIFIER_KEYS
    assert resolve_classifier_names(("random_forest", "linear_svm")) == (
        "random_forest",
        "linear_svm",
    )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ((), "empty"),
        (("unknown",), "unknown"),
        (("linear_svm", "linear_svm"), "duplicate"),
    ],
)
def test_classifier_selection_rejects_invalid_names(names, message):
    with pytest.raises(ValueError, match=message):
        resolve_classifier_names(names)


def test_random_forest_factory_is_fresh_and_enforces_one_internal_job():
    first = make_classifier(
        "random_forest", parameters=RF_PARAMETERS, random_state=7
    )
    second = make_classifier(
        "random_forest", parameters=RF_PARAMETERS, random_state=7
    )
    assert first is not second
    assert first.n_jobs == second.n_jobs == 1
    assert first.random_state == second.random_state == 7
    with pytest.raises(ValueError, match="n_jobs must be 1"):
        make_classifier(
            "random_forest",
            parameters={**RF_PARAMETERS, "n_jobs": 2},
            random_state=7,
        )


def test_classifier_seed_v1_is_deterministic_and_replication_specific():
    assert CLASSIFIER_SEED_POLICY_VERSION == "classifier_seed_v1"
    first = derive_classifier_seed(19, "random_forest", 4)
    assert first == derive_classifier_seed(19, "random_forest", 4)
    assert first != derive_classifier_seed(19, "random_forest", 5)


def test_execution_plan_is_pickle_safe():
    plan = ClassifierExecutionPlan(
        names=("linear_svm", "random_forest"),
        parameters={"random_forest": RF_PARAMETERS},
        provenance={"source": "test"},
    )
    restored = pickle.loads(pickle.dumps(plan))
    assert restored == plan
