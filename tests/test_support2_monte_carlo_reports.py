import pytest

from coinfosim.datasets.support2 import SUPPORT2_CHANNELS, load_support2_data
from coinfosim.reports.support2_monte_carlo import (
    generate_support2_gmm_to_real_monte_carlo_report,
    generate_support2_real_monte_carlo_report,
    generate_support2_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios.support2 import (
    build_gaussian_anchored_support2_model,
    build_gmm_anchored_support2_model,
)
from coinfosim.scenarios.support2_rf_calibration import (
    classifier_execution_plan_from_calibration,
    load_and_validate_calibration_artifact,
)
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def _config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=19,
    )


@pytest.fixture(scope="module")
def arm_results():
    data = load_support2_data("data/raw/support2")
    classifier_plan = classifier_execution_plan_from_calibration(
        load_and_validate_calibration_artifact(
            "config/calibration/support2_random_forest.json", data
        )
    )
    # Unit reports exercise Random Forest integration with a reduced tree count;
    # the canonical 100-tree artifact is validated separately.
    classifier_plan.parameters["random_forest"]["n_estimators"] = 2
    classifier_plan.provenance["classifier_configurations"]["random_forest"][
        "parameters"
    ]["n_estimators"] = 2
    config = _config()
    real_sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=config.base_seed,
        channel_names=data.channel_names,
        name="support2_real_data",
    )
    real = CooperativeMonteCarloSimulator(
        real_sampler.model,
        config,
        sampler=real_sampler,
        metadata={"experiment_arm": "real_to_real", "channel_names": list(data.channel_names)},
        classifier_plan=classifier_plan,
    ).run()
    gaussian_model = build_gaussian_anchored_support2_model(data)
    gaussian_sampler = SyntheticTrainRealTestSampler(
        GaussianClassConditionalSampler(
            gaussian_model.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        ),
        data.test_dataset,
    )
    gaussian = CooperativeMonteCarloSimulator(
        gaussian_model.model,
        config,
        sampler=gaussian_sampler,
        metadata={"experiment_arm": "single_gaussian_to_real", "channel_names": list(data.channel_names)},
        classifier_plan=classifier_plan,
    ).run()
    gmm_model = build_gmm_anchored_support2_model(data, max_components=1, n_init=1)
    gmm_sampler = SyntheticTrainRealTestSampler(
        GMMClassConditionalSampler(
            gmm_model.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        ),
        data.test_dataset,
    )
    gmm = CooperativeMonteCarloSimulator(
        gmm_model.model,
        config,
        sampler=gmm_sampler,
        metadata={"experiment_arm": "gmm_to_real", "channel_names": list(data.channel_names)},
        classifier_plan=classifier_plan,
    ).run()
    return data, real, gaussian, gmm


def _assert_common(text, arm):
    assert arm in text
    assert "death within 180 days after SUPPORT2 study entry" in text
    assert "same fixed real SUPPORT2 test set" in text
    assert "127 subsets" in text
    assert "Winner matrix" in text
    assert "Progressive N-star matrix" in text
    for channel in SUPPORT2_CHANNELS:
        assert channel in text
    assert "Linear SVM" in text
    assert "Random Forest" in text
    assert "Logistic Regression" not in text
    assert "Gaussian Naive Bayes" not in text
    assert "classifier_seed_v1" in text
    assert "support2_random_forest.json" in text
    assert "Internal n_jobs" in text
    assert ">1</td>" in text


def test_support2_real_report_identifies_real_training_and_fixed_test(arm_results, tmp_path):
    data, real, _, _ = arm_results
    output = generate_support2_real_monte_carlo_report(
        real, data.channel_names, tmp_path, nstar_selection_result=real
    )
    text = output.read_text(encoding="utf-8")
    _assert_common(text, "Real → Real")
    assert "according to death_180d" in text


def test_support2_synthetic_reports_identify_training_only_fits(arm_results, tmp_path):
    data, real, gaussian, gmm = arm_results
    gaussian_path = generate_support2_single_gaussian_to_real_monte_carlo_report(
        gaussian, data.channel_names, tmp_path, nstar_selection_result=real
    )
    gmm_path = generate_support2_gmm_to_real_monte_carlo_report(
        gmm, data.channel_names, tmp_path, nstar_selection_result=real
    )
    gaussian_text = gaussian_path.read_text(encoding="utf-8")
    gmm_text = gmm_path.read_text(encoding="utf-8")
    _assert_common(gaussian_text, "Single Gaussian → Real")
    _assert_common(gmm_text, "GMM → Real")
    assert "Estimated Gaussian training parameters" in gaussian_text
    assert "Estimated GMM training model" in gmm_text
    assert "real training partition only" in gaussian_text
    assert "real training partition only" in gmm_text
