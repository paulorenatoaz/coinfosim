import pytest

from coinfosim.datasets.air_quality import AIR_QUALITY_CHANNELS, load_air_quality_data
from coinfosim.reports.air_quality_monte_carlo import (
    generate_air_quality_gmm_to_real_monte_carlo_report,
    generate_air_quality_real_monte_carlo_report,
    generate_air_quality_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios.air_quality import (
    build_gaussian_anchored_air_quality_model,
    build_gmm_anchored_air_quality_model,
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
    data = load_air_quality_data("data/raw/air_quality")
    config = _config()
    real_sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=config.base_seed,
        channel_names=data.channel_names,
        name="air_quality_real_data",
    )
    real_result = CooperativeMonteCarloSimulator(
        real_sampler.model,
        config,
        sampler=real_sampler,
        metadata={
            "experiment_arm": "real_to_real",
            "channel_names": list(data.channel_names),
        },
    ).run()

    gaussian = build_gaussian_anchored_air_quality_model(data)
    gaussian_sampler = SyntheticTrainRealTestSampler(
        GaussianClassConditionalSampler(
            gaussian.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        ),
        data.test_dataset,
    )
    gaussian_result = CooperativeMonteCarloSimulator(
        gaussian.model,
        config,
        sampler=gaussian_sampler,
        metadata={
            "experiment_arm": "single_gaussian_to_real",
            "channel_names": list(data.channel_names),
        },
    ).run()

    gmm = build_gmm_anchored_air_quality_model(
        data,
        max_components=1,
        n_init=1,
        random_state=0,
    )
    gmm_sampler = SyntheticTrainRealTestSampler(
        GMMClassConditionalSampler(
            gmm.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        ),
        data.test_dataset,
    )
    gmm_result = CooperativeMonteCarloSimulator(
        gmm.model,
        config,
        sampler=gmm_sampler,
        metadata={
            "experiment_arm": "gmm_to_real",
            "channel_names": list(data.channel_names),
        },
    ).run()
    return data, real_result, gaussian_result, gmm_result


def _assert_shared_air_quality_content(text: str, arm_label: str):
    lowered = text.lower()
    assert arm_label in text
    assert "fixed future real Air Quality evaluation set" in text
    for channel in AIR_QUALITY_CHANNELS:
        assert channel in text
    assert "31 subsets" in text
    assert "10. Loss curves" in text
    assert "Nested cardinality" in text
    assert "12. Subset ranking by sample size" in text
    assert "13. Structural dynamics" in text
    assert "Winner matrix" in text
    assert "Progressive N-star matrix" in text
    assert "14. N-star diagnostics" in text
    assert "9. Monte Carlo precision diagnostics" in text
    assert "Full-5 reference" in text
    assert "occupancy" not in lowered
    assert "synthetic test" not in lowered


def test_air_quality_real_report_is_structured_and_dataset_specific(
    arm_results, tmp_path
):
    data, real, _, _ = arm_results
    output = generate_air_quality_real_monte_carlo_report(
        real,
        data.channel_names,
        tmp_path,
        nstar_selection_result=real,
    )
    text = output.read_text(encoding="utf-8")

    assert output.name == "air_quality_real_monte_carlo_report.html"
    assert "Air Quality Real → Real Monte Carlo Report" in text
    _assert_shared_air_quality_content(text, "Real → Real")


def test_air_quality_gaussian_report_uses_real_selection_and_model_section(
    arm_results, tmp_path
):
    data, real, gaussian, _ = arm_results
    output = generate_air_quality_single_gaussian_to_real_monte_carlo_report(
        gaussian,
        data.channel_names,
        tmp_path,
        nstar_selection_result=real,
    )
    text = output.read_text(encoding="utf-8")

    assert "Air Quality Single Gaussian → Real Monte Carlo Report" in text
    assert "Estimated Gaussian training parameters" in text
    assert "<strong>Mean:</strong>" in text
    _assert_shared_air_quality_content(text, "Single Gaussian → Real")


def test_air_quality_gmm_report_uses_real_selection_and_model_section(
    arm_results, tmp_path
):
    data, real, _, gmm = arm_results
    output = generate_air_quality_gmm_to_real_monte_carlo_report(
        gmm,
        data.channel_names,
        tmp_path,
        nstar_selection_result=real,
    )
    text = output.read_text(encoding="utf-8")

    assert "Air Quality GMM → Real Monte Carlo Report" in text
    assert "Estimated GMM training model" in text
    assert "Model-selection configuration" in text
    assert "Selected components" in text
    _assert_shared_air_quality_content(text, "GMM → Real")
