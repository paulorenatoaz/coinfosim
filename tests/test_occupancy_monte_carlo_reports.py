from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_monte_carlo import (
    generate_occupancy_gaussian_anchored_monte_carlo_report,
    generate_occupancy_real_monte_carlo_report,
    generate_occupancy_single_gaussian_to_real_monte_carlo_report,
)
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios.occupancy import build_gaussian_anchored_occupancy_model
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def _tiny_config(test_samples_per_class=20):
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4,),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=test_samples_per_class,
        ci_half_width_target=0.05,
        base_seed=17,
    )


def _real_result():
    data = load_occupancy_data("data/raw/occupancy")
    sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=17,
        channel_names=data.channel_names,
        name="occupancy",
    )
    sim = CooperativeMonteCarloSimulator(
        sampler.model,
        _tiny_config(),
        sampler=sampler,
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "real_data",
            "channel_names": list(data.channel_names),
        },
    )
    return data, sim.run()


def _gaussian_result():
    data = load_occupancy_data("data/raw/occupancy")
    scenario = build_gaussian_anchored_occupancy_model(data)
    config = _tiny_config(test_samples_per_class=20)
    sampler = GaussianClassConditionalSampler(
        scenario.model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    sim = CooperativeMonteCarloSimulator(
        scenario.model,
        config,
        sampler=sampler,
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "gaussian_anchored",
            "channel_names": list(data.channel_names),
        },
    )
    return data, sim.run()


def test_real_monte_carlo_report_generated(tmp_path):
    _, result = _real_result()
    out = generate_occupancy_real_monte_carlo_report(result, tmp_path)
    text = out.read_text(encoding="utf-8")

    assert out.name == "occupancy_real_monte_carlo_report.html"
    assert "Occupancy Real-Data Monte Carlo Report" in text
    assert "Real-data Monte Carlo" in text
    assert "Loss curves" in text
    assert "Final ranking at largest sample size" in text
    assert "Interpolated N-star" in text
    assert "Temperature+Humidity" in text
    assert "data:image/png;base64," in text
    assert "Bayes error" not in text
    assert "theoretical loss" not in text.lower()
    assert "train loss" not in text.lower()


def test_gaussian_anchored_monte_carlo_report_generated(tmp_path):
    data, result = _gaussian_result()
    out = generate_occupancy_gaussian_anchored_monte_carlo_report(
        result, data.channel_names, tmp_path
    )
    text = out.read_text(encoding="utf-8")

    assert out.name == "occupancy_gaussian_anchored_monte_carlo_report.html"
    assert "Occupancy Gaussian-Anchored Monte Carlo Report" in text
    assert "Gaussian-anchored Monte Carlo" in text
    assert "Estimated Gaussian parameters" in text
    assert "Loss curves" in text
    assert "Final ranking at largest sample size" in text
    assert "Interpolated N-star" in text
    assert "Temperature+Humidity" in text
    assert "data:image/png;base64," in text
    assert "Bayes error" not in text
    assert "theoretical loss" not in text.lower()
    assert "train loss" not in text.lower()


def _single_gaussian_to_real_result():
    """Train on single-Gaussian synthetic data, test on the real eval split."""
    data = load_occupancy_data("data/raw/occupancy")
    scenario = build_gaussian_anchored_occupancy_model(data)
    config = _tiny_config(test_samples_per_class=20)
    train_sampler = GaussianClassConditionalSampler(
        scenario.model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    sampler = SyntheticTrainRealTestSampler(train_sampler, data.test_dataset)
    sim = CooperativeMonteCarloSimulator(
        scenario.model,
        config,
        sampler=sampler,
        metadata={
            "scenario_name": "Occupancy Detection",
            "experiment_arm": "single_gaussian_to_real",
            "channel_names": list(data.channel_names),
        },
    )
    return data, sampler, sim.run()


def test_single_gaussian_to_real_uses_synthetic_train_and_real_test():
    data, sampler, result = _single_gaussian_to_real_result()
    # The fixed test set is exactly the real Occupancy evaluation split.
    test = sampler.sample_test()
    assert test.n_samples == data.test_dataset.n_samples
    assert (test.X == data.test_dataset.X).all()
    assert (test.y == data.test_dataset.y).all()
    # Training samples come from the synthetic Gaussian model (balanced draws).
    train = sampler.sample_train(n_per_class=4, replication_id=0)
    assert train.n_samples == 4 * len(data.class_labels)
    assert result.metadata["experiment_arm"] == "single_gaussian_to_real"


def test_single_gaussian_to_real_monte_carlo_report_generated(tmp_path):
    data, _, result = _single_gaussian_to_real_result()
    out = generate_occupancy_single_gaussian_to_real_monte_carlo_report(
        result, data.channel_names, tmp_path
    )
    text = out.read_text(encoding="utf-8")

    assert out.name == (
        "occupancy_single_gaussian_to_real_monte_carlo_report.html"
    )
    assert "Single Gaussian to Real Monte Carlo Report" in text
    assert "Single Gaussian → Real Monte Carlo" in text
    # Training source is the estimated Gaussian; evaluation is the real split.
    assert "Estimated Gaussian training parameters" in text
    assert "fixed real Occupancy evaluation split" in text
    assert "Loss curves" in text
    assert "Final ranking at largest sample size" in text
