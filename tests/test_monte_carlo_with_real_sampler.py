import numpy as np

from coinfosim.classifiers.registry import default_classifiers
from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4,),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=50,
        ci_half_width_target=0.05,
        base_seed=11,
    )


def test_monte_carlo_runs_with_real_occupancy_sampler():
    data = load_occupancy_data("data/raw/occupancy")
    sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=11,
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
            "standardization": "train_pool_only",
        },
    )

    result = sim.run()

    assert result.sample_sizes == [4]
    assert len(result.subsets) == 31
    assert result.classifier_names == default_classifiers()
    assert result.metadata["metric"] == "empirical_test_loss"
    assert result.metadata["experiment_arm"] == "real_data"
    assert result.metadata["scenario_name"] == "Occupancy Detection"
    assert result.metadata["standardization"] == "train_pool_only"
    assert result.metadata["fixed_test_size"] == 12417
    assert result.metadata["channel_names"] == list(data.channel_names)
    assert result.metadata["d"] == 5
    assert result.metadata["class_labels"] == [0, 1]

    for subset in result.subsets:
        for clf in result.classifier_names:
            losses = result.accumulator.losses(4, subset, clf)
            assert losses.size == 2
            assert np.all((losses >= 0.0) & (losses <= 1.0))

    keys = " ".join(result.metadata.keys()).lower()
    assert "theoretical" not in keys
    assert "bayes" not in keys
