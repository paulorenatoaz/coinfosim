"""Exact scientific-product equivalence for sequential and process execution."""

import json

import numpy as np
import pandas as pd
import pytest

from coinfosim.classifiers.registry import available_classifiers
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.models.gmm import GMMSimulationModel
from coinfosim.reports.report_tables import (
    compact_precision_diagnostics,
    full_loss_table,
)
from coinfosim.results.persistence import (
    load_simulation_result,
    save_simulation_result,
)
from coinfosim.runs.report_data import (
    scenario_report_data,
    simulation_report_data,
)
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import ExecutionConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


SUBSETS = ((0,), (1,), (0, 1))
CHANNEL_NAMES = ("A", "B")


def _config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=4,
        ci_half_width_target=0.05,
        base_seed=23,
    )


def _real_datasets():
    class_zero = np.column_stack(
        (
            np.linspace(-3.5, -2.0, 10),
            np.linspace(-2.0, -3.5, 10),
        )
    )
    class_one = -class_zero
    train = Dataset(
        np.vstack((class_zero, class_one)),
        np.array([0] * len(class_zero) + [1] * len(class_one)),
    )
    test = Dataset(
        [
            [-3.2, -2.2],
            [-2.8, -2.8],
            [-2.2, -3.2],
            [3.2, 2.2],
            [2.8, 2.8],
            [2.2, 3.2],
        ],
        [0, 0, 0, 1, 1, 1],
    )
    return train, test


def _gaussian_model():
    return GaussianSimulationModel(
        means={0: [-3.0, -2.5], 1: [3.0, 2.5]},
        covariances={
            0: [[0.2, 0.02], [0.02, 0.2]],
            1: [[0.2, 0.02], [0.02, 0.2]],
        },
    )


def _gmm_model():
    return GMMSimulationModel(
        weights={0: [0.6, 0.4], 1: [0.4, 0.6]},
        means={
            0: [[-3.2, -2.7], [-2.6, -2.2]],
            1: [[2.6, 2.2], [3.2, 2.7]],
        },
        covariances={
            0: [
                [[0.15, 0.01], [0.01, 0.15]],
                [[0.12, 0.01], [0.01, 0.12]],
            ],
            1: [
                [[0.12, 0.01], [0.01, 0.12]],
                [[0.15, 0.01], [0.01, 0.15]],
            ],
        },
        channel_names=CHANNEL_NAMES,
        name="tiny_gmm",
    )


def _samplers():
    config = _config()
    train, test = _real_datasets()
    gaussian_train = GaussianClassConditionalSampler(
        _gaussian_model(),
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    gmm_train = GMMClassConditionalSampler(
        _gmm_model(),
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    return {
        "real_to_real": RealDatasetSampler(
            train,
            test,
            base_seed=config.base_seed,
            channel_names=CHANNEL_NAMES,
            name="tiny_real",
        ),
        "single_gaussian_to_real": SyntheticTrainRealTestSampler(
            gaussian_train,
            test,
            name="tiny_gaussian_to_real",
        ),
        "gmm_to_real": SyntheticTrainRealTestSampler(
            gmm_train,
            test,
            name="tiny_gmm_to_real",
        ),
    }


def _run(sampler, execution_config, arm):
    return CooperativeMonteCarloSimulator(
        sampler.model,
        _config(),
        subsets=SUBSETS,
        classifier_names=available_classifiers(),
        sampler=sampler,
        metadata={
            "experiment_arm": arm,
            "channel_names": list(CHANNEL_NAMES),
        },
        execution_config=execution_config,
    ).run()


@pytest.fixture(scope="module")
def equivalent_results():
    sequential = {}
    process = {}
    process_config = ExecutionConfig(
        backend="process",
        n_jobs=2,
        start_method="forkserver",
        worker_inner_threads=1,
    )
    for arm, sampler in _samplers().items():
        sequential[arm] = _run(sampler, ExecutionConfig(), arm)
        process[arm] = _run(sampler, process_config, arm)
    return sequential, process


def _without_execution_and_runtime(value):
    if isinstance(value, dict):
        return {
            key: _without_execution_and_runtime(item)
            for key, item in value.items()
            if key not in {"execution", "runtime_seconds"}
        }
    if isinstance(value, list):
        return [_without_execution_and_runtime(item) for item in value]
    return value


def _assert_simulation_products_equal(sequential, process, arm):
    assert sequential.sample_sizes == process.sample_sizes
    assert sequential.subsets == process.subsets
    assert sequential.classifier_names == process.classifier_names
    assert sequential.config == process.config
    assert sequential.stopping_info == process.stopping_info
    assert _without_execution_and_runtime(
        sequential.metadata
    ) == _without_execution_and_runtime(process.metadata)

    assert sequential.metadata["execution"]["backend"] == "sequential"
    assert process.metadata["execution"]["backend"] == "process"
    for n_per_class in sequential.sample_sizes:
        for subset in sequential.subsets:
            for classifier_name in sequential.classifier_names:
                assert np.array_equal(
                    sequential.accumulator.losses(
                        n_per_class, subset, classifier_name
                    ),
                    process.accumulator.losses(
                        n_per_class, subset, classifier_name
                    ),
                )

    sequential_report_data = simulation_report_data(sequential)
    process_report_data = simulation_report_data(process)
    assert sequential_report_data == process_report_data
    for key in (
        "summary_table",
        "best_subset_rankings",
        "threshold_comparisons",
        "structural_dynamics",
    ):
        assert sequential_report_data[key] == process_report_data[key]

    pd.testing.assert_frame_equal(
        full_loss_table(
            sequential,
            channel_names=CHANNEL_NAMES,
            arm=arm,
        ),
        full_loss_table(
            process,
            channel_names=CHANNEL_NAMES,
            arm=arm,
        ),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        compact_precision_diagnostics(sequential),
        compact_precision_diagnostics(process),
        check_exact=True,
    )


def test_all_arm_simulation_products_are_exactly_equal(equivalent_results):
    sequential, process = equivalent_results
    assert set(sequential) == {
        "real_to_real",
        "single_gaussian_to_real",
        "gmm_to_real",
    }
    for arm in sequential:
        _assert_simulation_products_equal(sequential[arm], process[arm], arm)


def test_scenario_report_data_is_exact_and_json_safe(equivalent_results):
    sequential, process = equivalent_results
    sequential_data = scenario_report_data(
        sequential["real_to_real"],
        sequential["single_gaussian_to_real"],
        sequential["gmm_to_real"],
        CHANNEL_NAMES,
    )
    process_data = scenario_report_data(
        process["real_to_real"],
        process["single_gaussian_to_real"],
        process["gmm_to_real"],
        CHANNEL_NAMES,
    )
    sequential_science = _without_execution_and_runtime(sequential_data)
    process_science = _without_execution_and_runtime(process_data)

    assert sequential_science == process_science
    assert {
        arm: (data["train_source"], data["test_source"])
        for arm, data in sequential_science["arms"].items()
    } == {
        "real_to_real": (
            "real_occupancy_training_pool",
            "real_occupancy_evaluation_split",
        ),
        "single_gaussian_to_real": (
            "single_gaussian_synthetic",
            "real_occupancy_evaluation_split",
        ),
        "gmm_to_real": (
            "gmm_synthetic",
            "real_occupancy_evaluation_split",
        ),
    }
    structural = sequential_science["structural_fidelity"]
    assert structural["ranking_fidelity_series"]
    assert structural["winner_agreement_series"]
    assert structural["nstar_similarity_series"]
    assert structural["reference_display_subsets_by_classifier"]
    encoded = json.dumps(sequential_science, allow_nan=False, sort_keys=True)
    assert encoded == json.dumps(process_science, allow_nan=False, sort_keys=True)


def test_persisted_results_regenerate_identical_report_data(
    equivalent_results,
    tmp_path,
):
    _, process = equivalent_results
    restored = {
        arm: load_simulation_result(
            save_simulation_result(result, tmp_path / f"{arm}.json.gz")
        )
        for arm, result in process.items()
    }

    for arm, result in process.items():
        assert simulation_report_data(restored[arm]) == simulation_report_data(result)
        pd.testing.assert_frame_equal(
            full_loss_table(restored[arm], CHANNEL_NAMES, arm),
            full_loss_table(result, CHANNEL_NAMES, arm),
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            compact_precision_diagnostics(restored[arm]),
            compact_precision_diagnostics(result),
            check_exact=True,
        )

    restored_scenario = scenario_report_data(
        restored["real_to_real"],
        restored["single_gaussian_to_real"],
        restored["gmm_to_real"],
        CHANNEL_NAMES,
    )
    original_scenario = scenario_report_data(
        process["real_to_real"],
        process["single_gaussian_to_real"],
        process["gmm_to_real"],
        CHANNEL_NAMES,
    )
    assert restored_scenario == original_scenario
