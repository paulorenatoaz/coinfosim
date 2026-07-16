"""Tests for coinfosim.scenarios.service (the CLI orchestration layer).

Wiring is tested with the underlying Monte Carlo runner mocked out (the
runner itself is already extensively covered by the existing scenario
runner test suites). One tiny, real end-to-end smoke run exercises the
full path against the tracked repository data.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from coinfosim.scenarios import service
from coinfosim.scenarios.catalog import UnknownScenarioError
from coinfosim.simulation.config import MonteCarloConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_registered_scenario_unknown_scenario_raises(tmp_path):
    with pytest.raises(UnknownScenarioError):
        service.run_registered_scenario(
            "does-not-exist", output_dir=tmp_path, allow_download=False
        )


def test_run_registered_scenario_resolves_dataset_and_delegates(tmp_path):
    occupancy_dir = REPO_ROOT / "data" / "raw" / "occupancy"
    with mock.patch.object(
        service, "run_dataset_anchored_scenario"
    ) as mocked_run:
        mocked_run.return_value = {"status": "completed"}
        result = service.run_registered_scenario(
            "occupancy-detection",
            mode="smoke",
            data_dir=occupancy_dir,
            output_dir=tmp_path,
            allow_download=False,
        )
    assert result == {"status": "completed"}
    assert mocked_run.call_count == 1
    _, kwargs = mocked_run.call_args
    assert kwargs["raw_dir"] == str(occupancy_dir)
    assert kwargs["mode"] == "smoke"
    assert kwargs["output_dir"] == str(tmp_path)
    assert kwargs["classifier_configuration"] is None


def test_run_registered_scenario_passes_support2_classifier_configuration(tmp_path):
    support2_dir = REPO_ROOT / "data" / "raw" / "support2"
    with mock.patch.object(
        service, "run_dataset_anchored_scenario"
    ) as mocked_run:
        mocked_run.return_value = {"status": "completed"}
        service.run_registered_scenario(
            "support2",
            data_dir=support2_dir,
            output_dir=tmp_path,
            allow_download=False,
        )
    _, kwargs = mocked_run.call_args
    assert kwargs["classifier_configuration"] is not None
    assert kwargs["classifier_configuration"]["classifier_names"] == (
        "linear_svm",
        "random_forest",
    )
    assert isinstance(kwargs["classifier_configuration"], dict)


def test_run_registered_scenario_no_download_missing_dataset_raises(tmp_path):
    from coinfosim.datasets.resolver import DatasetResolutionError

    empty_dir = tmp_path / "definitely-empty"
    empty_dir.mkdir()
    with mock.patch.object(service, "run_dataset_anchored_scenario") as mocked_run:
        with pytest.raises(DatasetResolutionError):
            service.run_registered_scenario(
                "occupancy",
                data_dir=empty_dir,
                output_dir=tmp_path,
                allow_download=False,
            )
    mocked_run.assert_not_called()


def test_regenerate_registered_scenario_never_calls_run_dataset_anchored_scenario(tmp_path):
    with mock.patch.object(
        service, "regenerate_dataset_anchored_scenario"
    ) as mocked_regenerate, mock.patch.object(
        service, "run_dataset_anchored_scenario"
    ) as mocked_run:
        mocked_regenerate.return_value = {"status": "completed"}
        result = service.regenerate_registered_scenario(
            "air_quality", scenario_run_id=3, output_dir=tmp_path
        )
    assert result == {"status": "completed"}
    mocked_run.assert_not_called()
    mocked_regenerate.assert_called_once()
    args, kwargs = mocked_regenerate.call_args
    assert args[1] == 3
    assert kwargs["output_dir"] == str(tmp_path)


@pytest.mark.slow
def test_run_registered_scenario_real_tiny_occupancy_end_to_end(tmp_path):
    """One real, extremely tiny end-to-end run through the new service layer."""

    tiny_config = MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4),
        min_replications=2,
        max_replications=4,
        replication_batch_size=2,
        test_samples_per_class=10,
        ci_half_width_target=0.5,
        base_seed=0,
    )
    result = service.run_registered_scenario(
        "occupancy",
        data_dir=REPO_ROOT / "data" / "raw" / "occupancy",
        output_dir=tmp_path,
        allow_download=False,
        visualize=False,
        config=tiny_config,
    )
    assert result["scenario_run_id"] == 0
    assert Path(result["scenario_registry"]).exists()
    assert Path(result["simulation_registry"]).exists()
    assert (tmp_path / "scenario_runs.json").exists()
    assert (tmp_path / "simulation_runs.json").exists()
