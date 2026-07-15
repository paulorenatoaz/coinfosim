"""Targeted CLI and audit-metadata tests for Monte Carlo execution controls."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.reports.monte_carlo import _execution_configuration_html
from coinfosim.runs.report_data import simulation_summary_snapshot
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import (
    ExecutionConfig,
    build_execution_metadata,
)
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module():
    script = REPO_ROOT / "scripts" / "run_occupancy_scenario.py"
    module_name = "run_occupancy_scenario_execution_metadata_test"
    spec = importlib.util.spec_from_file_location(module_name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_result():
    model = GaussianSimulationModel(
        means={0: [-1.0, -0.5], 1: [1.0, 0.5]},
        covariances={
            0: [[1.0, 0.1], [0.1, 1.0]],
            1: [[1.0, 0.1], [0.1, 1.0]],
        },
    )
    config = MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2,),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=10,
        ci_half_width_target=0.05,
        base_seed=11,
    )
    sampler = GaussianClassConditionalSampler(
        model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    return CooperativeMonteCarloSimulator(
        model,
        config,
        sampler=sampler,
        execution_config=ExecutionConfig(),
    ).run()


def test_execution_metadata_records_capacity_and_cache_size():
    test_dataset = Dataset(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        [0, 1, 1],
    )
    metadata = build_execution_metadata(
        ExecutionConfig(backend="process", n_jobs=4),
        replication_batch_size=2,
        test_dataset=test_dataset,
        subsets=((0,), (1, 2)),
    )

    assert metadata == {
        "backend": "process",
        "requested_workers": 4,
        "effective_workers": 2,
        "worker_inner_threads": 1,
        "start_method": "forkserver",
        "logical_cpus": os.cpu_count(),
        "fixed_test_cache_bytes_per_worker": 72,
    }


def test_execution_metadata_is_in_result_summary_and_report(tiny_result):
    execution = tiny_result.metadata["execution"]
    assert execution["backend"] == "sequential"
    assert execution["requested_workers"] == 1
    assert execution["effective_workers"] == 1
    assert execution["fixed_test_cache_bytes_per_worker"] == 640

    summary = simulation_summary_snapshot(tiny_result)
    assert summary["execution"] == execution

    report_html = _execution_configuration_html(tiny_result)
    assert "Execution backend" in report_html
    assert "sequential" in report_html
    assert "640 bytes" in report_html


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], ExecutionConfig()),
        (
            [
                "--execution-backend",
                "process",
                "--n-jobs",
                "3",
                "--worker-inner-threads",
                "2",
                "--multiprocessing-start-method",
                "fork",
            ],
            ExecutionConfig(
                backend="process",
                n_jobs=3,
                worker_inner_threads=2,
                start_method="fork",
            ),
        ),
    ],
)
def test_cli_builds_execution_configuration(monkeypatch, arguments, expected):
    module = _load_script_module()
    captured = {}

    def fake_run_scenario(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_occupancy_scenario.py", "--quiet", *arguments],
    )

    assert module.main() == 0
    assert captured["execution_config"] == expected
