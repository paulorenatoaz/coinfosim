"""Tests for cooperative Monte Carlo progress reporting (console-output layer).

These tests verify that:

- :class:`CooperativeMonteCarloSimulator` still runs without a progress reporter;
- passing a progress reporter does not change numerical results;
- the reporter emits the expected high-level messages;
- quiet mode suppresses progress output while still surfacing errors.

The simulator tests use a small Gaussian model so they do not depend on the
Occupancy raw data files.
"""

import numpy as np

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator
from coinfosim.simulation.progress import CooperativeProgressReporter


MU0 = [-0.7, -0.4]
MU1 = [0.7, 0.4]
SIGMA = [[1.0, 0.2], [0.2, 1.0]]


def _model():
    return GaussianSimulationModel(
        means={0: MU0, 1: MU1},
        covariances={0: SIGMA, 1: SIGMA},
    )


def _config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4, 8),
        min_replications=2,
        max_replications=4,
        replication_batch_size=2,
        test_samples_per_class=40,
        ci_half_width_target=0.05,
        base_seed=7,
    )


def _run(progress):
    config = _config()
    model = _model()
    sampler = GaussianClassConditionalSampler(
        model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    sim = CooperativeMonteCarloSimulator(
        model,
        config,
        sampler=sampler,
        metadata={"experiment_arm": "gaussian_anchored"},
        progress=progress,
    )
    return sim.run()


def _all_losses(result):
    values = {}
    for n in result.sample_sizes:
        for subset in result.subsets:
            for clf in result.classifier_names:
                values[(n, subset, clf)] = result.accumulator.losses(n, subset, clf)
    return values


class _RecordingReporter(CooperativeProgressReporter):
    """Reporter that records which lifecycle hooks were invoked."""

    def __init__(self):
        super().__init__(verbose=False)
        self.events = []

    def simulation_start(self, **kwargs):
        self.events.append(("simulation_start", kwargs))

    def sample_size_start(self, n_per_class, index, total):
        self.events.append(("sample_size_start", n_per_class))

    def batch_finish(self, n_per_class, replications, max_ci_half_width):
        self.events.append(("batch_finish", n_per_class, replications))

    def sample_size_finish(self, n_per_class, replications, reason, max_ci_half_width, elapsed=None):
        self.events.append(("sample_size_finish", n_per_class, reason))

    def simulation_finish(self, runtime):
        self.events.append(("simulation_finish",))


def test_simulator_runs_without_reporter():
    result = _run(None)
    assert result.sample_sizes == [4, 8]
    assert len(result.subsets) == 3  # 2 channels -> 3 non-empty subsets


def test_reporter_does_not_change_results():
    baseline = _all_losses(_run(None))
    with_reporter = _all_losses(_run(CooperativeProgressReporter(verbose=False)))

    assert baseline.keys() == with_reporter.keys()
    for key in baseline:
        assert np.array_equal(baseline[key], with_reporter[key])


def test_reporter_receives_lifecycle_events():
    reporter = _RecordingReporter()
    _run(reporter)

    kinds = [event[0] for event in reporter.events]
    assert kinds[0] == "simulation_start"
    assert kinds[-1] == "simulation_finish"
    assert "sample_size_start" in kinds
    assert "batch_finish" in kinds
    assert "sample_size_finish" in kinds

    finishes = [e for e in reporter.events if e[0] == "sample_size_finish"]
    assert {e[1] for e in finishes} == {4, 8}
    for _, _, reason in finishes:
        assert reason in {"converged", "max_budget"}


def test_verbose_reporter_emits_output(capsys):
    reporter = CooperativeProgressReporter(verbose=True, no_color=True)
    reporter.scenario_step_start("Loading Occupancy dataset")
    reporter.simulation_start(
        arm="real_data",
        n_sample_sizes=2,
        n_subsets=3,
        n_classifiers=3,
        n_cells=9,
        fixed_test_size=100,
        sample_sizes=[4, 8],
    )
    reporter.sample_size_finish(4, 10, "converged", 0.01, elapsed=0.5)
    out = capsys.readouterr().out
    assert "Loading Occupancy dataset" in out
    assert "real_data" in out
    assert "empirical test loss" in out
    assert "converged" in out


def test_quiet_reporter_suppresses_output(capsys):
    reporter = CooperativeProgressReporter(verbose=False, no_color=True)
    reporter.scenario_step_start("Loading Occupancy dataset")
    reporter.simulation_start(
        arm="real_data",
        n_sample_sizes=2,
        n_subsets=3,
        n_classifiers=3,
        n_cells=9,
        fixed_test_size=100,
        sample_sizes=[4, 8],
    )
    reporter.sample_size_finish(4, 10, "converged", 0.01)
    assert capsys.readouterr().out == ""


def test_quiet_reporter_still_reports_errors(capsys):
    reporter = CooperativeProgressReporter(verbose=False, no_color=True)
    reporter.error("something failed", RuntimeError("boom"))
    out = capsys.readouterr().out
    assert "something failed" in out
    assert "boom" in out
