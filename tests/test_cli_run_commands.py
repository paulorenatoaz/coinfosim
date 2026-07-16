"""Tests for ``coinfosim runs ...`` commands."""

from __future__ import annotations

from typer.testing import CliRunner

from coinfosim.cli.app import app

runner = CliRunner()


def test_runs_scenarios_empty_registry(tmp_path):
    result = runner.invoke(app, ["runs", "scenarios", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_runs_simulations_empty_registry(tmp_path):
    result = runner.invoke(app, ["runs", "simulations", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0


def test_runs_scenarios_lists_a_recorded_run(tmp_path):
    from coinfosim.runs.registry import ScenarioRunRegistry

    registry = ScenarioRunRegistry(base_output_dir=tmp_path)
    registry.start_run(
        scenario_slug="fixture_scenario",
        scenario_name="Fixture Scenario",
        scenario_family="dataset",
        question="q?",
        mode="smoke",
    )
    result = runner.invoke(app, ["runs", "scenarios", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    # Rich may truncate the Slug column at the test runner's default width.
    assert "fixture_sc" in result.output


def test_runs_scenario_detail_unknown_id_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["runs", "scenario", "999", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_runs_scenario_detail_shows_record(tmp_path):
    from coinfosim.runs.registry import ScenarioRunRegistry

    registry = ScenarioRunRegistry(base_output_dir=tmp_path)
    registry.start_run(
        scenario_slug="fixture_scenario",
        scenario_name="Fixture Scenario",
        scenario_family="dataset",
        question="q?",
        mode="smoke",
    )
    result = runner.invoke(app, ["runs", "scenario", "0", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Fixture Scenario" in result.output


def test_runs_simulation_detail_unknown_id_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["runs", "simulation", "999", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0
