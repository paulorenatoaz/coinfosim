"""Presence/smoke tests for legacy parameter-model commands.

Full functional coverage of these commands (Model/Simulator execution)
already exists elsewhere; this file only verifies they were moved intact
into ``coinfosim.cli.legacy_commands`` and remain reachable from the root
CLI with a visible deprecation notice.
"""

from __future__ import annotations

from typer.testing import CliRunner

from coinfosim.cli.app import app

runner = CliRunner()


def test_run_simulation_help_available():
    result = runner.invoke(app, ["run-simulation", "--help"])
    assert result.exit_code == 0
    assert "--params" in result.output


def test_run_experiment_help_available():
    result = runner.invoke(app, ["run-experiment", "--help"])
    assert result.exit_code == 0
    assert "--scenarios" in result.output


def test_make_report_help_available():
    result = runner.invoke(app, ["make-report", "--help"])
    assert result.exit_code == 0
    assert "--scenario" in result.output


def test_cleanup_logs_help_available():
    result = runner.invoke(app, ["cleanup-logs", "--help"])
    assert result.exit_code == 0
    assert "--older-than" in result.output


def test_cleanup_logs_runs_with_no_logs_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("COINFOSIM_OUTPUT_DIR", str(tmp_path))
    result = runner.invoke(app, ["cleanup-logs", "--dry-run"])
    assert result.exit_code == 0


def test_run_simulation_shows_legacy_notice_on_invalid_params():
    result = runner.invoke(app, ["run-simulation", "--params", "not-json"])
    assert "legacy" in result.output.lower()


def test_make_report_requires_scenario_or_params():
    result = runner.invoke(app, ["make-report"])
    assert result.exit_code != 0
