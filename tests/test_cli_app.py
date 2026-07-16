"""Tests for the root CoInfoSim Typer application."""

from __future__ import annotations

from typer.testing import CliRunner

from coinfosim.cli.app import app, main

runner = CliRunner()


def test_root_help_exits_zero_and_lists_command_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("scenario", "dataset", "runs", "config", "publish", "doctor"):
        assert name in result.output


def test_root_help_lists_legacy_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("run-simulation", "run-experiment", "make-report", "cleanup-logs"):
        assert name in result.output


def test_version_flag_prints_version_and_exits_zero():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "coinfosim" in result.output.lower()


def test_unknown_command_exits_with_usage_code():
    result = runner.invoke(app, ["not-a-real-command"])
    assert result.exit_code == 2


def test_main_callable_returns_int_for_version():
    code = main(["--version"])
    assert code == 0


def test_main_callable_returns_nonzero_for_unknown_command():
    code = main(["not-a-real-command"])
    assert code == 2


def test_no_traceback_in_normal_mode_on_dataset_error(tmp_path):
    result = runner.invoke(app, ["dataset", "verify", "occupancy", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_debug_flag_shows_traceback_when_an_exception_was_caught():
    # scenario show routes an UnknownScenarioError through an except block,
    # so --debug has an active exception to render a traceback for.
    result = runner.invoke(app, ["--debug", "scenario", "show", "does-not-exist"])
    assert result.exit_code != 0
    assert "Traceback" in result.output


def test_debug_flag_does_not_crash_on_a_validation_only_failure(tmp_path):
    # dataset verify's failure path is a plain validation check with no
    # active exception; --debug must not crash trying to render one.
    result = runner.invoke(
        app, ["--debug", "dataset", "verify", "occupancy", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_unknown_scenario_exits_with_usage_code():
    result = runner.invoke(app, ["scenario", "show", "does-not-exist"])
    assert result.exit_code == 2
    assert "unknown scenario" in result.output.lower()


def test_unknown_dataset_exits_with_usage_code():
    result = runner.invoke(app, ["dataset", "show", "does-not-exist"])
    assert result.exit_code == 2
