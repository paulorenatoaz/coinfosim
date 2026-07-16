"""Tests for ``coinfosim scenario ...`` commands."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from coinfosim.cli.app import app

runner = CliRunner()


def test_scenario_list_shows_all_three_canonical_slugs():
    result = runner.invoke(app, ["scenario", "list"])
    assert result.exit_code == 0
    for slug in ("occupancy", "air-quality", "support2"):
        assert slug in result.output


def test_scenario_show_occupancy():
    result = runner.invoke(app, ["scenario", "show", "occupancy"])
    assert result.exit_code == 0
    assert "Occupancy Detection" in result.output
    assert "scenario run occupancy" in result.output


def test_scenario_show_resolves_alias():
    result = runner.invoke(app, ["scenario", "show", "occupancy-detection"])
    assert result.exit_code == 0
    assert "Occupancy Detection" in result.output


def test_scenario_run_maps_options_to_service_call(tmp_path):
    with mock.patch("coinfosim.scenarios.service.run_registered_scenario") as mocked_run:
        mocked_run.return_value = {
            "scenario_run_id": 0,
            "scenario_report": str(tmp_path / "report.html"),
        }
        result = runner.invoke(
            app,
            [
                "scenario",
                "run",
                "occupancy",
                "--mode",
                "smoke",
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path),
                "--no-download",
                "--no-visualizations",
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mocked_run.call_count == 1
    _, kwargs = mocked_run.call_args
    assert kwargs["mode"] == "smoke"
    assert kwargs["data_dir"] == Path(str(tmp_path))
    assert kwargs["output_dir"] == Path(str(tmp_path))
    assert kwargs["allow_download"] is False
    assert kwargs["visualize"] is False
    assert kwargs["force_download"] is False


def test_scenario_run_rejects_conflicting_download_flags():
    result = runner.invoke(
        app,
        ["scenario", "run", "occupancy", "--no-download", "--refresh-data"],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()


def test_scenario_run_rejects_sequential_with_multiple_workers():
    result = runner.invoke(
        app, ["scenario", "run", "occupancy", "--backend", "sequential", "--workers", "4"]
    )
    assert result.exit_code == 2
    assert "sequential" in result.output.lower()


def test_scenario_run_rejects_unknown_mode():
    result = runner.invoke(app, ["scenario", "run", "occupancy", "--mode", "bogus-mode"])
    assert result.exit_code == 2


def test_scenario_regenerate_maps_options_to_service_call(tmp_path):
    with mock.patch(
        "coinfosim.scenarios.service.regenerate_registered_scenario"
    ) as mocked_regenerate:
        mocked_regenerate.return_value = {"scenario_report": str(tmp_path / "report.html")}
        result = runner.invoke(
            app,
            [
                "scenario",
                "regenerate",
                "occupancy",
                "--run-id",
                "3",
                "--output-dir",
                str(tmp_path),
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    mocked_regenerate.assert_called_once()
    _, kwargs = mocked_regenerate.call_args
    assert kwargs["scenario_run_id"] == 3
    assert kwargs["output_dir"] == Path(str(tmp_path))


def test_scenario_regenerate_does_not_invoke_run_registered_scenario(tmp_path):
    with mock.patch(
        "coinfosim.scenarios.service.regenerate_registered_scenario"
    ) as mocked_regenerate, mock.patch(
        "coinfosim.scenarios.service.run_registered_scenario"
    ) as mocked_run:
        mocked_regenerate.return_value = {"scenario_report": str(tmp_path / "report.html")}
        runner.invoke(
            app,
            ["scenario", "regenerate", "occupancy", "--run-id", "0", "--output-dir", str(tmp_path)],
        )
    mocked_run.assert_not_called()
