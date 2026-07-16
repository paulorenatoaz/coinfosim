"""Tests for ``coinfosim doctor``."""

from __future__ import annotations

from typer.testing import CliRunner

from coinfosim.cli.app import app

runner = CliRunner()


def test_doctor_runs_successfully():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "CoInfoSim version" in result.output
    assert "Python version" in result.output
    assert "Dataset cache directory" in result.output


def test_doctor_reports_all_three_datasets():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for slug in ("occupancy", "air-quality", "support2"):
        assert slug in result.output


def test_doctor_does_not_fetch_by_default(tmp_path, monkeypatch):
    from coinfosim.datasets import resolver as resolver_module
    from coinfosim.cli import doctor_command

    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: tmp_path)
    import unittest.mock as mock

    with mock.patch("coinfosim.datasets.download.fetch_dataset") as mocked_fetch:
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    mocked_fetch.assert_not_called()
