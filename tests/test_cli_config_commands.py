"""Tests for ``coinfosim config ...`` commands."""

from __future__ import annotations

from typer.testing import CliRunner

from coinfosim.cli.app import app

runner = CliRunner()


def test_config_show_runs_successfully():
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "paths" in result.output.lower() or "output_dir" in result.output.lower()


def test_config_init_creates_project_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code == 0
    assert (tmp_path / "coinfosim.toml").exists()


def test_config_init_refuses_overwrite_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coinfosim.toml").write_text("[paths]\n")
    result = runner.invoke(app, ["config", "init"])
    assert result.exit_code != 0


def test_config_init_force_overwrites(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coinfosim.toml").write_text("[paths]\n")
    result = runner.invoke(app, ["config", "init", "--force"])
    assert result.exit_code == 0


def test_config_validate_default_config_is_valid():
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_config_validate_explicit_file(tmp_path):
    config_path = tmp_path / "custom.toml"
    config_path.write_text('[paths]\noutput_dir = "./output"\n')
    result = runner.invoke(app, ["config", "validate", "--file", str(config_path)])
    assert result.exit_code == 0
