"""Tests for packaging metadata consistency (single-sourced version, etc.)."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_pyproject_declares_pep621_project_table():
    data = _load_pyproject()
    assert "project" in data
    assert data["project"]["name"] == "coinfosim"


def test_pyproject_requires_python_310_or_higher():
    data = _load_pyproject()
    assert data["project"]["requires-python"] == ">=3.10"


def test_pyproject_console_script_entry_point():
    data = _load_pyproject()
    assert data["project"]["scripts"]["coinfosim"] == "coinfosim.cli:main"


def test_pyproject_declares_dev_and_legacy_extras():
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "dev" in extras
    assert "legacy" in extras
    assert not any("jupyter" in dep.lower() for dep in data["project"]["dependencies"])


def test_no_setup_py_present():
    assert not (REPO_ROOT / "setup.py").exists()


def test_installed_version_matches_pyproject():
    from importlib.metadata import PackageNotFoundError, version

    data = _load_pyproject()
    pyproject_version = data["project"]["version"]
    try:
        installed_version = version("coinfosim")
    except PackageNotFoundError:
        return  # not installed in this environment; nothing to compare
    assert installed_version == pyproject_version


def test_coinfosim_dunder_version_matches_installed_metadata():
    import coinfosim
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_version = version("coinfosim")
    except PackageNotFoundError:
        return
    assert coinfosim.__version__ == installed_version


def test_manifest_in_excludes_raw_data():
    manifest_text = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "prune data" in manifest_text or "recursive-exclude data" in manifest_text
