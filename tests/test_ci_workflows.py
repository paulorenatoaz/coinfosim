"""Structural checks for the CI and release GitHub Actions workflows."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


def _load(name: str) -> dict:
    with (WORKFLOWS_DIR / name).open() as fh:
        return yaml.safe_load(fh)


# PyYAML (YAML 1.1) parses the unquoted `on:` key of a workflow file as the
# boolean True, not the string "on" — so trigger config is read back as
# workflow[_ON], not workflow["on"].
_ON = True


@pytest.mark.parametrize(
    "name",
    ["ci.yml", "package.yml", "publish-testpypi.yml", "publish-pypi.yml", "static.yml", "publish-pages.yml"],
)
def test_workflow_file_is_valid_yaml(name):
    workflow = _load(name)
    assert isinstance(workflow, dict)
    assert "jobs" in workflow


def test_ci_matrix_covers_required_os_and_python_combinations():
    workflow = _load("ci.yml")
    matrix_entries = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]
    combos = {(entry["os"], entry["python-version"]) for entry in matrix_entries}

    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert ("ubuntu-latest", version) in combos

    assert any(os == "windows-latest" and version in ("3.11", "3.12") for os, version in combos)
    assert any(os == "macos-latest" and version in ("3.11", "3.12") for os, version in combos)


def test_ci_does_not_run_full_scale_or_strict_modes_implicitly():
    text = (WORKFLOWS_DIR / "ci.yml").read_text(encoding="utf-8")
    assert "full-scale" not in text
    assert "--mode strict" not in text
    assert "-m \"not slow\"" in text or "-m 'not slow'" in text


def test_package_workflow_builds_checks_and_installs_the_wheel():
    text = (WORKFLOWS_DIR / "package.yml").read_text(encoding="utf-8")
    assert "python -m build" in text
    assert "twine check" in text
    assert "resources/datasets.json" in text
    assert "data/raw" in text  # exclusion check present
    assert "coinfosim --version" in text
    assert "coinfosim scenario list" in text
    assert "coinfosim dataset list" in text
    assert "coinfosim doctor" in text


@pytest.mark.parametrize(
    "name,environment,index_hint",
    [
        ("publish-testpypi.yml", "testpypi", "test.pypi.org"),
        ("publish-pypi.yml", "pypi", "pypi.org"),
    ],
)
def test_publish_workflows_use_oidc_trusted_publishing_no_stored_token(name, environment, index_hint):
    workflow = _load(name)
    text = (WORKFLOWS_DIR / name).read_text(encoding="utf-8")

    publish_job = workflow["jobs"]["publish"]
    assert publish_job["environment"]["name"] == environment
    assert publish_job["permissions"]["id-token"] == "write"
    assert "pypa/gh-action-pypi-publish" in text
    assert index_hint in text

    # No permanent package-index secret anywhere in the file.
    assert "PYPI_API_TOKEN" not in text
    assert "password:" not in text
    assert "secrets.PYPI" not in text


def test_publish_pypi_triggers_only_on_github_release():
    workflow = _load("publish-pypi.yml")
    assert set(workflow[_ON]) == {"release"}
    assert workflow[_ON]["release"]["types"] == ["published"]


def test_publish_testpypi_is_manual_or_prerelease_tag_only():
    workflow = _load("publish-testpypi.yml")
    triggers = set(workflow[_ON])
    assert "workflow_dispatch" in triggers
    assert "push" not in triggers or "tags" in workflow[_ON]["push"]
