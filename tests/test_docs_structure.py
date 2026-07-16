"""Structural checks for README.md and the supporting docs/ files."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"

REQUIRED_SECTIONS_IN_ORDER = [
    "## Installation",
    "## 60-second quick start",
    "## Three built-in scenarios",
    "## Generated outputs",
    "## Dataset download and cache behavior",
    "## Scientific modes and computational cost",
    "## Multiprocessing",
    "## Report regeneration",
    "## Development installation",
    "## Citation",
    "## License",
]


def test_readme_sections_appear_in_required_order():
    text = README.read_text(encoding="utf-8")
    positions = []
    for heading in REQUIRED_SECTIONS_IN_ORDER:
        index = text.find(heading)
        assert index != -1, f"README.md is missing required section: {heading}"
        positions.append(index)
    assert positions == sorted(positions), "README.md sections are out of the required order"


def test_readme_first_pip_install_line_is_pypi_install():
    text = README.read_text(encoding="utf-8")
    install_section = text.split("## Installation", 1)[1].split("## 60-second quick start", 1)[0]
    assert "python -m pip install coinfosim" in install_section


def test_readme_quick_start_uses_new_cli():
    text = README.read_text(encoding="utf-8")
    assert "coinfosim scenario list" in text
    assert "coinfosim scenario run occupancy --mode smoke" in text


def test_readme_warns_expensive_modes_never_run_implicitly():
    text = README.read_text(encoding="utf-8")
    modes_section = text.split("## Scientific modes and computational cost", 1)[1]
    modes_section = modes_section.split("## Multiprocessing", 1)[0]
    assert "never start implicitly" in modes_section or "never run implicitly" in modes_section.lower() or "explicit" in modes_section.lower()


def test_readme_links_to_supporting_docs():
    text = README.read_text(encoding="utf-8")
    for doc in (
        "docs/cli.md",
        "docs/datasets.md",
        "docs/installation.md",
        "docs/releasing.md",
        "docs/migration-cli-0.2.md",
    ):
        assert doc in text


@pytest.mark.parametrize(
    "doc",
    [
        "docs/cli.md",
        "docs/datasets.md",
        "docs/installation.md",
        "docs/releasing.md",
        "docs/migration-cli-0.2.md",
    ],
)
def test_supporting_doc_exists_and_is_non_empty(doc):
    path = REPO_ROOT / doc
    assert path.is_file(), f"missing doc: {doc}"
    assert len(path.read_text(encoding="utf-8")) > 200


def test_citation_cff_version_matches_pyproject():
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    citation_text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {pyproject['project']['version']}" in citation_text


def test_datasets_doc_never_labels_support2_cc_by():
    text = (REPO_ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")
    support2_section = text.split("### SUPPORT2", 1)[1]
    next_heading = support2_section.find("\n## ")
    if next_heading != -1:
        support2_section = support2_section[:next_heading]
    # A bare mention of "CC BY" is fine if it is explicitly negated (as in
    # "must never be labeled CC BY"); only an affirmative assignment like
    # "License: CC BY" would be the real bug.
    assert "license: cc by" not in support2_section.lower()
    assert "license is cc by" not in support2_section.lower()
