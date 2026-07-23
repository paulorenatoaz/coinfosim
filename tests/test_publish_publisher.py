"""Tests for coinfosim.publish.publisher (git worktree publish orchestration).

Uses an isolated temporary git repository seeded with the real tracked raw
dataset files (so catalog hash verification succeeds) — never touches the
actual coinfosim repository's branches.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from coinfosim.publish.publisher import PublishError, publish_pages

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(args, cwd):
    subprocess.run(args, cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def isolated_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)

    for rel_dir in ("data/raw/occupancy", "data/raw/air_quality", "data/raw/support2"):
        source = REPO_ROOT / rel_dir
        target = repo / rel_dir
        shutil.copytree(source, target)

    (repo / "README.md").write_text("fixture repo\n")
    _run(["git", "add", "-A"], cwd=repo)
    _run(["git", "commit", "-q", "-m", "initial"], cwd=repo)
    return repo


@pytest.fixture
def output_dir_with_one_report(tmp_path):
    output_dir = tmp_path / "output"
    (output_dir / "reports").mkdir(parents=True)
    (output_dir / "data").mkdir(parents=True)
    return output_dir


def test_publish_pages_creates_orphan_branch_and_commits(isolated_repo, output_dir_with_one_report, monkeypatch):
    monkeypatch.chdir(isolated_repo)
    result = publish_pages(
        output_dir_with_one_report,
        branch="gh-pages",
        remote="origin",
        push=False,
        dry_run=False,
        init_branch_if_missing=True,
    )
    assert result.changed is True
    assert result.committed is True
    assert result.pushed is False
    assert result.dataset_file_count == 5  # 3 occupancy + 1 air-quality + 1 support2

    branches = subprocess.run(
        ["git", "branch", "--list", "gh-pages"], cwd=str(isolated_repo), capture_output=True, text=True
    ).stdout
    assert "gh-pages" in branches


def test_publish_pages_worktree_is_always_removed(isolated_repo, output_dir_with_one_report, monkeypatch):
    monkeypatch.chdir(isolated_repo)
    publish_pages(output_dir_with_one_report, branch="gh-pages", init_branch_if_missing=True)
    worktrees = subprocess.run(
        ["git", "worktree", "list"], cwd=str(isolated_repo), capture_output=True, text=True
    ).stdout
    assert worktrees.strip().count("\n") == 0  # only the main worktree line, no extras


def test_publish_pages_missing_branch_without_init_raises(isolated_repo, output_dir_with_one_report, monkeypatch):
    monkeypatch.chdir(isolated_repo)
    with pytest.raises(PublishError, match="does not exist"):
        publish_pages(output_dir_with_one_report, branch="gh-pages", init_branch_if_missing=False)


def test_publish_pages_dry_run_never_commits(isolated_repo, output_dir_with_one_report, monkeypatch):
    monkeypatch.chdir(isolated_repo)
    # Seed the branch first (real commit), then verify dry-run afterward makes no new commit.
    publish_pages(output_dir_with_one_report, branch="gh-pages", init_branch_if_missing=True)
    log_before = subprocess.run(
        ["git", "log", "gh-pages", "--oneline"], cwd=str(isolated_repo), capture_output=True, text=True
    ).stdout

    result = publish_pages(output_dir_with_one_report, branch="gh-pages", dry_run=True)
    assert result.committed is False
    assert result.pushed is False

    log_after = subprocess.run(
        ["git", "log", "gh-pages", "--oneline"], cwd=str(isolated_repo), capture_output=True, text=True
    ).stdout
    assert log_before == log_after


def test_publish_pages_never_pushes_without_explicit_flag(isolated_repo, output_dir_with_one_report, monkeypatch):
    monkeypatch.chdir(isolated_repo)
    result = publish_pages(output_dir_with_one_report, branch="gh-pages", init_branch_if_missing=True, push=False)
    assert result.pushed is False


def test_publish_pages_copies_ontology_and_links_it_from_index(
    isolated_repo, output_dir_with_one_report, monkeypatch
):
    ontology_source = REPO_ROOT / "ontology" / "coinfosim.owl.ttl"
    ontology_target = isolated_repo / "ontology" / "coinfosim.owl.ttl"
    ontology_target.parent.mkdir(parents=True)
    shutil.copy2(ontology_source, ontology_target)
    _run(["git", "add", "-A"], cwd=isolated_repo)
    _run(["git", "commit", "-q", "-m", "add ontology"], cwd=isolated_repo)

    monkeypatch.chdir(isolated_repo)
    publish_pages(output_dir_with_one_report, branch="gh-pages", init_branch_if_missing=True)

    ontology_show = subprocess.run(
        ["git", "show", "gh-pages:ontology/coinfosim.owl.ttl"],
        cwd=str(isolated_repo),
        capture_output=True,
        text=True,
    )
    assert ontology_show.returncode == 0
    assert "owl:Ontology" in ontology_show.stdout

    index_show = subprocess.run(
        ["git", "show", "gh-pages:index.html"],
        cwd=str(isolated_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "ontology/coinfosim.owl.ttl" in index_show.stdout


def test_publish_pages_generates_manifest_and_index_in_branch(isolated_repo, output_dir_with_one_report, monkeypatch):
    monkeypatch.chdir(isolated_repo)
    publish_pages(output_dir_with_one_report, branch="gh-pages", init_branch_if_missing=True)

    show = subprocess.run(
        ["git", "show", "gh-pages:datasets/manifest.json"],
        cwd=str(isolated_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"schema_version"' in show.stdout

    index_show = subprocess.run(
        ["git", "show", "gh-pages:index.html"],
        cwd=str(isolated_repo),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Available datasets" in index_show.stdout

    nojekyll = subprocess.run(
        ["git", "show", "gh-pages:.nojekyll"],
        cwd=str(isolated_repo),
        capture_output=True,
        text=True,
    )
    assert nojekyll.returncode == 0
