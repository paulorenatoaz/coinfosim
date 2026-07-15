import subprocess
from pathlib import Path

import pytest

from coinfosim.publish.publisher import (
    DEFAULT_SITE_TITLE,
    PublishError,
    _collect_reports,
    _render_index,
    _validate_local_links,
    discover_scenario_reports,
    publish_to_pages,
)


def _write(path: Path, content: str = "<html><title>Report</title></html>") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_discovers_current_and_legacy_scenario_reports(tmp_path):
    reports = tmp_path / "reports"
    for relative in (
        "air_quality_scenario_report.html",
        "nested/occupancy_scenario_report.html",
        "nested/occupancy_scenario_report_full_000002.html",
        "scenario_1_report.html",
        "scenario_2_report[test].html",
        "air_quality_dataset_report.html",
        "air_quality_real_monte_carlo_report.html",
    ):
        _write(reports / relative)

    assert [path.as_posix() for path in discover_scenario_reports(reports)] == [
        "air_quality_scenario_report.html",
        "nested/occupancy_scenario_report.html",
        "nested/occupancy_scenario_report_full_000002.html",
        "scenario_1_report.html",
        "scenario_2_report[test].html",
    ]


def test_renders_academic_homepage_with_metadata_and_escaped_titles(tmp_path):
    reports_root = tmp_path / "reports"
    _write(
        reports_root / "air_quality_scenario_report.html",
        "<html><head><title>Air &amp; Quality Study</title></head></html>",
    )
    _write(
        reports_root / "nested" / "occupancy_scenario_report.html",
        "<html><head><title>Occupancy Scenario Results</title></head></html>",
    )
    reports = _collect_reports(reports_root)

    homepage = _render_index(
        reports,
        title=DEFAULT_SITE_TITLE,
        published_at="2026-07-15T12:00:00Z",
        source_sha="abc1234",
    )

    for expected in (
        DEFAULT_SITE_TITLE,
        "CoInfoSim is a research simulator",
        "When does cooperation among information channels improve supervised classification?",
        "Occupancy Detection",
        "UCI Air Quality",
        "10.24432/C59K5F",
        "Air &amp; Quality Study",
        "Occupancy Scenario Results",
        'href="reports/air_quality_scenario_report.html"',
        'href="reports/nested/occupancy_scenario_report.html"',
        "abc1234",
        "2026-07-15T12:00:00Z",
    ):
        assert expected in homepage


def test_broken_local_report_link_is_rejected(tmp_path):
    site = tmp_path / "site"
    reports_root = site / "reports"
    _write(
        reports_root / "air_quality_scenario_report.html",
        """<html><title>Air Quality</title><body>
        <a href="https://example.org">External</a>
        <a href="#results">Anchor</a>
        <a href="missing_report.html">Missing</a>
        </body></html>""",
    )

    reports = _collect_reports(reports_root)
    with pytest.raises(PublishError) as error:
        _validate_local_links(site, reports)

    message = str(error.value)
    assert "air_quality_scenario_report.html" in message
    assert "missing_report.html" in message


def test_git_publication_is_idempotent_and_updates_content(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    remote = tmp_path / "remote.git"
    output = tmp_path / "output"
    repository.mkdir()
    _git("init", "--initial-branch=main", cwd=repository)
    _git("config", "user.name", "Test User", cwd=repository)
    _git("config", "user.email", "test@example.com", cwd=repository)
    _write(repository / "README.md", "test repository")
    _git("add", "README.md", cwd=repository)
    _git("commit", "-m", "initial", cwd=repository)
    _git("init", "--bare", str(remote), cwd=tmp_path)
    _git("remote", "add", "origin", str(remote), cwd=repository)
    report = _write(
        output / "reports" / "nested" / "occupancy_scenario_report.html",
        "<html><title>Occupancy Results</title><body>first</body></html>",
    )
    monkeypatch.chdir(repository)

    first = publish_to_pages(output)
    assert first.changed is True
    assert first.pushed is True
    first_sha = _git("--git-dir", str(remote), "rev-parse", "gh-pages", cwd=tmp_path)
    tree = _git("--git-dir", str(remote), "ls-tree", "-r", "--name-only", "gh-pages", cwd=tmp_path)
    assert {"index.html", ".nojekyll", "reports/nested/occupancy_scenario_report.html"} <= set(tree.splitlines())

    second = publish_to_pages(output)
    assert second.changed is False
    assert second.pushed is False
    assert _git("--git-dir", str(remote), "rev-parse", "gh-pages", cwd=tmp_path) == first_sha

    report.write_text(
        "<html><title>Occupancy Results</title><body>updated</body></html>",
        encoding="utf-8",
    )
    third = publish_to_pages(output)
    assert third.changed is True
    assert third.pushed is True
    assert _git("--git-dir", str(remote), "rev-parse", "gh-pages", cwd=tmp_path) != first_sha
    published = _git(
        "--git-dir",
        str(remote),
        "show",
        "gh-pages:reports/nested/occupancy_scenario_report.html",
        cwd=tmp_path,
    )
    assert "updated" in published
