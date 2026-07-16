"""Tests for coinfosim.publish.site (home page and report index generation)."""

from __future__ import annotations

import json
from pathlib import Path

from coinfosim.datasets.catalog import list_datasets
from coinfosim.publish.site import (
    discover_json,
    discover_scenarios,
    sync_reports,
    write_index,
)


def test_discover_scenarios_empty_when_no_registry(tmp_path):
    assert discover_scenarios(tmp_path) == []


def test_discover_scenarios_reads_completed_runs_from_registry(tmp_path):
    registry = {
        "runs": [
            {
                "status": "completed",
                "scenario_slug": "occupancy_baseline",
                "scenario_name": "Occupancy Detection Baseline",
                "mode": "smoke",
                "question": "does cooperation help?",
                "artifacts": {"scenario_report": str(tmp_path / "scenarios" / "run0" / "report.html")},
            },
            {"status": "failed", "scenario_slug": "occupancy_baseline"},
        ]
    }
    (tmp_path / "scenario_runs.json").write_text(json.dumps(registry), encoding="utf-8")
    (tmp_path / "scenarios" / "run0").mkdir(parents=True)
    (tmp_path / "scenarios" / "run0" / "report.html").write_text("<html></html>")

    items = discover_scenarios(tmp_path)
    assert len(items) == 1
    assert items[0]["dataset"] == "Occupancy Detection"
    assert items[0]["title"] == "Occupancy Detection Baseline (smoke)"


def test_discover_json_lists_and_sorts_by_name(tmp_path):
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "a.json").write_text("{}")
    files = discover_json(tmp_path)
    assert [f.name for f in files] == ["a.json", "b.json"]


def test_sync_reports_copies_reports_and_data(tmp_path):
    output_dir = tmp_path / "output"
    (output_dir / "reports" / "scenarios").mkdir(parents=True)
    (output_dir / "reports" / "scenarios" / "report.html").write_text("<html></html>")
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "data" / "x.json").write_text("{}")

    site_dir = tmp_path / "site"
    sync_reports(output_dir, site_dir)
    assert (site_dir / "reports" / "scenarios" / "report.html").exists()
    assert (site_dir / "data" / "x.json").exists()


def test_sync_reports_tolerates_missing_output_dir(tmp_path):
    site_dir = tmp_path / "site"
    sync_reports(tmp_path / "does-not-exist", site_dir)
    assert site_dir.exists()


def test_write_index_contains_seven_required_sections(tmp_path):
    path = write_index(
        tmp_path,
        reports_rel=Path("reports"),
        data_rel=Path("data"),
        scenarios=[],
        json_files=[],
        datasets=list_datasets(),
    )
    html_text = path.read_text(encoding="utf-8")
    for section in (
        "Install CoInfoSim",
        "Quick start",
        "Available datasets",
        "Published scenario reports",
        "Machine-readable artifacts",
        "Citation and license",
    ):
        assert section in html_text
    assert "pip install coinfosim" in html_text
    assert "coinfosim scenario run occupancy --mode smoke" in html_text


def test_write_index_escapes_external_text(tmp_path):
    malicious_scenario = {
        "path": Path("evil.html"),
        "title": "<script>alert(1)</script>",
        "dataset": "Occupancy",
        "question": "<img src=x onerror=alert(1)>",
    }
    path = write_index(
        tmp_path,
        reports_rel=Path("reports"),
        data_rel=Path("data"),
        scenarios=[malicious_scenario],
        json_files=[],
        datasets=list_datasets(),
    )
    html_text = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    assert "<img src=x onerror=alert(1)>" not in html_text


def test_write_index_never_labels_support2_as_cc_by():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_index(
            Path(tmp),
            reports_rel=Path("reports"),
            data_rel=Path("data"),
            scenarios=[],
            json_files=[],
            datasets=list_datasets(),
        )
        html_text = path.read_text(encoding="utf-8")
        support2_index = html_text.index("SUPPORT2")
        # The nearby text (dataset card) must not claim CC BY for SUPPORT2.
        window = html_text[support2_index : support2_index + 800]
        assert "CC BY" not in window


def test_write_index_lists_all_dataset_files():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = write_index(
            Path(tmp),
            reports_rel=Path("reports"),
            data_rel=Path("data"),
            scenarios=[],
            json_files=[],
            datasets=list_datasets(),
        )
        html_text = path.read_text(encoding="utf-8")
        for dataset in list_datasets():
            for file in dataset.files:
                assert file.filename in html_text
                assert file.sha256 in html_text


def test_write_index_is_deterministic_given_fixed_timestamp(tmp_path, monkeypatch):
    import coinfosim.publish.site as site_module
    import datetime as dt

    class _FixedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(site_module.dt, "datetime", _FixedDatetime)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    path_a = write_index(
        dir_a, reports_rel=Path("reports"), data_rel=Path("data"), scenarios=[], json_files=[], datasets=list_datasets()
    )
    path_b = write_index(
        dir_b, reports_rel=Path("reports"), data_rel=Path("data"), scenarios=[], json_files=[], datasets=list_datasets()
    )
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")
