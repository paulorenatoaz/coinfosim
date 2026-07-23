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


def test_discover_scenarios_links_semantic_manifest_and_provenance_when_present(tmp_path):
    run_dir = tmp_path / "scenarios" / "000002_occupancy_baseline_full"
    run_dir.mkdir(parents=True)
    (run_dir / "scenario_report.html").write_text("<html></html>")
    (run_dir / "semantic_manifest.json").write_text("{}")
    (run_dir / "provenance.jsonld").write_text("{}")
    registry = {
        "runs": [
            {
                "status": "completed",
                "scenario_slug": "occupancy_baseline",
                "scenario_name": "Occupancy Detection Baseline",
                "mode": "full",
                "question": "does predictive cooperation help?",
                "run_dir": str(run_dir),
                "artifacts": {"scenario_report": str(run_dir / "scenario_report.html")},
            }
        ]
    }
    (tmp_path / "scenario_runs.json").write_text(json.dumps(registry), encoding="utf-8")

    items = discover_scenarios(tmp_path)
    assert len(items) == 1
    assert items[0]["semantic_manifest_path"] == Path(
        "scenarios/000002_occupancy_baseline_full/semantic_manifest.json"
    )
    assert items[0]["provenance_path"] == Path(
        "scenarios/000002_occupancy_baseline_full/provenance.jsonld"
    )


def test_discover_scenarios_prefers_registry_provenance_artifacts_map(tmp_path):
    run_dir = tmp_path / "scenarios" / "000003_occupancy_baseline_smoke"
    run_dir.mkdir(parents=True)
    (run_dir / "scenario_report.html").write_text("<html></html>")
    (run_dir / "semantic_manifest.json").write_text("{}")
    for name in ("provenance.provjson", "provenance.provn", "provenance.ttl"):
        (run_dir / name).write_text("x")
    registry = {
        "runs": [
            {
                "status": "completed",
                "scenario_slug": "occupancy_baseline",
                "scenario_name": "Occupancy Detection Baseline",
                "mode": "smoke",
                "question": "does predictive cooperation help?",
                "run_dir": str(run_dir),
                "artifacts": {"scenario_report": str(run_dir / "scenario_report.html")},
                "provenance_path": str(run_dir / "provenance.provjson"),
                "provenance_artifacts": {
                    "provjson": str(run_dir / "provenance.provjson"),
                    "provn": str(run_dir / "provenance.provn"),
                    "ttl": str(run_dir / "provenance.ttl"),
                },
            }
        ]
    }
    (tmp_path / "scenario_runs.json").write_text(json.dumps(registry), encoding="utf-8")

    items = discover_scenarios(tmp_path)
    assert items[0]["provenance_artifacts"] == {
        "provjson": Path("scenarios/000003_occupancy_baseline_smoke/provenance.provjson"),
        "provn": Path("scenarios/000003_occupancy_baseline_smoke/provenance.provn"),
        "ttl": Path("scenarios/000003_occupancy_baseline_smoke/provenance.ttl"),
    }


def test_discover_scenarios_falls_back_to_sibling_canonical_files_by_filename(tmp_path):
    run_dir = tmp_path / "scenarios" / "000004_occupancy_baseline_smoke"
    run_dir.mkdir(parents=True)
    (run_dir / "scenario_report.html").write_text("<html></html>")
    for name in ("provenance.provjson", "provenance.provn", "provenance.ttl", "provenance.png"):
        (run_dir / name).write_text("x")
    registry = {
        "runs": [
            {
                "status": "completed",
                "scenario_slug": "occupancy_baseline",
                "scenario_name": "Occupancy Detection Baseline",
                "mode": "smoke",
                "question": "",
                "run_dir": str(run_dir),
                "artifacts": {"scenario_report": str(run_dir / "scenario_report.html")},
                # No provenance_artifacts map on this (older) record -- discovered
                # via sibling filenames next to run_dir instead.
            }
        ]
    }
    (tmp_path / "scenario_runs.json").write_text(json.dumps(registry), encoding="utf-8")

    items = discover_scenarios(tmp_path)
    found = items[0]["provenance_artifacts"]
    assert found["provjson"] == Path(
        "scenarios/000004_occupancy_baseline_smoke/provenance.provjson"
    )
    assert found["png"] == Path("scenarios/000004_occupancy_baseline_smoke/provenance.png")
    assert "pdf" not in found


def test_discover_scenarios_omits_semantic_links_when_absent(tmp_path):
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
        ]
    }
    (tmp_path / "scenario_runs.json").write_text(json.dumps(registry), encoding="utf-8")
    (tmp_path / "scenarios" / "run0").mkdir(parents=True)
    (tmp_path / "scenarios" / "run0" / "report.html").write_text("<html></html>")

    items = discover_scenarios(tmp_path)
    assert items[0]["semantic_manifest_path"] is None
    assert items[0]["provenance_path"] is None


def test_scenario_card_renders_semantic_manifest_and_provenance_links():
    from coinfosim.publish.site import _scenario_card_html

    item = {
        "path": Path("scenarios/000002/scenario_report.html"),
        "title": "Occupancy Detection Baseline (full)",
        "dataset": "Occupancy Detection",
        "question": "",
        "semantic_manifest_path": Path("scenarios/000002/semantic_manifest.json"),
        "provenance_path": Path("scenarios/000002/provenance.jsonld"),
    }
    card = _scenario_card_html(item, Path("reports"))
    assert "reports/scenarios/000002/semantic_manifest.json" in card
    assert "reports/scenarios/000002/provenance.jsonld" in card
    assert "semantic manifest" in card
    assert "provenance (JSON-LD)" in card


def test_scenario_card_omits_semantic_block_when_absent():
    from coinfosim.publish.site import _scenario_card_html

    item = {
        "path": Path("evil.html"),
        "title": "Title",
        "dataset": "Occupancy",
        "question": "",
    }
    card = _scenario_card_html(item, Path("reports"))
    assert "semantic manifest" not in card
    assert "provenance (JSON-LD)" not in card


def test_scenario_card_renders_canonical_provenance_links_in_required_order():
    from coinfosim.publish.site import _scenario_card_html

    item = {
        "path": Path("scenarios/000002/scenario_report.html"),
        "title": "Occupancy Detection Baseline (full)",
        "dataset": "Occupancy Detection",
        "question": "",
        "semantic_manifest_path": Path("scenarios/000002/semantic_manifest.json"),
        "provenance_path": Path("scenarios/000002/provenance.provjson"),
        "provenance_artifacts": {
            "provjson": Path("scenarios/000002/provenance.provjson"),
            "provn": Path("scenarios/000002/provenance.provn"),
            "ttl": Path("scenarios/000002/provenance.ttl"),
            "png": Path("scenarios/000002/provenance.png"),
            "pdf": Path("scenarios/000002/provenance.pdf"),
        },
    }
    card = _scenario_card_html(item, Path("reports"))
    for label in (
        "semantic manifest",
        "PROV-JSON",
        "PROV-N",
        "PROV-O/Turtle",
        "provenance graph (PNG)",
        "provenance graph (PDF)",
    ):
        assert label in card
    # Discovery/rendering order per Section 13.
    positions = [
        card.index(label)
        for label in (
            "semantic manifest",
            "PROV-JSON",
            "PROV-N",
            "PROV-O/Turtle",
            "provenance graph (PNG)",
            "provenance graph (PDF)",
        )
    ]
    assert positions == sorted(positions)
    # No duplicate legacy JSON-LD link when canonical formats are present.
    assert "provenance (JSON-LD)" not in card


def test_scenario_card_falls_back_to_legacy_jsonld_when_no_canonical_artifacts():
    from coinfosim.publish.site import _scenario_card_html

    item = {
        "path": Path("scenarios/000001/scenario_report.html"),
        "title": "Occupancy Detection Baseline (full)",
        "dataset": "Occupancy Detection",
        "question": "",
        "provenance_path": Path("scenarios/000001/provenance.jsonld"),
        "provenance_artifacts": {},
    }
    card = _scenario_card_html(item, Path("reports"))
    assert "provenance (JSON-LD)" in card
    assert "reports/scenarios/000001/provenance.jsonld" in card


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


def test_sync_reports_default_merges_and_keeps_orphaned_files(tmp_path):
    output_dir = tmp_path / "output"
    (output_dir / "reports").mkdir(parents=True)
    (output_dir / "reports" / "new.html").write_text("<html>new</html>")

    site_dir = tmp_path / "site"
    (site_dir / "reports").mkdir(parents=True)
    (site_dir / "reports" / "orphan.html").write_text("<html>orphan</html>")

    sync_reports(output_dir, site_dir)
    assert (site_dir / "reports" / "new.html").exists()
    assert (site_dir / "reports" / "orphan.html").exists()


def test_sync_reports_mirror_removes_orphaned_files_not_in_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    (output_dir / "reports").mkdir(parents=True)
    (output_dir / "reports" / "new.html").write_text("<html>new</html>")

    site_dir = tmp_path / "site"
    (site_dir / "reports").mkdir(parents=True)
    (site_dir / "reports" / "orphan.html").write_text("<html>orphan</html>")

    sync_reports(output_dir, site_dir, mirror=True)
    assert (site_dir / "reports" / "new.html").exists()
    assert not (site_dir / "reports" / "orphan.html").exists()


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
