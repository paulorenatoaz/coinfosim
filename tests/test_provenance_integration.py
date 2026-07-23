"""Tests for normal-run provenance finalization (Block P3).

Exercises exactly the evidence-collection + emission functions that
``run_dataset_anchored_scenario`` calls after scientific results/reports have
already been persisted (see ``dataset_anchored_runner.py``). Uses a tiny,
fully synthetic fixture written directly to ``tmp_path`` -- no dataset load,
no Monte Carlo, no real scenario execution.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from coinfosim.provenance import (
    ArtifactEvidence,
    SimulationArmEvidence,
    collect_runtime_provenance_evidence,
    emit_scenario_semantic_and_provenance_artifacts,
    sha256_of_file,
    to_repo_relative,
)
from coinfosim.provenance.evidence import GAUSSIAN_ARM_ID, GMM_ARM_ID, REAL_ARM_ID


def _write_fake_run(tmp_path):
    scenario_dir = tmp_path / "output" / "reports" / "scenarios" / "000002"
    scenario_dir.mkdir(parents=True)

    result_paths = {}
    for arm, seed in ((REAL_ARM_ID, "a"), (GAUSSIAN_ARM_ID, "b"), (GMM_ARM_ID, "c")):
        path = scenario_dir.parent.parent / "simulations" / f"result_data_{arm}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake-result-data-{seed}".encode("utf-8"))
        result_paths[arm] = path

    report_path = scenario_dir / "occupancy_baseline_scenario_report_smoke_000002.html"
    report_path.write_text("<html>fake scenario report</html>", encoding="utf-8")

    return scenario_dir, result_paths, report_path


def _build_evidence(tmp_path, scenario_dir, result_paths, report_path):
    simulation_arms = [
        SimulationArmEvidence(
            arm_id=arm,
            simulation_run_id=run_id,
            result_data=ArtifactEvidence(
                path=to_repo_relative(path, tmp_path),
                sha256=sha256_of_file(path),
                role=f"result-data:{arm}",
            ),
        )
        for run_id, (arm, path) in enumerate(result_paths.items(), start=6)
    ]
    report_artifacts = [
        ArtifactEvidence(
            path=to_repo_relative(report_path, tmp_path),
            sha256=sha256_of_file(report_path),
            role="scenario-report",
        )
    ]
    return collect_runtime_provenance_evidence(
        scenario_run_id=2,
        scenario_slug="occupancy_baseline",
        dataset_metadata={"name": "Occupancy Detection", "slug": "occupancy"},
        target_metadata={"name": "Occupancy"},
        split_metadata={"strategy": "original UCI training and test files"},
        preprocessing_metadata={"method": "zscore", "fit_scope": "training_reservoir_only"},
        experiment_configuration={"mode": "smoke", "sample_sizes": [2, 4]},
        classifier_configuration={"names": ["linear_svm"]},
        simulation_arms=simulation_arms,
        report_artifacts=report_artifacts,
        gaussian_generator_metadata={"ridge": 0.1},
        gmm_generator_metadata={"n_components": 2},
        code_commit_sha="cafebabe0000000000000000000000000000000",
    )


def test_normal_run_finalization_emits_semantic_manifest_and_machine_provenance(tmp_path):
    scenario_dir, result_paths, report_path = _write_fake_run(tmp_path)
    evidence = _build_evidence(tmp_path, scenario_dir, result_paths, report_path)

    artifacts = emit_scenario_semantic_and_provenance_artifacts(
        evidence, scenario_dir=scenario_dir, repo_root=tmp_path
    )

    manifest_path = scenario_dir / "semantic_manifest.json"
    assert manifest_path.exists()
    assert (scenario_dir / "provenance.provjson").exists()
    assert (scenario_dir / "provenance.provn").exists()
    assert (scenario_dir / "provenance.ttl").exists()
    assert artifacts.semantic_manifest == to_repo_relative(manifest_path, tmp_path)


def test_result_data_hashes_in_manifest_match_actual_files(tmp_path):
    scenario_dir, result_paths, report_path = _write_fake_run(tmp_path)
    evidence = _build_evidence(tmp_path, scenario_dir, result_paths, report_path)

    emit_scenario_semantic_and_provenance_artifacts(
        evidence, scenario_dir=scenario_dir, repo_root=tmp_path
    )

    manifest = json.loads(
        (scenario_dir / "semantic_manifest.json").read_text(encoding="utf-8")
    )
    manifest_hashes = {
        entry["path"]: entry["sha256"] for entry in manifest["source_result_data"]
    }
    for arm, path in result_paths.items():
        relpath = to_repo_relative(path, tmp_path)
        assert manifest_hashes[relpath] == sha256_of_file(path)


def test_provenance_finalization_never_modifies_scientific_output_files(tmp_path):
    scenario_dir, result_paths, report_path = _write_fake_run(tmp_path)
    original_bytes = {arm: path.read_bytes() for arm, path in result_paths.items()}
    original_report = report_path.read_bytes()

    evidence = _build_evidence(tmp_path, scenario_dir, result_paths, report_path)
    emit_scenario_semantic_and_provenance_artifacts(
        evidence, scenario_dir=scenario_dir, repo_root=tmp_path
    )

    for arm, path in result_paths.items():
        assert path.read_bytes() == original_bytes[arm]
    assert report_path.read_bytes() == original_report


def test_graphviz_absence_does_not_fail_a_completed_scenario(tmp_path):
    scenario_dir, result_paths, report_path = _write_fake_run(tmp_path)
    evidence = _build_evidence(tmp_path, scenario_dir, result_paths, report_path)

    with patch("coinfosim.provenance.export.shutil.which", return_value=None):
        artifacts = emit_scenario_semantic_and_provenance_artifacts(
            evidence, scenario_dir=scenario_dir, repo_root=tmp_path
        )

    assert artifacts.graphviz_available is False
    assert artifacts.png is None
    assert artifacts.pdf is None
    # The three mandatory machine-readable formats are still written.
    assert (scenario_dir / "provenance.provjson").exists()
    assert (scenario_dir / "provenance.provn").exists()
    assert (scenario_dir / "provenance.ttl").exists()
