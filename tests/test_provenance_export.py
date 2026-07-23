"""Tests for the semantic manifest and PROV-O-compatible provenance export."""

from __future__ import annotations

import json

import pytest

from coinfosim.provenance import (
    build_provenance_graph,
    build_semantic_manifest,
    sha256_of_file,
    to_repo_relative,
    write_provenance_graph,
    write_semantic_manifest,
)
from coinfosim.provenance.jsonld import (
    commit_agent_id,
    profile_entity_id,
    report_entity_id,
    result_data_entity_id,
    simulation_run_entity_id,
)
from coinfosim.semantics import canonical_key_to_id, vocabulary_version


FAKE_RESULT_DATA = [
    {"path": "output/reports/simulations/000006_x/result_data_full_000006.json.gz", "sha256": "a" * 64},
    {"path": "output/reports/simulations/000007_x/result_data_full_000007.json.gz", "sha256": "b" * 64},
]


# --------------------------------------------------------------------------- #
# semantic_manifest.py
# --------------------------------------------------------------------------- #
def test_build_semantic_manifest_has_required_fields():
    manifest = build_semantic_manifest(
        scenario_run_id=2,
        scenario_slug="occupancy_baseline",
        dataset_id="occupancy_detection",
        classifier_ids=["logistic_regression", "linear_svm"],
        training_condition_ids=["real_to_real", "single_gaussian_to_real", "gmm_to_real"],
        sample_sizes=[2, 4, 8],
        source_simulation_run_ids=[6, 7, 8],
        source_result_data=FAKE_RESULT_DATA,
        code_commit_sha="deadbeef",
        recovered_source_commit_sha="c8f21045",
    )
    assert manifest["semantic_vocabulary_version"] == vocabulary_version()
    assert manifest["semantic_type"] == canonical_key_to_id("predictive_cooperation_profile")
    assert manifest["scenario_run_id"] == "000002"
    assert manifest["dataset_id"] == "occupancy_detection"
    assert manifest["classifier_ids"] == ["linear_svm", "logistic_regression"]
    assert manifest["sample_sizes"] == [2, 4, 8]
    assert manifest["source_simulation_run_ids"] == [6, 7, 8]
    assert manifest["code_commit_sha"] == "deadbeef"
    assert manifest["recovered_source_commit_sha"] == "c8f21045"
    assert manifest["fixed_real_test_set_statement"]
    assert len(manifest["source_result_data"]) == 2


def test_semantic_manifest_canonical_metric_ids_are_stable_semantic_ids():
    manifest = build_semantic_manifest(
        scenario_run_id=0,
        scenario_slug="s",
        dataset_id="d",
        classifier_ids=["clf"],
        training_condition_ids=["arm"],
        sample_sizes=[2],
        source_simulation_run_ids=[0],
        source_result_data=[],
        code_commit_sha=None,
    )
    for metric_id in manifest["canonical_metric_ids"]:
        assert metric_id.startswith("coinfosim:")


def test_semantic_manifest_never_contains_absolute_paths():
    manifest = build_semantic_manifest(
        scenario_run_id=0,
        scenario_slug="s",
        dataset_id="d",
        classifier_ids=["clf"],
        training_condition_ids=["arm"],
        sample_sizes=[2],
        source_simulation_run_ids=[0],
        source_result_data=FAKE_RESULT_DATA,
        code_commit_sha=None,
    )
    encoded = json.dumps(manifest)
    assert "/home/" not in encoded
    assert "/tmp/" not in encoded


def test_semantic_manifest_is_strict_json_safe_and_deterministic():
    kwargs = dict(
        scenario_run_id=5,
        scenario_slug="air_quality_baseline",
        dataset_id="air_quality",
        classifier_ids=["gaussian_nb"],
        training_condition_ids=["real_to_real"],
        sample_sizes=[2, 4],
        source_simulation_run_ids=[15],
        source_result_data=FAKE_RESULT_DATA,
        code_commit_sha="cccccccc",
    )
    first = build_semantic_manifest(**kwargs)
    second = build_semantic_manifest(**kwargs)
    assert first == second
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(
        second, allow_nan=False, sort_keys=True
    )


def test_write_semantic_manifest_round_trips(tmp_path):
    manifest = build_semantic_manifest(
        scenario_run_id=1,
        scenario_slug="s",
        dataset_id="d",
        classifier_ids=["clf"],
        training_condition_ids=["arm"],
        sample_sizes=[2],
        source_simulation_run_ids=[1],
        source_result_data=[],
        code_commit_sha="abc123",
    )
    path = tmp_path / "semantic_manifest.json"
    write_semantic_manifest(path, manifest)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == manifest


def test_sha256_of_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "sample.bin"
    path.write_bytes(b"predictive cooperation profile")
    assert sha256_of_file(path) == hashlib.sha256(b"predictive cooperation profile").hexdigest()


def test_to_repo_relative_normalizes_under_root(tmp_path):
    root = tmp_path
    nested = root / "output" / "reports" / "scenarios" / "000000"
    nested.mkdir(parents=True)
    file_path = nested / "scenario.json"
    file_path.write_text("{}", encoding="utf-8")
    relative = to_repo_relative(file_path, root)
    assert relative == "output/reports/scenarios/000000/scenario.json"
    assert not relative.startswith("/")


# --------------------------------------------------------------------------- #
# jsonld.py
# --------------------------------------------------------------------------- #
def test_provenance_graph_parses_as_strict_json():
    graph = build_provenance_graph(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7, 8],
        code_commit_sha="refactorsha",
        report_artifact_path="output/reports/scenarios/000002/scenario_report.html",
        report_artifact_sha256="f" * 64,
        recovered_source_commit_sha="ghpagessha",
    )
    encoded = json.dumps(graph, allow_nan=False)
    reloaded = json.loads(encoded)
    assert reloaded == graph


def test_provenance_graph_context_declares_required_prefixes():
    graph = build_provenance_graph(
        scenario_run_id=0,
        source_result_data=[],
        source_simulation_run_ids=[],
        code_commit_sha="abc",
        report_artifact_path="report.html",
        report_artifact_sha256="e" * 64,
    )
    for prefix in ("coinfosim", "prov", "xsd"):
        assert prefix in graph["@context"]


def test_provenance_graph_includes_entities_activities_and_agent():
    graph = build_provenance_graph(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7],
        code_commit_sha="refactorsha",
        report_artifact_path="output/reports/scenarios/000002/scenario_report.html",
        report_artifact_sha256="f" * 64,
        recovered_source_commit_sha="ghpagessha",
    )
    types_seen = set()
    for node in graph["@graph"]:
        node_types = node["@type"]
        if isinstance(node_types, str):
            node_types = [node_types]
        types_seen.update(node_types)
    assert "prov:Entity" in types_seen
    assert "prov:Activity" in types_seen
    assert "prov:SoftwareAgent" in types_seen
    assert "coinfosim:PredictiveCooperationProfile" in types_seen
    assert "coinfosim:ReportArtifact" in types_seen


def test_provenance_graph_uses_prov_relations():
    graph = build_provenance_graph(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7],
        code_commit_sha="refactorsha",
        report_artifact_path="report.html",
        report_artifact_sha256="f" * 64,
        recovered_source_commit_sha="ghpagessha",
    )
    all_keys = {key for node in graph["@graph"] for key in node}
    assert "prov:used" in all_keys
    assert "prov:wasGeneratedBy" in all_keys
    assert "prov:wasDerivedFrom" in all_keys
    assert "prov:wasAssociatedWith" in all_keys


def test_provenance_graph_distinguishes_source_and_code_commits():
    graph = build_provenance_graph(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7],
        code_commit_sha="refactorsha",
        report_artifact_path="report.html",
        report_artifact_sha256="f" * 64,
        recovered_source_commit_sha="ghpagessha",
        original_simulation_commit_sha="originalsha",
    )
    agent_ids = {
        node["@id"] for node in graph["@graph"] if node.get("@type") == "prov:SoftwareAgent"
    }
    assert commit_agent_id("refactorsha") in agent_ids
    assert commit_agent_id("ghpagessha") in agent_ids
    assert commit_agent_id("originalsha") in agent_ids
    assert len(agent_ids) == 3  # three distinct commits, never collapsed into one


def test_provenance_graph_never_contains_absolute_or_temp_paths():
    graph = build_provenance_graph(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7],
        code_commit_sha="refactorsha",
        report_artifact_path="output/reports/scenarios/000002/scenario_report.html",
        report_artifact_sha256="f" * 64,
    )
    encoded = json.dumps(graph)
    assert "/home/" not in encoded
    assert "/tmp/" not in encoded


def test_provenance_graph_ids_are_deterministic_for_same_inputs():
    kwargs = dict(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7],
        code_commit_sha="refactorsha",
        report_artifact_path="report.html",
        report_artifact_sha256="f" * 64,
    )
    first = build_provenance_graph(**kwargs)
    second = build_provenance_graph(**kwargs)
    assert first == second
    assert profile_entity_id(2) == profile_entity_id(2)
    assert report_entity_id(2) == report_entity_id(2)
    assert simulation_run_entity_id(6) == simulation_run_entity_id(6)
    assert result_data_entity_id("a" * 64) == result_data_entity_id("a" * 64)


def test_provenance_graph_has_no_deprecated_scientific_term_as_semantic_type():
    graph = build_provenance_graph(
        scenario_run_id=2,
        source_result_data=FAKE_RESULT_DATA,
        source_simulation_run_ids=[6, 7],
        code_commit_sha="refactorsha",
        report_artifact_path="report.html",
        report_artifact_sha256="f" * 64,
    )
    encoded = json.dumps(graph)
    for deprecated in ("structural_fidelity", "structural_dynamics", "nstar", "N-star"):
        assert deprecated not in encoded


def test_write_provenance_graph_round_trips(tmp_path):
    graph = build_provenance_graph(
        scenario_run_id=0,
        source_result_data=[],
        source_simulation_run_ids=[],
        code_commit_sha="abc",
        report_artifact_path="report.html",
        report_artifact_sha256="e" * 64,
    )
    path = tmp_path / "provenance.jsonld"
    write_provenance_graph(path, graph)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == graph
