"""Tests for persisted predictive-profile schema versioning and legacy readers."""

from __future__ import annotations

import json

import pytest

from coinfosim.results.profile_schema import (
    SCHEMA_VERSION,
    ProfileSchemaError,
    load_predictive_profile_payload,
    upgrade_predictive_profile_payload,
)
from coinfosim.runs.registry import ScenarioRunRecord, SimulationRunRecord
from coinfosim.semantics import canonical_key_to_id, vocabulary_version


def test_schema_version_constant_is_3():
    assert SCHEMA_VERSION == 3


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_load_canonical_scenario_payload():
    report_data = {
        "predictive_cooperation_profile": {
            "schema_version": 3,
            "semantic_vocabulary_version": "1.0.0",
            "final_summary": [],
        }
    }
    loaded = load_predictive_profile_payload(report_data, level="scenario")
    assert loaded.schema_version == 3
    assert loaded.is_canonical is True
    assert loaded.is_historical is False


def test_load_legacy_schema2_scenario_payload():
    report_data = {
        "structural_fidelity": {
            "schema_version": 2,
            "reversal_fidelity_series": [],
            "final_summary": [],
        }
    }
    loaded = load_predictive_profile_payload(report_data, level="scenario")
    assert loaded.schema_version == 2
    assert loaded.is_canonical is False
    assert loaded.is_historical is True


def test_load_legacy_schema1_scenario_payload_with_composite():
    report_data = {
        "structural_fidelity": {
            "schema_version": 1,
            "final_summary": [{"nstar_similarity": 0.8, "crossing_jaccard": 1.0}],
        }
    }
    loaded = load_predictive_profile_payload(report_data, level="scenario")
    assert loaded.schema_version == 1
    assert loaded.is_historical is True


def test_load_missing_payload_raises():
    with pytest.raises(ProfileSchemaError):
        load_predictive_profile_payload({}, level="scenario")


def test_load_rejects_unknown_level():
    with pytest.raises(ProfileSchemaError):
        load_predictive_profile_payload({"predictive_cooperation_profile": {}}, level="bogus")


# --------------------------------------------------------------------------- #
# Upgrading
# --------------------------------------------------------------------------- #
def test_upgrade_schema3_is_a_passthrough():
    payload = {"schema_version": 3, "final_summary": []}
    loaded = load_predictive_profile_payload(
        {"predictive_cooperation_profile": payload}, level="scenario"
    )
    result = upgrade_predictive_profile_payload(loaded)
    assert result.upgraded is False
    assert result.payload == payload


def test_upgrade_schema2_scenario_renames_fields_losslessly():
    payload = {
        "schema_version": 2,
        "reversal_fidelity_series": [
            {
                "classifier": "linear_svm",
                "arm": "arm",
                "n_prefix": 4,
                "n_reference_reversal_pairs": 1,
                "n_arm_reversal_pairs": 1,
                "n_shared_reversal_pairs": 1,
                "n_union_reversal_pairs": 1,
                "reversal_existence_agreement": 1.0,
                "mean_log2_reversal_distance": 0.0,
                "reversal_sample_size_similarity": 1.0,
                "status": "ok",
            }
        ],
        "final_summary": [
            {
                "classifier": "linear_svm",
                "arm": "arm",
                "n_reference_reversal_pairs": 1,
                "n_arm_reversal_pairs": 1,
                "n_shared_reversal_pairs": 1,
                "n_union_reversal_pairs": 1,
            }
        ],
    }
    loaded = load_predictive_profile_payload(
        {"structural_fidelity": payload}, level="scenario"
    )
    result = upgrade_predictive_profile_payload(loaded)
    assert result.upgraded is True
    assert result.non_upgradable_fields == []
    upgraded = result.payload
    assert upgraded["schema_version"] == 3
    assert upgraded["semantic_vocabulary_version"] == vocabulary_version()
    assert upgraded["semantic_type"] == canonical_key_to_id("predictive_cooperation_profile")
    assert "reversal_fidelity_series" not in upgraded
    row = upgraded["reversal_agreement_series"][0]
    assert row["reference_reversal_pair_count"] == 1
    assert row["arm_reversal_pair_count"] == 1
    assert row["shared_reversal_pair_count"] == 1
    assert row["union_reversal_pair_count"] == 1
    assert "n_reference_reversal_pairs" not in row
    summary_row = upgraded["final_summary"][0]
    assert summary_row["reference_reversal_pair_count"] == 1
    # numeric values themselves are untouched (lossless rename, no recomputation)
    assert row["reversal_existence_agreement"] == 1.0
    assert json.dumps(upgraded, allow_nan=False)  # strict-JSON safe


def test_upgrade_schema2_simulation_renames_effective_winner_container():
    payload = {
        "schema_version": 2,
        "subset_catalog": [[0], [1]],
        "sample_sizes": [2, 4],
        "classifiers": {
            "clf": {
                "effective_winner_pairs_by_n": {
                    "2": [{"i": 0, "j": 1, "outcome": -1}],
                    "4": [{"i": 0, "j": 1, "outcome": 1}],
                },
                "winner_reversal_events": [{"i": 0, "j": 1, "n_reversal": 4}],
            }
        },
    }
    loaded = load_predictive_profile_payload(
        {"structural_dynamics": payload}, level="simulation"
    )
    result = upgrade_predictive_profile_payload(loaded)
    assert result.upgraded is True
    upgraded = result.payload
    assert upgraded["schema_version"] == 3
    assert upgraded["semantic_type"] == canonical_key_to_id("pairwise_profile_dynamics")
    classifier = upgraded["classifiers"]["clf"]
    assert "effective_winner_pairs_by_n" not in classifier
    assert classifier["effective_winner_relations_by_n"] == [
        {"n_per_class": 2, "relations": [{"i": 0, "j": 1, "outcome": -1}]},
        {"n_per_class": 4, "relations": [{"i": 0, "j": 1, "outcome": 1}]},
    ]
    assert classifier["reversal_matrices_by_prefix"] == []
    assert json.dumps(upgraded, allow_nan=False)


def test_upgrade_schema1_refuses_composite_conversion():
    payload = {
        "schema_version": 1,
        "final_summary": [
            {
                "classifier": "linear_svm",
                "arm": "arm",
                "nstar_similarity": 0.8,
                "crossing_jaccard": 1.0,
                "timing_similarity": 0.6,
            }
        ],
    }
    loaded = load_predictive_profile_payload(
        {"structural_fidelity": payload}, level="scenario"
    )
    with pytest.raises(ProfileSchemaError, match="non-decomposable composite"):
        upgrade_predictive_profile_payload(loaded)


def test_upgrade_schema1_error_lists_non_upgradable_fields():
    payload = {
        "schema_version": 1,
        "final_summary": [{"nstar_similarity": 0.8, "crossing_jaccard": 1.0}],
    }
    loaded = load_predictive_profile_payload(
        {"structural_fidelity": payload}, level="scenario"
    )
    with pytest.raises(ProfileSchemaError) as excinfo:
        upgrade_predictive_profile_payload(loaded)
    assert "crossing_jaccard" in str(excinfo.value)
    assert "nstar_similarity" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Registry backward compatibility
# --------------------------------------------------------------------------- #
def test_scenario_run_record_from_dict_ignores_unknown_future_keys():
    record = ScenarioRunRecord.from_dict(
        {
            "scenario_run_id": 0,
            "scenario_slug": "occupancy",
            "scenario_name": "Occupancy",
            "scenario_family": "dataset",
            "question": "?",
            "mode": "smoke",
            "some_future_field_not_yet_invented": "value",
        }
    )
    assert record.scenario_run_id == 0
    assert record.semantic_schema_version is None


def test_scenario_run_record_round_trips_new_optional_semantic_fields():
    record = ScenarioRunRecord(
        scenario_run_id=1,
        scenario_slug="occupancy",
        scenario_name="Occupancy",
        scenario_family="dataset",
        question="?",
        mode="full",
        semantic_schema_version="1.0.0",
        semantic_manifest_path="output/reports/scenarios/000001/semantic_manifest.json",
        provenance_path="output/reports/scenarios/000001/provenance.jsonld",
        scientific_object_type="coinfosim:PredictiveCooperationProfile",
    )
    restored = ScenarioRunRecord.from_dict(record.to_dict())
    assert restored.semantic_schema_version == "1.0.0"
    assert restored.provenance_path == "output/reports/scenarios/000001/provenance.jsonld"
    assert restored.scientific_object_type == "coinfosim:PredictiveCooperationProfile"


def test_simulation_run_record_from_dict_ignores_unknown_future_keys_and_defaults_semantic_fields():
    record = SimulationRunRecord.from_dict(
        {
            "simulation_run_id": 0,
            "simulation_slug": "occupancy_real_data",
            "simulation_family": "real_dataset",
            "mode": "smoke",
            "not_yet_invented": True,
        }
    )
    assert record.simulation_run_id == 0
    assert record.semantic_manifest_path is None
