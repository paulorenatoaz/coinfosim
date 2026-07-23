"""Tests for the canonical CoInfoSim PROV document (Block P1).

Uses small synthetic evidence fixtures only -- no dataset load, no Monte
Carlo, no serialization concerns (see ``test_provenance_export.py`` for
serialization).
"""

from __future__ import annotations

import json

import pytest
from prov.model import (
    PROV_TYPE,
    ProvActivity,
    ProvAgent,
    ProvEntity,
    ProvGeneration,
    ProvUsage,
)

from coinfosim.provenance.evidence import (
    GAUSSIAN_ARM_ID,
    GMM_ARM_ID,
    REAL_ARM_ID,
    ArtifactEvidence,
    ExecutionEnvironmentEvidence,
    ProvenanceEvidence,
    SimulationArmEvidence,
)
from coinfosim.provenance.model import build_scenario_prov_document


def _artifact(role: str, seed: str) -> ArtifactEvidence:
    return ArtifactEvidence(
        path=f"output/reports/scenarios/000002/{role}.bin",
        sha256=(seed * 64)[:64],
        role=role,
    )


def _base_evidence(**overrides) -> ProvenanceEvidence:
    kwargs = dict(
        scenario_run_id=2,
        scenario_slug="occupancy_baseline",
        dataset_metadata={"name": "Occupancy Detection", "slug": "occupancy"},
        target_metadata={"name": "Occupancy"},
        split_metadata={"strategy": "original UCI training and test files"},
        preprocessing_metadata={"method": "zscore", "fit_scope": "training_reservoir_only"},
        experiment_configuration={"mode": "smoke", "base_seed": 0},
        classifier_configuration={"names": ["linear_svm"]},
        simulation_arms=[
            SimulationArmEvidence(REAL_ARM_ID, 6, _artifact("result-data:real_to_real", "a")),
            SimulationArmEvidence(
                GAUSSIAN_ARM_ID, 7, _artifact("result-data:single_gaussian_to_real", "b")
            ),
            SimulationArmEvidence(GMM_ARM_ID, 8, _artifact("result-data:gmm_to_real", "c")),
        ],
        gaussian_generator_metadata={"ridge": 0.1},
        gmm_generator_metadata={"n_components": 2},
        report_artifacts=[_artifact("scenario-report", "d")],
        current_code_revision="deadbeefcafebabe0000000000000000000000",
        execution_environment=ExecutionEnvironmentEvidence(
            stable_hash="abc123", python_version="3.12.3", platform="Linux-x86_64"
        ),
    )
    kwargs.update(overrides)
    return ProvenanceEvidence(**kwargs)


def _records_by_prov_type(document):
    by_type = {}
    for record in document.get_records():
        for prov_type in record.get_attribute(PROV_TYPE):
            by_type.setdefault(str(prov_type), []).append(record)
    return by_type


def _relations(document, cls):
    return [r for r in document.get_records() if isinstance(r, cls)]


# --------------------------------------------------------------------------- #
# Entity / Activity / Agent classes exist
# --------------------------------------------------------------------------- #
def test_expected_entity_activity_agent_classes_exist():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    expected_entities = {
        "coinfosim:SourceDataset",
        "coinfosim:PreparedDataset",
        "coinfosim:TrainingReservoir",
        "coinfosim:FixedRealTestSet",
        "coinfosim:TargetSpecification",
        "coinfosim:PreprocessingSpecification",
        "coinfosim:ExperimentConfiguration",
        "coinfosim:FittedSingleGaussianGenerator",
        "coinfosim:FittedGMMGenerator",
        "coinfosim:ResultData",
        "coinfosim:PredictiveCooperationProfile",
        "coinfosim:ReportArtifact",
        "coinfosim:CodeRevision",
        "coinfosim:ExecutionEnvironment",
    }
    for prov_type in expected_entities:
        assert prov_type in by_type, prov_type
        for record in by_type[prov_type]:
            assert isinstance(record, ProvEntity)

    expected_activities = {
        "coinfosim:DatasetPreparation",
        "coinfosim:FitSingleGaussian",
        "coinfosim:FitGMM",
        "coinfosim:RealSimulationRun",
        "coinfosim:SingleGaussianSimulationRun",
        "coinfosim:GMMSimulationRun",
        "coinfosim:ProfileComputation",
        "coinfosim:ReportGeneration",
    }
    for prov_type in expected_activities:
        assert prov_type in by_type, prov_type
        for record in by_type[prov_type]:
            assert isinstance(record, ProvActivity)

    agents = [r for r in document.get_records() if isinstance(r, ProvAgent)]
    assert len(agents) == 1
    assert str(next(iter(agents[0].get_attribute(PROV_TYPE)))) == "prov:SoftwareAgent"


def test_random_forest_calibration_artifact_only_when_evidenced():
    without_rf = build_scenario_prov_document(_base_evidence())
    assert "coinfosim:RandomForestCalibrationArtifact" not in _records_by_prov_type(
        without_rf
    )

    with_rf = build_scenario_prov_document(
        _base_evidence(
            random_forest_calibration=_artifact("random-forest-calibration", "e")
        )
    )
    by_type = _records_by_prov_type(with_rf)
    assert "coinfosim:RandomForestCalibrationArtifact" in by_type


def test_recovered_artifact_set_only_when_recovery_evidenced():
    without_recovery = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(without_recovery)
    assert "coinfosim:RecoveredArtifactSet" not in by_type
    assert "coinfosim:ArtifactRecovery" not in by_type

    with_recovery = build_scenario_prov_document(
        _base_evidence(
            recovery_source_commit="feedfacecafebeef0000000000000000000000",
            is_historical_regeneration=True,
        )
    )
    by_type = _records_by_prov_type(with_recovery)
    assert "coinfosim:RecoveredArtifactSet" in by_type
    assert "coinfosim:ArtifactRecovery" in by_type


# --------------------------------------------------------------------------- #
# Simulation run is an Activity; commit is an Entity, not an Agent
# --------------------------------------------------------------------------- #
def test_simulation_run_is_activity_not_entity():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    for prov_type in (
        "coinfosim:RealSimulationRun",
        "coinfosim:SingleGaussianSimulationRun",
        "coinfosim:GMMSimulationRun",
    ):
        records = by_type[prov_type]
        assert records
        for record in records:
            assert isinstance(record, ProvActivity)
            assert not isinstance(record, ProvEntity)


def test_commit_is_entity_not_agent():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    revisions = by_type["coinfosim:CodeRevision"]
    assert revisions
    for record in revisions:
        assert isinstance(record, ProvEntity)
        assert not isinstance(record, ProvAgent)
    agents = [r for r in document.get_records() if isinstance(r, ProvAgent)]
    assert all("CodeRevision" not in str(list(a.get_attribute(PROV_TYPE))) for a in agents)


# --------------------------------------------------------------------------- #
# Shared fixed test set / no-leakage invariant (Section 6.5)
# --------------------------------------------------------------------------- #
def _usage_pairs(document):
    result = []
    for record in _relations(document, ProvUsage):
        values = {str(k): v for k, v in record.formal_attributes}
        result.append((values["prov:activity"], values["prov:entity"]))
    return result


def test_all_three_simulation_activities_use_same_fixed_test_entity():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    fixed_test_entities = by_type["coinfosim:FixedRealTestSet"]
    assert len(fixed_test_entities) == 1
    fixed_test_id = fixed_test_entities[0].identifier

    usage_pairs = _usage_pairs(document)
    simulation_activity_ids = {
        record.identifier
        for prov_type in (
            "coinfosim:RealSimulationRun",
            "coinfosim:SingleGaussianSimulationRun",
            "coinfosim:GMMSimulationRun",
        )
        for record in by_type[prov_type]
    }
    assert len(simulation_activity_ids) == 3
    for activity_id in simulation_activity_ids:
        assert (activity_id, fixed_test_id) in usage_pairs


def test_gaussian_and_gmm_fitting_use_training_reservoir():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    training_reservoir_id = by_type["coinfosim:TrainingReservoir"][0].identifier
    usage_pairs = _usage_pairs(document)

    for prov_type in ("coinfosim:FitSingleGaussian", "coinfosim:FitGMM"):
        fitting_activity_id = by_type[prov_type][0].identifier
        assert (fitting_activity_id, training_reservoir_id) in usage_pairs


def test_gaussian_and_gmm_fitting_never_use_fixed_test_set():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    fixed_test_id = by_type["coinfosim:FixedRealTestSet"][0].identifier
    usage_pairs = _usage_pairs(document)

    for prov_type in ("coinfosim:FitSingleGaussian", "coinfosim:FitGMM"):
        fitting_activity_id = by_type[prov_type][0].identifier
        assert (fitting_activity_id, fixed_test_id) not in usage_pairs


# --------------------------------------------------------------------------- #
# Profile / report causal chain
# --------------------------------------------------------------------------- #
def test_profile_uses_all_three_result_data_entities():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    result_data_ids = {r.identifier for r in by_type["coinfosim:ResultData"]}
    assert len(result_data_ids) == 3

    profile_computation_id = by_type["coinfosim:ProfileComputation"][0].identifier
    usage_pairs = _usage_pairs(document)
    used_by_profile_computation = {
        entity for activity, entity in usage_pairs if activity == profile_computation_id
    }
    assert result_data_ids <= used_by_profile_computation


def test_report_derives_from_profile():
    document = build_scenario_prov_document(_base_evidence())
    by_type = _records_by_prov_type(document)
    profile_id = by_type["coinfosim:PredictiveCooperationProfile"][0].identifier
    report_generation_id = by_type["coinfosim:ReportGeneration"][0].identifier

    usage_pairs = _usage_pairs(document)
    assert (report_generation_id, profile_id) in usage_pairs

    generation_pairs = {
        (str(values["prov:entity"]), str(values["prov:activity"]))
        for values in (
            {str(k): v for k, v in r.formal_attributes}
            for r in _relations(document, ProvGeneration)
        )
    }
    report_ids = {r.identifier for r in by_type["coinfosim:ReportArtifact"]}
    assert report_ids
    for report_id in report_ids:
        assert (str(report_id), str(report_generation_id)) in generation_pairs


# --------------------------------------------------------------------------- #
# Determinism and path safety
# --------------------------------------------------------------------------- #
def test_identifiers_are_deterministic_for_same_inputs():
    first = build_scenario_prov_document(_base_evidence())
    second = build_scenario_prov_document(_base_evidence())
    first_ids = sorted(str(r.identifier) for r in first.get_records() if r.identifier)
    second_ids = sorted(str(r.identifier) for r in second.get_records() if r.identifier)
    assert first_ids == second_ids
    assert first_ids  # sanity: non-empty


def test_no_machine_specific_absolute_path_is_persisted():
    document = build_scenario_prov_document(_base_evidence())
    encoded = document.serialize(format="json")
    assert "/home/" not in encoded
    assert "/tmp/" not in encoded
    payload = json.loads(encoded)
    assert payload  # valid JSON


def test_historical_regeneration_never_associates_current_commit_with_simulation_run():
    document = build_scenario_prov_document(
        _base_evidence(
            current_code_revision="current0000000000000000000000000000000",
            original_code_revision=None,
            is_historical_regeneration=True,
        )
    )
    by_type = _records_by_prov_type(document)
    current_revision_id = None
    for record in document.get_records():
        if isinstance(record, ProvEntity) and "current0000" in str(record.identifier):
            current_revision_id = record.identifier
    assert current_revision_id is not None

    usage_pairs = _usage_pairs(document)
    simulation_activity_ids = {
        record.identifier
        for prov_type in (
            "coinfosim:RealSimulationRun",
            "coinfosim:SingleGaussianSimulationRun",
            "coinfosim:GMMSimulationRun",
            "coinfosim:DatasetPreparation",
            "coinfosim:FitSingleGaussian",
            "coinfosim:FitGMM",
        )
        for record in by_type[prov_type]
    }
    for activity_id in simulation_activity_ids:
        assert (activity_id, current_revision_id) not in usage_pairs

    # But profile computation and report generation DO run now.
    profile_computation_id = by_type["coinfosim:ProfileComputation"][0].identifier
    report_generation_id = by_type["coinfosim:ReportGeneration"][0].identifier
    assert (profile_computation_id, current_revision_id) in usage_pairs
    assert (report_generation_id, current_revision_id) in usage_pairs
