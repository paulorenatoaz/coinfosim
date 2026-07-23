"""Canonical ``prov.model.ProvDocument`` builder for CoInfoSim.

Exactly one canonical provenance model is built here from normalized
:class:`~coinfosim.provenance.evidence.ProvenanceEvidence`. All serialized
provenance artifacts (PROV-JSON, PROV-N, PROV-O/Turtle, PNG, PDF) must be
derived from the ``ProvDocument`` returned by
:func:`build_scenario_prov_document` -- see
``docs/semantics/provenance_mapping.md`` for the role table.

A simulation run is represented as a ``prov:Activity``, never as an
``prov:Entity``; a Git commit is represented as a ``prov:Entity`` (a
:class:`coinfosim:CodeRevision`), never as an ``prov:Agent``.
"""

from __future__ import annotations

from typing import Optional

from prov.identifier import Namespace, QualifiedName
from prov.model import ProvDocument

from coinfosim.provenance.evidence import (
    GAUSSIAN_ARM_ID,
    GMM_ARM_ID,
    REAL_ARM_ID,
    ProvenanceEvidence,
)

COINFOSIM_NAMESPACE_URI = "https://paulorenatoaz.github.io/coinfosim/ns#"
COINFOSIM_URN_PREFIX = "urn:coinfosim:"
SOFTWARE_AGENT_ID = "software:coinfosim"

_ARM_SIMULATION_ACTIVITY_TYPE = {
    REAL_ARM_ID: "RealSimulationRun",
    GAUSSIAN_ARM_ID: "SingleGaussianSimulationRun",
    GMM_ARM_ID: "GMMSimulationRun",
}


# --------------------------------------------------------------------------- #
# Deterministic identifier family (task plan Section 8) -- do not invent a
# second identifier strategy anywhere else in this package.
# --------------------------------------------------------------------------- #
def scenario_identifier(scenario_run_id: int, kind: str) -> str:
    return f"{COINFOSIM_URN_PREFIX}scenario:{int(scenario_run_id):06d}:{kind}"


def simulation_identifier(simulation_run_id: int, arm: str) -> str:
    return f"{COINFOSIM_URN_PREFIX}simulation:{int(simulation_run_id):06d}:{arm}"


def artifact_identifier(sha256: str, role: str) -> str:
    return f"{COINFOSIM_URN_PREFIX}artifact:{sha256[:16]}:{role}"


def revision_identifier(commit_sha: str) -> str:
    return f"{COINFOSIM_URN_PREFIX}revision:{commit_sha}"


def environment_identifier(stable_hash: str) -> str:
    return f"{COINFOSIM_URN_PREFIX}environment:{stable_hash}"


def software_agent_identifier() -> str:
    return f"{COINFOSIM_URN_PREFIX}{SOFTWARE_AGENT_ID}"


class _GraphBuilder:
    """Internal helper carrying the namespaces used while building one document."""

    def __init__(self, document: ProvDocument) -> None:
        self.document = document
        self.coinfosim = document.add_namespace(
            Namespace("coinfosim", COINFOSIM_NAMESPACE_URI)
        )
        self.urn = document.add_namespace(Namespace("cfurn", COINFOSIM_URN_PREFIX))

    def _qname(self, identifier: str) -> QualifiedName:
        assert identifier.startswith(COINFOSIM_URN_PREFIX)
        return self.urn[identifier[len(COINFOSIM_URN_PREFIX) :]]

    def coinfosim_type(self, local_name: str) -> QualifiedName:
        return self.coinfosim[local_name]

    def entity(self, identifier: str, *, type_: str, label: str):
        return self.document.entity(
            self._qname(identifier),
            other_attributes={
                "prov:type": self.coinfosim_type(type_),
                "prov:label": label,
            },
        )

    def activity(self, identifier: str, *, type_: str, label: str):
        return self.document.activity(
            self._qname(identifier),
            other_attributes={
                "prov:type": self.coinfosim_type(type_),
                "prov:label": label,
            },
        )

    def code_revision(self, commit_sha: str):
        return self.document.entity(
            self._qname(revision_identifier(commit_sha)),
            other_attributes={
                "prov:type": self.coinfosim_type("CodeRevision"),
                "prov:label": f"Git commit {commit_sha}",
            },
        )

    def execution_environment(self, stable_hash: str, *, label: str):
        return self.document.entity(
            self._qname(environment_identifier(stable_hash)),
            other_attributes={
                "prov:type": self.coinfosim_type("ExecutionEnvironment"),
                "prov:label": label,
            },
        )

    def software_agent(self):
        return self.document.agent(
            self._qname(software_agent_identifier()),
            other_attributes={
                "prov:type": "prov:SoftwareAgent",
                "prov:label": "CoInfoSim",
            },
        )


def build_scenario_prov_document(evidence: ProvenanceEvidence) -> ProvDocument:
    """Build the one canonical ``ProvDocument`` for a dataset-anchored scenario.

    Produces scenario/arm-level provenance (never one activity per Monte
    Carlo replication/sample size/classifier fit): three simulation-run
    activities, one dataset-preparation activity, up to two generator-fitting
    activities, one profile-computation activity, one report-generation
    activity, and -- only when recovery evidence is present -- one
    artifact-recovery activity.
    """

    document = ProvDocument()
    g = _GraphBuilder(document)
    run_id = evidence.scenario_run_id

    agent = g.software_agent()

    current_revision = (
        g.code_revision(evidence.current_code_revision)
        if evidence.current_code_revision
        else None
    )
    original_revision = (
        g.code_revision(evidence.original_code_revision)
        if evidence.original_code_revision
        else None
    )
    recovery_revision = (
        g.code_revision(evidence.recovery_source_commit)
        if evidence.recovery_source_commit
        else None
    )
    environment = (
        g.execution_environment(
            evidence.execution_environment.stable_hash,
            label=(
                f"Python {evidence.execution_environment.python_version} on "
                f"{evidence.execution_environment.platform}"
            ),
        )
        if evidence.execution_environment
        else None
    )

    # Historical (already-executed) computational activities are associated
    # with the original revision only when it was actually persisted; the
    # current commit/environment is never guessed onto them. Activities that
    # actually execute now (profile computation, report generation, artifact
    # recovery) always use the current commit/environment when known.
    upstream_revision = (
        original_revision if evidence.is_historical_regeneration else current_revision
    )
    upstream_environment = None if evidence.is_historical_regeneration else environment

    def _associate(activity, revision) -> None:
        document.wasAssociatedWith(activity, agent)
        if revision is not None:
            document.used(activity, revision)

    def _use_environment(activity, env) -> None:
        if env is not None:
            document.used(activity, env)

    # ----------------------------------------------------------------- #
    # Dataset preparation
    # ----------------------------------------------------------------- #
    dataset_name = evidence.dataset_metadata.get("name") or evidence.scenario_slug
    source_dataset = g.entity(
        scenario_identifier(run_id, "source-dataset"),
        type_="SourceDataset",
        label=f"{dataset_name} source dataset",
    )
    target_spec = g.entity(
        scenario_identifier(run_id, "target-specification"),
        type_="TargetSpecification",
        label=f"{dataset_name} target specification",
    )
    preprocessing_spec = g.entity(
        scenario_identifier(run_id, "preprocessing-specification"),
        type_="PreprocessingSpecification",
        label=f"{dataset_name} preprocessing specification",
    )

    dataset_preparation = g.activity(
        scenario_identifier(run_id, "dataset-preparation"),
        type_="DatasetPreparation",
        label=f"{dataset_name} dataset preparation",
    )
    _associate(dataset_preparation, upstream_revision)
    _use_environment(dataset_preparation, upstream_environment)
    document.used(dataset_preparation, source_dataset)
    document.used(dataset_preparation, target_spec)
    document.used(dataset_preparation, preprocessing_spec)

    prepared_dataset = g.entity(
        scenario_identifier(run_id, "prepared-dataset"),
        type_="PreparedDataset",
        label=f"{dataset_name} prepared dataset",
    )
    training_reservoir = g.entity(
        scenario_identifier(run_id, "training-reservoir"),
        type_="TrainingReservoir",
        label=f"{dataset_name} training reservoir",
    )
    fixed_test_set = g.entity(
        scenario_identifier(run_id, "fixed-real-test-set"),
        type_="FixedRealTestSet",
        label=f"{dataset_name} fixed real test set",
    )
    for generated in (prepared_dataset, training_reservoir, fixed_test_set):
        document.wasGeneratedBy(generated, dataset_preparation)

    experiment_configuration = g.entity(
        scenario_identifier(run_id, "experiment-configuration"),
        type_="ExperimentConfiguration",
        label=f"{dataset_name} experiment configuration",
    )

    rf_calibration = None
    if evidence.random_forest_calibration is not None:
        rf_calibration = g.entity(
            artifact_identifier(
                evidence.random_forest_calibration.sha256,
                evidence.random_forest_calibration.role,
            ),
            type_="RandomForestCalibrationArtifact",
            label="Random Forest calibration artifact",
        )

    # ----------------------------------------------------------------- #
    # Generator fitting (training reservoir only -- never the fixed test set)
    # ----------------------------------------------------------------- #
    fitted_gaussian = None
    if evidence.gaussian_generator_metadata is not None:
        fit_gaussian = g.activity(
            scenario_identifier(run_id, "fit-single-gaussian"),
            type_="FitSingleGaussian",
            label=f"{dataset_name} single Gaussian fitting",
        )
        _associate(fit_gaussian, upstream_revision)
        _use_environment(fit_gaussian, upstream_environment)
        document.used(fit_gaussian, training_reservoir)
        fitted_gaussian = g.entity(
            scenario_identifier(run_id, "fitted-single-gaussian-generator"),
            type_="FittedSingleGaussianGenerator",
            label=f"{dataset_name} fitted single Gaussian generator",
        )
        document.wasGeneratedBy(fitted_gaussian, fit_gaussian)

    fitted_gmm = None
    if evidence.gmm_generator_metadata is not None:
        fit_gmm = g.activity(
            scenario_identifier(run_id, "fit-gmm"),
            type_="FitGMM",
            label=f"{dataset_name} GMM fitting",
        )
        _associate(fit_gmm, upstream_revision)
        _use_environment(fit_gmm, upstream_environment)
        document.used(fit_gmm, training_reservoir)
        fitted_gmm = g.entity(
            scenario_identifier(run_id, "fitted-gmm-generator"),
            type_="FittedGMMGenerator",
            label=f"{dataset_name} fitted GMM generator",
        )
        document.wasGeneratedBy(fitted_gmm, fit_gmm)

    # ----------------------------------------------------------------- #
    # Simulation runs (all three share the same fixed real test set)
    # ----------------------------------------------------------------- #
    arm_train_entity = {
        REAL_ARM_ID: training_reservoir,
        GAUSSIAN_ARM_ID: fitted_gaussian,
        GMM_ARM_ID: fitted_gmm,
    }
    result_data_by_arm = {}
    for arm_evidence in evidence.simulation_arms:
        arm_id = arm_evidence.arm_id
        activity_type = _ARM_SIMULATION_ACTIVITY_TYPE[arm_id]
        simulation_activity = g.activity(
            simulation_identifier(arm_evidence.simulation_run_id, arm_id),
            type_=activity_type,
            label=f"{dataset_name} {arm_id} simulation run",
        )
        _associate(simulation_activity, upstream_revision)
        _use_environment(simulation_activity, upstream_environment)

        train_entity = arm_train_entity.get(arm_id)
        if train_entity is not None:
            document.used(simulation_activity, train_entity)
        document.used(simulation_activity, fixed_test_set)
        document.used(simulation_activity, experiment_configuration)
        if rf_calibration is not None:
            document.used(simulation_activity, rf_calibration)

        result_data = g.entity(
            artifact_identifier(
                arm_evidence.result_data.sha256, arm_evidence.result_data.role
            ),
            type_="ResultData",
            label=f"{dataset_name} {arm_id} result data",
        )
        document.wasGeneratedBy(result_data, simulation_activity)
        result_data_by_arm[arm_id] = result_data

    # ----------------------------------------------------------------- #
    # Profile computation and report generation (always run now)
    # ----------------------------------------------------------------- #
    profile = None
    if result_data_by_arm:
        profile_computation = g.activity(
            scenario_identifier(run_id, "profile-computation"),
            type_="ProfileComputation",
            label=f"{dataset_name} predictive cooperation profile computation",
        )
        _associate(profile_computation, current_revision)
        _use_environment(profile_computation, environment)
        for result_data in result_data_by_arm.values():
            document.used(profile_computation, result_data)

        profile = g.entity(
            scenario_identifier(run_id, "predictive-cooperation-profile"),
            type_="PredictiveCooperationProfile",
            label=f"{dataset_name} predictive cooperation profile",
        )
        document.wasGeneratedBy(profile, profile_computation)

    if profile is not None and evidence.report_artifacts:
        report_generation = g.activity(
            scenario_identifier(run_id, "report-generation"),
            type_="ReportGeneration",
            label=f"{dataset_name} report generation",
        )
        _associate(report_generation, current_revision)
        _use_environment(report_generation, environment)
        document.used(report_generation, profile)
        for report_artifact in evidence.report_artifacts:
            report_entity = g.entity(
                artifact_identifier(report_artifact.sha256, report_artifact.role),
                type_="ReportArtifact",
                label=f"{dataset_name} {report_artifact.role}",
            )
            document.wasGeneratedBy(report_entity, report_generation)

    # ----------------------------------------------------------------- #
    # Historical artifact recovery (only when recovery evidence is present)
    # ----------------------------------------------------------------- #
    if evidence.recovery_source_commit:
        artifact_recovery = g.activity(
            scenario_identifier(run_id, "artifact-recovery"),
            type_="ArtifactRecovery",
            label=f"{dataset_name} historical artifact recovery",
        )
        _associate(artifact_recovery, current_revision)
        _use_environment(artifact_recovery, environment)
        if recovery_revision is not None:
            document.used(artifact_recovery, recovery_revision)
        recovered_set = g.entity(
            scenario_identifier(run_id, "recovered-artifact-set"),
            type_="RecoveredArtifactSet",
            label=f"{dataset_name} recovered gh-pages artifact set",
        )
        document.wasGeneratedBy(recovered_set, artifact_recovery)
        for result_data in result_data_by_arm.values():
            document.wasDerivedFrom(result_data, recovered_set)

    return document
