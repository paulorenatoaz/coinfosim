"""Evidence normalization for the canonical CoInfoSim PROV document.

These types and collectors normalize already-persisted/collected facts
(dataset metadata, split/preprocessing metadata, simulation run ids, result
hashes, commit SHAs, execution environment) into the shape consumed by
:func:`coinfosim.provenance.model.build_scenario_prov_document`. Nothing here
implements scientific logic: values are copied/normalized, never recomputed.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

# The three dataset-anchored training conditions (see task plan Section 4).
# Kept as plain string constants here (not imported from the scenario runner)
# to avoid a provenance -> scenarios import cycle; the values are fixed and
# already used verbatim throughout the codebase.
REAL_ARM_ID = "real_to_real"
GAUSSIAN_ARM_ID = "single_gaussian_to_real"
GMM_ARM_ID = "gmm_to_real"


@dataclass(frozen=True)
class ArtifactEvidence:
    """One content-addressed, repo-relative persisted artifact."""

    path: str
    sha256: str
    role: str


@dataclass(frozen=True)
class SimulationArmEvidence:
    """Persisted evidence for one dataset-anchored simulation arm."""

    arm_id: str
    simulation_run_id: int
    result_data: ArtifactEvidence
    sampler_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionEnvironmentEvidence:
    """Evidence describing the environment that executed an activity."""

    stable_hash: str
    python_version: str
    platform: str


@dataclass(frozen=True)
class ProvenanceEvidence:
    """Normalized evidence used to build the canonical scenario ``ProvDocument``.

    Carries only already-persisted/collected facts; never derives new
    scientific metadata. ``simulation_arms`` must contain exactly the arms for
    which evidence is available (normally all three: :data:`REAL_ARM_ID`,
    :data:`GAUSSIAN_ARM_ID`, :data:`GMM_ARM_ID`).
    """

    scenario_run_id: int
    scenario_slug: str
    dataset_metadata: Mapping[str, Any]
    target_metadata: Mapping[str, Any] = field(default_factory=dict)
    split_metadata: Mapping[str, Any] = field(default_factory=dict)
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)
    experiment_configuration: Mapping[str, Any] = field(default_factory=dict)
    classifier_configuration: Mapping[str, Any] = field(default_factory=dict)
    simulation_arms: Sequence[SimulationArmEvidence] = field(default_factory=tuple)
    gaussian_generator_metadata: Optional[Mapping[str, Any]] = None
    gmm_generator_metadata: Optional[Mapping[str, Any]] = None
    report_artifacts: Sequence[ArtifactEvidence] = field(default_factory=tuple)
    random_forest_calibration: Optional[ArtifactEvidence] = None
    current_code_revision: Optional[str] = None
    original_code_revision: Optional[str] = None
    recovery_source_commit: Optional[str] = None
    execution_environment: Optional[ExecutionEnvironmentEvidence] = None
    # When True, this evidence describes a regeneration from historical/
    # recovered data: the current commit/environment must never be
    # associated with the (historical) dataset-preparation, generator-fitting
    # or simulation-run activities -- only with the activities actually
    # executed now (profile recomputation, report generation, recovery).
    is_historical_regeneration: bool = False


def collect_runtime_provenance_evidence(
    *,
    scenario_run_id: int,
    scenario_slug: str,
    dataset_metadata: Mapping[str, Any],
    target_metadata: Mapping[str, Any],
    split_metadata: Mapping[str, Any],
    preprocessing_metadata: Mapping[str, Any],
    experiment_configuration: Mapping[str, Any],
    classifier_configuration: Mapping[str, Any],
    simulation_arms: Sequence[SimulationArmEvidence],
    report_artifacts: Sequence[ArtifactEvidence],
    gaussian_generator_metadata: Optional[Mapping[str, Any]] = None,
    gmm_generator_metadata: Optional[Mapping[str, Any]] = None,
    random_forest_calibration: Optional[ArtifactEvidence] = None,
    code_commit_sha: Optional[str] = None,
) -> ProvenanceEvidence:
    """Normalize evidence collected while a normal scenario run just completed.

    All computational activities (dataset preparation, generator fitting,
    simulation runs) as well as profile computation/report generation are
    associated with the current commit and the current execution
    environment, since everything actually ran now.
    """

    return ProvenanceEvidence(
        scenario_run_id=scenario_run_id,
        scenario_slug=scenario_slug,
        dataset_metadata=dataset_metadata,
        target_metadata=target_metadata,
        split_metadata=split_metadata,
        preprocessing_metadata=preprocessing_metadata,
        experiment_configuration=experiment_configuration,
        classifier_configuration=classifier_configuration,
        simulation_arms=simulation_arms,
        gaussian_generator_metadata=gaussian_generator_metadata,
        gmm_generator_metadata=gmm_generator_metadata,
        report_artifacts=report_artifacts,
        random_forest_calibration=random_forest_calibration,
        current_code_revision=code_commit_sha,
        execution_environment=collect_execution_environment(),
        is_historical_regeneration=False,
    )


def collect_execution_environment() -> ExecutionEnvironmentEvidence:
    """Collect the current execution environment as normalized evidence.

    Only records a small, stable fingerprint (Python version + platform) --
    never an absolute or machine-specific path.
    """

    python_version = sys.version.split()[0]
    platform_label = f"{platform.system()}-{platform.machine()}"
    fingerprint = f"{python_version}|{platform_label}"
    stable_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return ExecutionEnvironmentEvidence(
        stable_hash=stable_hash,
        python_version=python_version,
        platform=platform_label,
    )
