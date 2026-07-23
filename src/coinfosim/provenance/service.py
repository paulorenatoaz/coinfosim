"""Shared orchestration for emitting semantic manifest and provenance artifacts.

Both normal scenario execution (``run_dataset_anchored_scenario``) and report
regeneration (``regenerate_dataset_anchored_scenario``) call
:func:`emit_scenario_semantic_and_provenance_artifacts` -- there is no
separate provenance implementation for either path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from coinfosim.provenance.evidence import ProvenanceEvidence
from coinfosim.provenance.export import export_provenance_artifacts
from coinfosim.provenance.model import build_scenario_prov_document
from coinfosim.provenance.semantic_manifest import (
    build_semantic_manifest,
    to_repo_relative,
    write_semantic_manifest,
)


@dataclass(frozen=True)
class ScenarioProvenanceArtifacts:
    """Repository-relative paths to everything emitted for one scenario."""

    semantic_manifest: str
    provjson: str
    provn: str
    ttl: str
    png: Optional[str]
    pdf: Optional[str]
    graphviz_available: bool


def _semantic_manifest_payload(evidence: ProvenanceEvidence) -> Dict[str, Any]:
    """Map normalized ``ProvenanceEvidence`` onto ``build_semantic_manifest``'s contract."""

    dataset_id = str(
        evidence.dataset_metadata.get("slug")
        or evidence.dataset_metadata.get("name")
        or evidence.scenario_slug
    )
    classifier_ids = list(evidence.classifier_configuration.get("names", []))
    training_condition_ids = [arm.arm_id for arm in evidence.simulation_arms]
    sample_sizes = list(evidence.experiment_configuration.get("sample_sizes", []))
    source_simulation_run_ids = [
        arm.simulation_run_id for arm in evidence.simulation_arms
    ]
    source_result_data = [
        {"path": arm.result_data.path, "sha256": arm.result_data.sha256}
        for arm in evidence.simulation_arms
    ]
    report_artifact_hashes = {
        artifact.path: artifact.sha256 for artifact in evidence.report_artifacts
    }
    return build_semantic_manifest(
        scenario_run_id=evidence.scenario_run_id,
        scenario_slug=evidence.scenario_slug,
        dataset_id=dataset_id,
        classifier_ids=classifier_ids,
        training_condition_ids=training_condition_ids,
        sample_sizes=sample_sizes,
        source_simulation_run_ids=source_simulation_run_ids,
        source_result_data=source_result_data,
        code_commit_sha=evidence.current_code_revision,
        recovered_source_commit_sha=evidence.recovery_source_commit,
        original_simulation_commit_sha=evidence.original_code_revision,
        report_artifact_hashes=report_artifact_hashes,
    )


def emit_scenario_semantic_and_provenance_artifacts(
    evidence: ProvenanceEvidence,
    *,
    scenario_dir: Path,
    repo_root: Path,
) -> ScenarioProvenanceArtifacts:
    """Write the semantic manifest and canonical provenance artifacts for one scenario.

    Never runs a simulation and never changes scientific result data: this
    only normalizes already-collected evidence into
    ``semantic_manifest.json`` and ``provenance.provjson``/``.provn``/``.ttl``
    (plus ``.png``/``.pdf`` when Graphviz is available). Graphviz being
    unavailable is not an error here -- see
    :func:`coinfosim.provenance.export.export_provenance_artifacts`.
    """

    scenario_dir = Path(scenario_dir)
    repo_root = Path(repo_root)

    manifest = _semantic_manifest_payload(evidence)
    manifest_path = scenario_dir / "semantic_manifest.json"
    write_semantic_manifest(manifest_path, manifest)

    document = build_scenario_prov_document(evidence)
    artifact_set = export_provenance_artifacts(document, scenario_dir, stem="provenance")

    return ScenarioProvenanceArtifacts(
        semantic_manifest=to_repo_relative(manifest_path, repo_root),
        provjson=to_repo_relative(artifact_set.provjson, repo_root),
        provn=to_repo_relative(artifact_set.provn, repo_root),
        ttl=to_repo_relative(artifact_set.ttl, repo_root),
        png=to_repo_relative(artifact_set.png, repo_root) if artifact_set.png else None,
        pdf=to_repo_relative(artifact_set.pdf, repo_root) if artifact_set.pdf else None,
        graphviz_available=artifact_set.graphviz_available,
    )
