"""Semantic manifest and canonical W3C PROV provenance export for CoInfoSim.

The canonical provenance model is exactly one ``prov.model.ProvDocument``,
built by :func:`coinfosim.provenance.model.build_scenario_prov_document` from
normalized :class:`coinfosim.provenance.evidence.ProvenanceEvidence`, and
exported to disk by
:func:`coinfosim.provenance.export.export_provenance_artifacts`.
``coinfosim.provenance.jsonld`` is retained only as a legacy-compatibility
module for historical ``provenance.jsonld`` publications.
"""

from __future__ import annotations

from coinfosim.provenance.evidence import (
    ArtifactEvidence,
    ExecutionEnvironmentEvidence,
    ProvenanceEvidence,
    SimulationArmEvidence,
    collect_execution_environment,
    collect_runtime_provenance_evidence,
)
from coinfosim.provenance.export import (
    ProvenanceArtifactSet,
    export_provenance_artifacts,
)
from coinfosim.provenance.jsonld import build_provenance_graph, write_provenance_graph
from coinfosim.provenance.model import build_scenario_prov_document
from coinfosim.provenance.semantic_manifest import (
    build_semantic_manifest,
    sha256_of_file,
    to_repo_relative,
    write_semantic_manifest,
)
from coinfosim.provenance.service import (
    ScenarioProvenanceArtifacts,
    emit_scenario_semantic_and_provenance_artifacts,
)

__all__ = [
    # Canonical PROV model and export (current)
    "ArtifactEvidence",
    "ExecutionEnvironmentEvidence",
    "ProvenanceEvidence",
    "SimulationArmEvidence",
    "collect_execution_environment",
    "collect_runtime_provenance_evidence",
    "build_scenario_prov_document",
    "ProvenanceArtifactSet",
    "export_provenance_artifacts",
    "ScenarioProvenanceArtifacts",
    "emit_scenario_semantic_and_provenance_artifacts",
    # Semantic manifest
    "build_semantic_manifest",
    "sha256_of_file",
    "to_repo_relative",
    "write_semantic_manifest",
    # Legacy JSON-LD compatibility (see coinfosim.provenance.jsonld)
    "build_provenance_graph",
    "write_provenance_graph",
]
