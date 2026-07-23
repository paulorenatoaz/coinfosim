"""PROV-O-compatible provenance JSON-LD export.

Builds a minimum provenance-ready graph (not a complete domain ontology, no
RDF/OWL runtime dependency) covering: recovered result-data entities, the
predictive cooperation profile entity, the report artifact entity, the
original-simulation / profile-recomputation / report-regeneration
activities, and the CoInfoSim software/commit agent(s). See
``docs/semantics/provenance_mapping.md`` for the role table.

Deterministic IDs are derived from scenario/simulation run IDs, artifact
hashes, and repo-relative paths -- never from absolute or machine-specific
paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_CONTEXT = {
    "coinfosim": "https://paulorenatoaz.github.io/coinfosim/ns#",
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def _urn(kind: str, key: str) -> str:
    return f"urn:coinfosim:{kind}:{key}"


def result_data_entity_id(sha256: str) -> str:
    return _urn("result-data", sha256[:16])


def simulation_run_entity_id(simulation_run_id: int) -> str:
    return _urn("simulation-run", f"{int(simulation_run_id):06d}")


def profile_entity_id(scenario_run_id: int) -> str:
    return _urn("predictive-cooperation-profile", f"{int(scenario_run_id):06d}")


def report_entity_id(scenario_run_id: int) -> str:
    return _urn("report", f"{int(scenario_run_id):06d}")


def commit_agent_id(commit_sha: str) -> str:
    return _urn("commit", commit_sha)


def build_provenance_graph(
    *,
    scenario_run_id: int,
    source_result_data: Sequence[Mapping[str, str]],
    source_simulation_run_ids: Sequence[int],
    code_commit_sha: Optional[str],
    report_artifact_path: str,
    report_artifact_sha256: str,
    recovered_source_commit_sha: Optional[str] = None,
    original_simulation_commit_sha: Optional[str] = None,
    report_regeneration_commit_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the minimum PROV-O-compatible provenance graph for one scenario.

    ``source_result_data`` items are ``{"path": <repo-relative>, "sha256": <hex>}``.
    Distinguishes, and never collapses, the original-simulation commit,
    the ``gh-pages`` recovery commit, the code commit that recomputed the
    profile, and the (possibly later) report-regeneration commit.
    """

    report_regeneration_commit_sha = report_regeneration_commit_sha or code_commit_sha

    result_data_entities: List[Dict[str, Any]] = []
    result_data_ids: List[str] = []
    for item in sorted(source_result_data, key=lambda item: str(item["path"])):
        entity_id = result_data_entity_id(str(item["sha256"]))
        result_data_ids.append(entity_id)
        entity: Dict[str, Any] = {
            "@id": entity_id,
            "@type": ["prov:Entity", "coinfosim:ResultData"],
            "coinfosim:relativePath": str(item["path"]),
            "coinfosim:sha256": str(item["sha256"]),
        }
        if recovered_source_commit_sha:
            entity["prov:wasAttributedTo"] = commit_agent_id(recovered_source_commit_sha)
        result_data_entities.append(entity)

    simulation_run_entities = [
        {
            "@id": simulation_run_entity_id(sim_id),
            "@type": ["prov:Entity", "coinfosim:SimulationRun"],
        }
        for sim_id in sorted(int(i) for i in source_simulation_run_ids)
    ]

    original_simulation_activity_id = _urn(
        "activity:original-simulation", f"{int(scenario_run_id):06d}"
    )
    original_simulation_activity: Dict[str, Any] = {
        "@id": original_simulation_activity_id,
        "@type": "prov:Activity",
        "prov:generated": [e["@id"] for e in simulation_run_entities],
    }
    if original_simulation_commit_sha:
        original_simulation_activity["prov:wasAssociatedWith"] = commit_agent_id(
            original_simulation_commit_sha
        )

    recomputation_activity_id = _urn(
        "activity:profile-recomputation", f"{int(scenario_run_id):06d}"
    )
    recomputation_activity: Dict[str, Any] = {
        "@id": recomputation_activity_id,
        "@type": "prov:Activity",
        "prov:used": result_data_ids,
    }
    if code_commit_sha:
        recomputation_activity["prov:wasAssociatedWith"] = commit_agent_id(code_commit_sha)

    profile_id = profile_entity_id(scenario_run_id)
    profile_entity = {
        "@id": profile_id,
        "@type": ["prov:Entity", "coinfosim:PredictiveCooperationProfile"],
        "prov:wasGeneratedBy": recomputation_activity_id,
        "prov:wasDerivedFrom": result_data_ids,
    }

    regeneration_activity_id = _urn(
        "activity:report-regeneration", f"{int(scenario_run_id):06d}"
    )
    regeneration_activity: Dict[str, Any] = {
        "@id": regeneration_activity_id,
        "@type": "prov:Activity",
        "prov:used": [profile_id],
    }
    if report_regeneration_commit_sha:
        regeneration_activity["prov:wasAssociatedWith"] = commit_agent_id(
            report_regeneration_commit_sha
        )

    report_id = report_entity_id(scenario_run_id)
    report_entity = {
        "@id": report_id,
        "@type": ["prov:Entity", "coinfosim:ReportArtifact"],
        "coinfosim:relativePath": str(report_artifact_path),
        "coinfosim:sha256": str(report_artifact_sha256),
        "prov:wasGeneratedBy": regeneration_activity_id,
        "prov:wasDerivedFrom": [profile_id] + result_data_ids,
    }

    agents: Dict[str, Dict[str, Any]] = {}
    for commit_sha in (
        code_commit_sha,
        recovered_source_commit_sha,
        original_simulation_commit_sha,
        report_regeneration_commit_sha,
    ):
        if commit_sha and commit_agent_id(commit_sha) not in agents:
            agents[commit_agent_id(commit_sha)] = {
                "@id": commit_agent_id(commit_sha),
                "@type": "prov:SoftwareAgent",
                "coinfosim:commitSha": commit_sha,
            }

    graph = (
        result_data_entities
        + simulation_run_entities
        + [
            original_simulation_activity,
            recomputation_activity,
            profile_entity,
            regeneration_activity,
            report_entity,
        ]
        + list(agents.values())
    )

    return {
        "@context": dict(_CONTEXT),
        "@graph": graph,
    }


def write_provenance_graph(path: str | Path, graph: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
