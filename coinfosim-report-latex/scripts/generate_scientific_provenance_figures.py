#!/usr/bin/env python3
"""Generate the report's scientific W3C PROV Graphviz figures.

This script does **not** implement a second provenance model. For each
target scenario it:

1. reuses already-persisted evidence (``scenario.json``, ``semantic_manifest.json``,
   and the persisted simulation ``result_data`` files) -- no Monte Carlo is
   rerun and no scientific value is recomputed;
2. normalizes that evidence into a
   :class:`coinfosim.provenance.evidence.ProvenanceEvidence` using the
   existing project collector (:func:`collect_persisted_provenance_evidence`);
3. applies a non-mutating ``dataclasses.replace(...)`` projection that clears
   only the five *engineering-only* fields the academic report intentionally
   excludes (current/original code revision, recovery source commit,
   execution environment, and the historical-regeneration flag) -- every
   scientific field (dataset, target, split, preprocessing, generator
   fitting, simulation arms, results, profile, report, RF calibration) is
   preserved unchanged;
4. builds the one canonical ``ProvDocument`` via
   :func:`build_scenario_prov_document` (the same function used by normal
   scenario runs);
5. renders it with :func:`prov.dot.prov_to_dot` / Graphviz.

The resulting PNG/PDF pair is a *visual projection* of the canonical
provenance model for reporting purposes only; it never alters the
machine-readable ``provenance.provjson``/``.provn``/``.ttl`` persisted next to
each scenario.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from prov.dot import prov_to_dot

from coinfosim.provenance import (
    ArtifactEvidence,
    ProvenanceEvidence,
    SimulationArmEvidence,
    build_scenario_prov_document,
    collect_persisted_provenance_evidence,
    sha256_of_file,
    to_repo_relative,
)
from coinfosim.provenance.evidence import GAUSSIAN_ARM_ID, GMM_ARM_ID, REAL_ARM_ID
from coinfosim.results.persistence import load_simulation_result

# Same rendering contract as coinfosim.provenance.export._render_graphviz --
# the report figures must look like the canonical export, not a bespoke style.
_DOT_RENDER_KWARGS = dict(
    direction="LR",
    use_labels=True,
    show_element_attributes=False,
    show_relation_attributes=False,
    show_nary=False,
)

# Engineering-only PROV types/labels that must never appear in the scientific
# reporting projection (task plan Section 6.1).
_FORBIDDEN_MARKERS = (
    "CodeRevision",
    "ExecutionEnvironment",
    "RecoveredArtifactSet",
    "ArtifactRecovery",
    "Git commit",
)

_ARM_FAMILY_TO_ID = {
    "real_dataset": REAL_ARM_ID,
    "single_gaussian_to_real": GAUSSIAN_ARM_ID,
    "gmm_to_real": GMM_ARM_ID,
}


@dataclasses.dataclass(frozen=True)
class ScenarioTarget:
    scenario_dir: str
    output_stem: str
    label: str


SCENARIOS = (
    ScenarioTarget(
        scenario_dir="output/reports/scenarios/000002_occupancy_baseline_full",
        output_stem="provenance_occupancy_000002",
        label="Occupancy (scenario 000002)",
    ),
    ScenarioTarget(
        scenario_dir="output/reports/scenarios/000005_air_quality_baseline_full",
        output_stem="provenance_air_quality_000005",
        label="Air Quality (scenario 000005)",
    ),
    ScenarioTarget(
        scenario_dir="output/reports/scenarios/000008_support2_baseline_full",
        output_stem="provenance_support2_000008",
        label="SUPPORT2 (scenario 000008)",
    ),
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dataset_metadata(report_data: Mapping[str, Any], scenario_json: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset_metadata = report_data.get("dataset")
    if dataset_metadata:
        return dataset_metadata
    scenario_section = report_data.get("scenario") or {}
    name = scenario_section.get("dataset_name") or scenario_json.get("scenario_slug")
    return {"name": name}


def _report_artifact_evidence(
    manifest: Mapping[str, Any], repo_root: Path
) -> list[ArtifactEvidence]:
    artifacts = []
    for path, sha256 in manifest.get("report_artifact_hashes", {}).items():
        artifacts.append(
            ArtifactEvidence(
                path=str(path),
                sha256=str(sha256),
                role=f"report-artifact:{Path(path).name}",
            )
        )
    return artifacts


def _rf_calibration_evidence(scenario_json: Mapping[str, Any]) -> Optional[ArtifactEvidence]:
    rf_configuration = (
        scenario_json.get("config", {})
        .get("classifier_configurations", {})
        .get("random_forest", {})
    )
    calibration = rf_configuration.get("calibration", {}) if rf_configuration else {}
    calibration_path = calibration.get("artifact_path")
    calibration_sha256 = calibration.get("artifact_sha256")
    if calibration_path and calibration_sha256:
        return ArtifactEvidence(
            path=str(calibration_path),
            sha256=str(calibration_sha256),
            role="random-forest-calibration",
        )
    return None


def build_scientific_evidence(target: ScenarioTarget, repo_root: Path) -> ProvenanceEvidence:
    """Reconstruct the persisted evidence for one scenario and clear only the
    engineering-only fields, per the reporting projection (Section 6.1)."""

    scenario_dir = repo_root / target.scenario_dir
    scenario_json = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
    manifest = json.loads((scenario_dir / "semantic_manifest.json").read_text(encoding="utf-8"))
    report_data = scenario_json.get("report_data", {})

    simulation_arms = []
    gaussian_generator_metadata = None
    gmm_generator_metadata = None
    for ref_key, ref in scenario_json["simulation_refs"].items():
        arm_id = _ARM_FAMILY_TO_ID[ref["simulation_family"]]
        result_data_path = repo_root / ref["result_data_path"]
        simulation_arms.append(
            SimulationArmEvidence(
                arm_id=arm_id,
                simulation_run_id=int(ref["simulation_run_id"]),
                result_data=ArtifactEvidence(
                    path=to_repo_relative(result_data_path, repo_root),
                    sha256=sha256_of_file(result_data_path),
                    role=f"result-data:{arm_id}",
                ),
            )
        )
        if arm_id == GAUSSIAN_ARM_ID:
            gaussian_result = load_simulation_result(result_data_path)
            gaussian_generator_metadata = gaussian_result.metadata.get(
                "gaussian_ridge_by_class"
            )
        elif arm_id == GMM_ARM_ID:
            gmm_result = load_simulation_result(result_data_path)
            gmm_generator_metadata = gmm_result.metadata.get(
                "gmm_model_selection"
            ) or (report_data.get("arms", {}).get(GMM_ARM_ID, {}).get("gmm_model_selection"))

    evidence = collect_persisted_provenance_evidence(
        scenario_run_id=int(scenario_json["scenario_run_id"]),
        scenario_slug=scenario_json["scenario_slug"],
        dataset_metadata=_dataset_metadata(report_data, scenario_json),
        target_metadata=report_data.get("target") or {},
        split_metadata=report_data.get("split") or {},
        preprocessing_metadata=report_data.get("preprocessing") or {},
        experiment_configuration={
            "sample_sizes": scenario_json.get("config", {}).get("sample_sizes", [])
        },
        classifier_configuration={"names": manifest.get("classifier_ids", [])},
        simulation_arms=simulation_arms,
        report_artifacts=_report_artifact_evidence(manifest, repo_root),
        gaussian_generator_metadata=gaussian_generator_metadata,
        gmm_generator_metadata=gmm_generator_metadata,
        random_forest_calibration=_rf_calibration_evidence(scenario_json),
        code_commit_sha=manifest.get("code_commit_sha"),
        original_code_revision=manifest.get("original_simulation_commit_sha"),
        recovery_source_commit=manifest.get("recovered_source_commit_sha"),
    )

    # The scientific reporting projection (Section 6.1): clear only the
    # engineering-only fields. Every scientific field above is untouched.
    return dataclasses.replace(
        evidence,
        current_code_revision=None,
        original_code_revision=None,
        recovery_source_commit=None,
        execution_environment=None,
        is_historical_regeneration=False,
    )


def render_scientific_provenance_figure(
    target: ScenarioTarget, repo_root: Path, output_dir: Path
) -> tuple[Path, Path]:
    scientific_evidence = build_scientific_evidence(target, repo_root)
    document = build_scenario_prov_document(scientific_evidence)

    dot = prov_to_dot(document, **_DOT_RENDER_KWARGS)
    dot_source = dot.to_string()
    for marker in _FORBIDDEN_MARKERS:
        if marker in dot_source:
            raise RuntimeError(
                f"Engineering-only marker {marker!r} leaked into the scientific "
                f"provenance projection for {target.label}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{target.output_stem}.png"
    pdf_path = output_dir / f"{target.output_stem}.pdf"
    dot.write_png(str(png_path))
    dot.write_pdf(str(pdf_path))
    return png_path, pdf_path


def main() -> None:
    root = repository_root()
    output_dir = root / "coinfosim-report-latex/figures/provenance"
    for target in SCENARIOS:
        png_path, pdf_path = render_scientific_provenance_figure(target, root, output_dir)
        for path in (png_path, pdf_path):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Empty or missing provenance figure: {path}")
        print(f"Generated {png_path}")
        print(f"Generated {pdf_path}")


if __name__ == "__main__":
    main()
