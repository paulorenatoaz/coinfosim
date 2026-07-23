#!/usr/bin/env python3
"""Single permitted multi-minute end-to-end predictive-profile validation.

Verifies the ``gh-pages`` recovery manifest and critical result-data gzip
integrity, regenerates the five academic scenarios (``000002``, ``000005``,
``000006``, ``000007``, ``000008``) from persisted results into an isolated
output directory, validates the canonical schema / semantic-manifest /
provenance-JSON-LD output, audits regenerated report HTML for retired
terminology, and runs a fixed set of targeted regression tests.

This script never executes a scenario or a Monte Carlo simulation: it never
imports ``run_dataset_anchored_scenario`` / ``run_registered_scenario``, and
it monkeypatches ``CooperativeMonteCarloSimulator`` to raise if anything
along the regeneration path ever attempts to construct one. It never opens
``output/reports`` or the recovered ``gh-pages`` snapshot for writing --
every write happens under the isolated ``--output-dir`` tree, built by
copying and path-rewriting the minimum required scenario/simulation
directories and registries. It never touches git or the publish module, so
it cannot reach ``gh-pages`` or publish remotely.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

SCENARIO_SLUGS = {
    "000002": "occupancy",
    "000005": "air-quality",
    "000006": "air-quality",
    "000007": "support2",
    "000008": "support2",
}

FORBIDDEN_ACTIVE_TERMS = (
    "Progressive N-star",
    "Timing similarity",
    "Interpolated N-star",
    "N-star diagnostics",
    "structural_fidelity",
    "structural_dynamics",
    "nstar_similarity",
    "crossing_jaccard",
)

TARGETED_TESTS = [
    "tests/test_semantic_vocabulary.py",
    "tests/test_predictive_profile_metrics.py",
    "tests/test_predictive_profile_schema.py",
    "tests/test_provenance_export.py",
    "tests/test_occupancy_scenario_report.py",
    "tests/test_publish_site.py",
    "tests/test_parallel_scientific_equivalence.py",
]


class ValidationError(RuntimeError):
    """Raised for any validation failure; the process exits non-zero."""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _rewrite_paths(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v, old, new) for v in obj]
    if isinstance(obj, str) and old in obj:
        return obj.replace(old, new)
    return obj


def _verify_gzip(path: Path) -> None:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1024 * 1024):
                pass
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ValidationError(f"invalid gzip stream: {path} ({exc})") from exc


def verify_recovery_manifest(
    manifest_path: Path, reports_dir: Path, scenario_ids: Sequence[str]
) -> Dict[str, Any]:
    print(f"[1/6] Verifying recovery manifest {manifest_path} ...")
    if not manifest_path.exists():
        raise ValidationError(f"recovery manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)
    missing = [sid for sid in scenario_ids if sid not in manifest.get("scenarios", {})]
    if missing:
        raise ValidationError(f"manifest is missing academic scenarios: {missing}")

    checked = 0
    for sid in scenario_ids:
        scenario_entry = manifest["scenarios"][sid]
        if scenario_entry.get("critical_result_data_ok") is not True:
            raise ValidationError(
                f"manifest reports critical_result_data_ok != True for scenario {sid}"
            )
        for sim_id in scenario_entry["linked_simulation_ids"]:
            sim_key = f"{sim_id:06d}"
            sim_dir = reports_dir / "simulations"
            matches = sorted(sim_dir.glob(f"{sim_key}_*"))
            if not matches:
                raise ValidationError(f"simulation directory {sim_key} not found on disk")
            for gz in sorted(matches[0].glob("result_data_*.json.gz")):
                _verify_gzip(gz)
                checked += 1
    print(f"    OK: {len(scenario_ids)} scenarios, {checked} result_data files gzip-valid.")
    return manifest


def _collect_simulation_ids(reports_dir: Path, scenario_id: str) -> List[int]:
    scenario_dirs = sorted((reports_dir / "scenarios").glob(f"{scenario_id}_*"))
    if not scenario_dirs:
        raise ValidationError(f"scenario directory {scenario_id} not found")
    scenario_json = _load_json(scenario_dirs[0] / "scenario.json")
    return [int(i) for i in scenario_json["simulation_run_ids"]]


def build_isolated_snapshot(
    reports_dir: Path, output_dir: Path, scenario_ids: Sequence[str]
) -> None:
    """Copy exactly the required scenario/simulation directories and
    registries into ``output_dir``, then rewrite embedded paths to point at
    the isolated tree. Never writes into ``reports_dir``.
    """

    print(f"[2/6] Building isolated snapshot under {output_dir} ...")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "scenarios").mkdir(parents=True)
    (output_dir / "simulations").mkdir(parents=True)

    scenario_registry = _load_json(reports_dir / "scenario_runs.json")
    simulation_registry = _load_json(reports_dir / "simulation_runs.json")
    registered_scenario_ids = {int(r["scenario_run_id"]) for r in scenario_registry["runs"]}
    registered_simulation_ids = {int(r["simulation_run_id"]) for r in simulation_registry["runs"]}

    all_simulation_ids: List[int] = []
    for scenario_id in scenario_ids:
        scenario_dirs = sorted((reports_dir / "scenarios").glob(f"{scenario_id}_*"))
        shutil.copytree(scenario_dirs[0], output_dir / "scenarios" / scenario_dirs[0].name)
        if int(scenario_id) not in registered_scenario_ids:
            # Known, documented registry inconsistency: the scenario
            # directory and its scenario.json exist and are complete, but
            # the run was never appended to scenario_runs.json (see
            # docs/provenance/gh_pages_recovery_manifest.json). Reconstruct
            # the missing registry row from the already-persisted
            # scenario.json (not fabricated data) so the isolated copy is
            # self-consistent; the original output/reports registry is left
            # untouched and the inconsistency remains reported, not
            # silently repaired.
            scenario_record = _load_json(scenario_dirs[0] / "scenario.json")
            scenario_registry["runs"].append(scenario_record)
            registered_scenario_ids.add(int(scenario_id))
            scenario_registry["next_scenario_run_id"] = max(
                int(scenario_registry.get("next_scenario_run_id", 0)),
                int(scenario_id) + 1,
            )
            print(
                f"    NOTE: reconstructed missing scenario registry row "
                f"{scenario_id} in the isolated copy only (documented registry "
                f"inconsistency; output/reports/scenario_runs.json unchanged)."
            )
        sim_ids = _collect_simulation_ids(reports_dir, scenario_id)
        all_simulation_ids.extend(sim_ids)
        for sim_id in sim_ids:
            sim_key = f"{sim_id:06d}"
            sim_dirs = sorted((reports_dir / "simulations").glob(f"{sim_key}_*"))
            if not sim_dirs:
                raise ValidationError(f"simulation directory {sim_key} not found")
            target = output_dir / "simulations" / sim_dirs[0].name
            if not target.exists():
                shutil.copytree(sim_dirs[0], target)
            if sim_id not in registered_simulation_ids:
                # Known, documented registry inconsistency (see
                # docs/provenance/gh_pages_recovery_manifest.json): the
                # simulation directory and its simulation.json exist and are
                # complete, but the run was never appended to
                # simulation_runs.json. Reconstruct the missing registry row
                # from the already-persisted simulation.json (not fabricated
                # data) so the isolated copy is self-consistent; the
                # original output/reports registry is left untouched and
                # the inconsistency remains reported, not silently repaired.
                sim_record = _load_json(sim_dirs[0] / "simulation.json")
                simulation_registry["runs"].append(sim_record)
                registered_simulation_ids.add(sim_id)
                print(
                    f"    NOTE: reconstructed missing simulation registry row "
                    f"{sim_key} in the isolated copy only (documented registry "
                    f"inconsistency; output/reports/simulation_runs.json unchanged)."
                )

    if all_simulation_ids:
        simulation_registry["next_simulation_run_id"] = max(
            int(simulation_registry.get("next_simulation_run_id", 0)),
            max(all_simulation_ids) + 1,
        )

    old_prefix = str(reports_dir)
    new_prefix = str(output_dir)
    # scenario_runs.json / simulation_runs.json paths are relative
    # ("output/reports/...") rather than the absolute reports_dir; rewrite
    # both spellings defensively.
    for old, new in ((old_prefix, new_prefix), ("output/reports", str(output_dir))):
        scenario_registry = _rewrite_paths(scenario_registry, old, new)
        simulation_registry = _rewrite_paths(simulation_registry, old, new)

    _write_json(output_dir / "scenario_runs.json", scenario_registry)
    _write_json(output_dir / "simulation_runs.json", simulation_registry)

    for scenario_id in scenario_ids:
        scenario_dirs = sorted((output_dir / "scenarios").glob(f"{scenario_id}_*"))
        scenario_json_path = scenario_dirs[0] / "scenario.json"
        data = _load_json(scenario_json_path)
        for old, new in ((old_prefix, new_prefix), ("output/reports", str(output_dir))):
            data = _rewrite_paths(data, old, new)
        _write_json(scenario_json_path, data)

    for sim_id in set(all_simulation_ids):
        sim_key = f"{sim_id:06d}"
        sim_dirs = sorted((output_dir / "simulations").glob(f"{sim_key}_*"))
        sim_json_path = sim_dirs[0] / "simulation.json"
        data = _load_json(sim_json_path)
        for old, new in ((old_prefix, new_prefix), ("output/reports", str(output_dir))):
            data = _rewrite_paths(data, old, new)
        _write_json(sim_json_path, data)

    print(f"    OK: isolated snapshot contains {len(scenario_ids)} scenarios, "
          f"{len(set(all_simulation_ids))} simulations.")


def regenerate_scenarios(output_dir: Path, scenario_ids: Sequence[str]) -> Dict[str, str]:
    print(f"[3/6] Regenerating {len(scenario_ids)} academic scenarios (no Monte Carlo) ...")

    import coinfosim.scenarios.dataset_anchored_runner as runner_module
    from coinfosim.scenarios.service import regenerate_registered_scenario

    def _forbidden_simulator(*_args: Any, **_kwargs: Any) -> Any:
        raise ValidationError(
            "refused: an attempt was made to construct CooperativeMonteCarloSimulator "
            "during report regeneration; Monte Carlo must never rerun in this validation"
        )

    runner_module.CooperativeMonteCarloSimulator = _forbidden_simulator  # type: ignore[assignment]

    report_paths: Dict[str, str] = {}
    for scenario_id in scenario_ids:
        slug = SCENARIO_SLUGS[scenario_id]
        run_id = int(scenario_id)
        result = regenerate_registered_scenario(
            slug, scenario_run_id=run_id, output_dir=output_dir
        )
        report_paths[scenario_id] = result["scenario_report"]
        print(f"    regenerated {scenario_id} ({slug}) -> {result['scenario_report']}")
    return report_paths


def _assert_no_forbidden_terms(text: str, source: str) -> None:
    for term in FORBIDDEN_ACTIVE_TERMS:
        if term in text:
            raise ValidationError(f"forbidden active term {term!r} found in {source}")


def validate_regenerated_scenario(
    output_dir: Path, scenario_id: str, report_path: str, recovered_source_commit_sha: str
) -> None:
    scenario_dirs = sorted((output_dir / "scenarios").glob(f"{scenario_id}_*"))
    scenario_json = _load_json(scenario_dirs[0] / "scenario.json")
    report_data = scenario_json["report_data"]

    if "structural_fidelity" in report_data:
        raise ValidationError(f"{scenario_id}: legacy structural_fidelity key still present")

    # Some scenarios (e.g. SUPPORT2) intentionally use
    # structural_snapshot_policy == "regenerate_from_result_data" and do not
    # embed the predictive_cooperation_profile/pairwise_profile_dynamics
    # snapshots in persisted JSON (only the HTML report, recomputed live
    # from result_data, reflects them). Only require the embedded schema-3
    # payload when the scenario's own policy says it should be embedded.
    embeds_snapshot = report_data.get("structural_snapshot_policy") != "regenerate_from_result_data"

    if embeds_snapshot:
        if "predictive_cooperation_profile" not in report_data:
            raise ValidationError(f"{scenario_id}: missing predictive_cooperation_profile key")
        profile = report_data["predictive_cooperation_profile"]
        if profile["schema_version"] != 3:
            raise ValidationError(f"{scenario_id}: schema_version != 3")
        if not profile.get("semantic_vocabulary_version"):
            raise ValidationError(f"{scenario_id}: missing semantic_vocabulary_version")
        if profile.get("semantic_type") != "coinfosim:PredictiveCooperationProfile":
            raise ValidationError(f"{scenario_id}: unexpected semantic_type")
        for row in profile["reversal_agreement_series"]:
            if "product" in "".join(row.keys()).lower():
                raise ValidationError(f"{scenario_id}: composite/product field present in row {row}")
            for forbidden in ("nstar_similarity", "crossing_jaccard", "timing_similarity"):
                if forbidden in row:
                    raise ValidationError(f"{scenario_id}: forbidden field {forbidden!r} in row")

    for sim_id in scenario_json["simulation_run_ids"]:
        sim_key = f"{sim_id:06d}"
        sim_dirs = sorted((output_dir / "simulations").glob(f"{sim_key}_*"))
        sim_json = _load_json(sim_dirs[0] / "simulation.json")
        sim_result_data = sim_json["result_data"]
        if "structural_dynamics" in sim_result_data:
            raise ValidationError(f"{sim_key}: legacy structural_dynamics key still present")
        if embeds_snapshot:
            dynamics = sim_result_data["pairwise_profile_dynamics"]
            if dynamics["schema_version"] != 3:
                raise ValidationError(f"{sim_key}: pairwise_profile_dynamics schema_version != 3")

    manifest_path = scenario_dirs[0] / "semantic_manifest.json"
    if not manifest_path.exists():
        raise ValidationError(f"{scenario_id}: semantic_manifest.json missing")
    manifest = _load_json(manifest_path)
    if manifest.get("recovered_source_commit_sha") != recovered_source_commit_sha:
        raise ValidationError(
            f"{scenario_id}: semantic manifest recovered_source_commit_sha mismatch"
        )
    if not manifest.get("code_commit_sha"):
        raise ValidationError(f"{scenario_id}: semantic manifest missing code_commit_sha")

    provenance_path = scenario_dirs[0] / "provenance.jsonld"
    if not provenance_path.exists():
        raise ValidationError(f"{scenario_id}: provenance.jsonld missing")
    provenance = _load_json(provenance_path)
    graph = provenance["@graph"]
    relations = {key for node in graph for key in node}
    for relation in ("prov:used", "prov:wasGeneratedBy", "prov:wasDerivedFrom", "prov:wasAssociatedWith"):
        if relation not in relations:
            raise ValidationError(f"{scenario_id}: provenance graph missing {relation}")

    report_html_path = Path(report_path)
    text = report_html_path.read_text(encoding="utf-8")
    _assert_no_forbidden_terms(text, str(report_html_path))
    for required in (
        "predictive cooperation profile",
        "Reversal existence agreement",
        "Reversal sample-size similarity",
        "Winner matrix",
        "Reversal matrix",
    ):
        if required not in text:
            raise ValidationError(f"{scenario_id}: report missing required text {required!r}")


def run_targeted_tests() -> None:
    print(f"[5/6] Running {len(TARGETED_TESTS)} targeted regression test files ...")
    cmd = [sys.executable, "-m", "pytest", "-q", *TARGETED_TESTS]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise ValidationError(f"targeted tests failed (exit code {result.returncode})")
    print("    OK: targeted tests passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    parser.add_argument("--scenario-ids", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-targeted-tests", action="store_true")
    args = parser.parse_args()

    scenario_ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
    if set(scenario_ids) != set(SCENARIO_SLUGS):
        raise ValidationError(
            f"expected exactly the five academic scenarios {sorted(SCENARIO_SLUGS)}, "
            f"got {sorted(scenario_ids)}"
        )

    reports_dir = args.reports_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == reports_dir:
        raise ValidationError("--output-dir must not be --reports-dir")

    manifest = verify_recovery_manifest(args.recovery_manifest, reports_dir, scenario_ids)
    recovered_source_commit_sha = manifest["source_commit_sha"]

    build_isolated_snapshot(reports_dir, output_dir, scenario_ids)

    report_paths = regenerate_scenarios(output_dir, scenario_ids)

    print("[4/6] Validating canonical schema, semantic manifests, and provenance JSON-LD ...")
    for scenario_id in scenario_ids:
        validate_regenerated_scenario(
            output_dir, scenario_id, report_paths[scenario_id], recovered_source_commit_sha
        )
        print(f"    OK: {scenario_id}")

    if args.run_targeted_tests:
        run_targeted_tests()
    else:
        print("[5/6] Skipped targeted tests (--run-targeted-tests not passed).")

    print("[6/6] All five academic scenarios validated successfully.")
    print(f"Recovered gh-pages commit: {recovered_source_commit_sha}")
    print(f"Regenerated into: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
