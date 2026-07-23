#!/usr/bin/env python3
"""Inventory recovered ``gh-pages`` scenario/simulation artifacts by directory scan.

This script does not trust the published homepage listing. It walks
``output/reports/scenarios`` and ``output/reports/simulations`` directly, cross
references each scenario/simulation against the local run registries
(``scenario_runs.json`` / ``simulation_runs.json``), validates the gzip
integrity of every ``result_data_*.json.gz`` file for the five academic
scenarios, and writes a deterministic, tracked recovery manifest.

Registry inconsistencies (a directory present on disk but absent from its
registry) are recorded, never silently repaired.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ACADEMIC_SCENARIO_IDS = ("000002", "000005", "000006", "000007", "000008")
SCENARIO_DIR_RE = re.compile(r"^(\d{6})_")
SIMULATION_DIR_RE = re.compile(r"^(\d{6})_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_gzip(path: Path) -> Optional[str]:
    """Return ``None`` if the gzip stream is valid, else an error message."""
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1024 * 1024):
                pass
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _resolve_ref(ref: str) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", ref],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ref


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _scan_result_data_files(sim_dir: Path, root: Path, verify_gzip: bool) -> List[Dict[str, Any]]:
    entries = []
    for gz_path in sorted(sim_dir.glob("result_data_*.json.gz")):
        entry: Dict[str, Any] = {
            "path": _relpath(gz_path, root),
            "size_bytes": gz_path.stat().st_size,
            "sha256": _sha256(gz_path),
        }
        if verify_gzip:
            error = _verify_gzip(gz_path)
            entry["gzip_valid"] = error is None
            if error is not None:
                entry["gzip_error"] = error
        entries.append(entry)
    return entries


def build_manifest(
    reports_dir: Path,
    source_ref: str,
    verify_gzip: bool,
) -> Dict[str, Any]:
    root = reports_dir.parent if reports_dir.name == "reports" else reports_dir
    scenarios_dir = reports_dir / "scenarios"
    simulations_dir = reports_dir / "simulations"

    scenario_registry_path = reports_dir / "scenario_runs.json"
    simulation_registry_path = reports_dir / "simulation_runs.json"
    scenario_registry_ids = set()
    simulation_registry_ids = set()
    if scenario_registry_path.exists():
        data = _load_json(scenario_registry_path)
        scenario_registry_ids = {int(r["scenario_run_id"]) for r in data.get("runs", [])}
    if simulation_registry_path.exists():
        data = _load_json(simulation_registry_path)
        simulation_registry_ids = {int(r["simulation_run_id"]) for r in data.get("runs", [])}

    simulation_inventory: Dict[str, Dict[str, Any]] = {}
    if simulations_dir.exists():
        for sim_dir in sorted(simulations_dir.iterdir()):
            if not sim_dir.is_dir():
                continue
            match = SIMULATION_DIR_RE.match(sim_dir.name)
            if not match:
                continue
            sim_id = match.group(1)
            sim_json_path = sim_dir / "simulation.json"
            record: Dict[str, Any] = {
                "simulation_run_id": int(sim_id),
                "directory": _relpath(sim_dir, root),
                "in_simulation_registry": int(sim_id) in simulation_registry_ids,
                "result_data_files": _scan_result_data_files(sim_dir, root, verify_gzip),
                "has_simulation_json": sim_json_path.exists(),
            }
            simulation_inventory[sim_id] = record

    scenario_inventory: Dict[str, Dict[str, Any]] = {}
    missing_registry_refs: List[Dict[str, Any]] = []
    if scenarios_dir.exists():
        for scenario_dir in sorted(scenarios_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            match = SCENARIO_DIR_RE.match(scenario_dir.name)
            if not match:
                continue
            scenario_id = match.group(1)
            scenario_json_path = scenario_dir / "scenario.json"
            linked_simulation_ids: List[int] = []
            scenario_slug = None
            mode = None
            status = None
            if scenario_json_path.exists():
                scenario_json = _load_json(scenario_json_path)
                linked_simulation_ids = [int(i) for i in scenario_json.get("simulation_run_ids", [])]
                scenario_slug = scenario_json.get("scenario_slug")
                mode = scenario_json.get("mode")
                status = scenario_json.get("status")

            in_scenario_registry = int(scenario_id) in scenario_registry_ids
            if not in_scenario_registry:
                missing_registry_refs.append(
                    {
                        "kind": "scenario",
                        "id": scenario_id,
                        "directory": _relpath(scenario_dir, root),
                        "issue": "directory present on disk but absent from scenario_runs.json",
                    }
                )

            missing_linked_simulations: List[int] = []
            for sim_id in linked_simulation_ids:
                sim_key = f"{sim_id:06d}"
                if sim_key not in simulation_inventory:
                    missing_linked_simulations.append(sim_id)
                elif not simulation_inventory[sim_key]["in_simulation_registry"]:
                    missing_registry_refs.append(
                        {
                            "kind": "simulation",
                            "id": sim_key,
                            "directory": simulation_inventory[sim_key]["directory"],
                            "issue": (
                                f"directory present on disk and referenced by scenario "
                                f"{scenario_id} but absent from simulation_runs.json"
                            ),
                        }
                    )

            is_academic = scenario_id in ACADEMIC_SCENARIO_IDS
            critical_result_data_ok = True
            if is_academic:
                for sim_id in linked_simulation_ids:
                    sim_key = f"{sim_id:06d}"
                    sim_record = simulation_inventory.get(sim_key)
                    if sim_record is None:
                        critical_result_data_ok = False
                        continue
                    if not sim_record["result_data_files"]:
                        critical_result_data_ok = False
                    for rd in sim_record["result_data_files"]:
                        if verify_gzip and not rd.get("gzip_valid", True):
                            critical_result_data_ok = False

            scenario_inventory[scenario_id] = {
                "scenario_run_id": int(scenario_id),
                "directory": _relpath(scenario_dir, root),
                "scenario_slug": scenario_slug,
                "mode": mode,
                "status": status,
                "in_scenario_registry": in_scenario_registry,
                "linked_simulation_ids": linked_simulation_ids,
                "missing_linked_simulations": missing_linked_simulations,
                "is_academic_scenario": is_academic,
                "critical_result_data_ok": critical_result_data_ok if is_academic else None,
            }

    academic_scenarios_present = sorted(
        sid for sid in ACADEMIC_SCENARIO_IDS if sid in scenario_inventory
    )
    academic_scenarios_missing = sorted(
        sid for sid in ACADEMIC_SCENARIO_IDS if sid not in scenario_inventory
    )

    manifest = {
        "recovery_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_ref": source_ref,
        "source_commit_sha": _resolve_ref(source_ref),
        "reports_dir": _relpath(reports_dir, root),
        "expected_scenario_ids": list(f"{i:06d}" for i in range(9)),
        "academic_scenario_ids": list(ACADEMIC_SCENARIO_IDS),
        "academic_scenarios_present": academic_scenarios_present,
        "academic_scenarios_missing": academic_scenarios_missing,
        "scenarios": dict(sorted(scenario_inventory.items())),
        "simulations": dict(sorted(simulation_inventory.items())),
        "registry_inconsistencies": missing_registry_refs,
        "gzip_verification_enabled": verify_gzip,
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--verify-gzip", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.reports_dir, args.source_ref, args.verify_gzip)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    print(f"Wrote manifest to {args.manifest}")
    print(f"Academic scenarios present: {manifest['academic_scenarios_present']}")
    print(f"Academic scenarios missing: {manifest['academic_scenarios_missing']}")
    print(f"Registry inconsistencies: {len(manifest['registry_inconsistencies'])}")
    for issue in manifest["registry_inconsistencies"]:
        print(f"  - {issue['kind']} {issue['id']}: {issue['issue']}")

    invalid_gzip = [
        rd["path"]
        for sim in manifest["simulations"].values()
        for rd in sim["result_data_files"]
        if args.verify_gzip and not rd.get("gzip_valid", True)
    ]
    if invalid_gzip:
        print("Invalid gzip files detected:")
        for path in invalid_gzip:
            print(f"  - {path}")
        return 1

    missing_academic = [
        sid
        for sid in manifest["academic_scenario_ids"]
        if sid not in manifest["scenarios"]
        or manifest["scenarios"][sid]["critical_result_data_ok"] is False
    ]
    if missing_academic:
        print(f"Academic scenarios with missing/invalid critical result data: {missing_academic}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
