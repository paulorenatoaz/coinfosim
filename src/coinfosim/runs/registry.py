"""
Global run registries for CoInfoSim scenarios and simulations.

This module implements a lightweight, file-based experiment-tracking layer that
is intentionally *not* tied to any particular scenario family. It underpins:

- dataset scenarios (e.g. Occupancy Detection);
- future synthetic Gaussian scenarios (varying correlation or class means);
- reuse of a single simulation run by multiple scenarios.

Two global registries live at the base output directory::

    output/reports/scenario_runs.json
    output/reports/simulation_runs.json

Each registry allocates monotonically increasing natural-number ids starting at
``0`` and never overwrites previous runs: every run gets a dedicated,
zero-padded directory under ``scenarios/`` or ``simulations/``. Registry writes
are atomic (temp file + ``os.replace``). Sequential local execution is assumed;
no cross-process locking is implemented.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# Recognized family vocabularies (kept open; values are not enforced so future
# families can be added without touching this module).
SCENARIO_FAMILIES = ("dataset", "synthetic")
SIMULATION_FAMILIES = (
    "real_dataset",
    "single_gaussian_to_real",
    "gmm_to_real",
    "gaussian_anchored",
    "synthetic_gaussian",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dirname(run_id: int, slug: str, mode: str) -> str:
    return f"{run_id:06d}_{slug}_{mode}"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioRunRecord:
    """One scenario run: a scientific/narrative execution over 1+ simulations."""

    scenario_run_id: int
    scenario_slug: str
    scenario_name: str
    scenario_family: str
    question: str
    mode: str
    status: str = "running"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_seconds: Optional[float] = None
    base_output_dir: str = "output/reports"
    run_dir: str = ""
    scenario_json_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    simulation_run_ids: List[int] = field(default_factory=list)
    simulation_refs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    report_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioRunRecord":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


@dataclass
class SimulationRunRecord:
    """One simulation run: a computational execution (model+sampler+config)."""

    simulation_run_id: int
    simulation_slug: str
    simulation_family: str
    mode: str
    scenario_run_id_origin: Optional[int] = None
    reused_by_scenario_run_ids: List[int] = field(default_factory=list)
    status: str = "running"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    runtime_seconds: Optional[float] = None
    base_output_dir: str = "output/reports"
    run_dir: str = ""
    simulation_json_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    sampler_metadata: Dict[str, Any] = field(default_factory=dict)
    result_data: Dict[str, Any] = field(default_factory=dict)
    summary_data: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationRunRecord":
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


# --------------------------------------------------------------------------- #
# Registry base
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically via a temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class _BaseRunRegistry:
    """Shared machinery for the scenario and simulation registries."""

    registry_filename: str = ""
    id_field: str = ""
    next_id_field: str = ""
    subdir: str = ""
    run_json_filename: str = ""
    record_cls: Any = None

    def __init__(self, base_output_dir: Path | str = "output/reports") -> None:
        self.base_output_dir = Path(base_output_dir)

    @property
    def registry_path(self) -> Path:
        return self.base_output_dir / self.registry_filename

    def _load(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            return {
                "schema_version": SCHEMA_VERSION,
                self.next_id_field: 0,
                "runs": [],
            }
        with self.registry_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault(self.next_id_field, len(data.get("runs", [])))
        data.setdefault("runs", [])
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        _atomic_write_json(self.registry_path, data)

    def ensure_registry(self) -> Path:
        """Create the registry file if missing and return its path."""
        if not self.registry_path.exists():
            self._save(self._load())
        return self.registry_path

    def _allocate_run(self, slug: str, mode: str) -> Dict[str, Any]:
        """Allocate an id and run directory; append a running record."""
        data = self._load()
        run_id = int(data[self.next_id_field])
        data[self.next_id_field] = run_id + 1

        run_dir = self.base_output_dir / self.subdir / _run_dirname(run_id, slug, mode)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_json_path = run_dir / self.run_json_filename

        record = {
            self.id_field: run_id,
            "run_dir": str(run_dir),
            "base_output_dir": str(self.base_output_dir),
            "status": "running",
            "started_at": _utcnow(),
        }
        return record, data, run_id, run_dir, run_json_path

    def _write_runs(self, data: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
        data["runs"] = records
        self._save(data)

    def get_run(self, run_id: int) -> Optional[Any]:
        data = self._load()
        for run in data["runs"]:
            if int(run[self.id_field]) == int(run_id):
                return self.record_cls.from_dict(run)
        return None

    def list_runs(self) -> List[Any]:
        data = self._load()
        return [self.record_cls.from_dict(run) for run in data["runs"]]

    def _update_run(self, run_id: int, updates: Dict[str, Any]) -> Any:
        data = self._load()
        found = None
        for run in data["runs"]:
            if int(run[self.id_field]) == int(run_id):
                run.update(updates)
                found = run
                break
        if found is None:
            raise KeyError(f"run id {run_id} not found in {self.registry_path}")
        self._save(data)
        return self.record_cls.from_dict(found)


# --------------------------------------------------------------------------- #
# Scenario registry
# --------------------------------------------------------------------------- #
class ScenarioRunRegistry(_BaseRunRegistry):
    """Registry for scenario runs (``scenario_runs.json``)."""

    registry_filename = "scenario_runs.json"
    id_field = "scenario_run_id"
    next_id_field = "next_scenario_run_id"
    subdir = "scenarios"
    run_json_filename = "scenario.json"
    record_cls = ScenarioRunRecord

    def start_run(
        self,
        *,
        scenario_slug: str,
        scenario_name: str,
        scenario_family: str,
        question: str,
        mode: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> ScenarioRunRecord:
        """Allocate a scenario run id + directory and append a running record."""
        base, data, run_id, run_dir, run_json_path = self._allocate_run(
            scenario_slug, mode
        )
        record = ScenarioRunRecord(
            scenario_run_id=run_id,
            scenario_slug=scenario_slug,
            scenario_name=scenario_name,
            scenario_family=scenario_family,
            question=question,
            mode=mode,
            status="running",
            started_at=base["started_at"],
            base_output_dir=str(self.base_output_dir),
            run_dir=str(run_dir),
            scenario_json_path=str(run_json_path),
            config=dict(config or {}),
        )
        runs = data["runs"]
        runs.append(record.to_dict())
        self._write_runs(data, runs)
        return record

    def complete_run(
        self, run_id: int, *, runtime_seconds: float, **updates: Any
    ) -> ScenarioRunRecord:
        payload = {
            "status": "completed",
            "finished_at": _utcnow(),
            "runtime_seconds": float(runtime_seconds),
            "error": None,
        }
        payload.update(updates)
        return self._update_run(run_id, payload)

    def fail_run(self, run_id: int, *, error: str, **updates: Any) -> ScenarioRunRecord:
        payload = {
            "status": "failed",
            "finished_at": _utcnow(),
            "error": str(error),
        }
        payload.update(updates)
        return self._update_run(run_id, payload)

    def update_run(self, run_id: int, **updates: Any) -> ScenarioRunRecord:
        return self._update_run(run_id, updates)


# --------------------------------------------------------------------------- #
# Simulation registry
# --------------------------------------------------------------------------- #
class SimulationRunRegistry(_BaseRunRegistry):
    """Registry for simulation runs (``simulation_runs.json``)."""

    registry_filename = "simulation_runs.json"
    id_field = "simulation_run_id"
    next_id_field = "next_simulation_run_id"
    subdir = "simulations"
    run_json_filename = "simulation.json"
    record_cls = SimulationRunRecord

    def start_run(
        self,
        *,
        simulation_slug: str,
        simulation_family: str,
        mode: str,
        scenario_run_id_origin: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> SimulationRunRecord:
        """Allocate a simulation run id + directory and append a running record."""
        base, data, run_id, run_dir, run_json_path = self._allocate_run(
            simulation_slug, mode
        )
        reused_by: List[int] = []
        if scenario_run_id_origin is not None:
            reused_by = [int(scenario_run_id_origin)]
        record = SimulationRunRecord(
            simulation_run_id=run_id,
            simulation_slug=simulation_slug,
            simulation_family=simulation_family,
            mode=mode,
            scenario_run_id_origin=scenario_run_id_origin,
            reused_by_scenario_run_ids=reused_by,
            status="running",
            started_at=base["started_at"],
            base_output_dir=str(self.base_output_dir),
            run_dir=str(run_dir),
            simulation_json_path=str(run_json_path),
            config=dict(config or {}),
        )
        runs = data["runs"]
        runs.append(record.to_dict())
        self._write_runs(data, runs)
        return record

    def complete_run(
        self, run_id: int, *, runtime_seconds: float, **updates: Any
    ) -> SimulationRunRecord:
        payload = {
            "status": "completed",
            "finished_at": _utcnow(),
            "runtime_seconds": float(runtime_seconds),
            "error": None,
        }
        payload.update(updates)
        return self._update_run(run_id, payload)

    def fail_run(
        self, run_id: int, *, error: str, **updates: Any
    ) -> SimulationRunRecord:
        payload = {
            "status": "failed",
            "finished_at": _utcnow(),
            "error": str(error),
        }
        payload.update(updates)
        return self._update_run(run_id, payload)

    def update_run(self, run_id: int, **updates: Any) -> SimulationRunRecord:
        return self._update_run(run_id, updates)

    def add_scenario_reuse(
        self, run_id: int, scenario_run_id: int
    ) -> SimulationRunRecord:
        """Record that ``scenario_run_id`` reuses this simulation run."""
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(f"simulation run id {run_id} not found")
        reused = list(record.reused_by_scenario_run_ids)
        if int(scenario_run_id) not in reused:
            reused.append(int(scenario_run_id))
        return self._update_run(run_id, {"reused_by_scenario_run_ids": reused})
