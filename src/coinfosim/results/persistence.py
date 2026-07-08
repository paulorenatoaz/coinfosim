"""
Persistence for cooperative Monte Carlo simulation results.

This module serializes a
:class:`~coinfosim.simulation.monte_carlo.SimulationResult` to a compact,
self-describing ``.json.gz`` payload and reconstructs it later. The persisted
data is sufficient to regenerate every downstream artifact that depends only on
the accumulated losses and result metadata: simulation-level Monte Carlo
reports, scenario-level reports, summary tables, ranking tables, threshold
comparisons, and loss-curve figures.

The scientific/statistical behavior is unaffected; this is purely an I/O layer.

Reconstructed models
--------------------
Two model families are supported, matching the two Occupancy experiment arms
and the future synthetic Gaussian scenarios:

- Gaussian models (:class:`~coinfosim.models.gaussian.GaussianSimulationModel`)
  are stored via their per-class means and covariances and rebuilt exactly.
- Finite real-dataset models
  (:class:`~coinfosim.samplers.real.RealDatasetModel`) expose only lightweight
  metadata (``d``, class labels, channel names, name), which is stored and
  rebuilt. The raw data pool itself is not part of a result and is not needed
  to regenerate reports.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.results.accumulator import LossAccumulator
from coinfosim.samplers.real import RealDatasetModel
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import SimulationResult, StoppingInfo

SCHEMA_VERSION = 1


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples to JSON-native types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _serialize_model(model: Any) -> Dict[str, Any]:
    """Serialize a simulation model into a self-describing dict.

    Gaussian models are detected by their ``mean``/``covariance`` methods; all
    other models are treated as finite real-dataset models exposing only
    metadata.
    """
    is_gaussian = (
        hasattr(model, "mean")
        and hasattr(model, "covariance")
        and hasattr(model, "class_labels")
    )
    if is_gaussian:
        labels = [int(label) for label in model.class_labels]
        return {
            "kind": "gaussian",
            "d": int(model.d),
            "class_labels": labels,
            "means": {str(label): model.mean(label).tolist() for label in labels},
            "covariances": {
                str(label): model.covariance(label).tolist() for label in labels
            },
        }
    return {
        "kind": "real_dataset",
        "d": int(getattr(model, "d")),
        "class_labels": [int(label) for label in getattr(model, "class_labels")],
        "channel_names": [str(c) for c in getattr(model, "channel_names", ())],
        "name": str(getattr(model, "name", "real dataset")),
    }


def _deserialize_model(spec: Dict[str, Any]) -> Any:
    """Rebuild a simulation model from a :func:`_serialize_model` payload."""
    kind = spec.get("kind")
    if kind == "gaussian":
        means = {
            int(label): np.asarray(vec, dtype=float)
            for label, vec in spec["means"].items()
        }
        covariances = {
            int(label): np.asarray(mat, dtype=float)
            for label, mat in spec["covariances"].items()
        }
        return GaussianSimulationModel(means=means, covariances=covariances)
    if kind == "real_dataset":
        return RealDatasetModel(
            d=int(spec["d"]),
            class_labels=tuple(int(label) for label in spec["class_labels"]),
            channel_names=tuple(str(c) for c in spec.get("channel_names", ())),
            name=str(spec.get("name", "real dataset")),
        )
    raise ValueError(f"unknown model kind: {kind!r}")


def _serialize_accumulator(accumulator: LossAccumulator) -> List[Dict[str, Any]]:
    """Serialize every recorded cell and its ordered replication losses."""
    records: List[Dict[str, Any]] = []
    for n in accumulator.sample_sizes():
        for cell in accumulator.cells_for(n):
            _n, subset, clf = cell
            losses = accumulator.losses(_n, subset, clf)
            records.append(
                {
                    "n_per_class": int(_n),
                    "subset": [int(i) for i in subset],
                    "classifier": str(clf),
                    "losses": [float(v) for v in losses.tolist()],
                }
            )
    return records


def _deserialize_accumulator(records: List[Dict[str, Any]]) -> LossAccumulator:
    """Rebuild a :class:`LossAccumulator` from serialized cell records."""
    accumulator = LossAccumulator()
    for record in records:
        n = int(record["n_per_class"])
        subset = tuple(int(i) for i in record["subset"])
        clf = str(record["classifier"])
        for replication_id, loss in enumerate(record["losses"]):
            accumulator.add(n, subset, clf, replication_id, float(loss))
    return accumulator


def _serialize_config(config: MonteCarloConfig) -> Dict[str, Any]:
    return {
        "mode": config.mode,
        "sample_sizes": [int(n) for n in config.sample_sizes],
        "min_replications": int(config.min_replications),
        "max_replications": int(config.max_replications),
        "replication_batch_size": int(config.replication_batch_size),
        "test_samples_per_class": int(config.test_samples_per_class),
        "ci_half_width_target": float(config.ci_half_width_target),
        "base_seed": int(config.base_seed),
    }


def _deserialize_config(spec: Dict[str, Any]) -> MonteCarloConfig:
    return MonteCarloConfig(
        mode=str(spec["mode"]),
        sample_sizes=tuple(int(n) for n in spec["sample_sizes"]),
        min_replications=int(spec["min_replications"]),
        max_replications=int(spec["max_replications"]),
        replication_batch_size=int(spec["replication_batch_size"]),
        test_samples_per_class=int(spec["test_samples_per_class"]),
        ci_half_width_target=float(spec["ci_half_width_target"]),
        base_seed=int(spec["base_seed"]),
    )


def result_to_dict(result: SimulationResult) -> Dict[str, Any]:
    """Return a fully JSON-serializable dict for ``result``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model": _serialize_model(result.model),
        "config": _serialize_config(result.config),
        "subsets": [[int(i) for i in subset] for subset in result.subsets],
        "classifier_names": [str(c) for c in result.classifier_names],
        "stopping_info": [
            {
                "n_per_class": int(info.n_per_class),
                "replications": int(info.replications),
                "reason": info.reason,
                "max_ci_half_width": float(info.max_ci_half_width),
            }
            for info in result.stopping_info.values()
        ],
        "runtime_seconds": float(result.runtime_seconds),
        "metadata": _to_jsonable(result.metadata),
        "accumulator": _serialize_accumulator(result.accumulator),
    }


def result_from_dict(payload: Dict[str, Any]) -> SimulationResult:
    """Reconstruct a :class:`SimulationResult` from :func:`result_to_dict` output."""
    config = _deserialize_config(payload["config"])
    subsets = [tuple(int(i) for i in subset) for subset in payload["subsets"]]
    classifier_names = [str(c) for c in payload["classifier_names"]]
    stopping_info = {
        int(entry["n_per_class"]): StoppingInfo(
            n_per_class=int(entry["n_per_class"]),
            replications=int(entry["replications"]),
            reason=entry["reason"],
            max_ci_half_width=float(entry["max_ci_half_width"]),
        )
        for entry in payload["stopping_info"]
    }
    return SimulationResult(
        model=_deserialize_model(payload["model"]),
        config=config,
        subsets=subsets,
        classifier_names=classifier_names,
        accumulator=_deserialize_accumulator(payload["accumulator"]),
        stopping_info=stopping_info,
        runtime_seconds=float(payload["runtime_seconds"]),
        metadata=dict(payload.get("metadata", {})),
    )


def save_simulation_result(result: SimulationResult, path: Path | str) -> Path:
    """Persist ``result`` to ``path`` as gzip-compressed JSON.

    The parent directory is created if missing. Returns the written path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result_to_dict(result)
    data = json.dumps(payload, indent=2).encode("utf-8")
    with gzip.open(path, "wb") as fh:
        fh.write(data)
    return path


def load_simulation_result(path: Path | str) -> SimulationResult:
    """Load a :class:`SimulationResult` previously written by
    :func:`save_simulation_result`."""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        payload = json.loads(fh.read().decode("utf-8"))
    return result_from_dict(payload)
