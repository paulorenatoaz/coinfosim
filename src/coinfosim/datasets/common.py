"""Shared primitives for dataset-anchored CoInfoSim scenarios."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Protocol, Tuple, runtime_checkable

import pandas as pd

from coinfosim.samplers.dataset import Dataset


@dataclass(frozen=True)
class StandardizationParameters:
    """Per-channel standardization parameters learned from training data."""

    means: pd.Series
    stds: pd.Series

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"mean": self.means, "std": self.stds})


@runtime_checkable
class DatasetAnchoredData(Protocol):
    """Small contract required by shared dataset-anchored model builders."""

    train_dataset: Dataset
    test_dataset: Dataset
    channel_names: Tuple[str, ...]
    class_labels: Tuple[int, ...]
    standardization: StandardizationParameters


def compute_file_hashes(source_files: Mapping[str, Path]) -> Dict[str, str]:
    """Return SHA-256 digests, using an empty string for a missing file."""

    hashes: Dict[str, str] = {}
    for name, path in source_files.items():
        if path.exists():
            sha = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha.update(chunk)
            hashes[name] = sha.hexdigest()
        else:
            hashes[name] = ""
    return hashes
