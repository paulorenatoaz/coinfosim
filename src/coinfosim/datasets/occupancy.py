"""Loader and summaries for the UCI Occupancy Detection dataset.

Sprint 2 uses the original dataset split:

- ``datatraining.txt`` as the finite training pool;
- ``datatest.txt`` plus ``datatest2.txt`` as the fixed test set.

Standardization parameters are estimated from the training pool only and then
applied to both splits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from coinfosim.samplers.dataset import Dataset

OCCUPANCY_CHANNELS: Tuple[str, ...] = (
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
)
OCCUPANCY_TARGET = "Occupancy"
OCCUPANCY_RAW_FILENAMES: Tuple[str, ...] = (
    "datatraining.txt",
    "datatest.txt",
    "datatest2.txt",
)

_RAW_COLUMNS = ("row_id", "date", *OCCUPANCY_CHANNELS, OCCUPANCY_TARGET)


@dataclass(frozen=True)
class StandardizationParameters:
    """Per-channel standardization parameters learned from training data."""

    means: pd.Series
    stds: pd.Series

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"mean": self.means, "std": self.stds})


@dataclass(frozen=True)
class OccupancyData:
    """Loaded Occupancy Detection data and Sprint 2 split metadata."""

    raw_dir: Path
    source_files: Mapping[str, Path]
    raw_by_file: Mapping[str, pd.DataFrame]
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    standardized_train: pd.DataFrame
    standardized_test: pd.DataFrame
    train_dataset: Dataset
    test_dataset: Dataset
    standardization: StandardizationParameters
    channel_names: Tuple[str, ...] = OCCUPANCY_CHANNELS
    target_name: str = OCCUPANCY_TARGET
    file_hashes: Mapping[str, str] = field(default_factory=dict)

    @property
    def class_labels(self) -> Tuple[int, ...]:
        labels = sorted(set(self.raw_train[self.target_name]) | set(self.raw_test[self.target_name]))
        return tuple(int(label) for label in labels)

    @property
    def d(self) -> int:
        return len(self.channel_names)

    def row_counts(self) -> Dict[str, int]:
        return {name: int(df.shape[0]) for name, df in self.raw_by_file.items()}

    def class_counts_by_file(self) -> Dict[str, Dict[int, int]]:
        counts: Dict[str, Dict[int, int]] = {}
        for name, df in self.raw_by_file.items():
            value_counts = df[self.target_name].value_counts().sort_index()
            counts[name] = {int(k): int(v) for k, v in value_counts.items()}
        return counts

    def raw_channel_summary(self) -> pd.DataFrame:
        return _channel_summary(self.raw_train, self.raw_test, self.channel_names)

    def standardized_channel_summary(self) -> pd.DataFrame:
        return _channel_summary(
            self.standardized_train, self.standardized_test, self.channel_names
        )

    def train_correlation(self, standardized: bool = True) -> pd.DataFrame:
        frame = self.standardized_train if standardized else self.raw_train
        return frame.loc[:, self.channel_names].corr()


def compute_file_hashes(
    source_files: Mapping[str, Path],
) -> Dict[str, str]:
    """Return a dict mapping each filename to its SHA-256 hex digest.

    Only files that exist are hashed; missing files map to the empty string.
    """
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


def load_occupancy_data(raw_dir: Path | str = "data/raw/occupancy") -> OccupancyData:
    """Load and standardize the Sprint 2 Occupancy Detection dataset.

    Raises
    ------
    FileNotFoundError
        If any required raw file is missing.
    ValueError
        If required columns contain missing values, labels are not binary
        ``{0, 1}``, or a training channel has zero standard deviation.
    """

    raw_dir = Path(raw_dir)
    source_files = {name: raw_dir / name for name in OCCUPANCY_RAW_FILENAMES}
    missing = [str(path) for path in source_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Occupancy Detection raw file(s): " + ", ".join(missing)
        )

    raw_by_file = {
        name: _read_occupancy_file(path) for name, path in source_files.items()
    }
    raw_train = raw_by_file["datatraining.txt"].copy()
    raw_test = pd.concat(
        [raw_by_file["datatest.txt"], raw_by_file["datatest2.txt"]],
        ignore_index=True,
    )

    _validate_occupancy_frame(raw_train, "datatraining.txt")
    _validate_occupancy_frame(raw_test, "datatest.txt + datatest2.txt")

    means = raw_train.loc[:, OCCUPANCY_CHANNELS].mean()
    stds = raw_train.loc[:, OCCUPANCY_CHANNELS].std(ddof=0)
    if (stds <= 0).any():
        zero_channels = list(stds[stds <= 0].index)
        raise ValueError(f"training channels have zero std: {zero_channels}")

    standardization = StandardizationParameters(means=means, stds=stds)
    standardized_train = _standardize_frame(raw_train, standardization)
    standardized_test = _standardize_frame(raw_test, standardization)

    train_dataset = Dataset(
        standardized_train.loc[:, OCCUPANCY_CHANNELS].to_numpy(dtype=float),
        standardized_train[OCCUPANCY_TARGET].to_numpy(dtype=int),
    )
    test_dataset = Dataset(
        standardized_test.loc[:, OCCUPANCY_CHANNELS].to_numpy(dtype=float),
        standardized_test[OCCUPANCY_TARGET].to_numpy(dtype=int),
    )

    file_hashes = compute_file_hashes(source_files)

    return OccupancyData(
        raw_dir=raw_dir,
        source_files=source_files,
        raw_by_file=raw_by_file,
        raw_train=raw_train,
        raw_test=raw_test,
        standardized_train=standardized_train,
        standardized_test=standardized_test,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        standardization=standardization,
        file_hashes=file_hashes,
    )


def _read_occupancy_file(path: Path) -> pd.DataFrame:
    """Read a raw Occupancy file with an explicit row-id column.

    The distributed files advertise ``date`` as the first header field, but
    data rows actually begin with a row/index field followed by the timestamp.
    Reading with explicit names preserves both fields.
    """

    df = pd.read_csv(path, header=0, names=_RAW_COLUMNS)
    df = df.loc[:, _RAW_COLUMNS].copy()
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    for channel in OCCUPANCY_CHANNELS:
        df[channel] = pd.to_numeric(df[channel], errors="raise")
    df[OCCUPANCY_TARGET] = pd.to_numeric(
        df[OCCUPANCY_TARGET], errors="raise"
    ).astype(int)
    return df


def _validate_occupancy_frame(df: pd.DataFrame, name: str) -> None:
    required = ("row_id", "date", *OCCUPANCY_CHANNELS, OCCUPANCY_TARGET)
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{name} is missing required columns: {missing_cols}")
    if df.loc[:, required].isna().any().any():
        raise ValueError(f"{name} contains missing values in required columns")
    labels = set(int(v) for v in df[OCCUPANCY_TARGET].unique())
    if labels != {0, 1}:
        raise ValueError(f"{name} must contain class labels {{0, 1}}, got {labels}")


def _standardize_frame(
    df: pd.DataFrame, params: StandardizationParameters
) -> pd.DataFrame:
    out = df.copy()
    out.loc[:, OCCUPANCY_CHANNELS] = (
        out.loc[:, OCCUPANCY_CHANNELS] - params.means
    ) / params.stds
    return out


def _channel_summary(
    train: pd.DataFrame, test: pd.DataFrame, channels: Tuple[str, ...]
) -> pd.DataFrame:
    rows = []
    for split_name, frame in (("train_pool", train), ("fixed_test", test)):
        stats = frame.loc[:, channels].agg(["mean", "std", "min", "max"]).T
        for channel, row in stats.iterrows():
            rows.append(
                {
                    "split": split_name,
                    "channel": channel,
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
            )
    return pd.DataFrame(rows)
