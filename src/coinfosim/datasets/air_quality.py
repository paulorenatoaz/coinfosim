"""Loader and audit summaries for the UCI Air Quality dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from coinfosim.datasets.common import (
    StandardizationParameters,
    compute_file_hashes,
)
from coinfosim.samplers.dataset import Dataset

AIR_QUALITY_RAW_FILENAME = "AirQualityUCI.csv"
AIR_QUALITY_CHANNELS: Tuple[str, ...] = (
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
)
AIR_QUALITY_REFERENCE = "C6H6(GT)"
AIR_QUALITY_TARGET = "benzene_elevated"
AIR_QUALITY_MISSING_SENTINEL = -200.0
AIR_QUALITY_TRAIN_FRACTION = 0.80
AIR_QUALITY_TARGET_QUANTILE = 0.75
AIR_QUALITY_REQUIRED_RAW_COLUMNS: Tuple[str, ...] = (
    "Date",
    "Time",
    *AIR_QUALITY_CHANNELS,
    AIR_QUALITY_REFERENCE,
)
AIR_QUALITY_MEASUREMENT_COLUMNS: Tuple[str, ...] = (
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
)


@dataclass(frozen=True)
class AirQualityData:
    """Prepared Air Quality cohort and chronological split metadata."""

    raw_dir: Path
    source_files: Mapping[str, Path]
    file_hashes: Mapping[str, str]
    raw_frame: pd.DataFrame
    cohort_frame: pd.DataFrame
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame
    standardized_train: pd.DataFrame
    standardized_test: pd.DataFrame
    train_dataset: Dataset
    test_dataset: Dataset
    standardization: StandardizationParameters
    threshold_value: float
    split_index: int
    train_first_timestamp: pd.Timestamp
    train_last_timestamp: pd.Timestamp
    test_first_timestamp: pd.Timestamp
    test_last_timestamp: pd.Timestamp
    channel_names: Tuple[str, ...] = AIR_QUALITY_CHANNELS
    reference_name: str = AIR_QUALITY_REFERENCE
    target_name: str = AIR_QUALITY_TARGET
    class_labels: Tuple[int, ...] = (0, 1)
    threshold_quantile: float = AIR_QUALITY_TARGET_QUANTILE
    train_fraction: float = AIR_QUALITY_TRAIN_FRACTION

    @property
    def d(self) -> int:
        return len(self.channel_names)

    @property
    def cutoff_timestamp(self) -> pd.Timestamp:
        """First timestamp assigned to the fixed future test split."""

        return self.test_first_timestamp

    def row_counts(self) -> Dict[str, int]:
        return {
            "raw_non_empty": int(len(self.raw_frame)),
            "complete_case_cohort": int(len(self.cohort_frame)),
            "discarded_incomplete": int(len(self.raw_frame) - len(self.cohort_frame)),
            "train": int(len(self.raw_train)),
            "test": int(len(self.raw_test)),
        }

    def missing_counts(self) -> Dict[str, int]:
        required = (*self.channel_names, self.reference_name)
        return {name: int(self.raw_frame[name].isna().sum()) for name in required}

    def class_counts(self) -> Dict[str, Dict[int, int]]:
        counts: Dict[str, Dict[int, int]] = {}
        for name, frame in (("train", self.raw_train), ("test", self.raw_test)):
            values = frame[self.target_name].value_counts().sort_index()
            counts[name] = {int(label): int(count) for label, count in values.items()}
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

    def train_sensor_reference_correlations(self) -> pd.Series:
        columns = (*self.channel_names, self.reference_name)
        correlations = self.raw_train.loc[:, columns].corr()[self.reference_name]
        return correlations.loc[list(self.channel_names)].rename("correlation")


def load_air_quality_data(
    raw_dir: Path | str = "data/raw/air_quality",
) -> AirQualityData:
    """Load, chronologically split, label, and standardize Air Quality data."""

    raw_dir = Path(raw_dir)
    source_path = raw_dir / AIR_QUALITY_RAW_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Missing UCI Air Quality raw file: {source_path}")

    raw_frame = _read_air_quality_file(source_path)
    return _prepare_air_quality_data(raw_frame, raw_dir, source_path)


def _read_air_quality_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", decimal=",")
    frame = _remove_trailing_empty_rows(frame)
    frame = _remove_trailing_unnamed_columns(frame)

    missing_columns = [
        name for name in AIR_QUALITY_REQUIRED_RAW_COLUMNS if name not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Air Quality CSV is missing required columns: {missing_columns}")

    dates = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="raise")
    time_text = frame["Time"].astype("string").str.strip()
    short_time = time_text.str.fullmatch(r"\d{1,2}\.\d{2}")
    full_time = time_text.where(~short_time, time_text + ".00")
    parsed_time = pd.to_datetime(full_time, format="%H.%M.%S", errors="raise")
    time_offset = (
        pd.to_timedelta(parsed_time.dt.hour, unit="h")
        + pd.to_timedelta(parsed_time.dt.minute, unit="m")
        + pd.to_timedelta(parsed_time.dt.second, unit="s")
    )
    frame["timestamp"] = dates + time_offset
    if frame["timestamp"].isna().any():
        raise ValueError("Air Quality CSV contains invalid timestamps")

    for column in AIR_QUALITY_MEASUREMENT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            frame[column] = frame[column].replace(AIR_QUALITY_MISSING_SENTINEL, np.nan)

    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    duplicates = frame["timestamp"].duplicated(keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, "timestamp"].astype(str).unique().tolist()
        raise ValueError(f"Air Quality CSV contains duplicate timestamps: {values}")
    return frame


def _remove_trailing_empty_rows(frame: pd.DataFrame) -> pd.DataFrame:
    nonempty = ~frame.isna().all(axis=1)
    if not nonempty.any():
        raise ValueError("Air Quality CSV contains no non-empty rows")
    last_nonempty = int(np.flatnonzero(nonempty.to_numpy())[-1])
    return frame.iloc[: last_nonempty + 1].copy()


def _remove_trailing_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep_count = len(frame.columns)
    while keep_count:
        column = frame.columns[keep_count - 1]
        if not str(column).startswith("Unnamed:") or not frame[column].isna().all():
            break
        keep_count -= 1
    return frame.iloc[:, :keep_count].copy()


def _prepare_air_quality_data(
    raw_frame: pd.DataFrame,
    raw_dir: Path,
    source_path: Path,
) -> AirQualityData:
    complete_columns = (*AIR_QUALITY_CHANNELS, AIR_QUALITY_REFERENCE)
    cohort = raw_frame.dropna(subset=list(complete_columns)).copy()
    split_index = int(np.floor(AIR_QUALITY_TRAIN_FRACTION * len(cohort)))
    if split_index <= 0 or split_index >= len(cohort):
        raise ValueError("Air Quality complete-case cohort cannot support an 80/20 split")

    train = cohort.iloc[:split_index].copy()
    test = cohort.iloc[split_index:].copy()
    threshold = float(
        train[AIR_QUALITY_REFERENCE].quantile(
            AIR_QUALITY_TARGET_QUANTILE, interpolation="linear"
        )
    )
    cohort[AIR_QUALITY_TARGET] = (
        cohort[AIR_QUALITY_REFERENCE] >= threshold
    ).astype(int)
    train = cohort.iloc[:split_index].copy()
    test = cohort.iloc[split_index:].copy()

    _validate_binary_split(train, "training")
    _validate_binary_split(test, "test")
    if not train["timestamp"].max() < test["timestamp"].min():
        raise ValueError("Air Quality training timestamps must precede test timestamps")

    means = train.loc[:, AIR_QUALITY_CHANNELS].mean()
    stds = train.loc[:, AIR_QUALITY_CHANNELS].std(ddof=0)
    if (stds <= 0).any():
        zero_channels = list(stds[stds <= 0].index)
        raise ValueError(f"training attributes have zero std: {zero_channels}")
    standardization = StandardizationParameters(means=means, stds=stds)
    standardized_train = _standardize_frame(train, standardization)
    standardized_test = _standardize_frame(test, standardization)

    train_dataset = Dataset(
        standardized_train.loc[:, AIR_QUALITY_CHANNELS].to_numpy(dtype=float),
        standardized_train[AIR_QUALITY_TARGET].to_numpy(dtype=int),
    )
    test_dataset = Dataset(
        standardized_test.loc[:, AIR_QUALITY_CHANNELS].to_numpy(dtype=float),
        standardized_test[AIR_QUALITY_TARGET].to_numpy(dtype=int),
    )
    source_files = {AIR_QUALITY_RAW_FILENAME: source_path}
    return AirQualityData(
        raw_dir=raw_dir,
        source_files=source_files,
        file_hashes=compute_file_hashes(source_files),
        raw_frame=raw_frame,
        cohort_frame=cohort,
        raw_train=train,
        raw_test=test,
        standardized_train=standardized_train,
        standardized_test=standardized_test,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        standardization=standardization,
        threshold_value=threshold,
        split_index=split_index,
        train_first_timestamp=pd.Timestamp(train["timestamp"].iloc[0]),
        train_last_timestamp=pd.Timestamp(train["timestamp"].iloc[-1]),
        test_first_timestamp=pd.Timestamp(test["timestamp"].iloc[0]),
        test_last_timestamp=pd.Timestamp(test["timestamp"].iloc[-1]),
    )


def _validate_binary_split(frame: pd.DataFrame, name: str) -> None:
    labels = set(int(value) for value in frame[AIR_QUALITY_TARGET].unique())
    if labels != {0, 1}:
        raise ValueError(
            f"Air Quality {name} split must contain class labels {{0, 1}}, got {labels}"
        )


def _standardize_frame(
    frame: pd.DataFrame, params: StandardizationParameters
) -> pd.DataFrame:
    output = frame.copy()
    output.loc[:, AIR_QUALITY_CHANNELS] = (
        output.loc[:, AIR_QUALITY_CHANNELS] - params.means
    ) / params.stds
    return output


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
