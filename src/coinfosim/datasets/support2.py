"""Validated SUPPORT2 loader for the fixed 180-day mortality scenario."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from coinfosim.datasets.common import StandardizationParameters, compute_file_hashes
from coinfosim.samplers.dataset import Dataset

SUPPORT2_RAW_FILENAME = "support2.csv"
SUPPORT2_RAW_SHA256 = "79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78"
SUPPORT2_RAW_HEADER_COLUMNS: Tuple[str, ...] = (
    "age", "death", "sex", "hospdead", "slos", "d.time", "dzgroup",
    "dzclass", "num.co", "edu", "income", "scoma", "charges", "totcst",
    "totmcst", "avtisst", "race", "sps", "aps", "surv2m", "surv6m",
    "hday", "diabetes", "dementia", "ca", "prg2m", "prg6m", "dnr",
    "dnrday", "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb",
    "bili", "crea", "sod", "ph", "glucose", "bun", "urine", "adlp",
    "adls", "sfdm2", "adlsc",
)
SUPPORT2_INTERNAL_COLUMNS: Tuple[str, ...] = ("id", *SUPPORT2_RAW_HEADER_COLUMNS)
SUPPORT2_CHANNELS: Tuple[str, ...] = (
    "meanbp", "hrt", "resp", "temp", "wblc", "crea", "sod",
)
SUPPORT2_TARGET = "death_180d"
SUPPORT2_TARGET_EVENT_COLUMN = "death"
SUPPORT2_TARGET_TIME_COLUMN = "d.time"
SUPPORT2_TARGET_HORIZON_DAYS = 180
SUPPORT2_STRATIFICATION_COLUMN = "dzgroup"
SUPPORT2_TRAIN_FRACTION = 0.80
SUPPORT2_SPLIT_SEED = 0
SUPPORT2_RAW_ROWS = 9_105

SUPPORT2_COHORT_FINGERPRINT = "5c42d0c15c34abaad9e81dce0c1749e1001e66ca8b663d680cd37c6fecd7c59e"
SUPPORT2_TRAIN_FINGERPRINT = "154809eb0f6759485342138c97f7ef7efc7d45bfc13d5bfdace19e366bea8979"
SUPPORT2_TEST_FINGERPRINT = "74731ff933b9a19cb77dc4c859e797020c7168497befccf4669680c86037f7a7"

_CATEGORICAL_COLUMNS = {
    "sex", "dzgroup", "dzclass", "income", "race", "ca", "dnr", "sfdm2"
}
SUPPORT2_NUMERIC_COLUMNS: Tuple[str, ...] = tuple(
    column for column in SUPPORT2_INTERNAL_COLUMNS if column not in _CATEGORICAL_COLUMNS
)
SUPPORT2_COMPLETE_CASE_COLUMNS: Tuple[str, ...] = (
    *SUPPORT2_CHANNELS,
    SUPPORT2_TARGET_EVENT_COLUMN,
    SUPPORT2_TARGET_TIME_COLUMN,
    SUPPORT2_STRATIFICATION_COLUMN,
)
SUPPORT2_EXCLUDED_PREDICTORS: Tuple[str, ...] = (
    "death", "d.time", "death_180d", "hospdead", "surv2m", "surv6m", "id", "dzgroup",
)


def _column_roles() -> Dict[str, str]:
    roles = {column: "audit_only_excluded" for column in SUPPORT2_INTERNAL_COLUMNS}
    roles["id"] = "identifier_only"
    for column in SUPPORT2_CHANNELS:
        roles[column] = "predictor"
    roles["death"] = "target_construction_only"
    roles["d.time"] = "target_construction_only"
    roles["hospdead"] = "excluded_alternative_outcome"
    roles["surv2m"] = "excluded_outcome_proxy"
    roles["surv6m"] = "excluded_outcome_proxy"
    roles["dzgroup"] = "stratification_and_reporting_only"
    return roles


SUPPORT2_COLUMN_ROLES: Mapping[str, str] = _column_roles()


@dataclass(frozen=True)
class Support2Data:
    """Prepared SUPPORT2 cohort, fixed split, and audit metadata."""

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
    split_seed: int = SUPPORT2_SPLIT_SEED
    channel_names: Tuple[str, ...] = SUPPORT2_CHANNELS
    target_name: str = SUPPORT2_TARGET
    class_labels: Tuple[int, ...] = (0, 1)
    train_fraction: float = SUPPORT2_TRAIN_FRACTION
    column_roles: Mapping[str, str] = field(default_factory=lambda: SUPPORT2_COLUMN_ROLES)

    @property
    def d(self) -> int:
        return len(self.channel_names)

    def row_counts(self) -> Dict[str, int]:
        return {
            "raw": int(len(self.raw_frame)),
            "complete_case_cohort": int(len(self.cohort_frame)),
            "discarded_incomplete": int(len(self.raw_frame) - len(self.cohort_frame)),
            "train": int(len(self.raw_train)),
            "test": int(len(self.raw_test)),
        }

    def class_counts(self) -> Dict[str, Dict[int, int]]:
        return {
            name: _class_counts(frame[self.target_name])
            for name, frame in (
                ("raw", self.raw_frame),
                ("cohort", self.cohort_frame),
                ("train", self.raw_train),
                ("test", self.raw_test),
            )
        }

    def missing_counts(self) -> Dict[str, int]:
        return {
            column: int(self.raw_frame[column].isna().sum())
            for column in SUPPORT2_COMPLETE_CASE_COLUMNS
        }

    def disease_group_counts(self, frame: pd.DataFrame | None = None) -> Dict[str, int]:
        selected = self.cohort_frame if frame is None else frame
        counts = selected[SUPPORT2_STRATIFICATION_COLUMN].value_counts().sort_index()
        return {str(group): int(count) for group, count in counts.items()}

    def joint_stratum_counts(self) -> Dict[str, Dict[str, int]]:
        return {
            name: _joint_stratum(frame).value_counts().sort_index().astype(int).to_dict()
            for name, frame in (
                ("cohort", self.cohort_frame),
                ("train", self.raw_train),
                ("test", self.raw_test),
            )
        }

    def id_fingerprints(self) -> Dict[str, str]:
        return {
            "cohort": _id_fingerprint(self.cohort_frame["id"]),
            "train": _id_fingerprint(self.raw_train["id"]),
            "test": _id_fingerprint(self.raw_test["id"]),
        }

    def split_manifest(self) -> Dict[str, object]:
        """Return the exact persisted cohort and partition membership."""

        return {
            "dataset": "SUPPORT2",
            "target": SUPPORT2_TARGET,
            "split_seed": int(self.split_seed),
            "train_fraction": float(self.train_fraction),
            "test_fraction": float(1.0 - self.train_fraction),
            "stratification_variables": [SUPPORT2_TARGET, SUPPORT2_STRATIFICATION_COLUMN],
            "id_fingerprints": self.id_fingerprints(),
            "cohort_ids": [int(value) for value in self.cohort_frame["id"]],
            "train_ids": [int(value) for value in self.raw_train["id"]],
            "test_ids": [int(value) for value in self.raw_test["id"]],
        }

    def raw_channel_summary(self) -> pd.DataFrame:
        return _channel_summary(self.raw_train, self.raw_test, self.channel_names)

    def standardized_channel_summary(self) -> pd.DataFrame:
        return _channel_summary(
            self.standardized_train, self.standardized_test, self.channel_names
        )

    def train_correlation(self, standardized: bool = True) -> pd.DataFrame:
        frame = self.standardized_train if standardized else self.raw_train
        return frame.loc[:, self.channel_names].corr()


def load_support2_data(raw_dir: Path | str = "data/raw/support2") -> Support2Data:
    """Load the canonical SUPPORT2 source under the approved fixed protocol."""

    raw_dir = Path(raw_dir)
    source_path = raw_dir / SUPPORT2_RAW_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Missing SUPPORT2 raw file: {source_path}")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != SUPPORT2_RAW_SHA256:
        raise ValueError(
            f"SUPPORT2 raw SHA-256 mismatch: expected {SUPPORT2_RAW_SHA256}, got {digest}"
        )
    raw_frame = _read_support2_file(source_path)
    return _prepare_support2_data(raw_frame, raw_dir, source_path)


def _read_support2_file(path: Path | str) -> pd.DataFrame:
    """Read the malformed 47-header/48-field source without implicit indexing."""

    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError("SUPPORT2 CSV is empty") from exc
        if len(header) != 47 or len(set(header)) != 47:
            raise ValueError("SUPPORT2 header must contain exactly 47 unique names")
        if header != SUPPORT2_RAW_HEADER_COLUMNS:
            raise ValueError("SUPPORT2 header does not match the canonical schema")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 48:
                raise ValueError(
                    f"SUPPORT2 row {line_number} has {len(row)} fields; expected 48"
                )
            rows.append(row)
    if len(rows) != SUPPORT2_RAW_ROWS:
        raise ValueError(
            f"SUPPORT2 must contain {SUPPORT2_RAW_ROWS} data rows, got {len(rows)}"
        )

    frame = pd.DataFrame(rows, columns=SUPPORT2_INTERNAL_COLUMNS)
    for column in SUPPORT2_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in _CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip().replace("", pd.NA)

    ids = frame["id"]
    if ids.isna().any() or not np.isfinite(ids.to_numpy(dtype=float)).all():
        raise ValueError("SUPPORT2 id values must be finite")
    if not np.equal(ids, np.floor(ids)).all():
        raise ValueError("SUPPORT2 id values must be integral")
    frame["id"] = ids.astype(np.int64)
    expected_ids = np.arange(1, SUPPORT2_RAW_ROWS + 1, dtype=np.int64)
    if frame["id"].duplicated().any() or not np.array_equal(frame["id"], expected_ids):
        raise ValueError("SUPPORT2 ids must be unique and sequential from 1 through 9105")
    return frame


def _derive_death_180d(death, d_time) -> pd.Series:
    """Validate target sources and derive inclusive 180-day mortality."""

    death_series = pd.Series(death, copy=True)
    time_series = pd.Series(d_time, copy=True, index=death_series.index)
    if death_series.isna().any():
        raise ValueError("death contains missing values")
    if time_series.isna().any():
        raise ValueError("d.time contains missing values")
    try:
        death_values = death_series.to_numpy(dtype=float)
        time_values = time_series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("death and d.time must be numeric") from exc
    if not np.isfinite(death_values).all():
        raise ValueError("death must be finite")
    if not np.isfinite(time_values).all():
        raise ValueError("d.time must be finite")
    labels = set(death_values.tolist())
    if not labels.issubset({0.0, 1.0}):
        raise ValueError(f"death must contain only 0 and 1, got {sorted(labels)}")
    if (time_values < 0).any():
        raise ValueError("d.time must be non-negative")
    insufficient_survivor = (death_values == 0) & (
        time_values <= SUPPORT2_TARGET_HORIZON_DAYS
    )
    if insufficient_survivor.any():
        raise ValueError("all death == 0 patients must have d.time > 180")
    target = (
        (death_values == 1) & (time_values <= SUPPORT2_TARGET_HORIZON_DAYS)
    ).astype(np.int64)
    return pd.Series(target, index=death_series.index, name=SUPPORT2_TARGET)


def _prepare_support2_data(
    raw_frame: pd.DataFrame,
    raw_dir: Path,
    source_path: Path,
    *,
    split_seed: int = SUPPORT2_SPLIT_SEED,
) -> Support2Data:
    """Derive the target, complete cohort, fixed split, and standardization."""

    missing_columns = [
        column for column in SUPPORT2_INTERNAL_COLUMNS if column not in raw_frame.columns
    ]
    if missing_columns:
        raise ValueError(f"SUPPORT2 frame is missing columns: {missing_columns}")
    frame = raw_frame.copy()
    frame[SUPPORT2_TARGET] = _derive_death_180d(
        frame[SUPPORT2_TARGET_EVENT_COLUMN], frame[SUPPORT2_TARGET_TIME_COLUMN]
    )
    if _class_counts(frame[SUPPORT2_TARGET]) != {0: 4_840, 1: 4_265}:
        raise ValueError("SUPPORT2 raw death_180d counts do not match the protocol")

    cohort = frame.dropna(subset=list(SUPPORT2_COMPLETE_CASE_COLUMNS)).copy()
    if len(cohort) != 8_873 or _class_counts(cohort[SUPPORT2_TARGET]) != {
        0: 4_711,
        1: 4_162,
    }:
        raise ValueError("SUPPORT2 complete-cohort counts do not match the protocol")

    indices = cohort.index.to_numpy()
    train_indices, test_indices = train_test_split(
        indices,
        test_size=1.0 - SUPPORT2_TRAIN_FRACTION,
        random_state=split_seed,
        stratify=_joint_stratum(cohort),
    )
    train = cohort.loc[train_indices].sort_values("id", kind="mergesort").copy()
    test = cohort.loc[test_indices].sort_values("id", kind="mergesort").copy()
    if len(train) != 7_098 or len(test) != 1_775:
        raise ValueError("SUPPORT2 split sizes do not match the protocol")
    if _class_counts(train[SUPPORT2_TARGET]) != {0: 3_768, 1: 3_330}:
        raise ValueError("SUPPORT2 training class counts do not match the protocol")
    if _class_counts(test[SUPPORT2_TARGET]) != {0: 943, 1: 832}:
        raise ValueError("SUPPORT2 test class counts do not match the protocol")
    if set(train["id"]) & set(test["id"]):
        raise ValueError("SUPPORT2 training and test ids overlap")
    if set(train["id"]) | set(test["id"]) != set(cohort["id"]):
        raise ValueError("SUPPORT2 split does not partition the complete cohort")
    if len(_joint_stratum(train).value_counts()) != 16 or len(
        _joint_stratum(test).value_counts()
    ) != 16:
        raise ValueError("all 16 SUPPORT2 joint strata must appear in both splits")
    if split_seed == SUPPORT2_SPLIT_SEED:
        observed = {
            "cohort": _id_fingerprint(cohort["id"]),
            "train": _id_fingerprint(train["id"]),
            "test": _id_fingerprint(test["id"]),
        }
        expected = {
            "cohort": SUPPORT2_COHORT_FINGERPRINT,
            "train": SUPPORT2_TRAIN_FINGERPRINT,
            "test": SUPPORT2_TEST_FINGERPRINT,
        }
        if observed != expected:
            raise ValueError(f"SUPPORT2 split fingerprints differ: {observed}")

    means = train.loc[:, SUPPORT2_CHANNELS].mean()
    stds = train.loc[:, SUPPORT2_CHANNELS].std(ddof=0)
    if means.isna().any() or stds.isna().any() or (stds <= 0).any():
        raise ValueError("SUPPORT2 training attributes require finite nonzero variance")
    standardization = StandardizationParameters(means=means, stds=stds)
    standardized_train = _standardize_frame(train, standardization)
    standardized_test = _standardize_frame(test, standardization)
    train_dataset = Dataset(
        standardized_train.loc[:, SUPPORT2_CHANNELS].to_numpy(dtype=float),
        standardized_train[SUPPORT2_TARGET].to_numpy(dtype=int),
    )
    test_dataset = Dataset(
        standardized_test.loc[:, SUPPORT2_CHANNELS].to_numpy(dtype=float),
        standardized_test[SUPPORT2_TARGET].to_numpy(dtype=int),
    )
    source_files = {SUPPORT2_RAW_FILENAME: source_path}
    return Support2Data(
        raw_dir=raw_dir,
        source_files=source_files,
        file_hashes=compute_file_hashes(source_files),
        raw_frame=frame,
        cohort_frame=cohort.sort_values("id", kind="mergesort").copy(),
        raw_train=train,
        raw_test=test,
        standardized_train=standardized_train,
        standardized_test=standardized_test,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        standardization=standardization,
        split_seed=split_seed,
    )


def _standardize_frame(
    frame: pd.DataFrame, params: StandardizationParameters
) -> pd.DataFrame:
    result = frame.copy()
    result.loc[:, SUPPORT2_CHANNELS] = (
        result.loc[:, SUPPORT2_CHANNELS] - params.means
    ) / params.stds
    return result


def _class_counts(values: pd.Series) -> Dict[int, int]:
    counts = values.value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def _joint_stratum(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[SUPPORT2_TARGET].astype("int64").astype(str)
        + "|"
        + frame[SUPPORT2_STRATIFICATION_COLUMN].astype("string")
    )


def _id_fingerprint(ids: pd.Series) -> str:
    payload = "".join(f"{int(value)}\n" for value in sorted(ids))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _channel_summary(
    train: pd.DataFrame, test: pd.DataFrame, channels: Tuple[str, ...]
) -> pd.DataFrame:
    rows = []
    for split_name, frame in (("train", train), ("fixed_test", test)):
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
