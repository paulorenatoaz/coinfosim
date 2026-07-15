import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coinfosim.datasets.air_quality import (
    AIR_QUALITY_CHANNELS,
    AIR_QUALITY_MISSING_SENTINEL,
    AIR_QUALITY_RAW_FILENAME,
    AIR_QUALITY_REFERENCE,
    AIR_QUALITY_REQUIRED_RAW_COLUMNS,
    AIR_QUALITY_TARGET,
    AIR_QUALITY_TARGET_QUANTILE,
    AIR_QUALITY_TRAIN_FRACTION,
    _read_air_quality_file,
    load_air_quality_data,
)
from coinfosim.datasets.common import DatasetAnchoredData


RAW_DIR = Path("data/raw/air_quality")
EXPECTED_SHA256 = "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996"
EXPECTED_OFFICIAL_COLUMNS = [
    "Date",
    "Time",
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
]


def _official_source_frame() -> pd.DataFrame:
    frame = pd.read_csv(RAW_DIR / AIR_QUALITY_RAW_FILENAME, sep=";", decimal=",")
    frame = frame.dropna(how="all")
    return frame.loc[:, EXPECTED_OFFICIAL_COLUMNS].copy()


def _write_fixture(frame: pd.DataFrame, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / AIR_QUALITY_RAW_FILENAME
    frame.to_csv(path, sep=";", decimal=",", index=False)
    return path


def test_required_csv_exists_and_has_frozen_sha256():
    path = RAW_DIR / AIR_QUALITY_RAW_FILENAME
    assert path.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_parser_reads_official_schema_and_removes_empty_trailers():
    data = load_air_quality_data(RAW_DIR)

    assert list(data.raw_frame.columns) == [*EXPECTED_OFFICIAL_COLUMNS, "timestamp"]
    assert len(data.raw_frame) == 9357
    assert not any(str(column).startswith("Unnamed:") for column in data.raw_frame)
    assert not data.raw_frame.isna().all(axis=1).any()


def test_timestamp_parsing_supports_source_time_and_stable_chronology(tmp_path):
    frame = _official_source_frame()
    frame.loc[0, "Time"] = "18.00"
    frame = frame.iloc[::-1].reset_index(drop=True)
    path = _write_fixture(frame, tmp_path)

    parsed = _read_air_quality_file(path)

    assert parsed["timestamp"].is_monotonic_increasing
    assert parsed["timestamp"].iloc[0] == pd.Timestamp("2004-03-10 18:00:00")
    assert parsed["timestamp"].iloc[-1] == pd.Timestamp("2005-04-04 14:00:00")


def test_duplicate_and_invalid_timestamps_are_rejected(tmp_path):
    frame = _official_source_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = _write_fixture(duplicate, tmp_path / "duplicate")
    with pytest.raises(ValueError, match="duplicate timestamps"):
        _read_air_quality_file(path)

    frame.loc[0, "Time"] = "25.00.00"
    path = _write_fixture(frame, tmp_path / "invalid")
    with pytest.raises(ValueError, match="time data|doesn't match format"):
        _read_air_quality_file(path)


def test_missing_sentinel_becomes_nan_and_complete_case_cohort_is_complete():
    data = load_air_quality_data(RAW_DIR)
    required = [*AIR_QUALITY_CHANNELS, AIR_QUALITY_REFERENCE]

    assert data.raw_frame.loc[:, required].isna().sum().to_dict() == {
        name: 366 for name in required
    }
    assert not (data.raw_frame.loc[:, required] == AIR_QUALITY_MISSING_SENTINEL).any().any()
    assert not data.cohort_frame.loc[:, required].isna().any().any()


def test_exact_channel_order_and_dataset_excludes_reference_and_context():
    data = load_air_quality_data(RAW_DIR)
    excluded = {
        "CO(GT)",
        "NMHC(GT)",
        "C6H6(GT)",
        "NOx(GT)",
        "NO2(GT)",
        "T",
        "RH",
        "AH",
        "Date",
        "Time",
        "timestamp",
    }

    assert data.channel_names == AIR_QUALITY_CHANNELS
    assert len(data.channel_names) == 5
    assert excluded.isdisjoint(data.channel_names)
    assert data.train_dataset.X.shape[1] == 5
    assert data.test_dataset.X.shape[1] == 5
    assert np.array_equal(
        data.train_dataset.X,
        data.standardized_train.loc[:, AIR_QUALITY_CHANNELS].to_numpy(),
    )
    assert isinstance(data, DatasetAnchoredData)


def test_reference_remains_in_audit_frames_but_never_in_dataset_x():
    data = load_air_quality_data(RAW_DIR)

    for frame in (
        data.raw_frame,
        data.cohort_frame,
        data.raw_train,
        data.raw_test,
        data.standardized_train,
        data.standardized_test,
    ):
        assert AIR_QUALITY_REFERENCE in frame
    assert data.train_dataset.X.shape == (data.split_index, len(AIR_QUALITY_CHANNELS))


def test_frozen_official_cohort_split_counts_and_threshold():
    data = load_air_quality_data(RAW_DIR)

    assert data.row_counts() == {
        "raw_non_empty": 9357,
        "complete_case_cohort": 8991,
        "discarded_incomplete": 366,
        "train": 7192,
        "test": 1799,
    }
    assert data.split_index == int(np.floor(AIR_QUALITY_TRAIN_FRACTION * 8991)) == 7192
    assert data.class_counts() == {
        "train": {0: 5374, 1: 1818},
        "test": {0: 1524, 1: 275},
    }
    assert data.threshold_value == pytest.approx(14.5)


def test_split_is_strictly_chronological_with_frozen_ranges():
    data = load_air_quality_data(RAW_DIR)

    assert data.train_first_timestamp == pd.Timestamp("2004-03-10 18:00:00")
    assert data.train_last_timestamp == pd.Timestamp("2005-01-16 01:00:00")
    assert data.test_first_timestamp == pd.Timestamp("2005-01-16 02:00:00")
    assert data.test_last_timestamp == pd.Timestamp("2005-04-04 14:00:00")
    assert data.cutoff_timestamp == data.test_first_timestamp
    assert data.raw_train["timestamp"].max() < data.raw_test["timestamp"].min()


def test_threshold_is_training_only_linear_quantile():
    data = load_air_quality_data(RAW_DIR)
    expected = data.raw_train[AIR_QUALITY_REFERENCE].quantile(
        AIR_QUALITY_TARGET_QUANTILE, interpolation="linear"
    )

    assert data.threshold_quantile == AIR_QUALITY_TARGET_QUANTILE
    assert data.threshold_value == expected


def test_changing_only_test_reference_values_does_not_change_threshold(tmp_path):
    canonical = load_air_quality_data(RAW_DIR)
    frame = _official_source_frame()
    required = [*AIR_QUALITY_CHANNELS, AIR_QUALITY_REFERENCE]
    complete = frame.loc[:, required].replace(AIR_QUALITY_MISSING_SENTINEL, np.nan).notna().all(axis=1)
    test_indices = frame.index[complete][canonical.split_index :]
    frame.loc[test_indices, AIR_QUALITY_REFERENCE] = frame.loc[
        test_indices[::-1], AIR_QUALITY_REFERENCE
    ].to_numpy()
    _write_fixture(frame, tmp_path)

    changed = load_air_quality_data(tmp_path)

    assert changed.threshold_value == canonical.threshold_value
    assert not np.array_equal(
        changed.raw_test[AIR_QUALITY_REFERENCE].to_numpy(),
        canonical.raw_test[AIR_QUALITY_REFERENCE].to_numpy(),
    )


def test_labels_use_greater_than_or_equal_threshold():
    data = load_air_quality_data(RAW_DIR)
    expected_train = (
        data.raw_train[AIR_QUALITY_REFERENCE] >= data.threshold_value
    ).astype(int)
    expected_test = (
        data.raw_test[AIR_QUALITY_REFERENCE] >= data.threshold_value
    ).astype(int)

    assert np.array_equal(data.raw_train[AIR_QUALITY_TARGET], expected_train)
    assert np.array_equal(data.raw_test[AIR_QUALITY_TARGET], expected_test)
    assert (data.raw_train.loc[data.raw_train[AIR_QUALITY_REFERENCE] == 14.5, AIR_QUALITY_TARGET] == 1).all()
    assert data.class_labels == (0, 1)
    assert set(data.raw_train[AIR_QUALITY_TARGET]) == {0, 1}
    assert set(data.raw_test[AIR_QUALITY_TARGET]) == {0, 1}


def test_standardization_is_fit_on_training_only_with_ddof_zero():
    data = load_air_quality_data(RAW_DIR)
    train = data.standardized_train.loc[:, AIR_QUALITY_CHANNELS]

    assert np.allclose(train.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(train.std(ddof=0).to_numpy(), 1.0, atol=1e-12)
    assert np.allclose(
        data.standardization.means.to_numpy(),
        data.raw_train.loc[:, AIR_QUALITY_CHANNELS].mean().to_numpy(),
    )
    assert np.allclose(
        data.standardization.stds.to_numpy(),
        data.raw_train.loc[:, AIR_QUALITY_CHANNELS].std(ddof=0).to_numpy(),
    )


def test_test_standardization_uses_training_parameters():
    data = load_air_quality_data(RAW_DIR)
    expected = (
        data.raw_test.loc[:, AIR_QUALITY_CHANNELS] - data.standardization.means
    ) / data.standardization.stds

    assert np.allclose(
        expected.to_numpy(),
        data.standardized_test.loc[:, AIR_QUALITY_CHANNELS].to_numpy(),
    )


def test_zero_variance_training_channel_is_rejected(tmp_path):
    frame = _official_source_frame()
    frame.loc[:, AIR_QUALITY_CHANNELS[0]] = 1.0
    _write_fixture(frame, tmp_path)

    with pytest.raises(ValueError, match=r"zero std.*PT08.S1\(CO\)"):
        load_air_quality_data(tmp_path)


def test_missing_file_and_missing_required_column_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match=AIR_QUALITY_RAW_FILENAME):
        load_air_quality_data(tmp_path / "missing")

    frame = _official_source_frame().drop(columns=[AIR_QUALITY_CHANNELS[2]])
    _write_fixture(frame, tmp_path / "column")
    with pytest.raises(ValueError, match="missing required columns.*PT08.S3"):
        load_air_quality_data(tmp_path / "column")


def test_required_columns_constant_matches_pipeline_inputs():
    assert AIR_QUALITY_REQUIRED_RAW_COLUMNS == (
        "Date",
        "Time",
        *AIR_QUALITY_CHANNELS,
        AIR_QUALITY_REFERENCE,
    )
