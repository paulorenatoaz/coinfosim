import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coinfosim.datasets import Support2Data, load_support2_data
from coinfosim.datasets.common import DatasetAnchoredData
from coinfosim.datasets.support2 import (
    SUPPORT2_CHANNELS,
    SUPPORT2_COHORT_FINGERPRINT,
    SUPPORT2_COLUMN_ROLES,
    SUPPORT2_COMPLETE_CASE_COLUMNS,
    SUPPORT2_EXCLUDED_PREDICTORS,
    SUPPORT2_INTERNAL_COLUMNS,
    SUPPORT2_RAW_HEADER_COLUMNS,
    SUPPORT2_TARGET,
    SUPPORT2_TEST_FINGERPRINT,
    SUPPORT2_TRAIN_FINGERPRINT,
    _derive_death_180d,
    _prepare_support2_data,
    _read_support2_file,
)

RAW_DIR = Path("data/raw/support2")
RAW_PATH = RAW_DIR / "support2.csv"
EXPECTED_SHA256 = "79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78"
CHANNELS = ("meanbp", "hrt", "resp", "temp", "wblc", "crea", "sod")


@pytest.fixture(scope="module")
def support2_data():
    return load_support2_data(RAW_DIR)


def _canonical_rows():
    with RAW_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def _write_rows(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_canonical_support2_file_structure_and_hash():
    assert RAW_PATH.exists()
    assert hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256

    with RAW_PATH.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        header = next(rows)
        data_rows = list(rows)

    assert len(header) == 47
    assert len(set(header)) == 47
    assert len(data_rows) == 9_105
    assert {len(row) for row in data_rows} == {48}
    assert {"death", "d.time", "dzgroup", *CHANNELS}.issubset(header)


def test_explicit_parser_reconstructs_sequential_ids_and_exact_schema(support2_data):
    frame = support2_data.raw_frame
    assert tuple(frame.columns) == (*SUPPORT2_INTERNAL_COLUMNS, SUPPORT2_TARGET)
    assert frame["id"].is_unique
    assert frame["id"].iloc[0] == 1
    assert frame["id"].iloc[-1] == 9_105
    assert np.array_equal(frame["id"], np.arange(1, 9_106))
    assert support2_data.channel_names == CHANNELS == SUPPORT2_CHANNELS
    assert len(SUPPORT2_COLUMN_ROLES) == len(SUPPORT2_INTERNAL_COLUMNS) == 48
    assert set(SUPPORT2_COLUMN_ROLES) == set(SUPPORT2_INTERNAL_COLUMNS)
    assert SUPPORT2_COLUMN_ROLES["death"] == "target_construction_only"
    assert SUPPORT2_COLUMN_ROLES["d.time"] == "target_construction_only"
    assert SUPPORT2_COLUMN_ROLES["hospdead"] == "excluded_alternative_outcome"
    assert SUPPORT2_COLUMN_ROLES["dzgroup"] == "stratification_and_reporting_only"


@pytest.mark.parametrize("corruption", ["short_row", "missing_header", "duplicate_header"])
def test_parser_rejects_structural_corruption(tmp_path, corruption):
    header, rows = _canonical_rows()
    if corruption == "short_row":
        rows[10] = rows[10][:-1]
    elif corruption == "missing_header":
        header = header[:-1]
    else:
        header[-1] = header[0]
    path = tmp_path / corruption / "support2.csv"
    _write_rows(path, header, rows)
    with pytest.raises(ValueError, match="47 unique|canonical schema|expected 48"):
        _read_support2_file(path)


def test_death_180d_inclusive_boundary_and_late_death_semantics():
    death = [1, 1, 1, 1, 1, 0]
    d_time = [1, 179, 180, 181, 900, 344]
    assert _derive_death_180d(death, d_time).tolist() == [1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize(
    ("death", "d_time", "message"),
    [
        ([1, None], [10, 20], "death contains missing"),
        ([1, 0], [10, None], "d.time contains missing"),
        ([1, 2], [10, 20], "only 0 and 1"),
        ([1, 0], [10, -1], "non-negative"),
        ([1, 0], [10, np.inf], "finite"),
        ([1, 0], [10, 180], "death == 0"),
    ],
)
def test_death_180d_rejects_invalid_target_sources(death, d_time, message):
    with pytest.raises(ValueError, match=message):
        _derive_death_180d(death, d_time)


def test_canonical_target_audit_and_generation_order(support2_data):
    frame = support2_data.raw_frame
    assert frame[SUPPORT2_TARGET].notna().all()
    assert set(frame[SUPPORT2_TARGET]) == {0, 1}
    assert frame[SUPPORT2_TARGET].value_counts().sort_index().to_dict() == {
        0: 4_840,
        1: 4_265,
    }
    assert frame["death"].value_counts().sort_index().to_dict() == {0: 2_904, 1: 6_201}
    survivors = frame.loc[frame["death"] == 0]
    assert len(survivors) == 2_904
    assert survivors["d.time"].min() == 344
    assert survivors["d.time"].isna().sum() == 0
    assert (survivors["d.time"] <= 180).sum() == 0
    cross = pd.crosstab(frame["death"], frame[SUPPORT2_TARGET])
    assert cross.to_dict() == {0: {0: 2_904, 1: 1_936}, 1: {0: 0, 1: 4_265}}
    incomplete = frame.loc[frame[list(SUPPORT2_COMPLETE_CASE_COLUMNS)].isna().any(axis=1)]
    assert len(incomplete) == 232
    assert incomplete[SUPPORT2_TARGET].notna().all()


def test_complete_cohort_fixed_split_counts_strata_and_fingerprints(support2_data):
    data = support2_data
    assert data.row_counts() == {
        "raw": 9_105,
        "complete_case_cohort": 8_873,
        "discarded_incomplete": 232,
        "train": 7_098,
        "test": 1_775,
    }
    assert data.class_counts() == {
        "raw": {0: 4_840, 1: 4_265},
        "cohort": {0: 4_711, 1: 4_162},
        "train": {0: 3_768, 1: 3_330},
        "test": {0: 943, 1: 832},
    }
    strata = data.joint_stratum_counts()
    assert {name: len(counts) for name, counts in strata.items()} == {
        "cohort": 16,
        "train": 16,
        "test": 16,
    }
    assert min(strata["cohort"].values()) == 143
    assert min(strata["train"].values()) == 114
    assert min(strata["test"].values()) == 29
    assert data.id_fingerprints() == {
        "cohort": SUPPORT2_COHORT_FINGERPRINT,
        "train": SUPPORT2_TRAIN_FINGERPRINT,
        "test": SUPPORT2_TEST_FINGERPRINT,
    }
    assert data.raw_train["id"].is_monotonic_increasing
    assert data.raw_test["id"].is_monotonic_increasing
    assert set(data.raw_train["id"]).isdisjoint(data.raw_test["id"])
    assert set(data.raw_train["id"]) | set(data.raw_test["id"]) == set(
        data.cohort_frame["id"]
    )


def test_fixed_split_is_deterministic_and_a_different_seed_changes_membership(support2_data):
    repeated = load_support2_data(RAW_DIR)
    assert repeated.id_fingerprints() == support2_data.id_fingerprints()
    changed = _prepare_support2_data(
        support2_data.raw_frame.drop(columns=[SUPPORT2_TARGET]),
        RAW_DIR,
        RAW_PATH,
        split_seed=1,
    )
    assert changed.id_fingerprints()["cohort"] == SUPPORT2_COHORT_FINGERPRINT
    assert changed.id_fingerprints()["train"] != SUPPORT2_TRAIN_FINGERPRINT
    assert changed.id_fingerprints()["test"] != SUPPORT2_TEST_FINGERPRINT


def test_training_only_population_standardization_and_predictor_boundary(support2_data):
    data = support2_data
    train = data.standardized_train.loc[:, SUPPORT2_CHANNELS]
    assert isinstance(data, (Support2Data, DatasetAnchoredData))
    assert np.allclose(train.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(train.std(ddof=0).to_numpy(), 1.0, atol=1e-12)
    assert not np.allclose(train.std(ddof=1).to_numpy(), 1.0, atol=1e-12)
    assert np.allclose(
        data.standardization.means,
        data.raw_train.loc[:, SUPPORT2_CHANNELS].mean(),
    )
    assert np.allclose(
        data.standardization.stds,
        data.raw_train.loc[:, SUPPORT2_CHANNELS].std(ddof=0),
    )
    expected_test = (
        data.raw_test.loc[:, SUPPORT2_CHANNELS] - data.standardization.means
    ) / data.standardization.stds
    assert np.allclose(data.standardized_test.loc[:, SUPPORT2_CHANNELS], expected_test)
    assert data.train_dataset.X.shape == (7_098, 7)
    assert data.test_dataset.X.shape == (1_775, 7)
    assert set(SUPPORT2_EXCLUDED_PREDICTORS).isdisjoint(data.channel_names)
    assert np.array_equal(data.train_dataset.X, train.to_numpy())
    assert np.array_equal(
        data.test_dataset.X,
        data.standardized_test.loc[:, SUPPORT2_CHANNELS].to_numpy(),
    )


def test_test_only_mutation_cannot_change_training_parameters(support2_data):
    raw = support2_data.raw_frame.drop(columns=[SUPPORT2_TARGET]).copy()
    test_ids = set(support2_data.raw_test["id"])
    mask = raw["id"].isin(test_ids)
    raw.loc[mask, SUPPORT2_CHANNELS] = raw.loc[mask, SUPPORT2_CHANNELS] + 1_000_000
    changed = _prepare_support2_data(raw, RAW_DIR, RAW_PATH)
    assert np.array_equal(changed.standardization.means, support2_data.standardization.means)
    assert np.array_equal(changed.standardization.stds, support2_data.standardization.stds)
    assert not np.array_equal(changed.test_dataset.X, support2_data.test_dataset.X)


def test_preprocessing_preserves_zero_and_extreme_values_without_imputation_or_clipping(
    support2_data,
):
    data = support2_data
    for split_raw, split_standardized in (
        (data.raw_train, data.standardized_train),
        (data.raw_test, data.standardized_test),
    ):
        assert not split_raw.loc[:, SUPPORT2_CHANNELS].isna().any().any()
        restored = (
            split_standardized.loc[:, SUPPORT2_CHANNELS] * data.standardization.stds
            + data.standardization.means
        )
        assert np.allclose(restored, split_raw.loc[:, SUPPORT2_CHANNELS])
