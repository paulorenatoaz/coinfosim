"""Verify tracked raw dataset files agree byte-for-byte with the packaged catalog."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from coinfosim.datasets.air_quality import load_air_quality_data
from coinfosim.datasets.catalog import list_datasets
from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.datasets.support2 import load_support2_data

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.parametrize("dataset", list_datasets(), ids=lambda d: d.slug)
def test_every_catalog_file_exists_on_disk(dataset):
    raw_dir = REPO_ROOT / dataset.repository_raw_directory
    for file in dataset.files:
        assert (raw_dir / file.filename).is_file(), (
            f"missing tracked raw file for {dataset.slug!r}: {raw_dir / file.filename}"
        )


@pytest.mark.parametrize("dataset", list_datasets(), ids=lambda d: d.slug)
def test_every_catalog_file_hash_matches_committed_bytes(dataset):
    raw_dir = REPO_ROOT / dataset.repository_raw_directory
    for file in dataset.files:
        path = raw_dir / file.filename
        assert path.stat().st_size == file.size_bytes, (
            f"{dataset.slug}/{file.filename}: size mismatch "
            f"(catalog={file.size_bytes}, disk={path.stat().st_size})"
        )
        assert _sha256_of(path) == file.sha256, (
            f"{dataset.slug}/{file.filename}: SHA-256 mismatch against pinned catalog"
        )


def test_occupancy_loader_reads_tracked_raw_files():
    data = load_occupancy_data(REPO_ROOT / "data/raw/occupancy")
    assert data.class_labels == (0, 1)
    assert data.d == 5


def test_air_quality_loader_reads_tracked_raw_file():
    data = load_air_quality_data(REPO_ROOT / "data/raw/air_quality")
    assert data.class_labels == (0, 1)
    assert data.d == 5


def test_support2_loader_reads_tracked_raw_file():
    data = load_support2_data(REPO_ROOT / "data/raw/support2")
    assert data.class_labels == (0, 1)
    assert data.d == 7
    assert data.row_counts()["raw"] == 9_105
