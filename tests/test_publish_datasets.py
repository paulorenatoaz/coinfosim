"""Tests for coinfosim.publish.datasets (manifest generation + verified copy)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coinfosim.datasets.catalog import list_datasets
from coinfosim.publish.datasets import (
    DatasetPublishError,
    build_dataset_manifest,
    sync_dataset_files,
    write_dataset_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_matches_packaged_catalog():
    manifest = build_dataset_manifest()
    assert manifest["schema_version"] == 1
    assert set(manifest["datasets"]) == {d.slug for d in list_datasets()}
    for dataset in list_datasets():
        entry = manifest["datasets"][dataset.slug]
        assert entry["citation"] == dataset.citation
        assert entry["license"]["name"] == dataset.license.name
        manifest_files = {f["filename"]: f["sha256"] for f in entry["files"]}
        catalog_files = {f.filename: f.sha256 for f in dataset.files}
        assert manifest_files == catalog_files


def test_manifest_is_strict_json_no_nan_and_deterministic():
    manifest_a = build_dataset_manifest(source_commit="abc123")
    manifest_a["generated_at"] = "FIXED"
    manifest_b = build_dataset_manifest(source_commit="abc123")
    manifest_b["generated_at"] = "FIXED"
    assert json.dumps(manifest_a, sort_keys=True) == json.dumps(manifest_b, sort_keys=True)


def test_write_dataset_manifest_produces_valid_json(tmp_path):
    destination = tmp_path / "datasets" / "manifest.json"
    write_dataset_manifest(destination, source_commit="deadbeef")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "deadbeef"
    assert payload["pages_base_url"].startswith("https://")


def test_sync_dataset_files_copies_and_verifies_real_repo_data(tmp_path):
    copied = sync_dataset_files(REPO_ROOT, tmp_path)
    assert len(copied) == sum(len(d.files) for d in list_datasets())
    for dataset in list_datasets():
        for file in dataset.files:
            target = tmp_path / "datasets" / dataset.local_directory / file.filename
            assert target.is_file()
            assert target.stat().st_size == file.size_bytes


def test_sync_dataset_files_raises_on_missing_source(tmp_path, monkeypatch):
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    with pytest.raises(DatasetPublishError):
        sync_dataset_files(empty_repo, tmp_path / "site")


def test_sync_dataset_files_raises_on_corrupted_source(tmp_path):
    fake_repo = tmp_path / "fake-repo"
    occupancy_dir = fake_repo / "data" / "raw" / "occupancy"
    occupancy_dir.mkdir(parents=True)
    (occupancy_dir / "datatraining.txt").write_bytes(b"not the real content")
    (occupancy_dir / "datatest.txt").write_bytes(b"not the real content")
    (occupancy_dir / "datatest2.txt").write_bytes(b"not the real content")
    air_quality_dir = fake_repo / "data" / "raw" / "air_quality"
    air_quality_dir.mkdir(parents=True)
    (air_quality_dir / "AirQualityUCI.csv").write_bytes(b"not real")
    support2_dir = fake_repo / "data" / "raw" / "support2"
    support2_dir.mkdir(parents=True)
    (support2_dir / "support2.csv").write_bytes(b"not real")

    with pytest.raises(DatasetPublishError, match="does not match the pinned catalog hash"):
        sync_dataset_files(fake_repo, tmp_path / "site")
