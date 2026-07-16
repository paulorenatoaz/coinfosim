"""Tests for coinfosim.datasets.integrity."""

from __future__ import annotations

import hashlib
import os

import pytest

from coinfosim.datasets.catalog import DatasetDefinition, DatasetFileDefinition, DatasetLicense
from coinfosim.datasets.integrity import (
    DatasetIntegrityError,
    FileVerificationStatus,
    sha256_file,
    verify_dataset,
    verify_file,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def dataset_def():
    content_a = b"alpha content\n"
    content_b = b"beta content\n"
    return DatasetDefinition(
        slug="fixture",
        display_name="Fixture Dataset",
        local_directory="fixture",
        repository_raw_directory="data/raw/fixture",
        license=DatasetLicense(name="CC BY 4.0", url="https://example.org", notice="n"),
        citation="cite",
        source_url="https://example.org/fixture",
        files=(
            DatasetFileDefinition(
                filename="a.txt", sha256=_sha256(content_a), size_bytes=len(content_a),
                url="https://example.org/a.txt",
            ),
            DatasetFileDefinition(
                filename="b.txt", sha256=_sha256(content_b), size_bytes=len(content_b),
                url="https://example.org/b.txt",
            ),
        ),
    ), content_a, content_b


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "sample.bin"
    payload = os.urandom(4096)
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_verify_file_valid(tmp_path):
    content = b"hello world"
    path = tmp_path / "file.txt"
    path.write_bytes(content)
    verify_file(path, hashlib.sha256(content).hexdigest(), len(content))  # no raise


def test_verify_file_missing_raises(tmp_path):
    path = tmp_path / "missing.txt"
    with pytest.raises(DatasetIntegrityError) as excinfo:
        verify_file(path, "0" * 64, 10)
    assert excinfo.value.result.status is FileVerificationStatus.MISSING


def test_verify_file_size_mismatch_raises(tmp_path):
    content = b"hello world"
    path = tmp_path / "file.txt"
    path.write_bytes(content)
    with pytest.raises(DatasetIntegrityError) as excinfo:
        verify_file(path, hashlib.sha256(content).hexdigest(), len(content) + 1)
    assert excinfo.value.result.status is FileVerificationStatus.SIZE_MISMATCH


def test_verify_file_hash_mismatch_raises(tmp_path):
    content = b"hello world"
    path = tmp_path / "file.txt"
    path.write_bytes(content)
    with pytest.raises(DatasetIntegrityError) as excinfo:
        verify_file(path, "0" * 64, len(content))
    assert excinfo.value.result.status is FileVerificationStatus.HASH_MISMATCH


def test_verify_dataset_all_valid(tmp_path, dataset_def):
    definition, content_a, content_b = dataset_def
    directory = tmp_path / "fixture"
    directory.mkdir()
    (directory / "a.txt").write_bytes(content_a)
    (directory / "b.txt").write_bytes(content_b)
    result = verify_dataset(directory, definition)
    assert result.is_valid
    assert result.invalid_files == ()


def test_verify_dataset_missing_file(tmp_path, dataset_def):
    definition, content_a, _content_b = dataset_def
    directory = tmp_path / "fixture"
    directory.mkdir()
    (directory / "a.txt").write_bytes(content_a)
    result = verify_dataset(directory, definition)
    assert not result.is_valid
    statuses = {f.filename: f.status for f in result.invalid_files}
    assert statuses == {"b.txt": FileVerificationStatus.MISSING}


def test_verify_dataset_corrupted_file(tmp_path, dataset_def):
    definition, content_a, content_b = dataset_def
    directory = tmp_path / "fixture"
    directory.mkdir()
    (directory / "a.txt").write_bytes(content_a)
    (directory / "b.txt").write_bytes(content_b + b"corrupted")
    result = verify_dataset(directory, definition)
    assert not result.is_valid
    invalid = {f.filename: f.status for f in result.invalid_files}
    assert invalid == {"b.txt": FileVerificationStatus.SIZE_MISMATCH}
