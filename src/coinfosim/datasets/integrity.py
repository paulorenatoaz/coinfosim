"""Hash and size verification for pinned dataset files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from coinfosim.datasets.catalog import DatasetDefinition


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``, streaming in fixed chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FileVerificationStatus(str, Enum):
    """Disjoint outcomes of verifying one on-disk file against a pin."""

    VALID = "valid"
    MISSING = "missing"
    SIZE_MISMATCH = "size_mismatch"
    HASH_MISMATCH = "hash_mismatch"
    INACCESSIBLE = "inaccessible"


@dataclass(frozen=True)
class FileVerificationResult:
    """Outcome of verifying one file against its pinned hash and size."""

    filename: str
    path: Path
    status: FileVerificationStatus
    expected_sha256: str
    actual_sha256: Optional[str] = None
    expected_size: Optional[int] = None
    actual_size: Optional[int] = None
    detail: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status is FileVerificationStatus.VALID


@dataclass(frozen=True)
class DatasetVerificationResult:
    """Outcome of verifying every pinned file for one dataset."""

    dataset_slug: str
    directory: Path
    files: tuple[FileVerificationResult, ...]

    @property
    def is_valid(self) -> bool:
        return all(file.is_valid for file in self.files)

    @property
    def invalid_files(self) -> tuple[FileVerificationResult, ...]:
        return tuple(file for file in self.files if not file.is_valid)


class DatasetIntegrityError(RuntimeError):
    """Raised when a file fails integrity verification and must not be used."""

    def __init__(self, result: FileVerificationResult) -> None:
        self.result = result
        message = f"integrity check failed for {result.filename!r} at {result.path}: {result.status.value}"
        if result.detail:
            message = f"{message} ({result.detail})"
        super().__init__(message)


def verify_file_result(
    path: Path,
    *,
    filename: str,
    expected_sha256: str,
    expected_size: Optional[int],
) -> FileVerificationResult:
    if not path.exists():
        return FileVerificationResult(
            filename=filename,
            path=path,
            status=FileVerificationStatus.MISSING,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return FileVerificationResult(
            filename=filename,
            path=path,
            status=FileVerificationStatus.INACCESSIBLE,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            detail=str(exc),
        )
    if expected_size is not None and actual_size != expected_size:
        return FileVerificationResult(
            filename=filename,
            path=path,
            status=FileVerificationStatus.SIZE_MISMATCH,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            actual_size=actual_size,
        )
    try:
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        return FileVerificationResult(
            filename=filename,
            path=path,
            status=FileVerificationStatus.INACCESSIBLE,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            actual_size=actual_size,
            detail=str(exc),
        )
    if actual_sha256 != expected_sha256:
        return FileVerificationResult(
            filename=filename,
            path=path,
            status=FileVerificationStatus.HASH_MISMATCH,
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            expected_size=expected_size,
            actual_size=actual_size,
        )
    return FileVerificationResult(
        filename=filename,
        path=path,
        status=FileVerificationStatus.VALID,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        expected_size=expected_size,
        actual_size=actual_size,
    )


def verify_file(path: Path, expected_sha256: str, expected_size: Optional[int] = None) -> None:
    """Raise :class:`DatasetIntegrityError` unless ``path`` matches the pin."""

    path = Path(path)
    result = verify_file_result(
        path, filename=path.name, expected_sha256=expected_sha256, expected_size=expected_size
    )
    if not result.is_valid:
        raise DatasetIntegrityError(result)


def verify_dataset(directory: Path, definition: DatasetDefinition) -> DatasetVerificationResult:
    """Verify every pinned file of ``definition`` under ``directory``."""

    directory = Path(directory)
    results = tuple(
        verify_file_result(
            directory / file.filename,
            filename=file.filename,
            expected_sha256=file.sha256,
            expected_size=file.size_bytes,
        )
        for file in definition.files
    )
    return DatasetVerificationResult(dataset_slug=definition.slug, directory=directory, files=results)
