"""Atomic, hash-verified download of pinned dataset files.

Uses only the Python standard library. Every download is streamed to a
temporary file inside the destination directory, verified against the
pinned SHA-256 and size, and installed with :func:`os.replace` only after
verification succeeds. Nothing is ever trusted, cached, or installed
unverified.
"""

from __future__ import annotations

import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path
from typing import Optional

from coinfosim.datasets.catalog import DatasetDefinition, DatasetFileDefinition
from coinfosim.datasets.integrity import FileVerificationStatus, sha256_file, verify_file_result

_CHUNK_SIZE = 1 << 16  # 64 KiB


def _user_agent() -> str:
    try:
        pkg_version = _package_version("coinfosim")
    except PackageNotFoundError:
        pkg_version = "0.0.0+dev"
    return f"CoInfoSim/{pkg_version} (+https://paulorenatoaz.github.io/coinfosim/)"


class FileFetchStatus(str, Enum):
    """Disjoint outcomes of fetching one pinned dataset file."""

    ALREADY_VALID = "already_valid"
    DOWNLOADED = "downloaded"
    QUARANTINED_AND_DOWNLOADED = "quarantined_and_downloaded"
    FAILED = "failed"


@dataclass(frozen=True)
class FileFetchResult:
    """Outcome of fetching one pinned file."""

    filename: str
    path: Path
    status: FileFetchStatus
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    quarantined_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is not FileFetchStatus.FAILED


@dataclass(frozen=True)
class DatasetFetchResult:
    """Outcome of fetching every pinned file of one dataset."""

    dataset_slug: str
    directory: Path
    files: tuple[FileFetchResult, ...]

    @property
    def success(self) -> bool:
        return all(file.ok for file in self.files)

    @property
    def failures(self) -> tuple[FileFetchResult, ...]:
        return tuple(file for file in self.files if not file.ok)


class DatasetDownloadError(RuntimeError):
    """Raised by callers that require a hard failure instead of a result object."""


def _quarantine(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    quarantined = path.with_name(f"{path.name}.invalid-{timestamp}")
    suffix = 0
    while quarantined.exists():
        suffix += 1
        quarantined = path.with_name(f"{path.name}.invalid-{timestamp}-{suffix}")
    os.replace(path, quarantined)
    return quarantined


def _download_to_temp(
    url: str,
    destination_dir: Path,
    filename: str,
    *,
    timeout_seconds: float,
) -> Path:
    fd, temp_name = tempfile.mkstemp(
        dir=str(destination_dir), prefix=filename + ".", suffix=".part"
    )
    temp_path = Path(temp_name)
    request = urllib.request.Request(
        url, headers={"User-Agent": _user_agent()}
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return temp_path


def _fetch_one_file(
    file_def: DatasetFileDefinition,
    destination_dir: Path,
    *,
    force: bool,
    timeout_seconds: float,
    require_https: bool,
) -> FileFetchResult:
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest_path = destination_dir / file_def.filename

    if require_https and not file_def.url.lower().startswith("https://"):
        return FileFetchResult(
            filename=file_def.filename,
            path=dest_path,
            status=FileFetchStatus.FAILED,
            error=f"refusing non-HTTPS download URL: {file_def.url}",
        )

    quarantined_path: Optional[Path] = None
    if dest_path.exists():
        existing = verify_file_result(
            dest_path,
            filename=file_def.filename,
            expected_sha256=file_def.sha256,
            expected_size=file_def.size_bytes,
        )
        if existing.status is FileVerificationStatus.VALID and not force:
            return FileFetchResult(
                filename=file_def.filename,
                path=dest_path,
                status=FileFetchStatus.ALREADY_VALID,
                sha256=existing.actual_sha256,
                size_bytes=existing.actual_size,
            )
        if existing.status is not FileVerificationStatus.VALID and not force:
            quarantined_path = _quarantine(dest_path)

    try:
        temp_path = _download_to_temp(
            file_def.url, destination_dir, file_def.filename, timeout_seconds=timeout_seconds
        )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return FileFetchResult(
            filename=file_def.filename,
            path=dest_path,
            status=FileFetchStatus.FAILED,
            quarantined_path=quarantined_path,
            error=str(exc),
        )

    try:
        actual_size = temp_path.stat().st_size
        if actual_size != file_def.size_bytes:
            raise DatasetDownloadError(
                f"downloaded size {actual_size} != expected {file_def.size_bytes}"
            )
        actual_sha256 = sha256_file(temp_path)
        if actual_sha256 != file_def.sha256:
            raise DatasetDownloadError(
                f"downloaded sha256 {actual_sha256} != expected {file_def.sha256}"
            )
        os.replace(temp_path, dest_path)
    except (DatasetDownloadError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        return FileFetchResult(
            filename=file_def.filename,
            path=dest_path,
            status=FileFetchStatus.FAILED,
            quarantined_path=quarantined_path,
            error=str(exc),
        )

    status = (
        FileFetchStatus.QUARANTINED_AND_DOWNLOADED
        if quarantined_path is not None
        else FileFetchStatus.DOWNLOADED
    )
    return FileFetchResult(
        filename=file_def.filename,
        path=dest_path,
        status=status,
        sha256=actual_sha256,
        size_bytes=actual_size,
        quarantined_path=quarantined_path,
    )


def fetch_dataset(
    definition: DatasetDefinition,
    destination: Path,
    *,
    force: bool = False,
    timeout_seconds: float = 60.0,
    require_https: bool = True,
) -> DatasetFetchResult:
    """Download and verify every pinned file of ``definition`` into ``destination``.

    Existing valid files are left untouched (no network access) unless
    ``force`` is set. An existing file that fails verification is moved to a
    timestamped quarantine name before the fresh download is installed,
    unless ``force`` is set, in which case it is simply overwritten
    atomically after the new download verifies.
    """

    destination = Path(destination)
    results = tuple(
        _fetch_one_file(
            file_def,
            destination,
            force=force,
            timeout_seconds=timeout_seconds,
            require_https=require_https,
        )
        for file_def in definition.files
    )
    return DatasetFetchResult(dataset_slug=definition.slug, directory=destination, files=results)
