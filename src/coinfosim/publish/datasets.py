"""Dataset manifest generation and verified copying for GitHub Pages.

Every file is verified against the packaged catalog before it is copied
into the Pages worktree, and verified again after the copy completes.
Nothing is ever published unverified.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Optional

from coinfosim.datasets.catalog import list_datasets, pages_base_url
from coinfosim.datasets.integrity import sha256_file

MANIFEST_SCHEMA_VERSION = 1


class DatasetPublishError(RuntimeError):
    """Raised when a tracked raw file fails verification before/after copying."""


def _coinfosim_version() -> str:
    try:
        return _pkg_version("coinfosim")
    except PackageNotFoundError:
        return "0.0.0+dev"


def build_dataset_manifest(*, source_commit: Optional[str] = None) -> dict[str, Any]:
    """Build the ``datasets/manifest.json`` payload from the packaged catalog."""

    datasets_payload: dict[str, Any] = {}
    for dataset in list_datasets():
        datasets_payload[dataset.slug] = {
            "display_name": dataset.display_name,
            "license": {
                "name": dataset.license.name,
                "url": dataset.license.url,
                "notice": dataset.license.notice,
            },
            "citation": dataset.citation,
            "source_url": dataset.source_url,
            "files": [
                {
                    "filename": file.filename,
                    "sha256": file.sha256,
                    "size_bytes": file.size_bytes,
                    "url": file.url,
                    "relative_path": f"{dataset.local_directory}/{file.filename}",
                }
                for file in dataset.files
            ],
        }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coinfosim_version": _coinfosim_version(),
        "source_commit": source_commit,
        "pages_base_url": pages_base_url(),
        "datasets": datasets_payload,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def write_dataset_manifest(destination: Path, *, source_commit: Optional[str] = None) -> Path:
    """Write the dataset manifest as strict, deterministically ordered JSON."""

    manifest = build_dataset_manifest(source_commit=source_commit)
    payload = json.dumps(manifest, indent=2, allow_nan=False, sort_keys=True) + "\n"
    _atomic_write_text(destination, payload)
    return destination


def sync_dataset_files(repo_root: Path, site_dir: Path) -> list[dict[str, str]]:
    """Copy every tracked raw dataset file into ``site_dir/datasets/<slug>/``.

    Verifies each source file against the packaged catalog before copying,
    and verifies the copy again afterward. Raises :class:`DatasetPublishError`
    immediately on any mismatch rather than publishing a bad file.
    """

    copied: list[dict[str, str]] = []
    for dataset in list_datasets():
        source_dir = repo_root / dataset.repository_raw_directory
        target_dir = site_dir / "datasets" / dataset.local_directory
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in dataset.files:
            source_path = source_dir / file.filename
            if not source_path.is_file():
                raise DatasetPublishError(
                    f"missing tracked raw file for {dataset.slug!r}: {source_path}"
                )
            actual_source_hash = sha256_file(source_path)
            if actual_source_hash != file.sha256:
                raise DatasetPublishError(
                    f"{source_path} does not match the pinned catalog hash "
                    f"(expected {file.sha256}, got {actual_source_hash})"
                )

            target_path = target_dir / file.filename
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target_dir), prefix=file.filename + ".", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "wb") as out_handle, source_path.open("rb") as in_handle:
                    for chunk in iter(lambda: in_handle.read(1 << 16), b""):
                        out_handle.write(chunk)
                os.replace(tmp_name, target_path)
            except BaseException:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise

            copied_hash = sha256_file(target_path)
            if copied_hash != file.sha256:
                raise DatasetPublishError(
                    f"copied file {target_path} does not match the pinned catalog "
                    f"hash after copying (expected {file.sha256}, got {copied_hash})"
                )
            copied.append({"dataset": dataset.slug, "filename": file.filename, "sha256": copied_hash})
    return copied
