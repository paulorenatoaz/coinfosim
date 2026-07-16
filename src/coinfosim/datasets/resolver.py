"""Resolve a verified on-disk directory for a built-in dataset.

Implements the fixed resolution order documented for ``coinfosim scenario
run``:

1. an explicit ``--data-dir``;
2. a dataset-specific path from the loaded CoInfoSim configuration;
3. ``COINFOSIM_DATA_DIR``, interpreted as a root containing dataset
   subdirectories;
4. the verified platform cache;
5. automatic download into the platform cache (unless downloads are
   disabled);
6. a compatibility fallback to ``data/raw/<dataset>`` when running from a
   source checkout.

A present but hash-invalid file is never used silently: any explicitly
named location (``--data-dir``, configuration, or ``COINFOSIM_DATA_DIR``)
that exists but fails verification raises :class:`DatasetIntegrityError`
immediately rather than falling through to the next candidate.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from platformdirs import user_data_dir

from coinfosim.datasets.catalog import DatasetDefinition
from coinfosim.datasets.download import fetch_dataset
from coinfosim.datasets.integrity import DatasetIntegrityError, verify_dataset

_APP_NAME = "coinfosim"


class DatasetResolutionError(RuntimeError):
    """Raised when a dataset directory cannot be located or obtained."""


def default_dataset_cache_root() -> Path:
    """Return ``<platform user data dir>/coinfosim/datasets``."""

    return Path(user_data_dir(_APP_NAME)) / "datasets"


def _candidate_in_root(root: Path, dataset: DatasetDefinition) -> Optional[Path]:
    """Resolve ``root`` as either the direct or nested form for ``dataset``.

    Returns the resolved, verified directory; raises
    :class:`DatasetIntegrityError` if a candidate exists but fails
    verification; returns ``None`` if neither form is present at all.
    """

    direct = root
    nested = root / dataset.local_directory

    direct_present = direct.is_dir() and any(
        (direct / file.filename).exists() for file in dataset.files
    )
    nested_present = nested.is_dir() and any(
        (nested / file.filename).exists() for file in dataset.files
    )

    if direct_present:
        result = verify_dataset(direct, dataset)
        if result.is_valid:
            return direct
        raise DatasetIntegrityError(result.invalid_files[0])

    if nested_present:
        result = verify_dataset(nested, dataset)
        if result.is_valid:
            return nested
        raise DatasetIntegrityError(result.invalid_files[0])

    return None


def _source_checkout_data_dir(dataset: DatasetDefinition) -> Optional[Path]:
    """Return ``<repo_root>/data/raw/<dataset>`` only inside a source checkout.

    Never consults the current working directory: the checkout root is
    located relative to this module's own installed location, so the
    fallback is a no-op for a normal (non-editable) wheel install.
    """

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "data" / "raw").is_dir():
            return ancestor / dataset.repository_raw_directory
    return None


def resolve_dataset_directory(
    dataset: DatasetDefinition,
    *,
    explicit_data_dir: Optional[Path] = None,
    config: Optional[Mapping[str, Any]] = None,
    allow_download: bool = True,
    force_download: bool = False,
) -> Path:
    """Resolve a verified directory containing every pinned file of ``dataset``.

    Raises
    ------
    DatasetIntegrityError
        If an explicitly named location (``--data-dir``, configuration, or
        ``COINFOSIM_DATA_DIR``) exists but fails hash/size verification.
    DatasetResolutionError
        If no candidate directory can be found or obtained.
    """

    if explicit_data_dir is not None:
        resolved = _candidate_in_root(Path(explicit_data_dir), dataset)
        if resolved is not None:
            return resolved
        raise DatasetResolutionError(
            f"--data-dir {explicit_data_dir} does not contain the required "
            f"{dataset.display_name!r} files: {', '.join(dataset.filenames)}"
        )

    config = config or {}
    configured_path = config.get("datasets", {}).get(dataset.slug) if isinstance(config, Mapping) else None
    if configured_path:
        resolved = _candidate_in_root(Path(configured_path), dataset)
        if resolved is not None:
            return resolved
        raise DatasetResolutionError(
            f"configured dataset path {configured_path} does not contain the "
            f"required {dataset.display_name!r} files: {', '.join(dataset.filenames)}"
        )

    env_root = os.environ.get("COINFOSIM_DATA_DIR")
    if env_root:
        resolved = _candidate_in_root(Path(env_root), dataset)
        if resolved is not None:
            return resolved
        raise DatasetResolutionError(
            f"COINFOSIM_DATA_DIR={env_root} does not contain the required "
            f"{dataset.display_name!r} files: {', '.join(dataset.filenames)}"
        )

    cache_dir = default_dataset_cache_root() / dataset.local_directory
    cache_result = verify_dataset(cache_dir, dataset)
    if cache_result.is_valid:
        return cache_dir

    download_error: Optional[str] = None
    if allow_download:
        fetch_result = fetch_dataset(dataset, cache_dir, force=force_download)
        if fetch_result.success:
            verified = verify_dataset(cache_dir, dataset)
            if verified.is_valid:
                return cache_dir
            raise DatasetIntegrityError(verified.invalid_files[0])
        download_error = "; ".join(
            f"{file.filename}: {file.error}" for file in fetch_result.failures
        )

    fallback_dir = _source_checkout_data_dir(dataset)
    if fallback_dir is not None:
        fallback_result = verify_dataset(fallback_dir, dataset)
        if fallback_result.is_valid:
            return fallback_dir
        if fallback_dir.is_dir() and any(
            (fallback_dir / file.filename).exists() for file in dataset.files
        ):
            raise DatasetIntegrityError(fallback_result.invalid_files[0])

    if download_error is not None:
        raise DatasetResolutionError(
            f"failed to download {dataset.display_name!r} from CoInfoSim "
            f"GitHub Pages: {download_error}"
        )
    if not allow_download:
        raise DatasetResolutionError(
            f"{dataset.display_name!r} is not available locally and downloads "
            "are disabled (--no-download); run "
            f"'coinfosim dataset fetch {dataset.slug}' or pass --data-dir to an "
            "existing verified copy."
        )
    raise DatasetResolutionError(
        f"{dataset.display_name!r} could not be resolved from any known location."
    )
