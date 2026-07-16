"""Pinned, installable catalog of built-in CoInfoSim dataset mirrors.

The catalog is the single source of truth for the filenames, sizes, SHA-256
hashes, licenses, and CoInfoSim GitHub Pages URLs of the raw files required by
the three built-in dataset-anchored scenarios. It is packaged as JSON under
``coinfosim.resources`` and loaded with :mod:`importlib.resources`, so it is
available identically whether CoInfoSim is run from a source checkout or an
installed wheel.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

_SCHEMA_VERSION_SUPPORTED = (1,)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ALIASES: Mapping[str, str] = {
    "occupancy": "occupancy",
    "occupancy-detection": "occupancy",
    "occupancy_detection": "occupancy",
    "air-quality": "air-quality",
    "air_quality": "air-quality",
    "airquality": "air-quality",
    "support2": "support2",
    "support-2": "support2",
    "support_2": "support2",
}


class DatasetCatalogError(Exception):
    """Raised when the packaged dataset catalog is missing or malformed."""


class UnknownDatasetError(KeyError):
    """Raised when a dataset slug or alias does not resolve to a definition."""

    def __init__(self, slug_or_alias: str) -> None:
        self.slug_or_alias = slug_or_alias
        super().__init__(
            f"unknown dataset {slug_or_alias!r}; known datasets: "
            f"{', '.join(sorted(set(_ALIASES.values())))}"
        )


@dataclass(frozen=True)
class DatasetFileDefinition:
    """One pinned raw file belonging to a dataset."""

    filename: str
    sha256: str
    size_bytes: int
    url: str


@dataclass(frozen=True)
class DatasetLicense:
    """License or acknowledgment status recorded for a dataset."""

    name: str
    url: str | None
    notice: str


@dataclass(frozen=True)
class DatasetDefinition:
    """Immutable, pinned description of one built-in dataset."""

    slug: str
    display_name: str
    local_directory: str
    repository_raw_directory: str
    license: DatasetLicense
    citation: str
    source_url: str
    files: tuple[DatasetFileDefinition, ...]
    doi: str = ""
    creator: str = ""

    @property
    def filenames(self) -> tuple[str, ...]:
        return tuple(file.filename for file in self.files)


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise DatasetCatalogError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _validate_filename(filename: Any, context: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise DatasetCatalogError(f"{context}: filename must be a non-empty string")
    if filename.startswith("/") or filename.startswith("\\"):
        raise DatasetCatalogError(f"{context}: filename must not be absolute: {filename!r}")
    if ".." in filename.replace("\\", "/").split("/"):
        raise DatasetCatalogError(f"{context}: filename must not contain '..': {filename!r}")
    return filename


def _validate_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise DatasetCatalogError(
            f"{context}: sha256 must be exactly 64 lowercase hex characters, got {value!r}"
        )
    return value


def _validate_size(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DatasetCatalogError(f"{context}: size_bytes must be a positive integer, got {value!r}")
    return value


def _validate_https_url(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.lower().startswith("https://"):
        raise DatasetCatalogError(f"{context}: url must be an HTTPS URL, got {value!r}")
    return value


def _build_file(payload: Mapping[str, Any], context: str) -> DatasetFileDefinition:
    filename = _validate_filename(_require(payload, "filename", context), context)
    sha256 = _validate_sha256(_require(payload, "sha256", context), context)
    size_bytes = _validate_size(_require(payload, "size_bytes", context), context)
    url = _validate_https_url(_require(payload, "url", context), context)
    return DatasetFileDefinition(filename=filename, sha256=sha256, size_bytes=size_bytes, url=url)


def _build_license(payload: Mapping[str, Any], context: str) -> DatasetLicense:
    name = _require(payload, "name", context)
    if not isinstance(name, str) or not name:
        raise DatasetCatalogError(f"{context}: license.name must be a non-empty string")
    url = payload.get("url")
    if url is not None and not isinstance(url, str):
        raise DatasetCatalogError(f"{context}: license.url must be a string or null")
    notice = _require(payload, "notice", context)
    if not isinstance(notice, str) or not notice:
        raise DatasetCatalogError(f"{context}: license.notice must be a non-empty string")
    return DatasetLicense(name=name, url=url, notice=notice)


def _build_dataset(slug: str, payload: Mapping[str, Any]) -> DatasetDefinition:
    context = f"dataset {slug!r}"
    display_name = _require(payload, "display_name", context)
    local_directory = _validate_filename(
        _require(payload, "local_directory", context), context
    )
    repository_raw_directory = _require(payload, "repository_raw_directory", context)
    license_payload = _require(payload, "license", context)
    if not isinstance(license_payload, Mapping):
        raise DatasetCatalogError(f"{context}: license must be an object")
    license_ = _build_license(license_payload, context)
    citation = _require(payload, "citation", context)
    source_url = _require(payload, "source_url", context)

    files_payload = _require(payload, "files", context)
    if not isinstance(files_payload, list) or not files_payload:
        raise DatasetCatalogError(f"{context}: files must be a non-empty array")
    files = tuple(
        _build_file(entry, f"{context} file[{index}]")
        for index, entry in enumerate(files_payload)
    )
    seen_filenames = set()
    for file in files:
        if file.filename in seen_filenames:
            raise DatasetCatalogError(f"{context}: duplicate filename {file.filename!r}")
        seen_filenames.add(file.filename)

    doi = payload.get("doi") or ""
    creator = payload.get("creator") or ""
    if doi and not isinstance(doi, str):
        raise DatasetCatalogError(f"{context}: doi must be a string")
    if creator and not isinstance(creator, str):
        raise DatasetCatalogError(f"{context}: creator must be a string")

    return DatasetDefinition(
        slug=slug,
        display_name=display_name,
        local_directory=local_directory,
        repository_raw_directory=repository_raw_directory,
        license=license_,
        citation=citation,
        source_url=source_url,
        files=files,
        doi=doi,
        creator=creator,
    )


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetCatalogError(f"duplicate key {key!r} in dataset catalog JSON")
        result[key] = value
    return result


@lru_cache(maxsize=1)
def _load_catalog_payload() -> Mapping[str, Any]:
    try:
        raw_text = (
            resources.files("coinfosim.resources")
            .joinpath("datasets.json")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise DatasetCatalogError("packaged dataset catalog resource not found") from exc

    try:
        return json.loads(raw_text, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise DatasetCatalogError(f"dataset catalog JSON is invalid: {exc}") from exc


@lru_cache(maxsize=1)
def load_dataset_catalog() -> Mapping[str, DatasetDefinition]:
    """Load and validate the packaged dataset catalog.

    Returns
    -------
    Mapping[str, DatasetDefinition]
        Canonical dataset slug -> definition, in catalog file order.

    Raises
    ------
    DatasetCatalogError
        If the packaged catalog is missing, malformed, or fails validation.
    """

    payload = _load_catalog_payload()

    schema_version = payload.get("schema_version")
    if schema_version not in _SCHEMA_VERSION_SUPPORTED:
        raise DatasetCatalogError(
            f"unsupported dataset catalog schema_version {schema_version!r}; "
            f"supported versions: {_SCHEMA_VERSION_SUPPORTED}"
        )

    datasets_payload = payload.get("datasets")
    if not isinstance(datasets_payload, Mapping) or not datasets_payload:
        raise DatasetCatalogError("dataset catalog must contain a non-empty 'datasets' object")

    catalog: dict[str, DatasetDefinition] = {}
    for slug, dataset_payload in datasets_payload.items():
        if not isinstance(dataset_payload, Mapping):
            raise DatasetCatalogError(f"dataset {slug!r} must be an object")
        catalog[slug] = _build_dataset(slug, dataset_payload)

    return catalog


def pages_base_url() -> str:
    """Return the configured CoInfoSim GitHub Pages dataset base URL."""

    payload = _load_catalog_payload()
    base_url = payload.get("pages_base_url")
    if not isinstance(base_url, str) or not base_url.lower().startswith("https://"):
        raise DatasetCatalogError("dataset catalog pages_base_url must be an HTTPS URL")
    return base_url


def list_datasets() -> tuple[DatasetDefinition, ...]:
    """Return all built-in dataset definitions, ordered as declared in the catalog."""

    return tuple(load_dataset_catalog().values())


def _normalize_slug(slug_or_alias: str) -> str | None:
    candidate = slug_or_alias.strip().lower()
    if candidate in _ALIASES:
        return _ALIASES[candidate]
    dashed = candidate.replace("_", "-")
    if dashed in _ALIASES:
        return _ALIASES[dashed]
    underscored = candidate.replace("-", "_")
    if underscored in _ALIASES:
        return _ALIASES[underscored]
    return None


def get_dataset(slug_or_alias: str) -> DatasetDefinition:
    """Resolve ``slug_or_alias`` (including known aliases) to a definition.

    Raises
    ------
    UnknownDatasetError
        If ``slug_or_alias`` does not resolve to any built-in dataset.
    """

    canonical = _normalize_slug(slug_or_alias)
    catalog = load_dataset_catalog()
    if canonical is None or canonical not in catalog:
        raise UnknownDatasetError(slug_or_alias)
    return catalog[canonical]


def with_base_url(dataset: DatasetDefinition, base_url: str) -> DatasetDefinition:
    """Return a copy of ``dataset`` with every file URL rewritten under ``base_url``.

    Intended for explicit advanced/testing use (private mirrors, local test
    servers). The production default is always the pinned CoInfoSim GitHub
    Pages URLs recorded in the catalog; this is never applied automatically.
    """

    from dataclasses import replace

    base = base_url.rstrip("/")
    files = tuple(
        replace(file, url=f"{base}/{dataset.local_directory}/{file.filename}")
        for file in dataset.files
    )
    return replace(dataset, files=files)
