"""Tests for coinfosim.datasets.resolver resolution order (section 2.4)."""

from __future__ import annotations

import hashlib
import http.server
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from coinfosim.datasets import resolver as resolver_module
from coinfosim.datasets.catalog import (
    DatasetDefinition,
    DatasetFileDefinition,
    DatasetLicense,
    get_dataset,
)
from coinfosim.datasets.integrity import DatasetIntegrityError
from coinfosim.datasets.resolver import DatasetResolutionError, resolve_dataset_directory


def _dataset() -> DatasetDefinition:
    content = b"fixture content\n"
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
                filename="a.txt",
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                url="https://example.org/a.txt",
            ),
        ),
    ), content


def _write_valid(directory: Path, content: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a.txt").write_bytes(content)


@contextmanager
def _serve_file(path: str, content: bytes):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != path:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_explicit_data_dir_direct_form(tmp_path):
    definition, content = _dataset()
    _write_valid(tmp_path, content)
    resolved = resolve_dataset_directory(definition, explicit_data_dir=tmp_path)
    assert resolved == tmp_path


def test_explicit_data_dir_nested_form(tmp_path):
    definition, content = _dataset()
    nested = tmp_path / definition.local_directory
    _write_valid(nested, content)
    resolved = resolve_dataset_directory(definition, explicit_data_dir=tmp_path)
    assert resolved == nested


def test_explicit_data_dir_prefers_direct_over_nested(tmp_path):
    definition, content = _dataset()
    _write_valid(tmp_path, content)
    nested = tmp_path / definition.local_directory
    _write_valid(nested, content)
    resolved = resolve_dataset_directory(definition, explicit_data_dir=tmp_path)
    assert resolved == tmp_path


def test_explicit_data_dir_invalid_raises_integrity_error_not_silent(tmp_path):
    definition, _content = _dataset()
    (tmp_path / "a.txt").write_bytes(b"wrong bytes")
    with pytest.raises(DatasetIntegrityError):
        resolve_dataset_directory(definition, explicit_data_dir=tmp_path)


def test_explicit_data_dir_missing_raises_resolution_error(tmp_path):
    definition, _content = _dataset()
    with pytest.raises(DatasetResolutionError):
        resolve_dataset_directory(definition, explicit_data_dir=tmp_path / "does-not-exist")


def test_config_path_used_when_no_explicit_dir(tmp_path):
    definition, content = _dataset()
    configured_dir = tmp_path / "configured"
    _write_valid(configured_dir, content)
    config = {"datasets": {"fixture": str(configured_dir)}}
    resolved = resolve_dataset_directory(definition, config=config, allow_download=False)
    assert resolved == configured_dir


def test_explicit_data_dir_takes_precedence_over_config(tmp_path):
    definition, content = _dataset()
    explicit_dir = tmp_path / "explicit"
    configured_dir = tmp_path / "configured"
    _write_valid(explicit_dir, content)
    _write_valid(configured_dir, content)
    config = {"datasets": {"fixture": str(configured_dir)}}
    resolved = resolve_dataset_directory(
        definition, explicit_data_dir=explicit_dir, config=config, allow_download=False
    )
    assert resolved == explicit_dir


def test_env_var_used_when_no_explicit_or_config(tmp_path, monkeypatch):
    definition, content = _dataset()
    env_root = tmp_path / "env-root"
    nested = env_root / definition.local_directory
    _write_valid(nested, content)
    monkeypatch.setenv("COINFOSIM_DATA_DIR", str(env_root))
    resolved = resolve_dataset_directory(definition, allow_download=False)
    assert resolved == nested


def test_config_takes_precedence_over_env_var(tmp_path, monkeypatch):
    definition, content = _dataset()
    env_root = tmp_path / "env-root" / definition.local_directory
    _write_valid(env_root, content)
    configured_dir = tmp_path / "configured"
    _write_valid(configured_dir, content)
    monkeypatch.setenv("COINFOSIM_DATA_DIR", str(env_root.parent))
    config = {"datasets": {"fixture": str(configured_dir)}}
    resolved = resolve_dataset_directory(definition, config=config, allow_download=False)
    assert resolved == configured_dir


def test_platform_cache_used_when_valid(tmp_path, monkeypatch):
    definition, content = _dataset()
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: cache_root)
    cache_dir = cache_root / definition.local_directory
    _write_valid(cache_dir, content)
    resolved = resolve_dataset_directory(definition, allow_download=False)
    assert resolved == cache_dir


def test_downloads_into_cache_when_allowed(tmp_path, monkeypatch):
    definition, content = _dataset()
    cache_root = tmp_path / "cache-root"
    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: cache_root)
    with _serve_file("/a.txt", content) as base_url:
        file_def = DatasetFileDefinition(
            filename="a.txt",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            url=f"{base_url}/a.txt",
        )
        downloadable = DatasetDefinition(
            slug=definition.slug,
            display_name=definition.display_name,
            local_directory=definition.local_directory,
            repository_raw_directory=definition.repository_raw_directory,
            license=definition.license,
            citation=definition.citation,
            source_url=definition.source_url,
            files=(file_def,),
        )
        # fetch_dataset defaults require_https=True; the local test server is
        # plain HTTP, so this exercises the resolver's own download call path
        # by monkeypatching fetch_dataset to disable the HTTPS requirement.
        original_fetch = resolver_module.fetch_dataset

        def _fetch_without_https(defn, destination, *, force=False):
            return original_fetch(defn, destination, force=force, require_https=False)

        monkeypatch.setattr(resolver_module, "fetch_dataset", _fetch_without_https)
        resolved = resolve_dataset_directory(downloadable, allow_download=True)
    assert resolved == cache_root / definition.local_directory
    assert (resolved / "a.txt").read_bytes() == content


def test_no_download_raises_when_nothing_available(tmp_path, monkeypatch):
    definition, _content = _dataset()
    cache_root = tmp_path / "empty-cache"
    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: cache_root)
    with pytest.raises(DatasetResolutionError):
        resolve_dataset_directory(definition, allow_download=False)


def test_default_dataset_cache_root_uses_platformdirs():
    root = resolver_module.default_dataset_cache_root()
    assert root.name == "datasets"
    assert "coinfosim" in str(root)


def test_source_checkout_fallback_resolves_real_occupancy_dataset(tmp_path, monkeypatch):
    """Running the test suite from a source checkout is itself the fixture."""

    occupancy = get_dataset("occupancy")
    empty_cache_root = tmp_path / "empty-cache"
    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: empty_cache_root)
    resolved = resolve_dataset_directory(occupancy, allow_download=False)
    assert resolved.name == "occupancy"
    for file in occupancy.files:
        assert (resolved / file.filename).exists()
