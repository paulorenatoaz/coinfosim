"""Tests for coinfosim.datasets.download using a local HTTP test server.

These tests must never depend on the live GitHub Pages site.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from coinfosim.datasets.catalog import DatasetDefinition, DatasetFileDefinition, DatasetLicense
from coinfosim.datasets.download import FileFetchStatus, fetch_dataset


@contextmanager
def _serve(routes: dict):
    """Serve ``routes`` (path -> bytes | ("status", code) | ("sleep", seconds))."""

    request_counts: dict[str, int] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            request_counts[self.path] = request_counts.get(self.path, 0) + 1
            route = routes.get(self.path)
            if route is None:
                self.send_response(404)
                self.end_headers()
                return
            if isinstance(route, tuple) and route[0] == "status":
                self.send_response(route[1])
                self.end_headers()
                return
            if isinstance(route, tuple) and route[0] == "sleep":
                time.sleep(route[1])
                self.send_response(200)
                self.end_headers()
                return
            body = route
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", request_counts
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _file_def(filename: str, content: bytes, base_url: str, path: str) -> DatasetFileDefinition:
    return DatasetFileDefinition(
        filename=filename,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        url=f"{base_url}{path}",
    )


def _dataset(files: tuple[DatasetFileDefinition, ...]) -> DatasetDefinition:
    return DatasetDefinition(
        slug="fixture",
        display_name="Fixture Dataset",
        local_directory="fixture",
        repository_raw_directory="data/raw/fixture",
        license=DatasetLicense(name="CC BY 4.0", url="https://example.org", notice="n"),
        citation="cite",
        source_url="https://example.org/fixture",
        files=files,
    )


def test_fetch_dataset_successful_download(tmp_path):
    content = b"alpha content\n"
    with _serve({"/a.txt": content}) as (base_url, counts):
        file_def = _file_def("a.txt", content, base_url, "/a.txt")
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.success
    file_result = result.files[0]
    assert file_result.status is FileFetchStatus.DOWNLOADED
    assert (tmp_path / "a.txt").read_bytes() == content
    assert file_result.sha256 == file_def.sha256


def test_fetch_dataset_missing_file_404(tmp_path):
    with _serve({}) as (base_url, _counts):
        file_def = DatasetFileDefinition(
            filename="a.txt", sha256="0" * 64, size_bytes=10, url=f"{base_url}/missing.txt"
        )
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert not result.success
    assert result.files[0].status is FileFetchStatus.FAILED
    assert not (tmp_path / "a.txt").exists()


def test_fetch_dataset_http_failure(tmp_path):
    with _serve({"/a.txt": ("status", 500)}) as (base_url, _counts):
        file_def = DatasetFileDefinition(
            filename="a.txt", sha256="0" * 64, size_bytes=10, url=f"{base_url}/a.txt"
        )
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.files[0].status is FileFetchStatus.FAILED


def test_fetch_dataset_timeout(tmp_path):
    with _serve({"/a.txt": ("sleep", 1.5)}) as (base_url, _counts):
        file_def = DatasetFileDefinition(
            filename="a.txt", sha256="0" * 64, size_bytes=10, url=f"{base_url}/a.txt"
        )
        result = fetch_dataset(
            _dataset((file_def,)), tmp_path, require_https=False, timeout_seconds=0.2
        )
    assert result.files[0].status is FileFetchStatus.FAILED
    assert not (tmp_path / "a.txt").exists()


def test_fetch_dataset_truncated_response_fails(tmp_path):
    actual_body = b"short"
    with _serve({"/a.txt": actual_body}) as (base_url, _counts):
        # Declares a larger expected size than the server actually sends.
        file_def = DatasetFileDefinition(
            filename="a.txt",
            sha256=hashlib.sha256(actual_body + b"padding").hexdigest(),
            size_bytes=len(actual_body) + len(b"padding"),
            url=f"{base_url}/a.txt",
        )
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.files[0].status is FileFetchStatus.FAILED
    assert not (tmp_path / "a.txt").exists()
    assert not list(tmp_path.glob("a.txt.*.part"))


def test_fetch_dataset_incorrect_hash_fails(tmp_path):
    served_body = b"wrong content xx"
    with _serve({"/a.txt": served_body}) as (base_url, _counts):
        file_def = DatasetFileDefinition(
            filename="a.txt",
            sha256=hashlib.sha256(b"different content").hexdigest(),
            size_bytes=len(served_body),
            url=f"{base_url}/a.txt",
        )
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.files[0].status is FileFetchStatus.FAILED
    assert not (tmp_path / "a.txt").exists()


def test_fetch_dataset_incorrect_size_fails(tmp_path):
    served_body = b"twelve bytes"
    with _serve({"/a.txt": served_body}) as (base_url, _counts):
        file_def = DatasetFileDefinition(
            filename="a.txt",
            sha256=hashlib.sha256(served_body).hexdigest(),
            size_bytes=len(served_body) + 5,
            url=f"{base_url}/a.txt",
        )
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.files[0].status is FileFetchStatus.FAILED
    assert not (tmp_path / "a.txt").exists()


def test_fetch_dataset_no_partial_file_left_behind_on_failure(tmp_path):
    with _serve({"/a.txt": ("status", 500)}) as (base_url, _counts):
        file_def = DatasetFileDefinition(
            filename="a.txt", sha256="0" * 64, size_bytes=10, url=f"{base_url}/a.txt"
        )
        fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    remaining = list(tmp_path.iterdir())
    assert remaining == []


def test_fetch_dataset_cache_hit_makes_no_network_request(tmp_path):
    content = b"already cached\n"
    (tmp_path / "a.txt").write_bytes(content)
    with _serve({"/a.txt": content}) as (base_url, counts):
        file_def = _file_def("a.txt", content, base_url, "/a.txt")
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.files[0].status is FileFetchStatus.ALREADY_VALID
    assert counts == {}


def test_fetch_dataset_force_refresh_redownloads(tmp_path):
    content = b"already cached\n"
    (tmp_path / "a.txt").write_bytes(content)
    with _serve({"/a.txt": content}) as (base_url, counts):
        file_def = _file_def("a.txt", content, base_url, "/a.txt")
        result = fetch_dataset(
            _dataset((file_def,)), tmp_path, require_https=False, force=True
        )
    assert result.files[0].status is FileFetchStatus.DOWNLOADED
    assert counts.get("/a.txt") == 1


def test_fetch_dataset_quarantines_invalid_existing_file_then_downloads(tmp_path):
    content = b"correct content\n"
    (tmp_path / "a.txt").write_bytes(b"stale wrong content")
    with _serve({"/a.txt": content}) as (base_url, _counts):
        file_def = _file_def("a.txt", content, base_url, "/a.txt")
        result = fetch_dataset(_dataset((file_def,)), tmp_path, require_https=False)
    assert result.files[0].status is FileFetchStatus.QUARANTINED_AND_DOWNLOADED
    assert (tmp_path / "a.txt").read_bytes() == content
    quarantined = list(tmp_path.glob("a.txt.invalid-*"))
    assert len(quarantined) == 1


def test_fetch_dataset_https_only_by_default(tmp_path):
    file_def = DatasetFileDefinition(
        filename="a.txt", sha256="0" * 64, size_bytes=10, url="http://example.org/a.txt"
    )
    result = fetch_dataset(_dataset((file_def,)), tmp_path)
    assert result.files[0].status is FileFetchStatus.FAILED
    assert "https" in result.files[0].error.lower()


def test_fetch_dataset_windows_compatible_paths(tmp_path):
    """Destination handling must go through pathlib, not POSIX-only string ops."""

    content = b"cross platform\n"
    nested_destination = Path(str(tmp_path)) / "sub" / "dir"
    with _serve({"/a.txt": content}) as (base_url, _counts):
        file_def = _file_def("a.txt", content, base_url, "/a.txt")
        result = fetch_dataset(_dataset((file_def,)), nested_destination, require_https=False)
    assert result.success
    assert (nested_destination / "a.txt").read_bytes() == content
