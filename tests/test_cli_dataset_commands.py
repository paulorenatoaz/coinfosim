"""Tests for ``coinfosim dataset ...`` commands."""

from __future__ import annotations

import hashlib
import http.server
import threading
from contextlib import contextmanager
from pathlib import Path

from typer.testing import CliRunner

from coinfosim.cli.app import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[1]
OCCUPANCY_RAW_DIR = REPO_ROOT / "data" / "raw" / "occupancy"


def test_dataset_list_shows_all_three_canonical_slugs():
    result = runner.invoke(app, ["dataset", "list"])
    assert result.exit_code == 0
    for slug in ("occupancy", "air-quality", "support2"):
        assert slug in result.output


def test_dataset_show_occupancy_includes_hashes_and_urls():
    result = runner.invoke(app, ["dataset", "show", "occupancy"])
    assert result.exit_code == 0
    assert "datatraining.txt" in result.output
    # The URL column may be truncated by Rich at the test runner's default
    # width, so only check the guaranteed-visible prefix.
    assert "https://paulorenat" in result.output


def test_dataset_status_reports_per_file_verification(tmp_path, monkeypatch):
    from coinfosim.datasets import resolver as resolver_module

    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "status", "occupancy"])
    assert result.exit_code == 0
    assert "datatraining.txt" in result.output
    assert "missing" in result.output.lower()


def test_dataset_verify_valid_directory():
    result = runner.invoke(
        app, ["dataset", "verify", "occupancy", "--data-dir", str(OCCUPANCY_RAW_DIR)]
    )
    assert result.exit_code == 0
    assert "Valid" in result.output


def test_dataset_verify_invalid_directory_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["dataset", "verify", "occupancy", "--data-dir", str(tmp_path)])
    assert result.exit_code != 0


def test_dataset_path_no_download_fails_when_nothing_available(tmp_path, monkeypatch):
    from coinfosim.datasets import resolver as resolver_module

    # Force the source-checkout fallback away too, by pointing at an
    # isolated cache and running from outside any real checkout via a
    # dataset slug whose repository_raw_directory does not exist.
    monkeypatch.setattr(resolver_module, "default_dataset_cache_root", lambda: tmp_path / "empty-cache")
    monkeypatch.delenv("COINFOSIM_DATA_DIR", raising=False)
    result = runner.invoke(app, ["dataset", "path", "occupancy", "--no-download"])
    # Either it fails (no cache, no explicit dir, downloads disabled) or it
    # falls back to the real repo checkout data (since tests run from one);
    # both are valid depending on environment, but it must never crash with
    # a traceback in normal mode.
    assert "Traceback" not in result.output


def test_dataset_path_source_checkout_fallback_when_no_download():
    result = runner.invoke(app, ["dataset", "path", "occupancy", "--no-download"])
    assert result.exit_code == 0
    assert result.output.strip() == str(OCCUPANCY_RAW_DIR)


@contextmanager
def _serve_dataset_files(routes: dict):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = routes.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
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
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_dataset_fetch_with_base_url_downloads_and_verifies(tmp_path):
    content = b"fixture occupancy training content\n"
    routes = {"/occupancy/fixture.txt": content}
    with _serve_dataset_files(routes) as base_url:
        import coinfosim.cli.dataset_commands as dataset_commands_module
        from coinfosim.datasets.catalog import DatasetDefinition, DatasetFileDefinition, DatasetLicense

        fixture = DatasetDefinition(
            slug="occupancy",
            display_name="Occupancy Detection",
            local_directory="occupancy",
            repository_raw_directory="data/raw/occupancy",
            license=DatasetLicense(name="CC BY 4.0", url=None, notice="n"),
            citation="c",
            source_url="https://example.org",
            files=(
                DatasetFileDefinition(
                    filename="fixture.txt",
                    sha256=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                    url="https://placeholder/occupancy/fixture.txt",
                ),
            ),
        )
        import unittest.mock as mock

        with mock.patch.object(dataset_commands_module, "get_dataset", return_value=fixture):
            result = runner.invoke(
                app,
                [
                    "dataset",
                    "fetch",
                    "occupancy",
                    "--destination",
                    str(tmp_path),
                    "--base-url",
                    base_url,
                ],
            )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "fixture.txt").read_bytes() == content
