"""Clean-wheel integration test: build, install in a fresh venv, and run.

Marked slow: builds the package, creates a venv, and installs it, which
takes real wall-clock time. Not part of the default fast test run.
"""

from __future__ import annotations

import http.server
import subprocess
import sys
import threading
import venv
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _serve_directory(directory: Path):
    handler_cls = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(directory), **kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    dist_dir = tmp_path_factory.mktemp("dist")
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_venv(tmp_path_factory, built_wheel):
    venv_dir = tmp_path_factory.mktemp("venv") / "env"
    venv.create(venv_dir, with_pip=True)
    pip = venv_dir / "bin" / "pip"
    subprocess.run(
        [str(pip), "install", "--quiet", str(built_wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return venv_dir


def _run_cli(wheel_venv: Path, args: list[str], cwd: Path, timeout: float = 60) -> subprocess.CompletedProcess:
    coinfosim = wheel_venv / "bin" / "coinfosim"
    return subprocess.run(
        [str(coinfosim), *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )


@pytest.mark.slow
def test_installed_wheel_version_and_listings(wheel_venv, tmp_path):
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()

    result = _run_cli(wheel_venv, ["--version"], cwd=unrelated_dir)
    assert result.returncode == 0, result.stderr
    assert "coinfosim" in result.stdout.lower()

    result = _run_cli(wheel_venv, ["scenario", "list"], cwd=unrelated_dir)
    assert result.returncode == 0, result.stderr
    for slug in ("occupancy", "air-quality", "support2"):
        assert slug in result.stdout

    result = _run_cli(wheel_venv, ["dataset", "list"], cwd=unrelated_dir)
    assert result.returncode == 0, result.stderr
    for slug in ("occupancy", "air-quality", "support2"):
        assert slug in result.stdout

    result = _run_cli(wheel_venv, ["doctor"], cwd=unrelated_dir)
    assert result.returncode == 0, result.stderr
    assert "CoInfoSim version" in result.stdout


@pytest.mark.slow
def test_installed_wheel_downloads_and_verifies_dataset(wheel_venv, tmp_path):
    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    destination = tmp_path / "fetched"

    with _serve_directory(REPO_ROOT / "data" / "raw") as base_url:
        result = _run_cli(
            wheel_venv,
            [
                "dataset",
                "fetch",
                "occupancy",
                "--base-url",
                base_url,
                "--destination",
                str(destination),
            ],
            cwd=unrelated_dir,
            timeout=60,
        )
    assert result.returncode == 0, result.stderr
    assert (destination / "datatraining.txt").exists()
    assert (destination / "datatest.txt").exists()
    assert (destination / "datatest2.txt").exists()


@pytest.mark.slow
def test_installed_wheel_runs_tiny_occupancy_scenario(wheel_venv, tmp_path):
    """Exercise the full installed-wheel scenario path with a real (tiny) run.

    Uses the smoke preset (the smallest built-in mode) rather than an
    internal test-only config, since the CLI does not expose custom
    Monte Carlo budgets; this is deliberately the one full real-mode
    invocation the wheel-install test performs.
    """

    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    output_dir = tmp_path / "reports"

    result = _run_cli(
        wheel_venv,
        [
            "scenario",
            "run",
            "occupancy",
            "--mode",
            "smoke",
            "--data-dir",
            str(REPO_ROOT / "data" / "raw" / "occupancy"),
            "--no-download",
            "--no-visualizations",
            "--output-dir",
            str(output_dir),
            "--quiet",
        ],
        cwd=unrelated_dir,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "scenario_runs.json").exists()
    assert (output_dir / "simulation_runs.json").exists()
