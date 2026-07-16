"""Tests for coinfosim.simulation.execution.resolve_start_method."""

from __future__ import annotations

import multiprocessing as mp
import platform

import pytest

from coinfosim.simulation.execution import resolve_start_method


def test_auto_never_selects_fork():
    resolved = resolve_start_method("auto", backend="process")
    assert resolved != "fork"


def test_auto_resolves_to_available_method():
    resolved = resolve_start_method("auto", backend="process")
    assert resolved in mp.get_all_start_methods()


@pytest.mark.skipif(platform.system() != "Linux", reason="platform-specific behavior")
def test_auto_prefers_forkserver_on_linux():
    if "forkserver" in mp.get_all_start_methods():
        assert resolve_start_method("auto", backend="process") == "forkserver"


def test_explicit_valid_method_passes_through():
    available = mp.get_all_start_methods()
    for method in ("spawn", "forkserver", "fork"):
        if method in available:
            assert resolve_start_method(method, backend="process") == method


def test_explicit_unavailable_method_fails_clearly_for_process_backend():
    # Use a method name that is syntactically valid but not on this platform, if any.
    available = mp.get_all_start_methods()
    unavailable = next((m for m in ("fork", "spawn", "forkserver") if m not in available), None)
    if unavailable is None:
        pytest.skip("all start methods available on this platform")
    with pytest.raises(ValueError, match="not available"):
        resolve_start_method(unavailable, backend="process")


def test_unknown_start_method_request_rejected():
    with pytest.raises(ValueError, match="unknown multiprocessing start method"):
        resolve_start_method("bogus", backend="process")


def test_sequential_backend_never_validates_start_method():
    # Even an explicit method name skips availability validation entirely for
    # the sequential backend, since it never uses multiprocessing.
    for method in ("spawn", "forkserver", "fork"):
        assert resolve_start_method(method, backend="sequential") == method


def test_sequential_backend_resolves_auto_consistently():
    resolved = resolve_start_method("auto", backend="sequential")
    assert resolved in ("spawn", "forkserver")
