"""Regression guard: gh-pages is the only Pages branch name in active code.

Historical planning documents are exempt (only source, scripts, and
workflow files are checked).
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CHECKED_PATHS = [
    REPO_ROOT / "src" / "coinfosim",
    REPO_ROOT / "scripts",
    REPO_ROOT / ".github" / "workflows",
]
_CHECKED_SUFFIXES = {".py", ".sh", ".yml", ".yaml"}


def _candidate_files():
    for root in _CHECKED_PATHS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in _CHECKED_SUFFIXES and "__pycache__" not in path.parts:
                yield path


@pytest.mark.parametrize("path", list(_candidate_files()), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_reports_pages_naming_in_active_publication_code(path):
    text = path.read_text(encoding="utf-8")
    assert "reports-pages" not in text, f"{path} still references the retired 'reports-pages' branch name"
    assert "reports_pages" not in text, f"{path} still references the retired 'reports_pages' naming"
