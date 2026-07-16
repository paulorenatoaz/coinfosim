"""Shared pytest configuration.

Forces deterministic, unstyled CLI output for the whole test session.
Rich (via Typer's rich_markup_mode) auto-detects GitHub Actions and other
CI systems and force-enables ANSI color there, which also changes how it
wraps/splits option names (e.g. "--older-than" gets rendered as separate
styled runs "-" + "-older" + "-than"), breaking plain substring
assertions against ``CliRunner`` output. Environment variables like
``NO_COLOR``/``FORCE_COLOR`` do not override that CI-specific detection,
but ``TERM=dumb`` does: it is the highest-priority, POSIX-standard signal
that the terminal supports no special formatting at all, and every CLI
test in this suite relies on it being set before any Rich ``Console`` is
constructed.
"""

from __future__ import annotations

import os

os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "200"
os.environ["NO_COLOR"] = "1"
