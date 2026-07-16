#!/usr/bin/env python3
"""Deprecated compatibility wrapper. Use ``coinfosim scenario run occupancy``.

The Occupancy Detection three-arm scenario now lives in the installable
``coinfosim`` package (see ``coinfosim.scenarios.definitions.occupancy`` and
``coinfosim.scenarios.service``). This script is kept only so existing
invocations from a source checkout keep working; it delegates entirely to
the package scenario service and does not reimplement any part of the
scientific protocol.

Prefer, after ``pip install coinfosim``:

    coinfosim scenario run occupancy --mode smoke

which additionally supports automatic dataset download/verification,
parallel execution backends, and report regeneration options not exposed
by this wrapper.
"""

from __future__ import annotations

import argparse
import sys

from coinfosim.scenarios.service import (
    regenerate_registered_scenario,
    run_registered_scenario,
)
from coinfosim.simulation.config import VALID_MODES
from coinfosim.simulation.progress import CooperativeProgressReporter

SCENARIO_SLUG = "occupancy"
DEFAULT_RAW_DIR = "data/raw/occupancy"

_DEPRECATION_NOTICE = (
    "scripts/run_occupancy_scenario.py is deprecated and will be removed in a "
    "future release.\nUse the installed CLI instead:\n\n"
    "    coinfosim scenario run occupancy --mode smoke\n"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, default="smoke")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", default="output/reports")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--report-from-scenario-run", type=int, default=None)
    args = parser.parse_args()

    print(_DEPRECATION_NOTICE, file=sys.stderr)

    reporter = CooperativeProgressReporter(verbose=not args.quiet, no_color=args.no_color)
    try:
        if args.report_from_scenario_run is not None:
            regenerate_registered_scenario(
                SCENARIO_SLUG,
                scenario_run_id=args.report_from_scenario_run,
                output_dir=args.output_dir,
                reporter=reporter,
            )
        else:
            run_registered_scenario(
                SCENARIO_SLUG,
                mode=args.mode,
                data_dir=args.raw_dir,
                output_dir=args.output_dir,
                reporter=reporter,
                allow_download=False,
            )
    except Exception as exc:  # noqa: BLE001
        reporter.error("Occupancy scenario command failed", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
