#!/usr/bin/env python3
"""Calibrate and freeze the SUPPORT2 Random Forest configuration once."""

from __future__ import annotations

import argparse

from coinfosim.datasets.support2 import load_support2_data
from coinfosim.scenarios.support2_rf_calibration import (
    CANONICAL_ARTIFACT_PATH,
    calibrate_support2_random_forest,
    support2_training_provenance,
    write_calibration_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/support2")
    parser.add_argument("--output", default=str(CANONICAL_ARTIFACT_PATH))
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data = load_support2_data(args.raw_dir)
    # The calibration engine receives only the training Dataset and training
    # provenance.  The fixed test Dataset is never passed or accessed.
    artifact = calibrate_support2_random_forest(
        data.train_dataset,
        support2_training_provenance(data),
        calibration_seed=args.calibration_seed,
    )
    output = write_calibration_artifact(artifact, args.output, force=args.force)
    print(f"Random Forest calibration artifact: {output}")
    print(f"Selected parameters: {artifact['selected_parameters']}")
    print(f"Candidates: {len(artifact['candidate_results'])}")
    print(f"Evaluations: {len(artifact['evaluation_results'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
