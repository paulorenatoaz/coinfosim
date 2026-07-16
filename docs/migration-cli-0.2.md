# Migrating to the 0.2 CLI

CoInfoSim 0.2 replaces direct invocation of `scripts/run_occupancy_scenario.py`, `scripts/run_air_quality_scenario.py`, and `scripts/run_support2_scenario.py` with the installable `coinfosim` CLI. The underlying scientific protocol, execution specs, dataset loaders, and reports are unchanged — only how you invoke them has changed.

## Command mapping

```text
python scripts/run_occupancy_scenario.py --mode smoke
→ coinfosim scenario run occupancy --mode smoke

python scripts/run_air_quality_scenario.py --mode smoke
→ coinfosim scenario run air-quality --mode smoke

python scripts/run_support2_scenario.py --mode smoke
→ coinfosim scenario run support2 --mode smoke
```

```text
python scripts/run_occupancy_scenario.py --report-from-scenario-run 6
→ coinfosim scenario regenerate occupancy --run-id 6

python scripts/run_occupancy_scenario.py --list-scenario-runs
→ coinfosim runs scenarios

python scripts/run_occupancy_scenario.py --list-simulation-runs
→ coinfosim runs simulations
```

```text
--raw-dir data/raw/occupancy
→ --data-dir data/raw/occupancy

--execution-backend process --n-jobs 8 --worker-inner-threads 1 --multiprocessing-start-method forkserver
→ --backend process --workers 8 --worker-threads 1 --start-method forkserver
```

The old scripts still exist as thin, deprecated compatibility wrappers — running them from a source checkout still works and prints a migration notice — but they no longer contain their own scientific logic; they delegate to `coinfosim.scenarios.service`, the same orchestration layer the CLI uses.

## Behavior differences to be aware of

- **Datasets are no longer assumed to be pre-placed.** The old scripts required `data/raw/<dataset>` to already exist. The CLI resolves and, by default, automatically downloads and hash-verifies the dataset if it is not found locally (see [`docs/datasets.md`](datasets.md)). Pass `--no-download` for the old fail-if-missing behavior.
- **`--start-method`** now accepts `auto` (new default), which resolves to a safe platform-appropriate method instead of requiring you to pick `forkserver` or `fork` explicitly. `fork` is never selected automatically.
- **SUPPORT2 classifier selection** (`--classifiers`, `--rf-calibration-file` in the old script) is not part of the public CLI surface in 0.2. The built-in `coinfosim scenario run support2` always uses the default classifier set (`linear_svm`, `random_forest`) and the canonical calibration artifact. If you need a custom classifier selection, build a `DatasetAnchoredExecutionSpec` directly via `coinfosim.scenarios.definitions.support2` and call `coinfosim.scenarios.dataset_anchored_runner.run_dataset_anchored_scenario` yourself — the underlying capability is unchanged, only the CLI convenience flag is gone.
- **`--dataset-report-only`** (Occupancy only, in the old script) is not exposed by the new CLI; use `coinfosim.scenarios.definitions.occupancy` and the dataset report callback directly if you need this.
- **Legacy parameter-model commands** (`run-simulation`, `run-experiment`, `make-report`, `cleanup-logs`) are unaffected by this migration — they are a separate, older workflow, preserved as-is under `coinfosim <command>`.

## What did not change

- The three-arm protocol (real, single-Gaussian synthetic, GMM synthetic vs. one fixed real test set).
- Dataset loaders, preprocessing, standardization, and train/test splits.
- Scenario and simulation slugs, report filenames, and registry (`scenario_runs.json` / `simulation_runs.json`) formats — existing run directories from before 0.2 remain readable by `coinfosim runs scenario <id>` / `coinfosim runs simulation <id>` and by `coinfosim scenario regenerate`.
- Report content and scientific conclusions.

## If you have existing `output/reports` from before 0.2

No migration step is required. Point `--output-dir` at the existing directory and the CLI will read the existing registries and extend them with new, monotonically increasing run IDs, exactly as the old scripts did.
