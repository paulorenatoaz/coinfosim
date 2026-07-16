# CoInfoSim CLI Reference

The installed console script is `coinfosim` (equivalently, `python -m coinfosim`).

```text
coinfosim --version
coinfosim --help

coinfosim scenario list
coinfosim scenario show <scenario>
coinfosim scenario run <scenario>
coinfosim scenario regenerate <scenario> --run-id <id>

coinfosim dataset list
coinfosim dataset show <dataset>
coinfosim dataset status <dataset>
coinfosim dataset fetch <dataset>
coinfosim dataset verify <dataset>
coinfosim dataset path <dataset>

coinfosim runs scenarios
coinfosim runs simulations
coinfosim runs scenario <run-id>
coinfosim runs simulation <run-id>

coinfosim doctor

coinfosim config show
coinfosim config init
coinfosim config validate

coinfosim publish pages
```

Every command and subcommand supports `--help`.

## Global options

Set once, before the subcommand:

```bash
coinfosim [--log-level LEVEL] [--log-file PATH] [--quiet] [--no-color] [--debug] [--config PATH] <command> ...
```

`--debug` shows full tracebacks on unexpected failures; without it, errors print a concise message, the relevant path/scenario/dataset, and one actionable next command.

## Scenario commands

### `scenario list`

Table of slug, display name, dataset, description, and supported modes. Does not import matplotlib, scikit-learn, or any dataset loader — safe to run even without a fully configured environment.

### `scenario show <scenario>`

Scientific-question detail for one scenario: real-training source, fixed-test source, and an example run command.

### `scenario run <scenario>`

```text
coinfosim scenario run SCENARIO
    --mode smoke|fast|full|full-scale|strict   (default: smoke)
    --data-dir PATH
    --output-dir PATH                          (default: output/reports)
    --backend sequential|process                (default: sequential)
    --workers INTEGER                           (default: 1)
    --worker-threads INTEGER                     (default: 1)
    --start-method auto|spawn|forkserver|fork    (default: auto)
    --no-download
    --refresh-data
    --no-visualizations
    --quiet
    --debug
```

`--workers > 1` requires `--backend process`; the sequential backend always uses exactly one worker and never validates a multiprocessing start method. `--no-download` and `--refresh-data` are mutually exclusive. `--refresh-data` forces a fresh download and re-verification even if a valid cached copy exists.

### `scenario regenerate <scenario> --run-id <id>`

Rebuilds the report hierarchy for a previously completed scenario run using its persisted results only. Never re-invokes the Monte Carlo simulator.

## Dataset commands

`dataset list` shows local availability from the platform cache only (a fast, network-free check); `dataset status` performs a full per-file verification against a specific resolved directory. `dataset fetch` supports `--force` (overwrite an existing valid file), `--destination` (advanced/testing: an explicit target directory instead of the platform cache), and `--base-url` (advanced/testing: override the CoInfoSim Pages base URL — e.g. for a private mirror or a local test server). `dataset path` prints only the resolved, verified directory on stdout, suitable for shell scripting:

```bash
data_dir="$(coinfosim dataset path support2)"
```

## Run registry commands

`runs scenarios` / `runs simulations` list every recorded run from `output/reports/scenario_runs.json` / `simulation_runs.json`. `runs scenario <id>` / `runs simulation <id>` show one record's full detail, including artifact paths.

## `doctor`

Diagnoses the installation: CoInfoSim/Python/OS versions, package install location, writable dataset-cache and output directories, availability of all three built-in datasets, key scientific-library versions, available multiprocessing start methods, logical CPU count, loaded config source, and Git availability (informational only — Git is never required to run a scenario). Pass `--fetch-missing` to download any missing dataset before reporting; by default `doctor` never downloads anything.

## `config` commands

`config show` prints the currently loaded, merged configuration. `config init [--project|--user] [--force]` writes a template `coinfosim.toml` (project) or `~/.config/coinfosim/config.toml` (user). `config validate [--file PATH]` validates a config file (or the currently resolved one).

## `publish pages`

```text
coinfosim publish pages
    --output-dir PATH      (default: output)
    --branch gh-pages       (default: gh-pages)
    --remote origin          (default: origin)
    --push/--no-push
    --dry-run
    --init-branch
```

Regenerates the CoInfoSim Pages site (scenario reports, dataset mirrors, `datasets/manifest.json`, home page) through a temporary git worktree, and pushes only when `--push` is given. See [Releasing](releasing.md) and the publisher module docstring for the exact guarantees (never force-pushes, always removes the worktree, commits only when something changed).

## Legacy commands

`run-simulation`, `run-experiment`, `make-report`, and `cleanup-logs` are the pre-0.2 parameter-model commands, preserved unchanged for this release and marked with a deprecation notice. They operate on synthetic parameter grids, not the dataset-anchored scenarios above. See `coinfosim <command> --help` for each.
