# CoInfoSim — Task Plan for Random Forest Integration into the SUPPORT2 Scenario

## 1. Objective

Implement the SUPPORT2 scenario in CoInfoSim with exactly two classifiers:

1. `linear_svm`
2. `random_forest`

The implementation must:

- preserve the current SUPPORT2 scientific protocol;
- preserve the historical default classifier set for all other scenarios;
- calibrate Random Forest once using only the real SUPPORT2 training reservoir;
- freeze the selected configuration in a versioned JSON artifact;
- reuse the same configuration across all arms, sample sizes, subsets, and replications;
- run Random Forest with `n_jobs=1`;
- retain process-level Monte Carlo parallelism;
- persist classifier selection, parameters, seed policy, and provenance;
- expose those details in generated reports;
- validate the implementation through targeted tests and exactly one final `smoke` execution.

Do not run `full`, `strict`, or `full-scale` experiments.

---

## 2. Fixed Decisions

The decisions in this section are final and must not be reinterpreted during implementation.

### 2.1 SUPPORT2 classifier set

The approved ordered classifier set is:

```python
("linear_svm", "random_forest")
```

The SUPPORT2 CLI must not expose an unrestricted `--classifiers` argument. The classifier set is part of the scientific scenario definition and must be declared in `DatasetAnchoredExecutionSpec`.

### 2.2 Compatibility with existing scenarios

Adding `random_forest` to the global registry must not silently add it to Occupancy, Air Quality, or any other existing scenario.

Introduce two distinct concepts:

```python
REGISTERED_CLASSIFIER_KEYS = (
    "linear_svm",
    "logistic_regression",
    "gaussian_nb",
    "random_forest",
)

DEFAULT_CLASSIFIER_KEYS = (
    "linear_svm",
    "logistic_regression",
    "gaussian_nb",
)
```

Required behavior:

- `available_classifiers()` returns every registered classifier.
- `default_classifiers()` returns only the historical default set.
- The simulator uses `default_classifiers()` when no explicit scenario-level classifier selection is provided.

### 2.3 Calibration

Random Forest calibration must:

- run through a separate command;
- use only the real SUPPORT2 training reservoir;
- never receive or access the fixed test set;
- use the approved search space;
- generate a canonical JSON artifact;
- never run automatically as part of a normal scenario execution;
- have no silent fallback;
- fail explicitly when the artifact is missing, malformed, stale, or incompatible.

### 2.4 Parallelism

- Outer parallelism: the existing `ProcessPoolExecutor`.
- Final smoke execution: five outer processes.
- `worker_inner_threads=1`.
- Random Forest: `n_jobs=1`.
- Nested Random Forest parallelism is prohibited.

### 2.5 One-class training samples

Before fitting any classifier, verify that the training sample contains at least two classes.

When only one class is present:

- fail explicitly;
- do not use a constant classifier fallback;
- do not add Random Forest-specific recovery behavior;
- preserve the process executor's existing error propagation behavior.

### 2.6 Random Forest seed policy

The estimator seed must be a deterministic function of:

- `base_seed`;
- a stable classifier namespace and seed policy version;
- `replication_id`.

The estimator seed must not depend on:

- PID;
- worker identity;
- execution backend;
- future completion order;
- feature subset;
- scenario arm;
- `n_per_class`;
- Python's `hash()`.

For a given replication, the same Random Forest seed must be shared across:

- all feature subsets;
- all sample sizes;
- all three scenario arms.

Persist this policy as:

```text
classifier_seed_v1
```

The historical `linear_svm` behavior using `random_state=0` must remain unchanged.

---

## 3. Expected File Changes

### 3.1 Required new files

```text
src/coinfosim/scenarios/support2_rf_calibration.py
scripts/calibrate_support2_random_forest.py
config/calibration/support2_random_forest.json
tests/test_classifier_selection.py
tests/test_support2_random_forest_calibration.py
```

### 3.2 Required modified files

```text
src/coinfosim/classifiers/registry.py
src/coinfosim/simulation/replication.py
src/coinfosim/simulation/execution.py
src/coinfosim/simulation/monte_carlo.py
src/coinfosim/scenarios/dataset_anchored_runner.py
scripts/run_support2_scenario.py
src/coinfosim/simulation/progress.py
src/coinfosim/runs/report_data.py
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/support2_scenario.py
src/coinfosim/reports/report_tables.py
README.md
```

### 3.3 Files that may be modified only when necessary

```text
src/coinfosim/classifiers/__init__.py
src/coinfosim/reports/support2_monte_carlo.py
tests/test_run_registry.py
tests/test_execution_metadata.py
tests/test_process_execution.py
tests/test_parallel_scientific_equivalence.py
tests/test_support2_scenario_runner.py
tests/test_support2_monte_carlo_reports.py
tests/test_support2_scenario_report.py
```

### 3.4 Areas that must remain unchanged

Do not change:

- the `death_180d` target definition;
- the fixed SUPPORT2 split;
- the seven-channel set or its order;
- preprocessing and standardization;
- Gaussian or GMM models;
- samplers;
- Monte Carlo stopping rules;
- mode budgets;
- structural metrics;
- ranking, winner agreement, or N-star calculations;
- scientific defaults for Occupancy or Air Quality;
- historical records and outputs;
- project dependencies;
- full-scale execution behavior.

---

# 4. Block 0 — Repository State and Branch Creation

## Goal

Start from a clean and current `main` branch.

## Commands

```bash
git status --short --branch
git fetch origin
git switch main
git merge --ff-only origin/main
git status --short
git switch -c feature/support2-random-forest
```

## Acceptance criteria

- The working tree is clean before creating the branch.
- The active branch is `feature/support2-random-forest`.
- The branch is based on the current `origin/main`.
- No unrelated local output or report artifacts are added.

## Tests

None.

---

# 5. Block 1 — Registry, Factory, Validation, and Seed Derivation

## Goal

Register Random Forest without changing the historical default classifier set.

## Implementation

Modify:

```text
src/coinfosim/classifiers/registry.py
```

### 5.1 Registry

Add the stable key:

```python
"random_forest"
```

Use the display label:

```text
Random Forest
```

### 5.2 Public APIs

Provide or extend APIs equivalent to:

```python
available_classifiers() -> list[str]
default_classifiers() -> list[str]
resolve_classifier_names(names: Sequence[str] | None) -> tuple[str, ...]
make_classifier(
    key: str,
    *,
    parameters: Mapping[str, object] | None = None,
    random_state: int | None = None,
)
derive_classifier_seed(
    base_seed: int,
    classifier_name: str,
    replication_id: int,
) -> int
```

### 5.3 Classifier selection validation

`resolve_classifier_names()` must:

- accept `None` and return the historical defaults;
- reject an empty selection;
- reject unknown identifiers;
- reject duplicates;
- preserve declaration order;
- return an immutable tuple;
- include the registered identifiers in relevant error messages.

### 5.4 Random Forest construction

Use:

```python
sklearn.ensemble.RandomForestClassifier
```

The factory must accept the frozen parameters and enforce:

```python
n_jobs=1
random_state=<derived seed>
```

The calibration artifact must not be able to override `n_jobs`.

### 5.5 Compatibility

- Preserve the existing behavior of `linear_svm`, `logistic_regression`, and `gaussian_nb`.
- Keep the historical trio as the simulator default.
- Ensure `make_classifier()` returns a fresh, unfitted instance on every call.

## Acceptance criteria

- `available_classifiers()` returns four classifiers.
- `default_classifiers()` returns exactly the historical three.
- Registry order is stable.
- Random Forest is recognized by scikit-learn as a classifier.
- Repeated factory calls return distinct estimator objects.
- Random Forest always has `n_jobs == 1`.
- Seed derivation is deterministic.
- Seed derivation does not use Python's `hash()`.

## Targeted tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_sprint1_subsets_classifiers_metrics.py \
  tests/test_classifier_selection.py
```

---

# 6. Block 2 — Classifier Execution Plan and Monte Carlo Plumbing

## Goal

Transport classifier selection, resolved parameters, provenance, and seed behavior to every estimator fit.

## Implementation

Introduce a small immutable and pickle-safe structure, for example:

```python
@dataclass(frozen=True)
class ClassifierExecutionPlan:
    names: tuple[str, ...]
    parameters: Mapping[str, Mapping[str, object]]
    provenance: Mapping[str, object]
```

Place it in an existing coherent classifier module or in one small new classifier module. Do not introduce a broad framework.

### 6.1 Simulator

`CooperativeMonteCarloSimulator` must:

- accept a resolved classifier execution plan;
- retain compatible support for `classifier_names`;
- use the historical default set when no explicit selection is supplied;
- preserve classifier order;
- persist classifier selection and configuration in the result metadata.

### 6.2 Replication evaluation

`evaluate_replication()` must:

1. obtain the training sample;
2. validate that the sample contains at least two classes;
3. derive the classifier seed for the replication;
4. create a fresh estimator;
5. fit the estimator;
6. evaluate empirical test loss.

For a given replication, Random Forest must receive the same seed across all feature subsets.

### 6.3 Sequential and process executors

Both execution paths must receive the same resolved classifier plan.

The worker context must be:

- immutable;
- pickle-safe;
- installed once per worker;
- independent of runtime reads from the calibration artifact.

## Acceptance criteria

- Sequential and process execution produce identical Random Forest results for deterministic fixtures.
- Future completion order does not alter results.
- A given replication produces the same RF seed every time.
- Different replications produce different RF seeds.
- One-class validation occurs before `fit`.
- No worker creates nested RF parallelism.
- Historical classifier behavior remains stable.

## Targeted tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_process_execution.py \
  tests/test_parallel_scientific_equivalence.py \
  tests/test_classifier_selection.py
```

---

# 7. Block 3 — One-Time SUPPORT2 Random Forest Calibration

## Goal

Select one global Random Forest configuration using only the real SUPPORT2 training reservoir.

## New files

```text
src/coinfosim/scenarios/support2_rf_calibration.py
scripts/calibrate_support2_random_forest.py
tests/test_support2_random_forest_calibration.py
```

## 7.1 Structural test-set isolation

The core calibration function must receive:

- the training `Dataset`;
- training provenance metadata;
- a calibration specification;
- a calibration seed.

It must not receive the full `Support2Data` object or the fixed test dataset.

The CLI script may load `Support2Data`, but it must pass only:

```python
data.train_dataset
```

to the calibration engine.

## 7.2 Search space

Use exactly:

```python
n_estimators = (100, 200)
max_depth = (12, None)
min_samples_leaf = (1, 5)
max_features = ("sqrt", 0.5)
```

This produces:

```text
16 candidates
```

Fixed parameters:

```python
n_jobs = 1
bootstrap = True
criterion = "gini"
```

## 7.3 Representative sample sizes

Use exactly:

```python
n_per_class = (32, 256, 1024)
```

## 7.4 Internal validation protocol

Use:

```python
StratifiedShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=calibration_seed,
)
```

For each split:

1. create development and validation partitions inside the training reservoir;
2. extract balanced training samples for each representative size;
3. evaluate on the complete internal validation partition;
4. reuse the same splits and sampled data for every candidate.

Canonical fit count:

```text
16 candidates × 3 sample sizes × 3 splits = 144 fits
```

## 7.5 Metric

Primary metric:

```text
misclassification rate on the internal validation partition
```

Aggregate using the unweighted mean of the nine evaluations.

Also record:

- standard deviation;
- standard error;
- mean loss by sample size;
- each individual evaluation;
- balanced accuracy as an optional diagnostic only.

## 7.6 Selection rule

Apply the one-standard-error rule:

1. identify the candidate with the minimum mean loss;
2. define as eligible every candidate satisfying:
   ```text
   mean_loss <= best_mean_loss + best_standard_error
   ```
3. break ties using:
   1. fewer `n_estimators`;
   2. finite depth before `None`;
   3. smaller finite `max_depth`;
   4. larger `min_samples_leaf`;
   5. `"sqrt"` before `0.5`.

Candidate enumeration order must not affect the selected result.

## 7.7 Calibration artifact

Canonical path:

```text
config/calibration/support2_random_forest.json
```

Required fields:

```text
schema_version
classifier_key
estimator_class
selected_parameters
enforced_parameters
search_space
representative_sample_sizes
validation_splitter
selection_metric
aggregation_rule
selection_rule
tie_breaking_rule
candidate_results
evaluation_results
calibration_seed
seed_policy
dataset_slug
target_name
channel_names
raw_file_sha256
training_partition_fingerprint
split_seed
training_rows
training_class_counts
git_commit
python_version
numpy_version
scikit_learn_version
created_at
calibration_command
```

Write the artifact:

- as strict JSON;
- with `allow_nan=False`;
- atomically;
- with stable ordering where appropriate;
- without overwriting unless `--force` is provided.

## 7.8 Artifact validation

Fail when:

- the file is missing;
- JSON is invalid;
- the schema is unsupported;
- a required field is missing;
- the classifier key is incorrect;
- the raw dataset SHA-256 differs;
- the training fingerprint differs;
- the target differs;
- channels or channel order differ;
- the split seed differs;
- an estimator parameter is invalid;
- `n_jobs != 1`;
- the scikit-learn major version differs;
- the estimator cannot be constructed.

Version policy:

- major mismatch: error;
- minor or patch mismatch: warning recorded in metadata.

## Canonical command

```bash
.venv/bin/python scripts/calibrate_support2_random_forest.py \
  --raw-dir data/raw/support2 \
  --output config/calibration/support2_random_forest.json \
  --calibration-seed 0
```

## Acceptance criteria

- The fixed test set is structurally inaccessible to calibration.
- Calibration is reproducible.
- The canonical artifact contains 16 candidates and 144 evaluations.
- Every Random Forest fit uses `n_jobs=1`.
- The selection rule is documented and applied.
- The project can validate its own artifact.
- Unit tests use a reduced search space and do not perform 144 fits.

## Targeted tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_support2_random_forest_calibration.py \
  tests/test_support2_loader.py
```

---

# 8. Block 4 — Scenario-Level Selection and SUPPORT2 Integration

## Goal

Make the generic dataset-anchored runner execute the same classifier plan in all three arms.

## 8.1 `DatasetAnchoredExecutionSpec`

Add optional fields equivalent to:

```python
classifier_names: tuple[str, ...] | None = None
classifier_configuration_resolver: Callable[..., ClassifierExecutionPlan] | None = None
```

When no selection is provided, use:

```python
default_classifiers()
```

When a selection is provided:

- validate it;
- preserve order;
- apply the same selection to every arm.

## 8.2 SUPPORT2 specification

Declare:

```python
classifier_names=("linear_svm", "random_forest")
```

Add a resolver that:

- locates the calibration artifact;
- validates it against the loaded SUPPORT2 training data;
- creates the execution plan;
- records the artifact path and SHA-256;
- runs before Gaussian/GMM fitting and before Monte Carlo execution.

## 8.3 CLI

Add:

```text
--rf-calibration-file
```

Default:

```text
config/calibration/support2_random_forest.json
```

Do not add `--classifiers`.

## 8.4 Failure order

The runner must:

1. validate static classifier names before expensive work;
2. load the dataset;
3. validate the calibration artifact against the loaded dataset;
4. only then generate dataset reports, fit Gaussian/GMM models, and begin simulations.

## Acceptance criteria

Every SUPPORT2 result must contain:

```python
["linear_svm", "random_forest"]
```

All three arms must share:

- the same classifier order;
- the same RF hyperparameters;
- the same seed policy;
- the same calibration artifact path;
- the same artifact SHA-256.

Scenarios without explicit selection must retain:

```python
["linear_svm", "logistic_regression", "gaussian_nb"]
```

## Targeted tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_support2_scenario_runner.py \
  tests/test_execution_metadata.py
```

Runner tests must use a synthetic valid calibration artifact and a small Random Forest, approximately 3 to 5 trees.

---

# 9. Block 5 — Metadata, Persistence, and Reports

## Goal

Make the scientific configuration auditable without requiring a persistence schema migration.

## 9.1 Result metadata

Persist fields equivalent to:

```json
{
  "classifier_selection": {
    "source": "scenario_spec",
    "ordered_keys": ["linear_svm", "random_forest"]
  },
  "classifier_configurations": {
    "linear_svm": {
      "estimator": "sklearn.svm.SVC",
      "parameters": {
        "kernel": "linear",
        "random_state": 0
      },
      "seed_policy": {
        "kind": "fixed",
        "value": 0
      }
    },
    "random_forest": {
      "estimator": "sklearn.ensemble.RandomForestClassifier",
      "parameters": {
        "n_estimators": 0,
        "max_depth": null,
        "min_samples_leaf": 0,
        "max_features": "sqrt",
        "n_jobs": 1
      },
      "seed_policy": {
        "kind": "per_replication",
        "version": "classifier_seed_v1"
      },
      "calibration": {
        "artifact_path": "",
        "artifact_sha256": "",
        "schema_version": 1,
        "training_partition_fingerprint": ""
      }
    }
  }
}
```

Replace illustrative numeric values with the selected values from the artifact.

## 9.2 Persisted summaries and run records

Include classifier configuration and provenance in:

- `result_data_*.json.gz`;
- `summary_*.json`;
- `simulation.json`;
- `scenario.json`;
- simulation run configuration;
- scenario run configuration.

Do not increment a schema version merely because optional dictionary fields were added.

## 9.3 Console progress

At scenario and simulation startup, display:

```text
Classifiers
Random Forest calibration path
Random Forest calibration SHA-256
Random Forest internal jobs
Execution backend
Workers requested/effective
Numeric threads per worker
```

## 9.4 Monte Carlo reports

Add a generic classifier configuration table containing:

- stable key;
- display label;
- estimator class;
- resolved parameters;
- seed policy;
- calibration artifact and hash;
- internal `n_jobs`.

## 9.5 SUPPORT2 scenario report

Remove the static historical classifier metadata.

Derive the classifier list from `real_result.classifier_names`.

Before rendering, verify that all three arms share the same classifier plan.

For historical records that do not contain the new metadata:

- regenerate reports normally;
- show the classifier names stored in the result;
- state that detailed classifier configuration was not recorded;
- do not require the new calibration artifact.

## 9.6 CSV outputs

Do not change the loss table schema. Existing classifier key and label columns are sufficient.

## Acceptance criteria

- Parameters and provenance are visible in HTML and JSON outputs.
- Historical run records remain readable.
- Historical report regeneration does not depend on the new artifact.
- Reports do not mention classifiers absent from a result.
- The new SUPPORT2 classifier order is SVM followed by Random Forest.

## Targeted tests

```bash
.venv/bin/python -m pytest -q \
  tests/test_support2_monte_carlo_reports.py \
  tests/test_support2_scenario_report.py \
  tests/test_support2_scenario_runner.py \
  tests/test_run_registry.py
```

---

# 10. Block 6 — Canonical Calibration Artifact and Documentation

## Goal

Run the approved calibration procedure and version the resulting artifact.

## Command

```bash
.venv/bin/python scripts/calibrate_support2_random_forest.py \
  --raw-dir data/raw/support2 \
  --output config/calibration/support2_random_forest.json \
  --calibration-seed 0
```

## Required inspection

Verify:

- supported schema;
- canonical SUPPORT2 raw-file hash;
- correct training partition fingerprint;
- 16 candidates;
- 144 evaluations;
- selected parameters;
- correct one-standard-error rule;
- `n_jobs=1`;
- Python, NumPy, and scikit-learn versions;
- Git commit;
- generation command.

## README updates

Document:

- the approved SUPPORT2 classifier set;
- the one-time calibration command;
- the purpose and location of the artifact;
- absence of fallback behavior;
- incompatibility and version policy;
- classifier seed policy;
- distinction between outer and inner parallelism;
- final smoke command.

## Acceptance criteria

- The calibration artifact is versioned.
- The artifact diff is reviewable.
- No large Monte Carlo outputs are committed.
- Calibration is not repeated by the test suite.

## Targeted test

```bash
.venv/bin/python -m pytest -q \
  tests/test_support2_random_forest_calibration.py
```

---

# 11. Block 7 — Targeted Regression Suite and Final Smoke

## 11.1 Final targeted test command

```bash
.venv/bin/python -m pytest -q \
  tests/test_sprint1_subsets_classifiers_metrics.py \
  tests/test_classifier_selection.py \
  tests/test_support2_random_forest_calibration.py \
  tests/test_process_execution.py \
  tests/test_execution_metadata.py \
  tests/test_parallel_scientific_equivalence.py \
  tests/test_support2_scenario_runner.py \
  tests/test_support2_monte_carlo_reports.py \
  tests/test_support2_scenario_report.py \
  tests/test_run_registry.py
```

Do not run the complete test suite unless a concrete technical reason requires it.

## 11.2 Exact final smoke command

Run exactly once after all targeted tests pass:

```bash
set -o pipefail

.venv/bin/python scripts/run_support2_scenario.py \
  --mode smoke \
  --raw-dir data/raw/support2 \
  --output-dir output/validation/support2-random-forest-smoke \
  --rf-calibration-file config/calibration/support2_random_forest.json \
  --execution-backend process \
  --n-jobs 5 \
  --worker-inner-threads 1 \
  --multiprocessing-start-method forkserver \
  --no-color \
  2>&1 | tee output/validation/support2-random-forest-smoke.log
```

## Smoke acceptance criteria

- mode is `smoke`;
- classifiers are exactly SVM and Random Forest;
- backend is `process`;
- requested/effective workers are `5 / 5`;
- numeric threads per worker are `1`;
- Random Forest uses `n_jobs=1`;
- calibration artifact is loaded and validated;
- all three arms complete;
- all arms use the same classifier plan;
- JSON outputs and reports contain parameters, seed policy, and provenance;
- reports and CSVs are generated;
- no `full`, `strict`, or `full-scale` execution occurs.

---

# 12. Final Repository Validation

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff -- \
  src/coinfosim \
  scripts \
  tests \
  config/calibration \
  README.md
```

Verify:

- no out-of-scope changes;
- no large experimental files;
- no temporary files;
- the canonical JSON artifact is versioned;
- smoke output and log are not added to the commit unless explicitly requested;
- history was not rewritten;
- no remote branch was modified without authorization.

---

# 13. Expected Agent Delivery

At completion, the implementation agent must report:

1. a concise architecture summary;
2. all created and modified files;
3. the selected Random Forest configuration;
4. the applied selection rule;
5. the calibration artifact SHA-256;
6. executed tests and results;
7. the smoke command and result;
8. generated report locations;
9. confirmation of five outer workers and RF `n_jobs=1`;
10. any deviation, limitation, or incomplete item;
11. final Git status.

Do not push, open a pull request, or merge without explicit authorization.
