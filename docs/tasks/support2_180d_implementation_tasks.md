# SUPPORT2 180-Day Mortality Scenario
## Block-Structured Implementation Task Document

**Recommended repository path:** `docs/tasks/support2_180d_implementation_tasks.md`

---

# 1. Objective

Implement a new dataset-anchored CoInfoSim scenario for the SUPPORT2 dataset using a fixed 180-day mortality endpoint:

```python
death_180d = (
    (death == 1)
    & (d_time <= 180)
).astype(int)
```

The completed implementation must produce the same classes of persisted artifacts and reports already available for the existing dataset-anchored scenarios:

1. SUPPORT2 dataset report;
2. Real → Real Monte Carlo report;
3. Single Gaussian → Real Monte Carlo report;
4. GMM → Real Monte Carlo report;
5. consolidated SUPPORT2 scenario-comparison report;
6. persisted scenario and simulation metadata;
7. reproducible report regeneration from persisted run artifacts.

The implementation must reuse the current dataset-anchored architecture and avoid introducing a parallel execution framework.

---

# 2. Authoritative scientific protocol

## 2.1 Target

```text
death_180d
```

Positive class:

```text
death == 1 and d.time <= 180
```

Negative class:

```text
death == 0 or d.time > 180
```

The day-180 boundary is inclusive.

## 2.2 Target-source variables

```text
death
d.time
```

These variables are used only to construct the target.

They must never appear in:

- `Dataset.X`;
- `channel_names`;
- standardization inputs;
- Gaussian fitting inputs;
- GMM fitting inputs;
- classifier inputs;
- subset enumeration;
- feature-selection logic.

## 2.3 Alternative outcome excluded from prediction

```text
hospdead
```

`hospdead` is not the primary target and must not be used as a predictor.

It may appear only in documentation or descriptive comparisons that clearly identify it as an excluded alternative outcome.

## 2.4 Channels

The channel order is fixed:

```python
SUPPORT2_CHANNELS = (
    "meanbp",
    "hrt",
    "resp",
    "temp",
    "wblc",
    "crea",
    "sod",
)
```

Canonical labels:

```text
X1 = meanbp
X2 = hrt
X3 = resp
X4 = temp
X5 = wblc
X6 = crea
X7 = sod
```

## 2.5 Cohort

Use complete cases for:

```text
meanbp
hrt
resp
temp
wblc
crea
sod
death
d.time
dzgroup
```

Expected cohort:

```text
8,873 patients
```

Expected class counts:

```text
death_180d = 0: 4,711
death_180d = 1: 4,162
```

## 2.6 Split

Use:

```text
test_size = 0.20
random_state = 0
stratification = death_180d × dzgroup
```

Sort both resulting partitions by ascending original patient `id`.

Expected split:

```text
Training rows: 7,098
Test rows:     1,775
```

Expected class counts:

```text
Training:
death_180d = 0: 3,768
death_180d = 1: 3,330

Test:
death_180d = 0: 943
death_180d = 1: 832
```

Expected ID fingerprints:

```text
Cohort:
5c42d0c15c34abaad9e81dce0c1749e1001e66ca8b663d680cd37c6fecd7c59e

Training:
154809eb0f6759485342138c97f7ef7efc7d45bfc13d5bfdace19e366bea8979

Test:
74731ff933b9a19cb77dc4c859e797020c7168497befccf4669680c86037f7a7
```

The fingerprints are computed from newline-separated, ascending decimal patient IDs.

## 2.7 Preprocessing

Use training-only z-score standardization:

```text
ddof = 0
```

Do not use:

- imputation;
- clipping;
- winsorization;
- outlier removal;
- logarithmic transformation;
- Box-Cox transformation;
- Yeo-Johnson transformation;
- missingness indicators;
- target-dependent preprocessing;
- dimensionality reduction.

## 2.8 Scenario arms

Implement:

```text
Real → Real
Single Gaussian → Real
GMM → Real
```

All arms must use the exact same fixed real test set.

## 2.9 Classifiers

Use the existing registered classifiers:

```text
linear_svm
logistic_regression
gaussian_nb
```

## 2.10 Subsets

Enumerate every non-empty subset of the seven channels:

```text
127 subsets
```

Use the existing cardinality-first, lexicographic ordering.

---

# 3. Mandatory Git workflow

## 3.1 Feature branch

All implementation work must occur on a dedicated feature branch:

```text
feature/support2-180d-scenario
```

A different branch name may be used only when the repository has a documented naming convention that requires it. The chosen name must be recorded in the task document and final handoff.

## 3.2 Base branch

The feature branch must be created from an up-to-date local `main` branch.

Before creating it:

1. verify the working tree;
2. preserve or resolve any pre-existing uncommitted work;
3. switch to `main`;
4. update `main` from its configured remote when network access and permissions permit;
5. record the base commit hash;
6. create the feature branch.

Do not discard unrelated user changes.

## 3.3 Development on the feature branch

During Blocks 2 through 17:

- remain on the feature branch;
- do not implement directly on `main`;
- commit logical units of work;
- use descriptive commit messages;
- do not rewrite unrelated repository history;
- do not force-push;
- do not merge incomplete work into `main`.

## 3.4 Final merge

After all implementation, smoke validation, report generation, regeneration checks, and regression tests succeed:

1. ensure the feature branch working tree is clean;
2. create a final implementation commit if needed;
3. switch to `main`;
4. update `main` from its remote when permitted;
5. merge the feature branch into `main`;
6. prefer a non-fast-forward merge unless repository policy explicitly requires another strategy;
7. resolve conflicts without changing the approved scientific protocol;
8. run the required post-merge smoke verification on `main`;
9. confirm that the resulting `main` contains the complete implementation and generated report workflow.

Recommended merge command when compatible with repository policy:

```bash
git switch main
git merge --no-ff feature/support2-180d-scenario
```

## 3.5 Push policy

Do not push `main` or the feature branch unless the user or repository workflow explicitly requires a push.

The required deliverable is a locally merged and validated `main` branch. If a push is performed under an existing explicit repository policy, record the remote and resulting commit.

## 3.6 Merge failure policy

If merge conflicts cannot be resolved confidently without altering approved behavior:

- abort the merge;
- return to the feature branch;
- mark the final merge block as blocked;
- report the conflict paths and why the resolution is ambiguous.

Do not make speculative scientific or architectural changes merely to complete the merge.

---

# 4. Mandatory execution policy

## 4.1 Smoke-only rule

Every automated test or manual command that executes Monte Carlo simulation must use:

```text
mode = smoke
```

or:

```bash
--mode smoke
```

The implementation agent must not run:

```text
fast
full
strict
```

The implementation agent must not perform a confirmatory or production-scale Monte Carlo execution.

Unit and integration fixtures may use an even smaller explicit configuration, such as:

```python
MonteCarloConfig(
    mode="smoke",
    sample_sizes=(2, 4),
    min_replications=2,
    max_replications=2,
    replication_batch_size=2,
    test_samples_per_class=20,
    ci_half_width_target=0.05,
    base_seed=3,
)
```

Such fixtures remain smoke-mode tests.

## 4.2 Full-mode ownership

Full-mode execution is explicitly outside the implementation agent’s responsibilities.

After the implementation has been merged into `main` and validated in smoke mode, the repository owner will independently run the scenario in `full` mode.

## 4.3 Prohibited actions

The implementation agent must not:

- run `--mode full`;
- run `--mode fast`;
- run `--mode strict`;
- increase smoke repetitions to approximate a full run;
- launch background simulations;
- alter the approved scientific protocol to reduce implementation difficulty;
- omit reports because smoke results are small;
- replace the generic architecture with a SUPPORT2-specific parallel framework.

---

# 5. Status protocol

The agent must update the status marker for each block in this document.

Allowed states:

```text
[ ] Pending
[~] In progress
[x] Complete
[!] Blocked
```

A block may be marked complete only after:

1. its code changes are implemented;
2. its required inexpensive tests pass;
3. its acceptance criteria are verified;
4. relevant documentation is updated;
5. its work has been committed when the block represents a logical commit boundary.

The agent should add a short completion note below each completed block containing:

- changed files;
- tests executed;
- important verified values;
- commit hash;
- unresolved non-blocking observations.

---

# Block 0 — Inspect the current repository

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Inspected the required dataset, scenario,
simulation, sampler, persistence, registry, report, script, and Air Quality test
paths. The current generic dataset-anchored runner, report-data snapshots, run
registry, and persisted-result regeneration flow remain available. No prior
SUPPORT2 implementation was found. The only pre-existing worktree change was
this untracked task document, which was preserved and moved to its recommended
path. Local `main` and `origin/main` were both
`c084f86e7a6f6ccde8bd2d430dca336a6074dfc6`; `git pull --ff-only` reported
already up to date. No Monte Carlo execution occurred. Commit: recorded with
the Block 2 provenance commit.

## Purpose

Confirm the current architecture and avoid implementing against stale assumptions.

## Required inspection

Inspect at minimum:

```text
src/coinfosim/datasets/common.py
src/coinfosim/datasets/occupancy.py
src/coinfosim/datasets/air_quality.py
src/coinfosim/scenarios/dataset_anchored.py
src/coinfosim/scenarios/dataset_anchored_runner.py
src/coinfosim/simulation/config.py
src/coinfosim/simulation/subsets.py
src/coinfosim/simulation/monte_carlo.py
src/coinfosim/samplers/real.py
src/coinfosim/samplers/transfer.py
src/coinfosim/results/persistence.py
src/coinfosim/results/structural.py
src/coinfosim/runs/registry.py
src/coinfosim/runs/report_data.py
src/coinfosim/reports/air_quality_dataset.py
src/coinfosim/reports/air_quality_monte_carlo.py
src/coinfosim/reports/air_quality_scenario.py
scripts/run_air_quality_scenario.py
tests/test_air_quality_loader.py
tests/test_air_quality_anchored_models.py
tests/test_air_quality_scenario_runner.py
tests/test_air_quality_dataset_report.py
tests/test_air_quality_scenario_report.py
```

## Required actions

- Record the current branch.
- Record the current commit.
- Record the current `main` commit.
- Inspect the working tree for staged, unstaged, and untracked files.
- Confirm whether paths or symbols have changed.
- Confirm the current report-generation and report-regeneration flow.
- Confirm the current run-registry artifact layout.
- Confirm how Air Quality registers its execution specification.
- Confirm whether any SUPPORT2 implementation already exists.
- Reuse current conventions when they differ only cosmetically from this document.
- Do not silently change scientific requirements.

## Acceptance criteria

- The implementation plan is mapped to actual current symbols.
- Any material incompatibility is documented before coding.
- Pre-existing user changes are preserved.
- No Monte Carlo execution occurs in this block.

## Stop conditions

Mark the block as blocked if:

- the current architecture no longer has a dataset-anchored runner;
- existing uncommitted changes conflict with the planned files;
- SUPPORT2 has already been implemented with incompatible target semantics;
- `main` cannot be identified safely.

---

# Block 1 — Create the feature branch

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Created and switched to
`feature/support2-180d-scenario` from base commit
`c084f86e7a6f6ccde8bd2d430dca336a6074dfc6` after confirming that no branch of
that name already existed. The pre-existing task document remained preserved.
Commit: recorded with the Block 2 provenance commit.

**Depends on:** Block 0

## Purpose

Isolate the implementation from `main` until all code, tests, reports, and smoke validations are complete.

## Required actions

1. Ensure pre-existing user work has been preserved.
2. Ensure the working tree is clean or contains only intentionally preserved changes that will not be mixed into the feature.
3. Switch to `main`.
4. Update `main` from its configured remote when permitted.
5. Record the feature base commit.
6. Create and switch to:

```text
feature/support2-180d-scenario
```

Recommended commands:

```bash
git switch main
git pull --ff-only
git switch -c feature/support2-180d-scenario
```

If `git pull --ff-only` is unavailable because there is no network or no remote, continue from the verified local `main` and record that limitation.

## Acceptance criteria

- The current branch is the dedicated feature branch.
- The base commit is recorded.
- `main` contains no SUPPORT2 implementation changes.
- All subsequent development blocks will execute on the feature branch.

## Stop conditions

Stop if:

- the working tree cannot be cleaned without discarding user work;
- local `main` contains unresolved conflicts;
- the branch already exists with unrelated or incompatible work.

---

# Block 2 — Add raw data and provenance

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added the byte-identical Vanderbilt/HBiostat
`support2.csv`, provenance README, and frozen structure/hash test. Verified SHA-256
`79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78`, 47
header names, 48 fields in every record, and 9,105 records. Test:
`pytest tests/test_support2_loader.py -q`. Commit: `7fbf5a1d`.

**Depends on:** Block 1

## Files

```text
data/raw/support2/support2.csv
data/raw/support2/README.md
tests/test_support2_loader.py
```

## Required work

1. Add or confirm the canonical raw CSV.
2. Keep the raw source file unchanged.
3. Document:
   - source;
   - DOI;
   - provenance;
   - raw filename;
   - SHA-256;
   - 47-column header and 48-field row anomaly;
   - internal reconstruction of the unnamed leading patient ID;
   - use of `death` and `d.time` for target construction;
   - exclusion of `hospdead` as primary target.
4. Verify SHA-256:

```text
79621945edf2a5c8dc36359684ff356d3c6025e773ba4fefac26f865f7894c78
```

## Tests

Add inexpensive tests for:

- file existence;
- source hash;
- 47 advertised header names;
- 48 values in every data row;
- 9,105 data rows;
- presence of `death`;
- presence of `d.time`;
- presence of all seven channels;
- presence of `dzgroup`.

## Commit guidance

Create a logical commit after this block, for example:

```text
data: add canonical SUPPORT2 source and provenance
```

## Acceptance criteria

- Raw data are immutable and reproducible.
- The malformed-header behavior is explicitly documented.
- No implicit pandas-index behavior is relied upon.
- No simulation is run.

## Stop conditions

Stop if:

- the hash differs;
- row lengths are inconsistent;
- the file contains a different schema or patient count.

---

# Block 3 — Implement explicit SUPPORT2 ingestion

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added explicit CSV record inspection,
canonical schema enforcement, ID reconstruction/validation, numeric parsing, and
an exhaustive 48-column role map in `datasets/support2.py`. Structural corruption
tests pass. Commit: `78385cf0`.

**Depends on:** Block 2

## Primary file

```text
src/coinfosim/datasets/support2.py
```

## Expected symbols

Implement or adapt equivalent symbols:

```python
SUPPORT2_RAW_FILENAME
SUPPORT2_RAW_HEADER_COLUMNS
SUPPORT2_INTERNAL_COLUMNS
SUPPORT2_CHANNELS
SUPPORT2_TARGET
SUPPORT2_TARGET_EVENT_COLUMN
SUPPORT2_TARGET_TIME_COLUMN
SUPPORT2_TARGET_HORIZON_DAYS
SUPPORT2_STRATIFICATION_COLUMN
SUPPORT2_TRAIN_FRACTION
SUPPORT2_SPLIT_SEED

Support2Data

_read_support2_file
_derive_death_180d
_prepare_support2_data
load_support2_data
```

Use repository naming conventions if they require small adjustments.

## Required ingestion behavior

`_read_support2_file` must:

1. read the header independently;
2. require exactly 47 unique advertised names;
3. inspect raw records before DataFrame construction;
4. require 48 fields per row;
5. assign:

```python
("id", *header_columns)
```

6. validate that `id`:
   - is numeric;
   - is integral;
   - is unique;
   - runs from 1 through 9,105;
7. parse required numeric fields explicitly;
8. preserve `dzgroup` as categorical text;
9. reject schema drift;
10. reject field shifting;
11. avoid silent coercion of malformed target values.

## Column-role requirements

Create an explicit exhaustive role mapping.

At minimum:

```text
death       → target-construction-only
d.time      → target-construction-only
hospdead    → excluded alternative outcome
dzgroup     → split stratification and reporting only
id          → identifier only
seven channels → predictors
```

The mapping must account for every raw internal column.

## Tests

Add tests for:

- sequential ID reconstruction;
- unique IDs;
- first and last IDs;
- full column count;
- exact selected-channel order;
- malformed row rejection;
- missing required-column rejection;
- invalid duplicate-header rejection;
- exhaustive role mapping.

## Acceptance criteria

- The parser returns correctly aligned named fields.
- No target is inferred from `hospdead`.
- The loader fails loudly on structural corruption.
- Tests do not execute Monte Carlo.

---

# Block 4 — Implement and validate `death_180d`

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Implemented independently testable inclusive
180-day target derivation with missing, finite, domain, non-negative-time, and
survivor-follow-up checks. Verified raw counts 4,840/4,265, 2,904 survivors,
minimum survivor follow-up 344, and all boundary cases. Commit: `78385cf0`.

**Depends on:** Block 3

## Required function

Implement a deterministic pure or independently testable target-construction function equivalent to:

```python
def _derive_death_180d(death, d_time):
    return ((death == 1) & (d_time <= 180)).astype(np.int64)
```

Validation must occur before or as part of derivation.

## Required validations

Assert:

- `death` has no missing values;
- `death` contains only `0` and `1`;
- `d.time` has no missing values;
- `d.time` is finite;
- `d.time` is non-negative;
- all `death == 0` patients have `d.time > 180`;
- the derived target has no missing values;
- the derived target contains only `0` and `1`.

## Boundary semantics

Required cases:

```text
death=1, d.time=1   → 1
death=1, d.time=179 → 1
death=1, d.time=180 → 1
death=1, d.time=181 → 0
death=1, d.time=900 → 0
death=0, d.time=344 → 0
```

## Invalid cases

Fixtures containing any of the following must fail clearly:

```text
death is missing
d.time is missing
death not in {0,1}
d.time is negative
d.time is non-finite
death == 0 and d.time <= 180
```

## Canonical audit values

Verify for the raw 9,105-patient dataset:

```text
death_180d = 0: 4,840
death_180d = 1: 4,265
```

Verify:

```text
Patients with death == 0:       2,904
Minimum survivor d.time:          344
Survivors with d.time <= 180:       0
Missing survivor d.time:             0
```

Verify the raw cross-tabulation:

| `death` | `death_180d` | Count |
|---:|---:|---:|
| 0 | 0 | 2,904 |
| 1 | 0 | 1,936 |
| 1 | 1 | 4,265 |

## Ordering requirement

The target must be generated:

```text
after raw schema validation
before complete-case filtering
before train/test splitting
before standardization
before model fitting
```

## Tests

Add focused tests for:

- every boundary case;
- invalid values;
- raw class counts;
- survivor follow-up validation;
- target generation before split construction;
- absence of undefined labels.

## Acceptance criteria

- The canonical raw target counts match exactly.
- Day 180 is positive.
- Deaths after day 180 are negative for this endpoint.
- Survivors are assigned negative only after follow-up sufficiency is verified.
- `hospdead` is not consulted.

---

# Block 5 — Build the complete cohort and fixed split

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Implemented complete-case construction and the
fixed seed-0 joint-stratified split. Verified cohort 8,873; train/test 7,098/1,775;
class counts 3,768/3,330 and 943/832; 16 strata in both partitions; and all three
approved ID fingerprints. Commit: `78385cf0`.

**Depends on:** Block 4

## Required cohort fields

Require complete values for:

```text
meanbp
hrt
resp
temp
wblc
crea
sod
death
d.time
dzgroup
```

Keep the derived `death_180d` associated with every retained row.

## Expected cohort

```text
Rows: 8,873

death_180d = 0: 4,711
death_180d = 1: 4,162
```

## Split implementation

Construct a deterministic joint stratum equivalent to:

```python
joint_stratum = (
    death_180d.astype("int64").astype(str)
    + "|"
    + dzgroup.astype("string")
)
```

Use:

```python
train_test_split(
    cohort_indices,
    test_size=0.20,
    random_state=0,
    stratify=joint_stratum,
)
```

Sort each partition by ascending `id` after selection.

## Expected split

```text
Training rows: 7,098
Test rows:     1,775
```

```text
Training:
0 = 3,768
1 = 3,330

Test:
0 = 943
1 = 832
```

All 16 observed joint strata must appear in both partitions.

Expected minimum joint-stratum counts:

```text
Complete cohort minimum: 143
Training minimum:        114
Test minimum:              29
```

## Fingerprint tests

Recompute and verify:

```text
Cohort:
5c42d0c15c34abaad9e81dce0c1749e1001e66ca8b663d680cd37c6fecd7c59e

Training:
154809eb0f6759485342138c97f7ef7efc7d45bfc13d5bfdace19e366bea8979

Test:
74731ff933b9a19cb77dc4c859e797020c7168497befccf4669680c86037f7a7
```

## Required tests

Test:

- cohort size;
- cohort class counts;
- split sizes;
- split class counts;
- all joint strata represented;
- deterministic membership with seed 0;
- different valid membership with a different seed;
- sorted partition IDs;
- disjoint train/test IDs;
- train/test union equals cohort;
- exact fingerprints.

## Acceptance criteria

Every canonical count and fingerprint matches.

## Stop conditions

Stop if any expected count or fingerprint differs. Do not compensate by changing the seed, filtering order, boundary rule, or stratification variables.

---

# Block 6 — Implement training-only preprocessing

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added training-only seven-channel z-score
standardization with `ddof=0`, unchanged application to test, exact channel order,
and leakage/no-imputation/no-clipping restoration tests. Commit: `78385cf0`.

**Depends on:** Block 5

## Required behavior

Fit z-score parameters only on the training reservoir:

```text
ddof = 0
```

Apply the same fitted parameters to:

- training predictors;
- fixed test predictors.

Preserve:

- zeros;
- extreme values;
- original seven-channel order.

## Predictor boundary

Only these fields may enter preprocessing:

```text
meanbp
hrt
resp
temp
wblc
crea
sod
```

Explicitly exclude:

```text
id
death
d.time
death_180d
hospdead
dzgroup
surv2m
surv6m
```

## Tests

Test:

- training means and standard deviations;
- `ddof=0`;
- application of identical parameters to test;
- no test-data contribution to fitted parameters;
- exact channel ordering;
- no excluded field in `X`;
- test-only predictor mutation does not alter training parameters;
- no imputation;
- no clipping;
- no transformations.

## Acceptance criteria

The returned `Support2Data` satisfies the existing dataset-anchored data contract.

---

# Block 7 — Export the SUPPORT2 dataset API

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Exported `Support2Data`, constants, and
`load_support2_data` through `coinfosim.datasets`; Occupancy and Air Quality
imports remain intact. The combined loader suite reports 18 passed. Commit:
`78385cf0`.

**Depends on:** Block 6

## Files

```text
src/coinfosim/datasets/__init__.py
```

Potentially update other package exports only when required by current repository conventions.

## Required work

Export the stable public SUPPORT2 loader and prepared-data object.

Avoid unrelated cleanup of existing package exports.

## Tests

Verify public import paths used by the scenario script and tests.

## Acceptance criteria

SUPPORT2 loading is available through the expected package API without breaking Occupancy or Air Quality imports.

## Commit guidance

Blocks 3 through 7 may be committed together when they form one coherent dataset implementation, for example:

```text
feat: add SUPPORT2 180-day mortality dataset pipeline
```

---

# Block 8 — Persist target, preprocessing, and split metadata

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Extended the generic runner/spec and report-data
snapshot with optional dataset artifacts, preprocessing, and exclusion metadata.
SUPPORT2 writes strict split-manifest, target-metadata, and preprocessing-metadata
JSON artifacts. Air Quality runner regression: 4 passed. Commit: `1b69887e`;
SUPPORT2 callback/artifact implementation: `9642fa09`.

**Depends on:** Blocks 5–7

## Likely files

```text
src/coinfosim/scenarios/dataset_anchored_runner.py
src/coinfosim/runs/report_data.py
src/coinfosim/results/persistence.py
relevant generic tests
tests/test_support2_scenario_runner.py
```

## Required generic extension

Add only the minimal generic capabilities needed to persist:

1. exact split manifest or equivalent dataset artifact;
2. derived-target protocol;
3. preprocessing metadata.

Prefer an optional callback or optional context field that leaves existing scenarios unchanged.

Do not hard-code SUPPORT2 behavior into the generic runner.

## Required persisted target metadata

Persist equivalent information:

```json
{
  "name": "death_180d",
  "description": "Death within 180 days after SUPPORT2 study entry",
  "type": "derived_binary_fixed_horizon",
  "source_event_column": "death",
  "source_time_column": "d.time",
  "horizon_days": 180,
  "positive_rule": "death == 1 and d.time <= 180",
  "negative_rule": "death == 0 or d.time > 180",
  "derived_before_split": true,
  "class_labels": [0, 1],
  "raw_class_counts": {
    "0": 4840,
    "1": 4265
  },
  "cohort_class_counts": {
    "0": 4711,
    "1": 4162
  }
}
```

## Required split metadata

Persist:

- train/test fraction;
- seed;
- stratification variables;
- split class counts;
- disease-group stratum counts;
- cohort/train/test ID fingerprints;
- split-manifest artifact path.

## Required preprocessing metadata

Persist:

```text
method = zscore
fit_scope = training_reservoir_only
ddof = 0
imputation = none
transformation = none
clipping = none
zero_policy = preserve
```

## Required exclusion metadata

Persist target-related exclusions:

```text
death
d.time
hospdead
```

## Regeneration requirement

Report regeneration must read the persisted target and preprocessing metadata.

It must not:

- reload the raw CSV;
- rederive `death_180d`;
- recreate the split;
- rerun Monte Carlo.

## Tests

Test:

- metadata serialization;
- metadata loading;
- backward compatibility with existing scenarios;
- report-data construction with and without optional new fields;
- regeneration without Monte Carlo execution.

## Acceptance criteria

Existing dataset-anchored scenarios continue to pass their tests unchanged or with only compatible fixture updates.

---

# Block 9 — Implement SUPPORT2 anchored-model wrappers

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added thin wrappers around the generic Gaussian
and GMM builders. Tests verify classes, 3,768/3,330 training sizes, seven-dimensional
direct estimates, training-only fitting, and leakage boundaries. 2 passed. Commit:
`9642fa09`.

**Depends on:** Blocks 6–8

## Files

```text
src/coinfosim/scenarios/support2.py
tests/test_support2_anchored_models.py
```

## Required work

Create SUPPORT2-specific wrappers around:

```text
build_gaussian_anchored_model
build_gmm_anchored_model
```

Fit using:

- standardized real training predictors;
- `death_180d`;
- classes `(0, 1)`.

## Gaussian requirements

Fit one seven-dimensional class-conditional Gaussian for each target class.

Use the existing generic covariance and regularization behavior.

## GMM requirements

Fit one seven-dimensional class-conditional GMM per target class.

Reuse the current generic:

- BIC/AIC selection behavior;
- component candidate logic;
- minimum-points-per-component logic;
- covariance configuration;
- regularization;
- random state;
- initialization count.

## Tests

Test:

- classes `(0,1)`;
- exact class training sizes 3,768 and 3,330;
- seven-dimensional parameters;
- Gaussian means and covariances equal direct training calculations;
- GMM receives only training data;
- test predictor mutation does not affect fitted models;
- `death`, `d.time`, and `hospdead` never enter model input.

## Acceptance criteria

No SUPPORT2-specific duplicate Gaussian or GMM implementation is introduced.

---

# Block 10 — Implement the SUPPORT2 dataset report

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added a 20-section scientific report covering
provenance, endpoint, survivor audit, raw/cohort counts, missingness, disease groups,
split/fingerprints, preprocessing, class-conditional diagnostics, limitations, and
citations. Dataset-report test: 1 passed. Commit: `9642fa09`.

**Depends on:** Blocks 4–9

## Files

```text
src/coinfosim/reports/support2_dataset.py
tests/test_support2_dataset_report.py
```

## Required output

Generate a complete SUPPORT2 dataset report comparable in quality and structure to existing dataset reports.

Recommended title:

```text
CoInfoSim — SUPPORT2 180-Day Mortality Dataset Report
```

## Required sections

The report must include:

1. dataset overview and provenance;
2. raw structural anomaly;
3. reconstructed patient ID;
4. original SUPPORT prognostic objective;
5. target-source fields `death` and `d.time`;
6. exact `death_180d` formula;
7. inclusive day-180 boundary;
8. survivor follow-up validation;
9. raw target counts;
10. complete-case cohort construction;
11. complete-cohort target counts;
12. relationship among `death`, `d.time`, and `death_180d`;
13. rationale for using a fixed 180-day horizon;
14. why `hospdead` is not the primary target;
15. channel definitions and measurement timing;
16. excluded variables and leakage controls;
17. raw missingness;
18. selected-field missingness;
19. missingness by `death_180d`;
20. disease-group composition;
21. 180-day mortality prevalence by disease group;
22. fixed split protocol;
23. split counts and fingerprints;
24. training-only preprocessing;
25. marginal channel diagnostics;
26. class-conditional channel diagnostics;
27. correlation diagnostics;
28. zero and extreme-value policy;
29. limitations;
30. citation and license information.

## Mandatory statements

The report must explicitly state:

```text
hospdead is not used as the primary target or as a predictor.
```

The report must explicitly state that the implementation follows the original SUPPORT six-month prognostic objective.

The report must not claim that `death_180d` is a raw CSV column.

## Mandatory values

The report must contain:

```text
9,105
8,873
4,840
4,265
4,711
4,162
7,098
1,775
3,768
3,330
943
832
180
344
```

## Tests

Test:

- report creation;
- required headings;
- required values;
- target formula;
- correct class semantics;
- exclusion statement for `hospdead`;
- no description of the target as raw;
- report artifact registration when applicable.

## Acceptance criteria

A reader can independently reconstruct the endpoint, cohort, split, and preprocessing protocol from the report.

---

# Block 11 — Implement arm-report wrappers

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added Real → Real, Single Gaussian → Real, and
GMM → Real wrappers using the shared structured report. Tiny smoke fixtures verify
target/fixed-test wording, 127 subsets, structural sections, and model metadata.
2 passed. Commit: `9642fa09`.

**Depends on:** Blocks 9–10

## Files

```text
src/coinfosim/reports/support2_monte_carlo.py
tests/test_support2_monte_carlo_reports.py
```

## Required reports

Provide wrappers for:

```text
SUPPORT2 Real → Real
SUPPORT2 Single Gaussian → Real
SUPPORT2 GMM → Real
```

Reuse the generic structured Monte Carlo report generator.

## Required scientific wording

Every arm report must identify the target as:

```text
death within 180 days after SUPPORT2 study entry
```

The Real → Real report must explain that balanced training samples are drawn from the real training reservoir according to `death_180d`.

The synthetic reports must explain that class-conditional distributions are fitted using only the real training partition.

Every report must state that evaluation uses the same fixed real test set.

## Tests

Generate reports from tiny smoke-compatible fixtures.

Do not run a large Monte Carlo experiment.

Test:

- report title;
- arm identity;
- target identity;
- fixed-test wording;
- source-model metadata;
- 127-subset metadata where available;
- required structural metrics;
- artifact generation.

## Acceptance criteria

All three reports are generated through the same generic reporting architecture used by existing scenarios.

---

# Block 12 — Implement the SUPPORT2 scenario-comparison report

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added the SUPPORT2 specification for the generic
scenario renderer, with endpoint/cohort/split context, near-balanced classes,
ranking fidelity, exact-tie-aware agreement, progressive N-star similarity, and
links to all four subordinate reports. Test: 1 passed. Commit: `9642fa09`.

**Depends on:** Block 11

## Files

```text
src/coinfosim/reports/support2_scenario.py
tests/test_support2_scenario_report.py
```

## Scenario identity

```text
Scenario slug:
support2_baseline

Scenario name:
SUPPORT2 180-Day Mortality Baseline
```

## Simulation slugs

```text
support2_real_data
support2_single_gaussian_to_real
support2_gmm_to_real
```

## Required scientific question

Use wording equivalent to:

> To what extent do class-conditional Single Gaussian and GMM synthetic training distributions preserve the cooperative channel-subset structure observed for 180-day mortality prediction on a fixed real SUPPORT2 test set?

## Required content

The scenario report must include:

- endpoint definition;
- fixed 180-day rationale;
- cohort and split counts;
- near-balanced classes;
- selected channels;
- three-arm design;
- identical fixed test set;
- classifier set;
- 127 subsets;
- predictive results;
- ranking fidelity;
- exact-tie-aware winner agreement;
- progressive \(N^\star\) similarity;
- limitations;
- report links to the dataset and three arm reports.

## Tests

Generate from persisted tiny smoke-compatible fixture results.

Test:

- target identity;
- all three arm identities;
- structural metrics;
- report links;
- no rerun during regeneration;
- scenario artifact registration.

## Acceptance criteria

The scenario report is comparable to the existing scenario-level reports and links coherently to all lower-level reports.

---

# Block 13 — Implement the SUPPORT2 execution specification and CLI script

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Added the module-level execution spec, smoke CLI,
persisted-result regeneration CLI, strict protocol metadata, and exact split
artifact callback. Runner tests validate shared test identity, three simulations,
127 subsets, three classifiers, persistence, regeneration with loader/simulator
blocked, and pre-expensive-work failure for `n_per_class=3331`. 3 passed in smoke
mode. Commit: `9642fa09`.

**Depends on:** Blocks 8–12

## Files

```text
scripts/run_support2_scenario.py
tests/test_support2_scenario_runner.py
```

Potentially:

```text
src/coinfosim/scenarios/__init__.py
src/coinfosim/reports/__init__.py
```

only when required by current import conventions.

## Required execution specification

Create a module-level SUPPORT2 execution specification following the Air Quality pattern.

It must connect:

- SUPPORT2 loader;
- Gaussian builder;
- GMM builder;
- dataset report generator;
- three arm-report generators;
- scenario report generator;
- metadata callbacks;
- split-manifest artifact callback.

## CLI requirements

At minimum support:

```bash
.venv/bin/python scripts/run_support2_scenario.py --mode smoke
```

and report regeneration equivalent to:

```bash
.venv/bin/python scripts/run_support2_scenario.py \
  --report-from-scenario-run <SCENARIO_RUN_ID>
```

Do not add a target-selection option.

`death_180d` is fixed by the scenario protocol.

## Smoke-only protection

All tests invoking the script must pass:

```bash
--mode smoke
```

The agent must manually invoke only:

```bash
--mode smoke
```

The script may support existing generic modes if architecture requires it, but the agent must not execute any mode other than smoke.

## Integration test configuration

Use a tiny smoke-mode configuration with:

```text
sample_sizes = (2, 4)
min_replications = 2
max_replications = 2
```

Use a one-component GMM in tests where necessary to keep execution inexpensive.

## Required runner assertions

Verify:

- three simulations are created;
- same fixed test IDs across arms;
- same fixed test `X` across arms;
- same fixed test `y` across arms;
- classes are `(0,1)`;
- target is `death_180d`;
- 127 subsets are present;
- three registered classifiers are used;
- persistence succeeds;
- report generation succeeds;
- report regeneration succeeds without simulation;
- `n_per_class=3331` fails before expensive work.

## Acceptance criteria

A smoke execution produces a complete registered SUPPORT2 scenario and all expected reports.

## Commit guidance

Blocks 8 through 13 should be committed in logical increments, for example:

```text
feat: persist dataset-anchored target and split metadata
feat: add SUPPORT2 anchored scenario and reports
feat: add SUPPORT2 smoke CLI workflow
```

---

# Block 14 — Structural-result scalability check

**Status:** `[x] Complete`

**Completion note (2026-07-15):** The initial smoke revealed an actual duplication
problem: `scenario.json` was 70.32 MB and simulation JSON files were 18.36–18.60
MB. Added an opt-in, backward-compatible policy that omits only duplicated
precomputed structural snapshots for SUPPORT2 and regenerates them from persisted
compressed results. The retained smoke produced a 3.01 MB `scenario.json` and
0.789–0.790 MB simulation JSON files; compressed result payloads were 54–61 KB,
arm reports 9.79–9.87 MB, and the scenario report 171 KB. Scenario execution was
419.67 s; timed report-only regeneration was 177.88 s. HTML parsing and all report
links passed. Size regression tests are included. Commit: `32da88dd` plus the
feature artifact/finalization commit.

**Depends on:** Block 13

## Purpose

Confirm that 127 subsets do not make smoke artifacts or reports unusable.

## Required measurement

Using only a smoke execution, record:

- `simulation.json` size for each arm;
- `scenario.json` size;
- report-data artifact sizes;
- scenario execution time;
- report-generation time;
- report-regeneration time;
- whether generated HTML opens and renders correctly.

## Decision rule

Do not redesign structural persistence preemptively.

Change the structural schema only if the smoke artifacts demonstrate an actual usability or serialization problem.

Any generic schema change must:

- preserve existing metrics;
- preserve existing reports;
- include backward-compatible loading or an explicit migration decision;
- pass existing structural tests.

## Acceptance criteria

Smoke artifacts are usable, or a narrowly scoped compatibility-preserving fix is implemented and tested.

---

# Block 15 — Regression test suite on the feature branch

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Confirmed the feature branch and inspected every
test-side `MonteCarloConfig`; all executable simulations use `mode="smoke"`.
SUPPORT2 modules passed 18, 2, 1, 2, 1, and 3 tests respectively. The targeted
Air Quality/Occupancy/generic regression set passed 79 tests; the Air Quality
runner passed 4 tests; the final post-scalability full repository suite passed
317 tests in 1,330.71 s.
Only existing Matplotlib pending-deprecation warnings were reported. Commits:
`78385cf0`, `1b69887e`, `9642fa09`; documentation test record committed in the
feature-finalization commit.

**Depends on:** Blocks 2–14

## Branch requirement

Verify that the current branch is still the feature branch before running the final feature-branch regression suite.

## SUPPORT2 tests

Run:

```bash
.venv/bin/python -m pytest tests/test_support2_loader.py -q
.venv/bin/python -m pytest tests/test_support2_anchored_models.py -q
.venv/bin/python -m pytest tests/test_support2_dataset_report.py -q
.venv/bin/python -m pytest tests/test_support2_monte_carlo_reports.py -q
.venv/bin/python -m pytest tests/test_support2_scenario_report.py -q
.venv/bin/python -m pytest tests/test_support2_scenario_runner.py -q
```

## Existing scenario regressions

Run relevant existing tests:

```bash
.venv/bin/python -m pytest tests/test_air_quality_loader.py -q
.venv/bin/python -m pytest tests/test_air_quality_anchored_models.py -q
.venv/bin/python -m pytest tests/test_air_quality_dataset_report.py -q
.venv/bin/python -m pytest tests/test_air_quality_scenario_report.py -q
.venv/bin/python -m pytest tests/test_air_quality_scenario_runner.py -q
```

Run relevant Occupancy tests and generic metadata/structural tests.

## Full pytest suite

Run the repository’s full pytest suite if its tests do not launch non-smoke production simulations.

The smoke-only restriction applies to every test that executes Monte Carlo.

If an existing test attempts `fast`, `full`, or `strict`, do not run that test without first understanding why it exists. Do not silently weaken an established regression test.

## Acceptance criteria

- SUPPORT2 tests pass.
- Existing dataset scenarios remain green.
- Generic persistence and reporting tests remain green.
- No Monte Carlo test uses a non-smoke mode.
- The feature branch working tree is clean after committing all intended changes.

---

# Block 16 — Execute one end-to-end smoke scenario on the feature branch

**Status:** `[x] Complete`

**Completion note (2026-07-15):** Ran the required retained CLI smoke scenario
after replacing one superseded run that exposed the Block 14 size defect. Scenario
run `6` completed in 419.67 s with simulation runs `18`, `19`, and `20`; each has
127 subsets, all three classifiers, and fixed test size 1,775. Dataset, three arm,
scenario, split-manifest, target, preprocessing, result, summary, registry, and
run JSON artifacts exist and are non-empty. All scenario links resolve. Timed
regeneration (`--report-from-scenario-run 6`) completed in 177.88 s and explicitly
reported that Monte Carlo was not rerun. Artifacts committed in the feature
artifact/finalization commit.

**Depends on:** Block 15

## Permitted command

Run exactly one final end-to-end SUPPORT2 validation in smoke mode:

```bash
.venv/bin/python scripts/run_support2_scenario.py --mode smoke
```

Do not run another mode.

A repeat smoke run is permitted only when the first run exposes a defect that must be corrected and revalidated.

## Required generated artifacts

The final smoke scenario must produce and register:

1. dataset report;
2. Real → Real report;
3. Single Gaussian → Real report;
4. GMM → Real report;
5. consolidated scenario report;
6. three persisted simulation results;
7. scenario metadata;
8. split manifest;
9. preprocessing metadata;
10. target metadata.

## Required verification

Verify manually or programmatically:

- report files exist;
- report files are non-empty;
- report links resolve;
- all reports identify `death_180d`;
- no report identifies `hospdead` as primary;
- all three arms use the same fixed test set;
- all 127 subsets are represented;
- all three classifiers are represented;
- scenario and simulation registries contain the new runs;
- persisted run IDs are recorded.

## Report regeneration

Regenerate reports from the persisted scenario run:

```bash
.venv/bin/python scripts/run_support2_scenario.py \
  --report-from-scenario-run <SCENARIO_RUN_ID>
```

Verify that regeneration:

- succeeds;
- does not rerun Monte Carlo;
- retains the same endpoint metadata;
- retains links to all expected reports.

## Acceptance criteria

The smoke run produces the same report hierarchy expected from the existing scenarios.

## Stop conditions

Do not proceed to merge if:

- any expected report is missing;
- any arm has a different test set;
- target counts differ from the audited values;
- `death`, `d.time`, or `hospdead` enters predictor data;
- regeneration reruns simulation;
- reports contain scientifically incorrect target wording.

---

# Block 17 — Documentation and feature-branch finalization

**Status:** `[~] In progress`

**Depends on:** Block 16

## Files

```text
README.md
data/raw/support2/README.md
docs/tasks/support2_180d_implementation_tasks.md
```

Update other documentation only when required.

## README requirements

Document:

- new SUPPORT2 scenario;
- exact `death_180d` definition;
- fixed 180-day horizon;
- seven channels;
- complete-case cohort;
- split protocol;
- three arms;
- smoke execution command;
- report-regeneration command;
- report hierarchy;
- explicit statement that the repository owner will perform later full-mode experiments.

## Code quality

- Remove temporary debugging code.
- Remove temporary fixture artifacts.
- Preserve meaningful comments.
- Avoid unrelated refactoring.
- Ensure new public functions have appropriate documentation and typing.
- Ensure scientific constants are centralized rather than duplicated.

## Feature-branch finalization

Before marking complete:

1. ensure every intended file is tracked;
2. ensure the feature branch working tree is clean;
3. create the final feature commit;
4. record:
   - branch name;
   - feature base commit;
   - feature tip commit;
   - ordered commit list;
5. rerun any inexpensive tests affected by final cleanup.

## Acceptance criteria

- Documentation matches actual behavior.
- Commands are copy-pasteable.
- No documentation instructs the agent to run full mode.
- Task-block status markers are current.
- The feature branch is ready to merge.

---

# Block 18 — Merge the feature branch into `main`

**Status:** `[ ] Pending`

**Depends on:** Block 17

## Purpose

Integrate the completed and smoke-validated SUPPORT2 implementation into `main`.

## Pre-merge requirements

Before switching branches, verify:

- all previous blocks are complete;
- all required tests passed on the feature branch;
- the end-to-end smoke scenario passed;
- report regeneration passed;
- the feature branch working tree is clean;
- the feature tip commit is recorded.

## Required merge procedure

1. Switch to `main`.
2. Update `main` using `git pull --ff-only` when network access and repository permissions permit.
3. Record the pre-merge `main` commit.
4. Merge the feature branch.
5. Prefer:

```bash
git merge --no-ff feature/support2-180d-scenario
```

unless repository policy explicitly requires fast-forward, squash, or another strategy.
6. Resolve conflicts carefully.
7. Do not alter the approved target, cohort, split, preprocessing, or smoke-only policy during conflict resolution.
8. Record the merge commit hash.

## Conflict policy

If conflicts are material or ambiguous:

```bash
git merge --abort
```

Then:

- mark this block `[!] Blocked`;
- report conflicting files;
- explain why a safe resolution could not be made;
- leave the validated feature branch intact.

## Acceptance criteria

- `main` contains the complete SUPPORT2 implementation.
- The merge commit is recorded.
- The working tree on `main` is clean.
- No full, fast, or strict run has occurred.

---

# Block 19 — Post-merge smoke verification on `main`

**Status:** `[ ] Pending`

**Depends on:** Block 18

## Purpose

Verify that the merged `main` behaves identically to the validated feature branch.

## Required checks

Confirm the current branch is:

```text
main
```

Run the critical SUPPORT2 tests and relevant regressions again.

At minimum:

```bash
.venv/bin/python -m pytest tests/test_support2_loader.py -q
.venv/bin/python -m pytest tests/test_support2_anchored_models.py -q
.venv/bin/python -m pytest tests/test_support2_dataset_report.py -q
.venv/bin/python -m pytest tests/test_support2_monte_carlo_reports.py -q
.venv/bin/python -m pytest tests/test_support2_scenario_report.py -q
.venv/bin/python -m pytest tests/test_support2_scenario_runner.py -q
```

Run one post-merge end-to-end smoke validation:

```bash
.venv/bin/python scripts/run_support2_scenario.py --mode smoke
```

Then regenerate its reports:

```bash
.venv/bin/python scripts/run_support2_scenario.py \
  --report-from-scenario-run <POST_MERGE_SCENARIO_RUN_ID>
```

## Required post-merge artifact verification

Confirm on `main`:

- dataset report exists;
- all three arm reports exist;
- scenario report exists;
- report links resolve;
- target metadata are correct;
- split metadata and fingerprints are correct;
- test identity is shared across arms;
- regeneration does not rerun Monte Carlo.

## Acceptance criteria

- Critical tests pass on merged `main`.
- One end-to-end smoke scenario succeeds on merged `main`.
- Report regeneration succeeds on merged `main`.
- `main` working tree is clean after recording any intentionally tracked documentation updates.

## Stop conditions

If post-merge verification fails:

1. do not run full mode;
2. diagnose on a new fix branch or revert the merge according to repository policy;
3. do not leave `main` knowingly broken;
4. report the failure and remediation performed.

---

# Block 20 — Final handoff

**Status:** `[ ] Pending`

**Depends on:** Block 19

## Required final response

The agent’s final response must include:

### Implementation summary

- scientific endpoint implemented;
- cohort and split verified;
- three scenario arms implemented;
- reports generated;
- regeneration verified;
- feature branch merged into `main`;
- post-merge smoke validation completed.

### Git summary

Report:

- feature branch name;
- feature base commit;
- feature tip commit;
- ordered feature commit list;
- pre-merge `main` commit;
- merge commit;
- final `main` commit;
- whether any remote push occurred.

### Changed files

Group files by:

- dataset;
- scenarios;
- reports;
- scripts;
- generic infrastructure;
- tests;
- documentation.

### Verified canonical values

Report:

```text
Raw rows: 9,105
Cohort rows: 8,873

Raw classes:
0 = 4,840
1 = 4,265

Cohort classes:
0 = 4,711
1 = 4,162

Training:
0 = 3,768
1 = 3,330

Test:
0 = 943
1 = 832
```

### Tests executed

List exact commands and results separately for:

- feature branch;
- post-merge `main`.

Explicitly confirm:

```text
No fast, full, or strict scenario execution was performed.
```

### Smoke scenario artifacts

Provide both feature-branch and post-merge run identifiers when both were retained:

- scenario run ID;
- three simulation run IDs;
- dataset report path;
- three arm-report paths;
- scenario-report path;
- split-manifest path;
- relevant JSON artifact paths.

### Regeneration

Confirm that report regeneration succeeded without rerunning Monte Carlo on both the feature branch and merged `main`, when both validations were performed.

### Remaining issues

List only actual remaining issues.

Do not describe the unexecuted full run as an implementation defect. Full-mode execution is intentionally reserved for the repository owner.

---

# Global definition of done

The implementation is complete only when all of the following are true:

- [ ] A dedicated feature branch was created from `main`.
- [ ] All implementation work occurred on the feature branch.
- [ ] SUPPORT2 raw ingestion explicitly handles the 47/48-field anomaly.
- [ ] `death_180d` is deterministically derived from `death` and `d.time`.
- [ ] Survivor follow-up beyond 180 days is validated.
- [ ] Raw and cohort class counts match the audited values.
- [ ] The fixed joint-stratified split matches expected counts and fingerprints.
- [ ] Training-only z-score standardization is verified.
- [ ] No target-source or alternative-outcome variable enters predictors.
- [ ] All 127 non-empty channel subsets are used.
- [ ] Real, Single Gaussian, and GMM arms use the same fixed real test set.
- [ ] All three registered classifiers are used.
- [ ] Dataset report is generated.
- [ ] Three arm reports are generated.
- [ ] Consolidated scenario report is generated.
- [ ] Reports are registered and linked.
- [ ] Report regeneration succeeds without Monte Carlo.
- [ ] SUPPORT2 tests pass.
- [ ] Relevant Air Quality, Occupancy, persistence, and reporting regressions pass.
- [ ] One end-to-end smoke scenario succeeds on the feature branch.
- [ ] The feature branch is merged into `main`.
- [ ] Critical tests pass on merged `main`.
- [ ] One end-to-end smoke scenario succeeds on merged `main`.
- [ ] Report regeneration succeeds on merged `main`.
- [ ] No fast, full, or strict execution is performed by the implementation agent.
- [ ] Documentation is complete.
- [ ] Task-block statuses and completion notes are updated.
- [ ] Final Git and artifact identifiers are reported.
