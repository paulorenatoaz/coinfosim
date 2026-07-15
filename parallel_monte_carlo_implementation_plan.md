# CoInfoSim Parallel Monte Carlo Implementation Plan

**Intended repository path:** `docs/development/parallel_monte_carlo_implementation_plan.md`<br>
**Repository:** `paulorenatoaz/coinfosim`<br>
**Base branch:** `main`<br>
**Development branch:** `feature/parallel-monte-carlo`<br>
**Target platform:** Linux/Ubuntu<br>
**Primary scientific reference:** Occupancy scenario run ID 2, mode `full`<br>
**Execution protocol:** Complete exactly one block, report the results, and stop. Continue only after explicit user approval.

---

## 1. Purpose

Implement process-based parallel execution for the cooperative Monte Carlo engine while preserving the scientific and statistical behavior of CoInfoSim.

The parallel implementation must preserve:

- the training sample generated for every `(n_per_class, replication_id)`;
- the fixed test set used by each simulation arm;
- one common training sample across every subset and classifier within one replication;
- ordered per-replication empirical test losses;
- the stopping decision and stopping boundary for every `n_per_class`;
- all summary statistics, rankings, threshold comparisons, winner matrices, progressive directed N-star matrices, and cross-arm structural fidelity metrics;
- the current persistence schema and report-regeneration capability;
- the sequential backend as the stable reference implementation and default behavior.

The feature must reduce runtime without changing the experiment being executed.

---

## 2. Mandatory workflow and review gates

1. Read this entire document before changing code.
2. Execute only the currently approved block.
3. Run only the tests explicitly authorized for that block.
4. Make one focused commit when the block passes its authorized tests.
5. Push the feature branch when authentication and repository access permit it.
6. Produce a block completion report.
7. Stop and wait for explicit user approval.
8. Do not begin the next block merely because the current block passed.

Each block completion report must include:

- current branch and commit SHA;
- files changed;
- concise implementation summary;
- exact test commands and results;
- tests intentionally not executed;
- generated artifacts, if any;
- deviations from this plan;
- unresolved risks or decisions requiring review.

---

## 3. Branching and concurrent-work rules

### 3.1 Required branch

Do not create a fork. Use:

```text
feature/parallel-monte-carlo
```

The branch must originate from the latest available `origin/main`.

### 3.2 Safe branch creation

```bash
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c feature/parallel-monte-carlo
git push -u origin feature/parallel-monte-carlo
```

Safety requirements:

- If the worktree contains uncommitted changes, stop and report them. Do not discard, reset, stash, or overwrite them automatically.
- Never commit directly to `main`.
- Never force-push.
- Never reset `main`.
- Do not merge or cherry-pick work from the other dataset branch without explicit approval.
- Do not alter code belonging only to the concurrent dataset experiment unless a generic interface change is explicitly required by this plan.
- At the beginning of every approved block, run `git fetch origin` and report whether `origin/main` or the remote feature branch has advanced. Do not rebase or merge automatically.

### 3.3 Work on the other computer

The other dataset experiment should use a separate branch, for example:

```text
feature/<dataset-name>-experiments
```

After this feature is reviewed and merged into `main`, that dataset branch can receive it through a user-approved merge or rebase from the updated `main`.

---

## 4. AI and compute budget policy

### 4.1 Prohibited unless explicitly authorized

Do not:

- run the Occupancy `full` scenario during development;
- run the Occupancy `fast` scenario merely as a regression test;
- run the complete test suite after every block;
- regenerate all HTML reports after every block;
- benchmark every possible worker count;
- introduce unrelated refactors;
- rewrite report modules whose numerical inputs are unchanged;
- redesign the persistence schema;
- implement distributed multi-node execution;
- implement Windows-specific multiprocessing support;
- implement retries, checkpoints, resume, or fault-tolerant scheduling;
- implement exhaustive evaluation of all high-dimensional subsets;
- install new libraries without first checking availability and the explicit requirements of this plan.

### 4.2 Required economy

Prefer:

- targeted file inspection;
- focused unit tests;
- small in-memory fixtures;
- tiny fixed-replication configurations;
- one end-to-end integration test only when its block requires it;
- exact numerical comparisons instead of repeated visual report inspection;
- one representative benchmark after correctness is established.

If an authorized test fails, run only the smallest additional diagnostic needed. Do not immediately run the full suite.

---

## 5. Approved architecture decisions

These decisions are fixed. Implement them; do not reopen architectural exploration.

### 5.1 Unit of parallelism

The parallel task is one complete Monte Carlo replication for one `n_per_class`:

```python
ReplicationTask(
    n_per_class=...,
    replication_id=...,
)
```

The worker generates the training sample internally and evaluates every canonical `(subset, classifier)` cell for that replication.

### 5.2 Complete replication semantics

For one replication:

1. Call `sampler.sample_train(n_per_class, replication_id)` exactly once.
2. Reuse that training sample for every subset.
3. Restrict that same sample to each subset.
4. Construct a fresh estimator through the existing classifier registry for every cell.
5. Fit and compute empirical test loss on the fixed test subset.
6. Return one complete vector of losses.

Do not parallelize individual classifiers or subsets in this implementation.

### 5.3 Compact result format

Use the simulator's canonical ordering:

```python
cells = [
    (subset, classifier_name)
    for subset in subsets
    for classifier_name in classifier_names
]
```

Return:

```python
@dataclass(frozen=True)
class ReplicationResult:
    n_per_class: int
    replication_id: int
    losses: tuple[float, ...]
```

For Occupancy, each result contains:

```text
31 subsets × 3 classifiers = 93 losses
```

Do not repeat subset tuples and classifier names inside every returned loss item.

### 5.4 Batch semantics

For each batch:

1. Submit exactly its replication IDs.
2. Wait for all tasks to finish.
3. Validate the complete returned batch.
4. Sort by `replication_id`.
5. Commit atomically to the accumulator.
6. Evaluate stopping.
7. Submit another batch only if stopping says to continue.

Do not submit speculative tasks from the next batch.

### 5.5 Process execution

Use `concurrent.futures.ProcessPoolExecutor`.

Use one persistent pool for the entire `CooperativeMonteCarloSimulator.run()` call. A simulator instance represents one scenario arm, so this is naturally one pool per arm.

Do not create a pool per replication, per batch, or per `n_per_class`.

### 5.6 Linux start method

The initial process backend must use:

```python
multiprocessing.get_context("forkserver")
```

Represent the start method in execution configuration and metadata. A configurable `fork` alternative may be accepted later for benchmarking, but `forkserver` is the initial target.

### 5.7 Worker context

Each worker receives fixed arm context once during initialization. The context includes, directly or through the sampler:

- arm-specific sampler;
- fixed test data;
- ordered subsets;
- ordered classifier names;
- canonical cells;
- numeric thread limit.

After initialization, each submitted task contains only `n_per_class` and `replication_id`.

Workers must not refit Gaussian or GMM generative models. Those models are fitted before Monte Carlo begins.

### 5.8 Numeric thread control

External process parallelism must not multiply uncontrolled BLAS/OpenMP threads.

The worker initializer must apply a persistent numeric thread limit, initially one thread per worker, using `threadpoolctl`.

If CoInfoSim imports `threadpoolctl` directly, declare it as a direct dependency.

### 5.9 Sequential backend

Sequential execution remains the default and scientific reference. Calling the simulator without explicit parallel configuration must preserve current behavior.

### 5.10 Persistence

Keep the current persistence schema. Ordered loss lists reconstruct IDs by position, so every cell must contain:

```text
0, 1, 2, ..., R - 1
```

Do not add explicit replication IDs to persisted payloads in this work.

### 5.11 Fixed-test subset materialization

Preserve the current precomputed fixed-test-subset behavior.

`Dataset.select_channels()` copies `X` through NumPy advanced indexing. For Occupancy, all 31 fixed-test subsets require approximately 7.58 MiB per worker, which is acceptable.

Add a helper that estimates this cache size and record it in execution metadata. Do not implement generalized shared-memory or low-memory test caches in this branch; that is separate future work for thousands of subsets.

### 5.12 Statistics and reports

Workers produce only individual empirical test losses. They must not calculate or persist report statistics. Existing main-process modules continue to compute all summaries, rankings, N-star diagnostics, matrices, and structural fidelity metrics.

---

## 6. Out of scope

- multi-node or cluster execution;
- Dask, Ray, MPI, or Slurm integration;
- GPU execution;
- parallel scenario arms;
- parallel `n_per_class` values;
- parallel subsets or classifiers within one replication;
- changing batch size for hardware utilization;
- new stopping rules or N-star definitions;
- changes to ranking, agreement, or structural similarity;
- persistence schema changes;
- checkpoint/resume or retries;
- automatic worker-count tuning;
- high-dimensional subset-search algorithms;
- shared-memory test matrices;
- report redesign;
- changes to the meaning of existing scenario arms.

---

## 7. Relevant repository artifacts

Inspect these files for the stated reasons. Do not substitute a vague repository-wide scan.

### 7.1 Core simulation

- `src/coinfosim/simulation/monte_carlo.py`
  - Main loop, fixed test set, subset cache, accumulator, batch boundary, stopping, progress, runtime, and metadata.
- `src/coinfosim/results/accumulator.py`
  - Individual loss storage. Currently permits duplicate-ID replacement and assumes aligned cell counts.
- `src/coinfosim/simulation/stopping.py`
  - Maximum `1.96 × SE` across all cells and batch-boundary stopping.
- `src/coinfosim/simulation/config.py`
  - Scientific configuration; execution configuration must remain separate.
- `src/coinfosim/simulation/metrics.py`
  - Empirical test loss.
- `src/coinfosim/simulation/subsets.py`
  - Stable subset enumeration and ordering.
- `src/coinfosim/simulation/progress.py`
  - Main-process-only progress reporting.

### 7.2 Classifiers

- `src/coinfosim/classifiers/registry.py`
  - Fresh `SVC(kernel="linear")`, `LogisticRegression`, and `GaussianNB` estimators. Workers use classifier keys and this registry.

### 7.3 Dataset and samplers

- `src/coinfosim/samplers/dataset.py`
- `src/coinfosim/samplers/gaussian.py`
- `src/coinfosim/samplers/real.py`
- `src/coinfosim/samplers/gmm.py`
- `src/coinfosim/samplers/transfer.py`

Important details include deterministic seed derivation, prefix nesting, fixed-test reuse, real balanced sampling, and GMM cumulative-weight/Cholesky caches.

### 7.4 Scenario orchestration

- `scripts/run_occupancy_scenario.py`
  - Runs the three arms sequentially, fits generative models before Monte Carlo, persists results, and builds reports.
- `src/coinfosim/scenarios/occupancy.py`
  - Gaussian and GMM fitting outside Monte Carlo workers.
- `src/coinfosim/datasets/occupancy.py`
  - Occupancy train/test protocol and dimensions.

### 7.5 Persistence and reports

- `src/coinfosim/results/persistence.py`
- `src/coinfosim/results/summary.py`
- `src/coinfosim/results/snapshots.py`
- `src/coinfosim/results/analysis.py`
- `src/coinfosim/results/structural.py`
- `src/coinfosim/reports/report_tables.py`
- `src/coinfosim/reports/monte_carlo.py`
- `src/coinfosim/reports/occupancy_monte_carlo.py`

These derive output from the completed `SimulationResult`. Numerical output must remain equal between sequential and process backends.

### 7.6 Existing tests

- `tests/test_sprint1_scenario_simulator.py`
- `tests/test_monte_carlo_with_real_sampler.py`
- `tests/test_real_sampler.py`
- `tests/test_gmm_sampler.py`
- `tests/test_run_registry.py`
- `tests/test_cooperative_progress.py`
- `tests/test_structural_metrics.py`
- `tests/test_occupancy_run_tracking.py`
- `tests/test_occupancy_monte_carlo_reports.py`
- `tests/test_occupancy_scenario_report.py`

Use only the smallest relevant subset in each block.

### 7.7 Full Occupancy reference artifacts

Scenario:

- `output/reports/scenarios/000002_occupancy_baseline_full/scenario.json`
- `output/reports/scenarios/000002_occupancy_baseline_full/occupancy_baseline_scenario_report_full_000002.html`

Real → Real:

- `output/reports/simulations/000006_occupancy_real_data_full/result_data_full_000006.json.gz`
- `output/reports/simulations/000006_occupancy_real_data_full/summary_full_000006.json`
- `output/reports/simulations/000006_occupancy_real_data_full/precision_diagnostics_real_to_real.csv`
- `output/reports/simulations/000006_occupancy_real_data_full/occupancy_real_data_monte_carlo_report_full_000006.html`

Single Gaussian → Real:

- `output/reports/simulations/000007_occupancy_single_gaussian_to_real_full/result_data_full_000007.json.gz`
- `output/reports/simulations/000007_occupancy_single_gaussian_to_real_full/summary_full_000007.json`
- `output/reports/simulations/000007_occupancy_single_gaussian_to_real_full/precision_diagnostics_single_gaussian_to_real.csv`
- `output/reports/simulations/000007_occupancy_single_gaussian_to_real_full/occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`

GMM → Real:

- `output/reports/simulations/000008_occupancy_gmm_to_real_full/result_data_full_000008.json.gz`
- `output/reports/simulations/000008_occupancy_gmm_to_real_full/summary_full_000008.json`
- `output/reports/simulations/000008_occupancy_gmm_to_real_full/precision_diagnostics_gmm_to_real.csv`
- `output/reports/simulations/000008_occupancy_gmm_to_real_full/occupancy_gmm_to_real_monte_carlo_report_full_000008.html`

Observed reference totals:

| Arm | Replications | Classifier fits | Monte Carlo runtime |
|---|---:|---:|---:|
| Real → Real | 6,920 | 643,560 | 2,071.20 s |
| Single Gaussian → Real | 5,180 | 481,740 | 1,254.74 s |
| GMM → Real | 6,160 | 572,880 | 1,428.63 s |
| **Total** | **18,260** | **1,698,180** | **4,754.57 s** |

These values explain the value of replication-level parallelism. They are not machine-independent performance targets.

---

# BLOCK 0 — Create the isolated development branch and record the baseline

## Goal

Create a safe branch from the latest `main` and record the repository and environment baseline. Do not modify simulator code.

## Required reading

- this entire plan;
- `src/coinfosim/simulation/monte_carlo.py`;
- `src/coinfosim/results/accumulator.py`;
- `src/coinfosim/simulation/stopping.py`;
- `scripts/run_occupancy_scenario.py`.

## Tasks

1. Verify the repository is `paulorenatoaz/coinfosim`.
2. Check the worktree.
3. Fetch `origin`.
4. Update local `main` by fast-forward only.
5. Create `feature/parallel-monte-carlo`.
6. Push it and configure upstream tracking, when credentials permit.
7. Record:
   - base commit SHA;
   - Python version;
   - NumPy, SciPy, and scikit-learn versions;
   - `threadpoolctl` version, if installed;
   - logical CPU count;
   - available multiprocessing start methods.
8. Confirm whether every reference artifact in Section 7.7 is present.
9. Do not edit implementation files.

## Authorized commands/tests

No test suite is required.

```bash
git status --short
git rev-parse HEAD
python --version
python -c "import os, multiprocessing as mp; print(os.cpu_count()); print(mp.get_all_start_methods())"
python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
python -c "import threadpoolctl; print(threadpoolctl.__version__)"
```

The final command may fail if the package is unavailable. Report that fact without installing anything.

## Deliverables

- local and preferably remote feature branch;
- baseline environment report;
- reference-artifact availability report;
- no source-code changes.

## Commit

Do not create an empty commit unless explicitly requested.

## Stop condition

Return the Block 0 completion report and stop.

---

# BLOCK 1 — Extract complete-replication evaluation without parallelism

## Goal

Refactor serial execution so one complete replication is evaluated through a dedicated, testable component, with no behavior change.

## Required reading

- `src/coinfosim/simulation/monte_carlo.py`;
- `src/coinfosim/simulation/metrics.py`;
- `src/coinfosim/classifiers/registry.py`;
- all sampler interfaces in Section 7.3;
- `tests/test_sprint1_scenario_simulator.py`.

## Required design

Create:

```text
src/coinfosim/simulation/replication.py
```

Define at minimum:

```python
Cell = tuple[tuple[int, ...], str]

@dataclass(frozen=True)
class ReplicationTask:
    n_per_class: int
    replication_id: int

@dataclass(frozen=True)
class ReplicationResult:
    n_per_class: int
    replication_id: int
    losses: tuple[float, ...]
```

Add a replication evaluation function or immutable evaluator that receives:

- sampler;
- canonical ordered cells;
- precomputed fixed test subsets;
- one task.

It must sample training exactly once, traverse cells in canonical order, create fresh estimators, and return the complete loss tuple.

Do not add multiprocessing in this block.

## Tasks

1. Add replication data structures.
2. Extract the current inner replication logic.
3. Keep the serial batch loop.
4. Use the extracted evaluator from the serial loop.
5. Do not change subset or classifier order, sampler calls, accumulator semantics, stopping, metadata, persistence, or report modules.
6. Add one small test verifying:
   - one training-sample call per replication;
   - output length equals canonical cell count;
   - output ordering matches the cells;
   - losses are finite and in `[0, 1]`.

## Authorized tests

```bash
pytest -q tests/test_sprint1_scenario_simulator.py
pytest -q <new replication unit test file>
```

If necessary, select only the end-to-end and shared-replication-count tests from the existing file. Do not run Occupancy integration tests.

## Acceptance criteria

- simulator remains sequential;
- synthetic smoke behavior passes unchanged;
- one compact `ReplicationResult` represents a complete replication;
- no report or persistence code is changed.

## Commit

```text
refactor: extract complete replication evaluation
```

## Stop condition

Push when possible, report, and stop.

---

# BLOCK 2 — Add atomic accumulator operations and stopping integrity checks

## Goal

Prevent partial, duplicate, missing, non-contiguous, or misaligned replication data from entering statistical calculations.

## Required reading

- `src/coinfosim/results/accumulator.py`;
- `src/coinfosim/simulation/stopping.py`;
- `src/coinfosim/results/persistence.py`;
- `tests/test_run_registry.py`;
- any existing accumulator/stopping tests.

## Required design

Preserve the low-level `add()` method for backward compatibility and persistence reconstruction.

Add higher-level safe operations equivalent to:

```python
add_replication(
    n_per_class: int,
    replication_id: int,
    cells: Sequence[Cell],
    losses: Sequence[float],
) -> None
```

and:

```python
add_batch(
    n_per_class: int,
    expected_replication_ids: Sequence[int],
    cells: Sequence[Cell],
    results: Sequence[ReplicationResult],
) -> None
```

Exact names may differ; behavior may not.

## Validation requirements

Before any mutation, validate:

- received IDs exactly match expected IDs;
- no duplicate IDs;
- all results use the current `n_per_class`;
- every loss vector has `len(cells)` entries;
- all losses are finite and within `[0, 1]`;
- no `(cell, replication_id)` already exists;
- IDs continue contiguously from the current completed count;
- all existing requested cells have equal counts.

On failure, raise a clear exception and leave the accumulator unchanged.

## Stopping integrity

Stopping evaluation must explicitly verify equal replication counts across all requested cells. Do not use the maximum count as evidence of completeness.

Do not change CI calculations or decision order.

## Persistence constraint

Keep schema version 1. Confirm exact loss-array equality after save/load.

## Authorized tests

Add focused tests for valid insertion and every validation failure above, including proof that state remains unchanged after failure.

Run only:

```bash
pytest -q <new accumulator integrity test file>
pytest -q tests/test_run_registry.py::test_save_and_load_simulation_result_roundtrip
pytest -q tests/test_sprint1_scenario_simulator.py::test_simulation_shared_replication_count
pytest -q tests/test_sprint1_scenario_simulator.py::test_simulation_stopping_info_recorded
```

## Acceptance criteria

- serial simulation uses complete-replication or complete-batch insertion;
- partial batches cannot affect stopping;
- persistence schema remains unchanged;
- round-trip loss arrays are exactly equal.

## Commit

```text
feat: add atomic Monte Carlo batch accumulation
```

## Stop condition

Push, report, and stop.

---

# BLOCK 3 — Introduce execution configuration and a sequential executor

## Goal

Separate scientific Monte Carlo configuration from computational execution configuration, without creating processes yet.

## Required reading

- `src/coinfosim/simulation/config.py`;
- `src/coinfosim/simulation/monte_carlo.py`;
- the new replication module;
- `src/coinfosim/simulation/progress.py`.

## Required design

Create:

```text
src/coinfosim/simulation/execution.py
```

Define an immutable configuration equivalent to:

```python
@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "sequential"
    n_jobs: int = 1
    start_method: str = "forkserver"
    worker_inner_threads: int = 1
```

Validation:

- `n_jobs > 0`;
- `worker_inner_threads > 0`;
- supported backend names;
- process start-method availability when process execution is selected.

Add an executor interface:

```python
run_batch(
    n_per_class: int,
    replication_ids: Sequence[int],
) -> list[ReplicationResult]
```

Implement only `SequentialReplicationExecutor` in this block.

## Tasks

1. Keep `MonteCarloConfig` unchanged.
2. Add optional execution configuration to the simulator.
3. Preserve sequential default behavior.
4. Route serial batches through the sequential executor.
5. Preserve one fixed test set and current precomputed test subsets.
6. Return results to the simulator, validate, atomically commit, then call stopping.
7. Keep progress in the main simulator.

## Authorized tests

```bash
pytest -q <new execution configuration and sequential executor tests>
pytest -q tests/test_sprint1_scenario_simulator.py
pytest -q tests/test_run_registry.py::test_save_and_load_simulation_result_roundtrip
```

## Acceptance criteria

- no process is created;
- default behavior is sequential;
- detailed replication mechanics are outside the simulator loop;
- exact loss arrays and stopping information match the prior serial implementation.

## Commit

```text
refactor: add Monte Carlo execution abstraction
```

## Stop condition

Push, report, and stop.

---

# BLOCK 4 — Implement the persistent process executor

## Goal

Execute complete replications in parallel while preserving batch, failure, and result semantics.

## Required reading

- `src/coinfosim/simulation/execution.py`;
- `src/coinfosim/simulation/replication.py`;
- every sampler implementation;
- `src/coinfosim/classifiers/registry.py`;
- `setup.py`.

## Required process design

Use:

```python
ProcessPoolExecutor(
    max_workers=effective_workers,
    mp_context=multiprocessing.get_context(execution_config.start_method),
    initializer=...,
    initargs=(...),
)
```

Compute:

```python
effective_workers = min(
    execution_config.n_jobs,
    monte_carlo_config.replication_batch_size,
)
```

Create the pool once for the complete simulator run and close it reliably.

## Worker initialization

Establish process-local context containing:

- sampler;
- canonical cells;
- fixed test set;
- precomputed fixed test subsets;
- persistent threadpool limit controller.

The worker callable must be defined at module level.

Do not pass fitted classifiers, factory callables, progress reporters, accumulators, stopping rules, registries, or report generators.

## Submission and collection

For each batch:

1. Create one task per replication ID.
2. Submit tasks individually for dynamic scheduling.
3. Collect all futures.
4. If one future fails:
   - cancel not-yet-running futures;
   - commit no result from that batch;
   - raise a contextual error containing `n_per_class` and `replication_id`;
   - allow existing scenario-level handling to mark the run failed.
5. If all succeed:
   - validate;
   - sort by ID;
   - atomically commit;
   - evaluate stopping.

Workers must not print.

## Dependency task

If imported directly, add `threadpoolctl` as a direct dependency with a compatible minimum version. Do not alter unrelated dependencies.

## Authorized tests

Use tiny in-memory fixtures only. Add tests for:

- two-worker process execution;
- out-of-order completion with ordered commit;
- worker exception with no partial commit;
- worker-count limit by batch size;
- serializability under `forkserver`.

Run only:

```bash
pytest -q <new process executor test file>
pytest -q <new accumulator integrity test file>
pytest -q tests/test_sprint1_scenario_simulator.py
```

Do not run Occupancy `fast` or `full`.

## Acceptance criteria

- process backend runs at least two replications concurrently;
- pool lifetime covers the full simulation arm;
- batch boundary and stopping remain in the main process;
- failures cannot produce partial results;
- sequential remains the default.

## Commit

```text
feat: add process-based Monte Carlo executor
```

## Stop condition

Push, report, and stop.

---

# BLOCK 5 — Validate every sampler and arm semantic

## Goal

Prove that process execution works for all real and synthetic sampling modes used by CoInfoSim.

## Required reading

- `src/coinfosim/samplers/gaussian.py`;
- `src/coinfosim/samplers/real.py`;
- `src/coinfosim/samplers/gmm.py`;
- `src/coinfosim/samplers/transfer.py`;
- `tests/test_real_sampler.py`;
- `tests/test_gmm_sampler.py`;
- `tests/test_monte_carlo_with_real_sampler.py`.

## Required tests

Use small in-memory fixtures whenever possible.

For each sampler family:

1. Run a tiny sequential simulation.
2. Run the identical configuration with process backend and two workers.
3. Compare:
   - sample-size grid;
   - subset ordering;
   - classifier ordering;
   - every accumulator loss array with `np.array_equal`;
   - every stopping-info field except runtime;
   - metadata fields unrelated to execution runtime.

Required coverage:

- `GaussianClassConditionalSampler`;
- `RealDatasetSampler` with a small finite in-memory dataset;
- `GMMClassConditionalSampler` with a small in-memory model;
- `SyntheticTrainRealTestSampler` wrapping Gaussian training and fixed real test;
- `SyntheticTrainRealTestSampler` wrapping GMM training and fixed real test.

Add a pickle round-trip test for every sampler used in worker context.

## GMM requirement

Verify through code structure and one focused instrumentation test that cumulative weights and Cholesky factors are not recomputed per replication. Do not add permanent production counters.

## Authorized tests

```bash
pytest -q tests/test_real_sampler.py
pytest -q tests/test_gmm_sampler.py
pytest -q <new serial-versus-process sampler equivalence tests>
```

If raw Occupancy files are available, also run:

```bash
pytest -q tests/test_monte_carlo_with_real_sampler.py
```

A skip caused by missing raw files is acceptable.

## Acceptance criteria

- every sampler works with the process backend;
- sequential and process loss arrays are exactly equal;
- real → real and synthetic → real semantics are preserved;
- no generative-model fitting occurs inside replication workers.

## Commit

```text
test: validate parallel execution across sampler families
```

## Stop condition

Push, report, and stop.

---

# BLOCK 6 — Integrate CLI, progress, metadata, persistence snapshots, and reports

## Goal

Expose process execution through the Occupancy command and make the execution environment auditable without changing statistical report logic.

## Required reading

- `scripts/run_occupancy_scenario.py`;
- `src/coinfosim/simulation/progress.py`;
- `src/coinfosim/results/snapshots.py`;
- `src/coinfosim/results/persistence.py`;
- `src/coinfosim/reports/monte_carlo.py`;
- `src/coinfosim/reports/occupancy_monte_carlo.py`;
- `tests/test_cooperative_progress.py`;
- relevant sections of `tests/test_occupancy_run_tracking.py`.

## CLI requirements

Add options equivalent to:

```text
--execution-backend {sequential,process}
--n-jobs N
--worker-inner-threads N
--multiprocessing-start-method {forkserver,fork}
```

Defaults must preserve current execution:

```text
backend = sequential
n_jobs = 1
```

Recommended Ubuntu full invocation after final approval:

```bash
python scripts/run_occupancy_scenario.py \
  --mode full \
  --execution-backend process \
  --n-jobs 5 \
  --worker-inner-threads 1 \
  --multiprocessing-start-method forkserver
```

Do not run this full command in this block.

## Metadata requirements

Record:

- execution backend;
- requested workers;
- effective workers;
- numeric threads per worker;
- start method;
- detected logical CPUs;
- estimated fixed-test cache bytes per worker.

Store execution metadata in `SimulationResult.metadata`, preserving the persistence schema.

Expose it in:

- `simulation.json`;
- compact simulation summary data;
- arm-report run configuration;
- simulation-start progress output.

Execution runtime and worker settings must not alter scientific identity comparisons.

## Report requirements

Do not change numerical tables, rankings, N-star logic, structural metrics, or section meaning. Add only concise execution configuration fields.

## Authorized tests

```bash
pytest -q tests/test_cooperative_progress.py
pytest -q <new CLI/execution metadata tests>
pytest -q tests/test_run_registry.py::test_save_and_load_simulation_result_roundtrip
```

If raw Occupancy files are available, run one specifically selected tiny orchestration test, not the complete file:

```bash
pytest -q tests/test_occupancy_run_tracking.py::<one approved tiny orchestration test>
```

## Acceptance criteria

- old commands remain sequential;
- process execution is explicitly selectable;
- all three arms receive the same execution configuration;
- execution metadata survives persistence and report regeneration;
- numerical report content is unchanged.

## Commit

```text
feat: expose parallel Monte Carlo execution controls
```

## Stop condition

Push, report, and stop.

---

# BLOCK 7 — Targeted numerical and structural regression validation

## Goal

Demonstrate that process execution produces the same scientific product as sequential execution.

## Required reading

- `src/coinfosim/results/snapshots.py`;
- `src/coinfosim/results/structural.py`;
- `src/coinfosim/reports/report_tables.py`;
- `tests/test_structural_metrics.py`;
- `tests/test_occupancy_scenario_report.py`;
- `tests/test_occupancy_monte_carlo_reports.py`.

## Required comparison

Using a tiny configuration, construct sequential and process results for:

- real → real;
- single Gaussian → real;
- GMM → real.

Use small in-memory data where possible. One raw Occupancy end-to-end test may be used if files are available.

Compare exactly, excluding runtime and execution metadata.

### Simulation-level comparison

- every loss array;
- stopping information;
- summary table;
- best-subset rankings;
- threshold comparisons;
- structural dynamics;
- full loss table;
- precision diagnostics.

### Scenario-level comparison

- arm semantics;
- ranking fidelity;
- winner agreement;
- progressive N-star similarity;
- selected structural subsets;
- all JSON-safe scenario report data.

Use existing functions:

```python
simulation_report_data(...)
scenario_report_data(...)
```

Do not compare HTML byte-for-byte because runtime and execution fields legitimately differ.

## Authorized tests

```bash
pytest -q <new numerical/report-data equivalence test file>
pytest -q tests/test_structural_metrics.py
pytest -q tests/test_run_registry.py::test_save_and_load_simulation_result_roundtrip
pytest -q tests/test_sprint1_scenario_simulator.py
pytest -q tests/test_real_sampler.py
pytest -q tests/test_gmm_sampler.py
pytest -q tests/test_cooperative_progress.py
```

Do not run the full suite yet.

## Acceptance criteria

- all scientific numerical outputs match exactly;
- process execution changes only execution metadata and runtime;
- persistence remains exact;
- report regeneration requires no Monte Carlo rerun and preserves numerical data.

## Commit

```text
test: verify parallel Monte Carlo scientific equivalence
```

## Stop condition

Push, report, and stop.

---

# BLOCK 8 — Representative benchmark on the six-core Ubuntu desktop

## Goal

Measure useful speedup with a bounded benchmark that does not run the full scenario.

## Machine reference

```text
CPU: Intel Core i5-9400
Physical cores: 6
Threads per core: 1
Memory: 16 GiB
OS: Ubuntu 24.04 LTS
```

## Benchmark design

Add a lightweight benchmark script or documented command that:

- runs one representative arm without reports or registry artifacts;
- uses all 31 Occupancy subsets and all 3 classifiers;
- uses sample sizes `(2, 32, 512)`;
- uses exactly 40 replications per sample size;
- uses batch size 20;
- uses the same seed for every worker configuration;
- measures simulator runtime only;
- verifies exact equality before reporting speedup.

Run one pass for:

```text
n_jobs = 1
n_jobs = 5
n_jobs = 6
```

Use one numeric thread per process.

Do not test 2, 3, or 4 workers unless the user requests a fuller scaling study.

If five versus six workers differs by less than 5%, run one additional repeat only for those two configurations.

## Required output

Produce a small Markdown or CSV table with:

- backend;
- requested and effective workers;
- runtime;
- speedup versus sequential;
- efficiency;
- exact-equality result;
- peak-memory observation if readily available without new tooling.

Do not add benchmark results to scientific report tables.

## Authorized commands

Only the dedicated benchmark and its exact-equality check. Do not run the full scenario.

## Acceptance criteria

- exact equality is confirmed;
- a default for this machine is documented;
- five workers remain the initial recommendation unless measurement clearly favors six.

## Commit

Commit reusable benchmark code or documentation, not volatile machine-specific output unless requested.

```text
perf: add bounded Monte Carlo parallel benchmark
```

## Stop condition

Report benchmark results and stop. Do not start Block 9 without explicit approval.

---

# BLOCK 9 — Optional full-scenario acceptance run

## Status

Completed and accepted by the user. The first attempt, scenario run ID 3, failed
during main-process visualization after completing the real-data arm. The
successful rerun is scenario run ID 4 with simulation run IDs 10, 11, and 12.
The failed diagnostic and successful acceptance artifacts are retained on
`feature/parallel-monte-carlo` for auditability.

## Goal

Run the full parallel Occupancy scenario and compare it against scenario run ID 2.

## Invocation

```bash
python scripts/run_occupancy_scenario.py \
  --mode full \
  --execution-backend process \
  --n-jobs 5 \
  --worker-inner-threads 1 \
  --multiprocessing-start-method forkserver
```

## Required comparison

Compare the new arms against simulation runs 6, 7, and 8:

- same sample-size grid;
- same subset and classifier order;
- exact equality of every persisted loss array;
- same replication counts and stopping reasons;
- same maximum CI half-width values;
- same summaries, rankings, thresholds, structural dynamics, and cross-arm structural metrics.

Runtime and execution metadata are expected to differ.

## Failure rule

If any scientific value differs:

- identify the first differing arm, `n_per_class`, subset, classifier, and replication ID;
- stop acceptance work;
- report the diagnostic;
- do not weaken equality tolerances without approval.

## Deliverable

A final acceptance report with old/new runtime per arm, total speedup, equality result, worker configuration, memory observations, and new artifact paths.

## Accepted artifact disposition

The Block 9 Occupancy artifacts are useful acceptance evidence but are less
important than the independently developed Air Quality artifacts whose run IDs
overlap them. Commit the Block 9 artifacts to
`feature/parallel-monte-carlo`, but retain them only in that branch's history
when building the combined integration branch.

The combined tree must:

- prefer the committed `origin/feature/air-quality` registries and generated
  artifacts;
- omit the Block 9 Occupancy scenario directories 3 and 4 and simulation
  directories 9 through 12 from the final integration tree;
- preserve the pre-existing Occupancy artifacts already present on `main`;
- never use a whole-repository merge strategy such as `-X theirs`, because
  source changes from both branches are required.

---

# BLOCK 10 — Archive Block 9 and record the approved integration plan

## Goal

Preserve the completed Block 9 evidence on `feature/parallel-monte-carlo`, add
this approved integration plan to version control, and leave a clean branch for
the cross-feature integration work.

## Required reading

- this complete plan;
- `output/reports/scenario_runs.json`;
- `output/reports/simulation_runs.json`;
- scenario records 3 and 4;
- simulation records 9 through 12.

## Tasks

1. Fetch `origin` and verify that `origin/main`,
   `origin/feature/parallel-monte-carlo`, and
   `origin/feature/air-quality` have not unexpectedly advanced.
2. Confirm that scenario run ID 4 and simulation run IDs 10, 11, and 12 are
   completed and that their referenced artifacts exist.
3. Retain failed scenario run ID 3 and simulation run ID 9 as diagnostic
   evidence; do not rewrite the registries.
4. Commit the Block 9 registries, generated artifacts, and this implementation
   plan on `feature/parallel-monte-carlo`.
5. Push the feature branch when credentials permit.
6. Do not change simulator, dataset, report, or scenario source code.

## Authorized commands/tests

No simulation or pytest command is authorized in this block. Use only
read-only registry/artifact validation, `git diff --check`, Git status, commit,
and push commands.

## Acceptance criteria

- the Block 9 evidence is committed and pushed on the parallel feature branch;
- scenario and simulation registry records remain internally consistent;
- this plan documents Blocks 10 through 14;
- the worktree is clean;
- no implementation source changed.

## Commit

```text
test: record full parallel Occupancy acceptance
```

## Stop condition

Push, return the Block 10 completion report, and stop. Do not create the
integration branch until explicit approval.

---

# BLOCK 11 — Integrate Air Quality with the parallel execution architecture

## Goal

Create a dedicated integration branch containing both features, resolve the
known source and artifact conflicts deliberately, and make the generic
dataset-anchored runner support the approved parallel execution architecture.

## Required branch

Create and push:

```text
feature/parallel-air-quality-integration
```

Create it from the completed `feature/parallel-monte-carlo` commit. Merge
`origin/feature/air-quality` into it with a merge commit. Do not merge either
feature into `main` in this block.

## Required reading

- this complete plan;
- `scripts/run_occupancy_scenario.py` from both feature tips;
- `scripts/run_air_quality_scenario.py`;
- `src/coinfosim/scenarios/dataset_anchored_runner.py`;
- `src/coinfosim/simulation/execution.py`;
- `src/coinfosim/simulation/monte_carlo.py`;
- `src/coinfosim/reports/monte_carlo.py` from both feature tips;
- `src/coinfosim/runs/report_data.py` from both feature tips;
- the Air Quality scenario-runner and report tests;
- the parallel execution-metadata and CLI tests.

## Merge and artifact rules

1. Fetch and require a clean worktree before creating the integration branch.
2. Merge with `--no-ff --no-commit` so every conflict can be inspected before
   the merge commit is created.
3. Resolve source files by preserving the Air Quality generic dataset
   architecture and the parallel branch's execution controls and metadata.
4. For generated output only, restore the Air Quality registries/artifacts and
   remove the six Block 9 Occupancy directories listed above from the combined
   tree. Those artifacts remain recoverable from the parallel branch history.
5. Do not resolve source files wholesale as either `ours` or `theirs`.
6. Do not merge, rebase, or cherry-pick any other concurrent branch.

## Integration requirements

- Add optional `ExecutionConfig` propagation to the generic
  `run_dataset_anchored_scenario()` path.
- Pass the same execution configuration to all three scenario arms.
- Preserve sequential defaults for existing callers.
- Preserve the Occupancy CLI execution options from Block 6.
- Add the equivalent execution options to the Air Quality CLI.
- Keep execution metadata in `SimulationResult.metadata`, summaries,
  persistence records, progress output, and reports.
- Preserve the Air Quality generic report extensions and all existing
  Occupancy public wrappers.
- Preserve deterministic sampling, canonical cell ordering, atomic batches,
  complete-replication semantics, stopping boundaries, and persistence schema.
- Workers remain prohibited from progress, accumulation, stopping, registry,
  persistence, or report work.

## Authorized tests

```bash
pytest -q tests/test_execution_metadata.py
pytest -q tests/test_air_quality_scenario_runner.py
pytest -q tests/test_occupancy_run_tracking.py
pytest -q tests/test_air_quality_monte_carlo_reports.py
pytest -q tests/test_occupancy_monte_carlo_reports.py
```

Add or select the smallest focused tests needed to prove execution-config
propagation through both CLIs and the generic three-arm runner. Do not run an
actual Air Quality scenario, Occupancy `fast`/`full`, or the full test suite in
this block.

## Acceptance criteria

- the integration branch contains both complete feature histories;
- the final generated-output tree follows the approved Air Quality preference;
- both dataset CLIs expose the same process controls and preserve sequential
  defaults;
- all three generic scenario arms receive one execution configuration;
- focused Occupancy, Air Quality, persistence, metadata, and report tests pass;
- no scientific equality requirement or schema is weakened.

## Commit

Create one reviewed merge commit containing the conflict resolutions and
required integration changes:

```text
merge: integrate Air Quality with parallel Monte Carlo
```

## Stop condition

Push, return the Block 11 completion report, and stop. Do not run the smoke
scenario until explicit approval.

---

# BLOCK 12 — Air Quality process smoke acceptance and exact comparison

## Goal

Run the complete Air Quality smoke pipeline with the process backend, generate
all reports outside the tracked output tree, and prove that its scientific
results exactly equal the committed sequential Air Quality smoke reference.

## Required reading

- `scripts/run_air_quality_scenario.py`;
- `src/coinfosim/scenarios/dataset_anchored_runner.py`;
- `src/coinfosim/results/persistence.py`;
- `src/coinfosim/runs/report_data.py`;
- committed Air Quality smoke scenario 4 and simulation runs 12, 13, and 14.

## Invocation

Use a fresh temporary output directory, never `output/reports`:

```bash
.venv/bin/python scripts/run_air_quality_scenario.py \
  --mode smoke \
  --output-dir /tmp/coinfosim-air-quality-process-smoke \
  --execution-backend process \
  --n-jobs 5 \
  --worker-inner-threads 1 \
  --multiprocessing-start-method forkserver
```

## Required validation

- scenario and all three simulations complete successfully;
- all expected JSON, compressed result, CSV, HTML, and graph artifacts exist;
- all three simulations record the requested process configuration and
  effective worker count;
- persisted loss arrays exactly equal committed sequential smoke simulation
  runs 12, 13, and 14, matched by arm;
- sample sizes, subsets, classifiers, replication counts, stopping information,
  summaries, report tables, rankings, thresholds, structural dynamics, and
  cross-arm structural report data are exactly equal, excluding runtime and
  execution-only metadata;
- the tracked worktree remains unchanged by the temporary acceptance run.

## Failure rule

On any difference, report the first differing arm, sample size, subset,
classifier, replication ID, or report-data field and stop. Do not loosen exact
comparisons.

## Authorized tests

Run only the command above and a focused comparison/validation command. Do not
run Occupancy `fast`/`full`, Air Quality `fast`/`full`, or the complete suite.

## Commit

Do not create an empty commit. Commit only a focused test or integration fix if
the authorized acceptance work demonstrates one is necessary and all
authorized checks then pass.

## Stop condition

Return the Block 12 completion report and stop. The expensive Air Quality full
scenario requires explicit approval as the next block.

---

# BLOCK 13 — Full parallel Air Quality scenario acceptance

## Status

Expensive. Requires separate explicit approval after Block 12 passes.

## Goal

Run one real Air Quality `full` scenario using process execution and retain its
completed scientific reports as the final combined-feature acceptance
artifacts.

## Invocation

```bash
.venv/bin/python scripts/run_air_quality_scenario.py \
  --mode full \
  --execution-backend process \
  --n-jobs 5 \
  --worker-inner-threads 1 \
  --multiprocessing-start-method forkserver
```

## Required validation

- the scenario and real, single-Gaussian, and GMM arms complete successfully;
- every sample size stops only on a complete batch boundary;
- every cell contains contiguous replication IDs reconstructed by ordered
  position and has the arm's recorded replication count;
- all losses are finite and in `[0, 1]`;
- persisted results load successfully without a schema change;
- execution metadata reports the process backend, requested/effective workers,
  thread limit, start method, logical CPUs, and fixed-test cache estimate;
- summaries, precision diagnostics, full loss tables, arm reports, dataset
  report, scenario report, and structural comparison artifacts are present and
  internally consistent;
- workers produced no direct progress, registry, persistence, or report writes;
- the completed full artifacts and updated registries are committed on the
  integration branch.

The Air Quality branch has no sequential `full` reference. Do not manufacture
one by running a second expensive full scenario. Exact sequential/process
equality is established by Block 12 and the existing focused equivalence tests;
Block 13 validates the full production path and complete persisted product.

## Failure rule

If execution or validation fails, retain diagnostic evidence, run only the
smallest useful diagnostic, report the failure, and stop. Do not rerun the full
scenario automatically and do not weaken numerical requirements.

## Commit

When validation passes:

```text
test: record full parallel Air Quality acceptance
```

## Stop condition

Push, return the Block 13 completion report, and stop. Do not merge to `main`
without explicit final approval.

---

# BLOCK 14 — Final reviewed merge to main

## Status

Requires explicit approval only after Block 13 succeeds.

## Goal

Land the reviewed integration branch on `main` without rewriting either feature
history or rerunning expensive scenarios.

## Tasks

1. Fetch `origin` and require a clean worktree.
2. Verify the integration branch and `origin/main` tips against the Block 13
   report.
3. If `origin/main` advanced, stop and prepare a separately approved update and
   focused regression plan; do not rebase or merge automatically.
4. Verify that the committed Air Quality full acceptance records and artifacts
   are present and that the Block 9 Occupancy artifacts are absent from the
   integration tree.
5. Use the repository's accepted pull-request workflow. If direct local
   integration was explicitly chosen and branch protection permits it,
   fast-forward `main` to the reviewed integration tip; do not create new
   implementation commits on `main`.
6. Push without force and verify the remote `main` SHA.

## Authorized tests

No expensive scenario rerun is authorized. Run only the final minimal checks
explicitly approved with this block, plus `git diff --check` and repository
state/history verification.

## Acceptance criteria

- `main` contains both feature histories and the reviewed integration commit;
- remote `main` points to the approved integration result;
- the Air Quality full parallel acceptance artifacts are present;
- the superseded Block 9 Occupancy artifacts remain only in parallel-branch
  history;
- no force push, rebase, cherry-pick, or history rewrite occurred.

## Stop condition

Return the Block 14 completion report and stop.

---

## 8. Final definition of done

The feature is complete when:

- work occurred on `feature/parallel-monte-carlo`;
- sequential remains the default;
- process tasks are complete replications;
- one persistent pool is used per arm;
- worker context initializes once per process;
- numeric worker threads are controlled;
- batches are validated and committed atomically;
- stopping runs only after complete batches;
- duplicate, missing, partial, non-finite, or non-contiguous results are rejected;
- all sampler families show exact sequential/process equality;
- persistence schema remains unchanged;
- report and structural data remain scientifically identical;
- CLI and metadata expose execution settings;
- bounded benchmark results are documented;
- the full parallel Occupancy acceptance is archived on its feature branch;
- the combined output tree prefers the Air Quality branch artifacts where run
  IDs overlap;
- the generic dataset-anchored runner and both dataset CLIs support the same
  parallel execution configuration;
- exact Air Quality sequential/process smoke equivalence is demonstrated;
- one full parallel Air Quality scenario and its reports complete successfully;
- neither full scenario runs nor the final `main` merge occur without their
  explicit approval gates;
- the user approves every block before the next begins.

---

## 9. Agent reporting template

```markdown
## Block N completion report

### Repository state
- Branch:
- Base/main status:
- Commit:
- Remote push:

### Files changed
- `path`: purpose

### Implementation completed
- ...

### Tests executed
- `command`
  - Result:
  - Duration, if available:

### Tests intentionally not executed
- ...

### Scientific-equivalence status
- ...

### Deviations from the plan
- None / details

### Risks or questions for review
- ...

**Stopped after Block N. Awaiting explicit approval before Block N+1.**
```
