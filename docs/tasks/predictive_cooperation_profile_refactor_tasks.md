# CoInfoSim Predictive-Cooperation Profile Refactor
## Implementation Task Specification

**Repository:** `paulorenatoaz/coinfosim`  
**Observed remote baseline during planning:** `main` at commit `4ceba2f8ea2d97ec99e72f4a7558b91d53cc6bd8`  
**Planned feature branch:** `feature/predictive-cooperation-profile-refactor`  
**Primary objective:** replace the active `N*` / directed-crossing / composite-similarity framework with a mathematically explicit pairwise winner-reversal framework, update generated HTML reports and persisted report data, and align the repository's current scientific terminology and documentation.

---

# 1. Execution policy

This document is the implementation contract. Do not reinterpret the scientific method independently.

## 1.1 Credit and computation budget

The implementation agent must:

1. inspect only the files named in the active block;
2. use targeted symbol searches rather than broad repository exploration;
3. run only the tests explicitly named in the active block;
4. never run the full test suite after every block;
5. never run `fast`, `full`, `full-scale`, or `strict` scenarios;
6. never rerun a Monte Carlo experiment merely to regenerate a report;
7. use persisted result data for report regeneration when it is already available;
8. skip optional report regeneration when persisted local data is unavailable;
9. stop after each block, commit the block, report the result, and wait for approval;
10. avoid unrelated refactors, formatting sweeps, dependency upgrades, version bumps, or repository-wide renames.

No network downloads are required for this refactor.

## 1.2 Git policy

Before source changes:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git fetch origin main
git switch main
git pull --ff-only origin main
git switch -c feature/predictive-cooperation-profile-refactor
```

Rules:

- Do not discard, stash, overwrite, or commit pre-existing user changes without explicit approval.
- If the worktree is not clean, stop and report the exact paths.
- Record the actual starting SHA; do not require it to equal the planning SHA above.
- Create one focused commit after each approved block.
- Do not merge into `main` until the owner explicitly approves the final result.

## 1.3 Mandatory review checkpoint

After every block, report only:

- block completed;
- files changed;
- commands/tests run;
- pass/fail result;
- commit SHA;
- unresolved decisions or blockers.

Then stop.

---

# 2. Fixed scientific and terminology decisions

These decisions are already approved and must not be redesigned by the agent.

## 2.1 Core terminology

Use the following English terminology in active code documentation, report prose, README material, and academic sources:

- **attribute**: formal model-input variable;
- **information channel**: conceptual or operational interpretation of an attribute or attribute group;
- **predictive cooperation**: umbrella phenomenon describing how subset composition affects predictive performance;
- **predictive cooperation profile**: quantitative evolution of relative subset performance over sample size for a fixed dataset, classifier, and training condition;
- **predictive cooperation pattern**: an interpreted regularity identified in one or more profiles;
- **estimated mean test loss**: Monte Carlo estimate used for subset comparisons;
- **pairwise winner relation**: which of two subsets has lower estimated mean test loss;
- **pairwise predictive-superiority reversal**, shortened after first definition to **winner reversal**: a valid change from one defined pairwise winner to the other;
- **reversal sample size**: the evaluated sample size assigned to a valid winner reversal;
- **last observed reversal sample size**: the latest reversal sample size in the evaluated prefix;
- **winner matrix**: `W`;
- **reversal matrix**: `R`;
- **reversal existence agreement**: Jaccard agreement over unordered subset pairs for which `R` is defined;
- **reversal sample-size similarity**: normalized agreement between last observed reversal sample sizes for shared pairs;
- **mean log2 reversal distance**: unnormalized mean absolute difference on the `log2(n)` scale.

Do not use these as active principal terms:

- cooperative advantage threshold;
- `N*`, `N-star`, `nstar`;
- crossing point;
- directed crossing matrix;
- timing similarity;
- location similarity;
- structural fidelity as the name of the central scientific object;
- a multiplicative/composite reversal metric.

Historical changelog entries and explicitly legacy APIs may retain historical names when rewriting them would falsify history or break an intentionally retained compatibility surface.

## 2.2 Estimated pairwise winner

For subsets `S_i` and `S_j`, classifier `f`, training condition `g`, and sample size `n_k`, let:

\[
\widehat L_i^{(g,f)}(n_k)
\]

be the estimated mean test loss.

The observed pairwise outcome is:

\[
U_{ij}(n_k)=
\begin{cases}
+1, & \widehat L_i(n_k)<\widehat L_j(n_k),\\
-1, & \widehat L_i(n_k)>\widehat L_j(n_k),\\
0, & \widehat L_i(n_k)=\widehat L_j(n_k).
\end{cases}
\]

Use only unordered pairs `i < j` for reversal analysis. The sign is interpreted relative to the row/index `i`:

- `+1`: `S_i` wins;
- `-1`: `S_j` wins;
- `0`: no effective winner has yet been established.

## 2.3 Tie propagation rule

Construct the effective winner relation `W_ij(n_k)` as follows:

\[
W_{ij}(n_k)=
\begin{cases}
U_{ij}(n_k), & U_{ij}(n_k)\neq 0,\\
W_{ij}(n_{k-1}), & U_{ij}(n_k)=0\text{ and a previous winner exists},\\
0, & U_{ij}(n_k)=0\text{ and no previous winner exists}.
\end{cases}
\]

Consequences:

- initial ties remain undefined (`0`);
- the first strict winner initializes the relation;
- an exact tie after initialization preserves the previous winner;
- a tie does not create, erase, or reverse the relation;
- the first strict winner after initial ties is not a reversal.

This carry-forward rule applies to the effective pairwise winner matrices used by:

- winner-reversal extraction;
- reversal matrices;
- Winner Agreement.

It does **not** replace average-rank tie handling in Spearman ranking fidelity. Ranking fidelity remains a separate metric based on the current loss vector at each `n`.

## 2.4 Valid winner reversal and existence rule

A valid winner reversal occurs at `n_k`, `k >= 2`, only when:

\[
W_{ij}(n_{k-1})\neq 0,\qquad
W_{ij}(n_k)\neq 0,\qquad
W_{ij}(n_k)\neq W_{ij}(n_{k-1}).
\]

For a prefix ending at `n_t`, define:

\[
\mathcal E_{ij}(n_t)
=
\left\{
n_k\leq n_t:
W_{ij}(n_{k-1})\neq 0,\;
W_{ij}(n_k)\neq 0,\;
W_{ij}(n_k)\neq W_{ij}(n_{k-1})
\right\}.
\]

Then:

\[
R_{ij}(n_t)=
\begin{cases}
\max \mathcal E_{ij}(n_t), &
\mathcal E_{ij}(n_t)\neq\varnothing,\\
\varnothing, &
\mathcal E_{ij}(n_t)=\varnothing.
\end{cases}
\]

The scalar `R_ij(n_t)` exists **if and only if** the pair has undergone at least one valid winner reversal in the evaluated prefix.

It does not exist merely because:

- one subset wins at the first sample size;
- a first winner appears after initial ties;
- curves approach each other;
- an exact tie occurs;
- one subset remains the winner over the complete prefix.

## 2.5 Matrix representation

For `q` subsets:

- `W(n_k)` may remain a full antisymmetric display matrix:
  - diagonal: unavailable;
  - upper triangle: effective outcomes;
  - lower triangle: opposite signs.
- `R(n_t)` must represent each unordered pair once:
  - use the upper triangle (`i < j`);
  - diagonal and lower triangle are unavailable;
  - a defined cell stores the latest observed reversal sample size;
  - `R` does not separately store the new winner because that information is recoverable from `W`.

The numerical metrics must use all unordered pairs, regardless of any reduced display subset selection.

## 2.6 Reversal metrics

For arm `g` and prefix `n_t`, define:

\[
\mathcal C_g(n_t)
=
\left\{
(i,j): i<j,\;R_{ij}^{(g)}(n_t)\neq\varnothing
\right\}.
\]

### Reversal existence agreement

\[
A_R(n_t)
=
\frac{
|\mathcal C_{\mathrm{ref}}(n_t)\cap\mathcal C_g(n_t)|
}{
|\mathcal C_{\mathrm{ref}}(n_t)\cup\mathcal C_g(n_t)|
}.
\]

Conventions:

- first prefix `n_1`: unavailable, because no reversal can yet occur;
- empty union after `n_1`: `A_R = 1.0`, with status `no_reversals_in_either`;
- nonempty union and empty intersection: `A_R = 0.0`;
- otherwise use ordinary Jaccard.

### Mean log2 reversal distance

For shared pairs:

\[
\mathcal H(n_t)
=
\mathcal C_{\mathrm{ref}}(n_t)\cap\mathcal C_g(n_t),
\]

\[
D_R(n_t)
=
\frac{1}{|\mathcal H(n_t)|}
\sum_{(i,j)\in\mathcal H(n_t)}
\left|
\log_2 R_{ij}^{(g)}(n_t)
-
\log_2 R_{ij}^{(\mathrm{ref})}(n_t)
\right|.
\]

`D_R` is measured in levels of the power-of-two sample-size scale.

### Reversal sample-size similarity

For `n_t > n_1` and at least one shared reversal pair:

\[
S_R(n_t)
=
\operatorname{clip}
\left(
1-
\frac{D_R(n_t)}
{\log_2(n_t)-\log_2(n_1)},
0,1
\right).
\]

Conventions:

- first prefix: unavailable;
- no shared reversal pair: unavailable;
- no reversals in either arm: unavailable, not `1.0`, because there are no reversal sample sizes to compare;
- identical shared reversal sample sizes: `S_R = 1.0`.

## 2.7 No composite metric

Delete the active calculation and presentation of:

```python
composite = reversal_existence_agreement * reversal_sample_size_similarity
```

The two metrics answer different questions and must be displayed separately.

---

# 3. Current implementation map

The active implementation observed during planning is concentrated in these files.

## 3.1 Scientific post-processing

`src/coinfosim/results/structural.py`

Existing symbols to replace or refactor:

- `winner_matrix`
- `winner_agreement_series`
- `directed_crossing_events`
- `progressive_directed_nstar`
- `_crossing_cells`
- `progressive_nstar_similarity`
- `simulation_structural_dynamics`
- `scenario_structural_fidelity`

## 3.2 Persistence/report snapshots

`src/coinfosim/runs/report_data.py`

Relevant symbols:

- `simulation_report_data`
- `dataset_anchored_scenario_report_data`
- `generic_scenario_structural_report_data`

## 3.3 Visualizations

`src/coinfosim/reports/structural_visualization.py`

Existing symbols:

- `metric_series_figure`
- `winner_matrix_figure`
- `progressive_nstar_matrix_figure`

## 3.4 Consolidated scenario reports

`src/coinfosim/reports/occupancy_scenario.py`

Shared components used by generic dataset-anchored reports:

- `_structural_fidelity_section`
- `_nstar_section`
- structural graph/tab helpers

`src/coinfosim/reports/dataset_anchored_scenario.py`

Relevant call sites:

- `_structural_fidelity_section(...)`
- `_nstar_section(...)`
- current nine-section report composition

Dataset wrappers should be inspected only when a targeted test or stale dataset-specific phrase requires it:

- `src/coinfosim/reports/occupancy_scenario.py`
- `src/coinfosim/reports/air_quality_scenario.py`
- `src/coinfosim/reports/support2_scenario.py`

## 3.5 Per-arm Monte Carlo HTML reports

`src/coinfosim/reports/monte_carlo.py`

Relevant symbols:

- `_last_competitor_crossing`
- `_nstar_graph_image`
- `_nstar_panel`
- `_nstar_diagnostics_panel`
- `_structural_dynamics_panel`
- `generate_structured_monte_carlo_report`

## 3.6 Historical threshold code

`src/coinfosim/results/analysis.py`

This module contains first-threshold and interpolation APIs used by older/Sprint-1 reporting.

Policy:

- remove these APIs from the **current active report path**;
- do not delete them solely for terminology cleanup if they remain an intentional backward-compatibility API;
- clearly mark them as legacy/historical if retained;
- do not use interpolation in the new `W/R` framework.

Potentially legacy directories such as `src/coinfosim/reporting/` must not be explored or refactored unless a direct import from the current CLI/report path is demonstrated by a targeted search.

## 3.7 Primary tests

- `tests/test_structural_metrics.py`
- `tests/test_occupancy_scenario_report.py`
- `tests/test_air_quality_scenario_report.py`
- `tests/test_support2_scenario_report.py`
- `tests/test_occupancy_monte_carlo_reports.py`
- `tests/test_air_quality_monte_carlo_reports.py`
- `tests/test_support2_monte_carlo_reports.py`
- `tests/test_parallel_scientific_equivalence.py`
- `tests/test_occupancy_run_tracking.py`

## 3.8 Repository-facing documentation and metadata

Review only these known stale-title or stale-terminology files:

- `README.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `pyproject.toml`
- `src/coinfosim/__init__.py`
- `src/coinfosim/cli/app.py`
- `src/coinfosim/publish/site.py`
- `coinfosim_research_proposal_v4.tex`

Do not rewrite historical task documents in `docs/tasks/` unless the owner separately requests it.

---

# 4. Implementation blocks

The blocks are deliberately balanced so that no single block combines core math, report rendering, documentation, and expensive validation.

---

## Block 0 — Repository state and bounded baseline audit
**Complexity:** low  
**Purpose:** establish a safe branch and confirm only the known active call graph.

### Read

- this task document;
- `src/coinfosim/results/structural.py`;
- `src/coinfosim/runs/report_data.py`;
- import sections and structural call sites only in:
  - `src/coinfosim/reports/occupancy_scenario.py`;
  - `src/coinfosim/reports/dataset_anchored_scenario.py`;
  - `src/coinfosim/reports/monte_carlo.py`;
  - `src/coinfosim/reports/structural_visualization.py`;
- import list and affected tests in `tests/test_structural_metrics.py`.

### Allowed targeted searches

```bash
rg -n "progressive_directed_nstar|progressive_nstar_similarity|directed_crossing_events|nstar_similarity|timing_similarity|crossing_jaccard" \
  src/coinfosim/results src/coinfosim/runs src/coinfosim/reports tests
```

```bash
rg -n "_nstar_section|_nstar_diagnostics_panel|progressive_nstar_matrix_figure" \
  src/coinfosim/reports tests
```

Do not perform a repository-wide architecture survey.

### Change

- create the feature branch;
- add no scientific behavior changes;
- optionally add this task document under:
  `docs/tasks/predictive_cooperation_profile_refactor_tasks.md`
  if it is not already present.

### Validation

No test run is required.

### Acceptance criteria

- clean feature branch exists;
- actual base SHA is reported;
- active call sites are confirmed;
- no unrelated files were inspected or modified.

### Commit

```text
chore: start predictive cooperation profile refactor
```

### Stop for review

Mandatory.

---

## Block 1 — Effective winner relation and reversal extraction
**Complexity:** medium  
**Purpose:** implement the approved `W` trajectory, tie propagation, unordered reversal events, and progressive triangular `R`.

### Primary files

- `src/coinfosim/results/structural.py`
- `tests/test_structural_metrics.py`

### Required implementation

Preserve `winner_matrix(...)` as the instantaneous observed comparison helper if it is still useful for compatibility.

Add or rename narrowly scoped helpers with clear docstrings. Recommended interface:

```python
effective_winner_matrices(
    result,
    classifier,
    subsets=None,
) -> list[dict[str, object]]
```

Each item should include:

```python
{
    "n_per_class": int,
    "matrix": list[list[Optional[int]]],
}
```

Effective matrix encoding:

- diagonal: `None`;
- row winner: `+1`;
- row loser: `-1`;
- unresolved initial tie: `0`;
- exact tie after initialization: carry the previous `+1/-1`.

Add:

```python
winner_reversal_events(
    result,
    classifier,
    subsets=None,
) -> list[dict[str, int]]
```

Each event must represent one unordered pair:

```python
{"i": i, "j": j, "n_reversal": n}
```

with `i < j`.

Do not persist a redundant `winner_after` field. It can be recovered from `W`.

Add:

```python
progressive_reversal_matrices(
    result,
    classifier,
    subsets=None,
) -> list[dict[str, object]]
```

Each prefix item:

```python
{
    "n_prefix": int,
    "matrix": triangular_matrix,
}
```

Matrix rules:

- upper triangle stores the last observed reversal sample size through the prefix;
- diagonal and lower triangle are `None`;
- the first prefix may be represented as an empty `R` matrix when needed by a report, but reversal extraction must never treat initialization as reversal.

Remove active dependence on:

- `directed_crossing_events`;
- `progressive_directed_nstar`.

Do not implement interpolation.

### Required unit cases

Update/add only focused tests in `tests/test_structural_metrics.py`:

1. initial tie, then first winner: no reversal;
2. repeated initial ties, then first winner: no reversal;
3. winner `i`, tie, winner `i`: no reversal;
4. winner `i`, tie, winner `j`: one reversal at the strict `j` win;
5. winner `i`, winner `j`: one reversal;
6. multiple alternations: all events retained and progressive `R` keeps only the last;
7. pair with no reversal remains undefined;
8. events use only `i < j`;
9. `R` has values only in the upper triangle;
10. `W` remains antisymmetric after carry-forward.

### Targeted test

```bash
pytest -q tests/test_structural_metrics.py \
  -k "winner or reversal or tie or matrix"
```

Do not run report tests in this block.

### Acceptance criteria

- tie behavior exactly matches Section 2.3;
- existence exactly matches Section 2.4;
- no directed duplicate reversal cells;
- no interpolation;
- focused tests pass.

### Commit

```text
refactor: define effective winners and pairwise reversals
```

### Stop for review

Mandatory.

---

## Block 2 — Separate reversal metrics and schema v2
**Complexity:** medium  
**Purpose:** implement the two approved metrics, remove the product, and update JSON-safe report snapshots.

### Primary files

- `src/coinfosim/results/structural.py`
- `src/coinfosim/runs/report_data.py`
- `tests/test_structural_metrics.py`
- only the serialization-specific assertions in:
  - `tests/test_occupancy_run_tracking.py`;
  - `tests/test_parallel_scientific_equivalence.py`

### Required implementation

Replace `progressive_nstar_similarity(...)` with a function named consistently with the new object, preferably:

```python
progressive_reversal_fidelity(
    arm_results,
    reference_arm,
) -> list[dict[str, object]]
```

Each row must contain:

```text
classifier
arm
n_prefix
n_reference_reversal_pairs
n_arm_reversal_pairs
n_shared_reversal_pairs
n_union_reversal_pairs
reversal_existence_agreement
mean_log2_reversal_distance
reversal_sample_size_similarity
status
```

Status values:

- `unavailable_first_prefix`
- `no_reversals_in_either`
- `no_shared_reversals`
- `ok`

Rules:

- first prefix: both metrics `None`;
- empty union after first prefix:
  - existence agreement `1.0`;
  - distance `None`;
  - sample-size similarity `None`;
- no shared pairs but nonempty union:
  - existence agreement `0.0`;
  - distance `None`;
  - sample-size similarity `None`;
- shared pairs:
  - calculate all three numeric fields;
  - clip similarity to `[0, 1]`;
- do not calculate or emit a composite/product metric.

Update `winner_agreement_series(...)` to use effective `W`:

- carry-forward exact ties after initialization;
- skip only unresolved pairs (`0`) in either arm;
- keep total/valid/matching/skipped counts.

Update:

- `simulation_structural_dynamics(...)`
- `scenario_structural_fidelity(...)`

Recommended persisted structure:

```text
structural_dynamics:
  schema_version: 2
  subset_catalog
  sample_sizes
  classifiers:
    <classifier>:
      effective_winner_pairs_by_n
      winner_reversal_events
```

```text
structural_fidelity:
  schema_version: 2
  ranking_fidelity_series
  winner_agreement_series
  reversal_fidelity_series
  final_summary
  reference_display_subsets_by_classifier
```

Remove from active schema:

- `directed_crossing_events`;
- `nstar_similarity_series`;
- `nstar_similarity`;
- `crossing_jaccard`;
- `timing_similarity`.

Keep strict JSON safety:

```python
json.dumps(payload, allow_nan=False)
```

### Required tests

Add/update focused cases for:

- first prefix unavailable;
- both supports empty;
- disjoint supports;
- partial overlap;
- full overlap with identical reversal sample sizes;
- full overlap with one-level `log2` displacement;
- sample-size similarity undefined without shared pairs;
- no product field anywhere in rows or snapshots;
- schema version equals `2`;
- serialization is deterministic and strict JSON-safe;
- Winner Agreement uses carried-forward winners.

### Targeted tests

```bash
pytest -q tests/test_structural_metrics.py
```

Then only if snapshot assertions are affected:

```bash
pytest -q \
  tests/test_occupancy_run_tracking.py \
  tests/test_parallel_scientific_equivalence.py \
  -k "structural or report_data or scientific_equivalence"
```

### Acceptance criteria

- `A_R`, `D_R`, and `S_R` follow Section 2.6;
- no composite metric remains in active result rows;
- all pair counts are unordered;
- schema version 2 is strict-JSON-safe;
- targeted tests pass.

### Commit

```text
refactor: separate reversal agreement and sample-size similarity
```

### Stop for review

Mandatory.

---

## Block 3 — Structural figures and paired W/R visualization
**Complexity:** medium  
**Purpose:** replace N-star figures with explicit paired winner/reversal figures.

### Primary files

- `src/coinfosim/reports/structural_visualization.py`
- `tests/test_structural_metrics.py`
- report tests only if they directly assert figure labels

### Required implementation

Update `metric_series_figure(...)` to support:

- `rho_rank` on `n_per_class`;
- `winner_agreement` on `n_per_class`;
- `reversal_existence_agreement` on `n_prefix`;
- `reversal_sample_size_similarity` on `n_prefix`.

Use exact y-axis labels:

- `Ranking fidelity`
- `Winner agreement`
- `Reversal existence agreement`
- `Reversal sample-size similarity`

Replace:

```python
progressive_nstar_matrix_figure(...)
```

with:

```python
reversal_matrix_figure(...)
```

Required visual behavior:

- upper-triangular values only;
- lower triangle and diagonal masked;
- cell annotation is the integer sample size;
- colorbar label: `Last observed reversal sample size`;
- missing cells are visually unavailable, not zero;
- title must use `Reversal matrix`, never `N-star`, `crossing`, or `timing`.

Keep `winner_matrix_figure(...)`, but update its prose/legend if needed:

- `row wins`;
- `unresolved`;
- `row loses`.

A current exact tie after initialization will already be carried forward before reaching this figure; `0` means unresolved effective winner.

Do not change colors or general figure styling unless required by the new triangular representation.

### Targeted tests

Prefer pure figure-construction tests already present. If none exist, add only lightweight assertions that figures construct without error and contain the expected labels.

```bash
pytest -q tests/test_structural_metrics.py -k "figure or matrix"
```

If no matching tests exist, run no additional test and rely on Block 4 report tests.

### Acceptance criteria

- four metric types render;
- `R` is visibly triangular;
- labels use the new terminology;
- no N-star label remains in active structural visualization code.

### Commit

```text
refactor: visualize winner and reversal matrices
```

### Stop for review

Mandatory.

---

## Block 4 — Consolidated scenario HTML reports
**Complexity:** medium-high  
**Purpose:** update the shared scenario renderer used by Occupancy, Air Quality, and SUPPORT2.

### Primary files

- `src/coinfosim/reports/occupancy_scenario.py`
- `src/coinfosim/reports/dataset_anchored_scenario.py`
- `tests/test_occupancy_scenario_report.py`
- `tests/test_air_quality_scenario_report.py`
- `tests/test_support2_scenario_report.py`

Inspect dataset-specific report wrappers only when a failing targeted test identifies stale text:

- `src/coinfosim/reports/air_quality_scenario.py`
- `src/coinfosim/reports/support2_scenario.py`

### Required scenario-report structure

Keep the existing tab-based UI. Do not replace tabs with dropdown selectors.

Update the current structural section so it explains:

1. ranking fidelity compares complete subset rankings;
2. Winner Agreement compares effective pairwise winner relations;
3. reversal existence agreement compares which unordered pairs have a valid last observed reversal;
4. reversal sample-size similarity compares the reversal sample sizes only for shared pairs;
5. no composite index is reported.

### Summary table

Remove active columns/labels:

- `S_N*`
- `Progressive N-star similarity`
- `Crossing Jaccard`
- `Timing similarity`
- `N-star status`
- directed crossing counts

Add:

- `A_R(<=Nmax)` or the full label `Reversal existence agreement`;
- `S_R(<=Nmax)` or the full label `Reversal sample-size similarity`;
- `Reference reversal pairs`;
- `Arm reversal pairs`;
- `Shared reversal pairs`;
- `Union reversal pairs`;
- `Mean log2 reversal distance`;
- `Reversal status`.

### Metric curves

Present four separate tabs/curves:

- Ranking fidelity;
- Winner agreement;
- Reversal existence agreement;
- Reversal sample-size similarity.

Do not multiply or merge the reversal metrics.

### Matrix panels

Replace the separated winner and progressive-N-star sections with one coherent section:

**Pairwise winner and reversal dynamics**

For each classifier, arm, and selected prefix:

- show `W(n_t)`;
- show `R(n_t)` beside or directly below it;
- explain:
  - `W` says who currently wins;
  - `R` says when the pair last changed winner;
  - the direction/new winner is recovered from `W`;
  - `R` stores one upper-triangular value per unordered pair;
  - display subset reduction does not alter all-subset numerical metrics.

At the first prefix, show the winner matrix and an empty/unavailable reversal matrix or an explicit note that no reversal can yet exist.

### Remove old active section

Remove the call to `_nstar_section(...)` from the generic dataset-anchored scenario report.

Delete or leave private dead helpers only after confirming they have no active imports. Do not retain a second interpolated metric section in the report.

Renumber sections consistently.

### Targeted tests

Run only:

```bash
pytest -q \
  tests/test_occupancy_scenario_report.py \
  tests/test_air_quality_scenario_report.py \
  tests/test_support2_scenario_report.py
```

### Text assertions

Tests must establish that generated active scenario HTML:

- contains `predictive cooperation profile` where the central object is explained;
- contains `Reversal existence agreement`;
- contains `Reversal sample-size similarity`;
- contains `Winner matrix`;
- contains `Reversal matrix`;
- does not contain `Progressive N-star similarity`;
- does not contain `Timing similarity`;
- does not contain `Interpolated N-star`;
- does not contain a composite/product metric label.

Do not assert that historical source modules contain no old strings.

### Acceptance criteria

- all three scenario report tests pass;
- shared renderer is used rather than three duplicated implementations;
- tabs are preserved;
- W/R interpretation is explicit;
- old metric section is absent.

### Commit

```text
refactor: update scenario reports for winner reversals
```

### Stop for review

Mandatory.

---

## Block 5 — Per-arm Monte Carlo HTML reports
**Complexity:** medium-high  
**Purpose:** remove the duplicated historical N-star diagnostics from the 17-section arm report and present arm-local W/R dynamics.

### Primary files

- `src/coinfosim/reports/monte_carlo.py`
- `tests/test_occupancy_monte_carlo_reports.py`
- `tests/test_air_quality_monte_carlo_reports.py`
- `tests/test_support2_monte_carlo_reports.py`

Dataset-specific Monte Carlo wrappers may be inspected only if a focused test shows stale injected prose:

- `src/coinfosim/reports/occupancy_monte_carlo.py`
- `src/coinfosim/reports/air_quality_monte_carlo.py`
- `src/coinfosim/reports/support2_monte_carlo.py`

### Required changes

Refactor `_structural_dynamics_panel(...)` into an arm-local paired W/R panel.

Remove active report use of:

- `_last_competitor_crossing`;
- `_nstar_graph_image`;
- `_nstar_panel`;
- `_nstar_diagnostics_panel`;
- interpolated vertical crossing markers;
- the separate `N-star diagnostics` section.

The arm report should have one section named:

**Pairwise winner and reversal dynamics**

It must:

- retain the current classifier tabs;
- retain sample-size/prefix tabs;
- use one arm-local display subset per cardinality as before;
- state that persisted numerical structures use all subsets;
- show `W` and `R` together;
- explain the tie propagation rule concisely;
- state the exact existence condition for an `R` cell.

Renumber later sections after removing the obsolete diagnostics section. Do not add a replacement section merely to preserve the old count of 17.

### Legacy generic report

`generate_monte_carlo_report(...)` is marked as backward-compatible Sprint-1 reporting. Do not rewrite or delete it in this block unless it is used by the current three dataset scenarios or a named target test fails because of it.

### Targeted tests

```bash
pytest -q \
  tests/test_occupancy_monte_carlo_reports.py \
  tests/test_air_quality_monte_carlo_reports.py \
  tests/test_support2_monte_carlo_reports.py
```

### Acceptance criteria

- current structured arm reports contain paired W/R dynamics;
- no active interpolated N-star diagnostics remain;
- all three targeted arm-report tests pass;
- no simulation is executed.

### Commit

```text
refactor: replace arm N-star diagnostics with W-R dynamics
```

### Stop for review

Mandatory.

---

## Block 6 — README, package metadata, publication text, and proposal LaTeX
**Complexity:** medium  
**Purpose:** align the current repository-facing scientific description without rewriting history or unrelated documentation.

### Primary files

- `README.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `pyproject.toml`
- `src/coinfosim/__init__.py`
- `src/coinfosim/cli/app.py`
- `src/coinfosim/publish/site.py`
- `coinfosim_research_proposal_v4.tex`

### Bounded terminology search

Run exactly:

```bash
rg -n "cooperative advantage|cooperative advantage threshold|N-star|N\*|nstar|structural fidelity|timing similarity|crossing Jaccard" \
  README.md CHANGELOG.md CITATION.cff pyproject.toml \
  src/coinfosim/__init__.py src/coinfosim/cli/app.py src/coinfosim/publish/site.py \
  coinfosim_research_proposal_v4.tex
```

### Required documentation changes

#### README

Update the active scientific framing to:

- predictive cooperation;
- predictive cooperation profile;
- real versus synthetic training-condition comparison;
- same fixed real test set;
- `W` and `R`;
- separate reversal existence and sample-size metrics;
- no composite metric;
- explicit statement that an `R` cell exists only after at least one valid winner reversal.

Replace the active mathematical definition of first-threshold `N*`.

Do not remove practical installation, CLI, dataset, licensing, or cache instructions.

#### CHANGELOG

Add one new `Unreleased` entry summarizing:

- effective winner tie propagation;
- unordered pairwise reversal matrix;
- separate reversal metrics;
- HTML/report schema updates;
- removal of the composite metric from active reporting.

Do not rewrite historical release notes.

#### CITATION.cff

Update the title/keywords only as needed to reflect predictive cooperation profiles. Preserve:

- author;
- affiliation;
- URL;
- license;
- release date;
- DOI placeholder.

Do not bump version unless the owner explicitly requests a release change.

#### pyproject.toml

Update only the project description/keywords needed for the new scientific framing.

Do not:

- change package version;
- alter dependencies;
- reformat the complete file.

#### CLI/package/site descriptions

Update stale one-line taglines and current scientific descriptions only. Do not change CLI behavior or published-site layout.

#### `coinfosim_research_proposal_v4.tex`

This is the only known LaTeX source currently tracked in the repository.

Update:

- title and PDF metadata;
- abstract;
- central research question;
- modeling terminology (`attribute` formal, `information channel` conceptual);
- predictive cooperation profile definition;
- estimated mean test loss wording;
- `W` and `R` definitions;
- tie propagation;
- existence rule;
- two reversal metrics;
- removal of the multiplicative metric;
- research questions and conclusions that still treat `N*` as a first threshold.

Preserve the proposal's overall structure unless a local edit is necessary for logical consistency.

### Validation

No Python test is required solely for prose.

Run bounded checks:

```bash
python -m compileall -q src/coinfosim
```

If a LaTeX engine is already installed, compile the proposal once:

```bash
pdflatex -interaction=nonstopmode -halt-on-error coinfosim_research_proposal_v4.tex
```

Do not install TeX packages. If unavailable, report `pdflatex not available` and continue.

### Acceptance criteria

- current repository-facing descriptions use the new terminology;
- README includes the exact reversal existence rule;
- no package version/dependency change;
- historical changelog entries remain intact;
- proposal compiles when the local toolchain supports it;
- no broad docs rewrite.

### Commit

```text
docs: align repository with predictive cooperation profiles
```

### Stop for review

Mandatory.

---

## Block 7 — Conditional final report and presentation LaTeX
**Complexity:** medium, conditional  
**Purpose:** update the final academic report and presentation only when their actual source paths are available.

### Known limitation

The remote repository inspection found only:

```text
coinfosim_research_proposal_v4.tex
```

The LaTeX sources for the final long report and the presentation were not found in the remote repository.

### Bounded local discovery

Run once:

```bash
find . -maxdepth 4 -type f \( -name "*.tex" -o -name "*.bib" \) -print | sort
```

Do not inspect unrelated binary PDFs as substitute source code.

### Branch A — sources found

Before editing, report the exact candidate paths and wait for owner confirmation that they are:

- the final report source;
- the presentation source.

After confirmation, update only those files and their directly imported local `.tex`/`.bib` dependencies.

Required academic updates:

- title: predictive cooperation profile preservation;
- profile versus pattern distinction;
- attribute versus information-channel distinction;
- estimated predictive superiority based on estimated mean test loss;
- formal `W` trajectory and tie rule;
- exact `R` existence condition;
- triangular unordered-pair `R`;
- reversal existence agreement;
- mean log2 reversal distance;
- reversal sample-size similarity;
- removal of multiplicative similarity;
- replacement of all affected tables, figure captions, results, discussion, limitations, and conclusions;
- no reuse of old composite values as if they were new metrics.

Compile each artifact once with the repository's existing build command. Do not install dependencies.

### Branch B — sources absent

Do not fabricate LaTeX sources and do not edit generated PDFs.

Report the blocker:

```text
Final report/presentation LaTeX sources are not present in the repository.
Exact source paths or files are required before Block 7 can continue.
```

Do not commit an empty or speculative change.

### Targeted validation

Only the exact build commands for the confirmed sources.

### Acceptance criteria

- either confirmed sources are updated and compile;
- or the missing-source blocker is reported precisely.

### Commit, only when sources are edited

```text
docs: update academic report and presentation terminology
```

### Stop for review

Mandatory.

---

## Block 8 — Economical integration validation and report regeneration
**Complexity:** medium  
**Purpose:** validate the complete refactor once, without a full suite or Monte Carlo rerun.

### 8.1 Focused test set

Run once:

```bash
pytest -q \
  tests/test_structural_metrics.py \
  tests/test_occupancy_scenario_report.py \
  tests/test_air_quality_scenario_report.py \
  tests/test_support2_scenario_report.py \
  tests/test_occupancy_monte_carlo_reports.py \
  tests/test_air_quality_monte_carlo_reports.py \
  tests/test_support2_monte_carlo_reports.py \
  tests/test_occupancy_run_tracking.py
```

Run `tests/test_parallel_scientific_equivalence.py` only when Block 2 changed assertions or payloads used by that file and it was not already validated after the final modifications:

```bash
pytest -q tests/test_parallel_scientific_equivalence.py
```

Do not run `pytest` without explicit test paths.

### 8.2 Targeted stale-name audit

Run:

```bash
rg -n "nstar_similarity|timing_similarity|crossing_jaccard|Progressive N-star|Interpolated N-star|N-star diagnostics" \
  src/coinfosim/results/structural.py \
  src/coinfosim/runs/report_data.py \
  src/coinfosim/reports/structural_visualization.py \
  src/coinfosim/reports/occupancy_scenario.py \
  src/coinfosim/reports/dataset_anchored_scenario.py \
  src/coinfosim/reports/monte_carlo.py \
  README.md
```

Expected result: no active occurrence.

Do not require zero matches in:

- historical changelog entries;
- `docs/tasks/`;
- explicitly retained legacy APIs;
- generated output directories.

### 8.3 Persisted-data report regeneration

Check only:

```bash
test -f output/reports/scenario_runs.json && echo "registry available"
coinfosim runs scenarios
```

When a completed **Occupancy smoke** scenario with persisted result payloads exists locally, regenerate exactly one representative hierarchy:

```bash
coinfosim scenario regenerate occupancy \
  --run-id <existing-occupancy-smoke-run-id> \
  --output-dir output/reports_refactor_validation
```

Rules:

- use an existing run ID from the registry;
- do not invent an ID;
- do not execute `coinfosim scenario run`;
- do not download a dataset;
- if no valid persisted run exists, skip regeneration and report that fact.

Audit the regenerated HTML with targeted string checks:

```bash
rg -n "Reversal existence agreement|Reversal sample-size similarity|Winner matrix|Reversal matrix" \
  output/reports_refactor_validation
```

```bash
rg -n "Progressive N-star|Timing similarity|Interpolated N-star|N-star diagnostics" \
  output/reports_refactor_validation
```

The second command should return no active report match.

Do not regenerate all scenarios until the owner separately approves publication/regeneration.

### 8.4 Final diff audit

```bash
git status --short
git diff --stat main...HEAD
git log --oneline --decorate main..HEAD
```

### Acceptance criteria

- focused test set passes;
- no active composite metric;
- no active old terminology in the bounded implementation paths;
- one existing report hierarchy regenerates correctly when persisted data exists;
- no Monte Carlo simulation was run;
- diff contains only approved scope.

### Commit

```text
test: validate predictive cooperation profile refactor
```

Create this commit only when Block 8 changes tracked fixtures, documentation, or test artifacts. Otherwise report validation without an empty commit.

### Stop for final review

Mandatory. Do not merge.

---

# 5. Final owner-approved merge procedure

After explicit owner approval:

```bash
git switch main
git pull --ff-only origin main
git merge --no-ff feature/predictive-cooperation-profile-refactor
```

Run only the focused Block 8 test command once after merge.

Do not run a full suite or any scenario simulation unless the owner separately requests it.

Report:

- merge commit SHA;
- final focused test result;
- whether academic report/presentation sources remain pending;
- whether full report publication/regeneration remains pending.

---

# 6. Non-goals

This refactor must not:

- change datasets, targets, splits, selected attributes, or classifier configurations;
- change Monte Carlo sampling or stopping rules;
- change the fixed real evaluation-set protocol;
- rerun full scientific experiments;
- add acquisition costs or synergy modeling;
- introduce mutual information or Fisher information;
- redesign the CLI;
- replace the tab UI with dropdown selectors;
- upgrade dependencies;
- bump the package version;
- rewrite historical changelog entries;
- refactor unrelated legacy modules;
- fabricate missing academic LaTeX sources;
- infer new scientific results from old composite values.

---

# 7. Completion definition

The implementation is complete when:

1. `W` has one formal, tested carry-forward tie rule;
2. a reversal requires a defined winner to be replaced by the other winner;
3. first winner initialization does not count as reversal;
4. `R_ij(n_t)` exists if and only if at least one valid reversal occurred;
5. `R` stores each unordered pair once;
6. the direction/current winner is recovered from `W`;
7. reversal existence agreement and reversal sample-size similarity are separate;
8. mean log2 reversal distance is persisted for interpretation;
9. no multiplicative reversal metric is calculated or reported;
10. current scenario and arm HTML reports show paired W/R dynamics;
11. report data uses schema version 2;
12. README, package descriptions, citation metadata, publication text, and tracked proposal LaTeX use the new terminology;
13. final report/presentation sources are updated when available, otherwise precisely identified as missing;
14. all focused tests pass;
15. no Monte Carlo experiment is rerun during agent implementation.
