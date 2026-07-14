# CoInfoSim Implementation Tasks: New Structural Metrics and Tabbed Report Integration

## 0. Status, authority, and baseline

This document is the authoritative implementation plan for the next CoInfoSim reporting work.
It supersedes every earlier version of `tasks_report_new_metrics.md` and every earlier prompt derived from that file.

Repository:

```text
https://github.com/paulorenatoaz/coinfosim
```

This specification was rebuilt after re-inspecting the current remote `main` branch at:

```text
62f5193b9298d9861c29d00a5490519bd5255733
report: Add occupancy Monte Carlo reports and related improvements
```

The coding agent must begin by confirming that its local `HEAD` is this commit or a descendant of it:

```bash
git fetch origin
git checkout main
git pull --ff-only
git rev-parse HEAD
git log -1 --oneline
```

Do not reset or discard uncommitted user work. If `HEAD` is newer than the baseline above, inspect only the newer changes that touch files listed in this document before starting Block 1.

The current remote is now the source of truth. Do not reconstruct implementation strategy from older local files, old chat context, or the earlier new-metrics task document.

---

## 1. Objective

Extend CoInfoSim with three separate structural-fidelity metrics:

1. Ranking Structural Fidelity, \(\rho_{\mathrm{rank}}(N)\);
2. Winner Agreement, \(A_W(N)\);
3. Progressive \(N^*\)-Similarity, \(S_{N^*}(\leq N)\).

Also add:

- directed winner matrices;
- progressive directed \(N^*(\leq N)\) matrices;
- scenario-level comparative visualizations;
- arm-level structural-dynamics visualizations;
- additive report-ready persistence;
- no-Monte-Carlo report regeneration backfill.

The three metrics must remain separate. Do not create a composite index.

The implementation must be completed in sequential work blocks. After each block, stop and wait for user review before beginning the next block.

---

## 2. Non-negotiable constraints

### 2.1 Preserve the current implementation

The current remote already contains a substantial report redesign. Preserve it.

Do not remove, replace, or regress:

- the structured Occupancy arm reports;
- the sticky channel legend;
- compact subset notation;
- the current tab appearance and interaction model;
- nested-cardinality loss curves;
- ranking-by-sample-size tabs;
- GMM class/component tabs;
- current N-star diagnostics;
- the current scenario N-star availability logic;
- CSV table exports;
- existing report-ready keys;
- existing threshold functions;
- Monte Carlo behavior.

All new work is additive except where this document explicitly requires compacting an existing scenario-report layout.

### 2.2 Do not rerun Monte Carlo

The complete numerical results are already persisted in `result_data_*.json.gz` files.
All new metrics are post-processing operations over those persisted results.

Never run any of these for this project:

```bash
python scripts/run_occupancy_scenario.py --mode smoke
python scripts/run_occupancy_scenario.py --mode fast
python scripts/run_occupancy_scenario.py --mode full
python scripts/run_occupancy_scenario.py --mode strict
```

Do not invoke `CooperativeMonteCarloSimulator.run()` merely to validate report changes.
Use handcrafted fixtures for numerical unit tests and the persisted full run for integration validation.

### 2.3 Use tabs, not select boxes

Every selector introduced by this work must use the same tab-button layout already used by the current structured arm reports.

Do not use:

- HTML `<select>` controls;
- dropdowns;
- automatic carousels;
- external JavaScript frameworks;
- CDN dependencies.

Use static nested tab groups with the current visual language:

```text
.tab-bar
.tab-btn
.tab-btn.active
.tab-panel
```

Only one panel in each tab group should be visible at a time. Nested tab groups must remain independent through unique group identifiers.

### 2.4 Conserve coding-agent credits

Testing must be narrow and purposeful.

Do not run:

```bash
pytest
pytest tests
pytest --cov
```

Do not run an entire existing report test file merely because one assertion changed.
Many existing report tests construct tiny simulations and fit models; they are not the preferred integration path for this task.

For each block:

1. run only the small pure/component tests assigned to the block;
2. regenerate the existing persisted full reports once;
3. inspect the generated HTML and relevant JSON artifacts;
4. stop for user review.

The principal integration test is successful report regeneration from persisted full results.

### 2.5 Generic numerical implementation

Generic structural-metric code must not hardcode:

- Occupancy;
- five channels;
- 31 subsets;
- three classifiers;
- three arms;
- specific arm identifiers;
- the power-of-two sample grid.

Occupancy-specific files may provide labels and narrative text, but all formulas and cross-arm structural logic must remain generic.

---

## 3. Exact persisted full run to use

Use this existing completed scenario:

```text
scenario_run_id = 2
mode = full
```

Scenario directory:

```text
output/reports/scenarios/000002_occupancy_baseline_full/
```

Scenario report:

```text
output/reports/scenarios/000002_occupancy_baseline_full/
occupancy_baseline_scenario_report_full_000002.html
```

Referenced full simulation runs:

```text
simulation_run_id = 6  Real → Real
simulation_run_id = 7  Single Gaussian → Real
simulation_run_id = 8  GMM → Real
```

Simulation directories and arm reports:

```text
output/reports/simulations/000006_occupancy_real_data_full/
occupancy_real_data_monte_carlo_report_full_000006.html

output/reports/simulations/000007_occupancy_single_gaussian_to_real_full/
occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html

output/reports/simulations/000008_occupancy_gmm_to_real_full/
occupancy_gmm_to_real_monte_carlo_report_full_000008.html
```

Use the current regeneration command:

```bash
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
```

This command must regenerate the scenario report and all three arm reports without allocating new run IDs and without executing Monte Carlo.

Before the first block, the agent may verify the run with:

```bash
python scripts/run_occupancy_scenario.py --list-scenario-runs
python scripts/run_occupancy_scenario.py --list-simulation-runs
```

Do not search for another run unless run 2 is missing locally.

---

## 4. Current remote architecture: what already exists

The coding agent should start from the files and functions below. Do not wander through unrelated repository areas.

## 4.1 Persisted numerical source of truth

Files:

```text
src/coinfosim/results/accumulator.py
src/coinfosim/simulation/monte_carlo.py
src/coinfosim/results/persistence.py
```

Current responsibilities:

- `LossAccumulator` stores ordered replication losses by `(n_per_class, subset, classifier)`;
- `SimulationResult` exposes the sample-size grid, ordered subsets, classifiers, accumulator, stopping information, metadata, and runtime;
- `result_data_*.json.gz` serializes the complete accumulator and can reconstruct the full `SimulationResult`.

Consequence:

- new metrics must be computed after simulation;
- do not modify the simulation loop;
- do not add derived metrics to the gzip payload;
- do not rerun the experiment.

## 4.2 Existing analysis semantics to preserve

File:

```text
src/coinfosim/results/analysis.py
```

Existing public behavior includes:

```text
best_subset
best_subset_rankings
cooperative_threshold
cooperative_threshold_interpolated
standard_threshold_comparisons
```

Important distinction:

- existing threshold functions find selected first crossings and may interpolate;
- existing standard comparisons include Occupancy-specific comparisons;
- the new progressive directed matrix is an all-pairs, observed-grid, latest-crossing structure.

Do not alter old threshold semantics to implement the new metric.
Create a separate structural-analysis module.

## 4.3 Current structured arm reports

Files:

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_monte_carlo.py
src/coinfosim/reports/report_tables.py
```

The current remote already implements a structured Occupancy arm report through:

```python
generate_structured_monte_carlo_report(...)
```

Current arm-report section order:

```text
1. Arm summary
2. Scientific role of this arm
3. Dataset provenance summary
4. Run configuration
5. Training/evaluation protocol
6. Arm-specific data and model description
7. Reproducibility controls
8. Monte Carlo stopping rule
9. Monte Carlo precision diagnostics
10. Loss curves
11. Best subset comparison
12. Subset ranking by sample size
13. N-star diagnostics
14. Robustness notes
15. Validity and limitations for this arm
16. Technical appendix / exported tables
```

Current features that are already implemented and must not be redone:

- classifier tabs with Linear SVM first;
- best-by-cardinality and nested-cardinality loss views;
- sample-size tabs for complete subset rankings;
- reference-cardinality tabs for N-star diagnostics;
- GMM class tabs and component tabs;
- yellow full-reference highlighting;
- sticky channel legend;
- full loss table and CSV exports.

Relevant current helpers in `src/coinfosim/reports/monte_carlo.py`:

```text
_tab_group
_TAB_JS
_STRUCTURED_CSS
_loss_curves_panel
_ranking_by_sample_size_panel
_nstar_diagnostics_panel
generate_structured_monte_carlo_report
```

Do not replace these reports with the older legacy `generate_monte_carlo_report(...)` path.
The legacy generator should remain backward compatible.

## 4.4 Current scenario report

File:

```text
src/coinfosim/reports/occupancy_scenario.py
```

Current scenario section order:

```text
1. Scientific question
2. Scenario summary
3. Experimental protocol
4. Data visualization
5. Best subset comparison at largest N
6. Top-ranked subsets
7. N-star availability
8. Interpretation notes
```

Current implementation functions that matter:

```text
_carousel_html
_best_comparison_html
_top_ranked_html
_nstar_section
_STYLE
generate_occupancy_scenario_report
```

Current problem:

- Section 4 uses an automatic carousel with bespoke controls;
- Section 5 emits three arm graphs sequentially;
- Section 6 emits classifier × arm tables and graphs sequentially;
- Section 7 emits classifier × reference-cardinality × arm tables and graphs sequentially;
- the scenario report does not use the tab layout already used by the arm reports.

Block 1 must correct this before new metrics are added.

## 4.5 Current run/report persistence

Files:

```text
src/coinfosim/runs/report_data.py
src/coinfosim/runs/registry.py
scripts/run_occupancy_scenario.py
```

Current behavior:

- `simulation_report_data(...)` embeds report-ready summary, rankings, and threshold comparisons in `simulation.json`;
- `scenario_report_data(...)` currently accepts three positional Occupancy results and nests per-arm report data;
- `regenerate_from_scenario_run(...)` loads persisted gzip results and regenerates HTML without Monte Carlo;
- registry writes are already atomic at the individual-file level;
- current regeneration does not backfill the proposed structural fields into standalone JSON and registry records.

The implementation must extend this behavior additively.

## 4.6 Current wrapper behavior that must remain

`scripts/run_occupancy_scenario.py` and `src/coinfosim/reports/occupancy_monte_carlo.py` already pass the Real → Real result into synthetic-arm report generation through:

```text
nstar_selection_result=real_result
```

This preserves common Real-selected N-star competitors in the current arm reports.
Do not remove or reinterpret this behavior.

---

## 5. Required architecture

## 5.1 Shared tab component

Create:

```text
src/coinfosim/reports/html_tabs.py
```

The module must expose a small shared API, for example:

```python
tab_group(group_id, tabs, default_key)
TAB_CSS
TAB_JS
```

Where each tab is conceptually:

```python
(key, visible_label, panel_html)
```

Requirements:

- reproduce the current arm-report tab appearance;
- use `.tab-bar`, `.tab-btn`, `.tab-btn.active`, and `.tab-panel`;
- support arbitrarily nested groups;
- use deterministic, unique group IDs;
- HTML-escape group IDs, keys, and visible labels;
- leave panel HTML unescaped because it is already rendered HTML;
- show exactly one panel per group by default;
- work in locally opened static HTML;
- include no external dependency;
- include shared CSS and JavaScript only once per document.

Migration strategy:

1. move or copy the current tab implementation from `monte_carlo.py` into `html_tabs.py`;
2. make `monte_carlo.py` import it while preserving private aliases if useful:

```python
from coinfosim.reports.html_tabs import (
    TAB_CSS as _TAB_CSS,
    TAB_JS as _TAB_JS,
    tab_group as _tab_group,
)
```

3. ensure generated arm-report HTML remains visually and functionally unchanged;
4. use the same component in `occupancy_scenario.py`.

Do not create a second visual design for scenario tabs.

## 5.2 Generic structural-analysis module

Create:

```text
src/coinfosim/results/structural.py
```

This module owns all numerical definitions and validation.
It must not import Occupancy-specific report modules.

Recommended public responsibilities:

```python
validate_structural_compatibility(...)
loss_vector(...)
rank_vector(...)
ranking_fidelity_series(...)
winner_matrix(...)
winner_agreement_series(...)
directed_crossing_events(...)
progressive_directed_nstar(...)
progressive_nstar_similarity(...)
simulation_structural_dynamics(...)
scenario_structural_fidelity(...)
select_display_subsets_by_cardinality(...)
```

The exact internal decomposition may differ, but formula logic, serialization logic, and report formatting must remain separate.

## 5.3 Structural visualization module

Create:

```text
src/coinfosim/reports/structural_visualization.py
```

This module should provide reusable Matplotlib figure builders for:

- structural metric curves;
- directed winner matrices;
- progressive directed N-star matrices.

Recommended design:

```python
metric_series_figure(...)
winner_matrix_figure(...)
progressive_nstar_matrix_figure(...)
save_figure(...)
figure_to_data_uri(...)
```

Scenario reports should save PNG files and register them through `graphs_out`.
Structured arm reports may embed data URIs, matching their current self-contained design.

Do not add a plotting dependency.

## 5.4 Generic arm mapping

All new cross-arm analysis must accept an ordered mapping:

```python
arm_results: Mapping[str, SimulationResult]
reference_arm: str
arm_labels: Mapping[str, str]
```

The generic layer must not accept exactly three positional arms.

Compatibility strategy:

- preserve the public signature of `generate_occupancy_scenario_report(...)` unless changing it is demonstrably necessary;
- inside the Occupancy wrapper, immediately normalize the three current arguments into an ordered mapping;
- add a generic structural-report-data helper in `src/coinfosim/runs/report_data.py`;
- keep the existing `scenario_report_data(real_result, gaussian_result, gmm_result, ...)` wrapper and have it call the generic helper.

Do not perform a broad scenario API rewrite merely for style.

## 5.5 Lightweight tests

Create:

```text
tests/test_report_tabs.py
tests/test_structural_metrics.py
```

These tests must use:

- pure HTML strings;
- handcrafted accumulators or minimal fake result objects;
- persisted full results when an integration fixture is needed.

They must not call `CooperativeMonteCarloSimulator.run()`.

Existing test files may be updated for future compatibility, but do not run their complete contents during the staged work unless a block explicitly authorizes a specific node ID.

---

## 6. Mathematical definitions and implementation rules

Let:

- \(\mathcal S=\{S_1,\ldots,S_M\}\) be the common ordered set of all non-empty channel subsets;
- \(\mathcal N=\{N_1,\ldots,N_T\}\) be the common strictly increasing sample-size grid;
- \(a_0\) be the reference arm;
- \(a\) be a comparison arm;
- \(c\) be a classifier.

Let:

\[
L^{(a,c)}(S_i,N_t)
\]

be the empirical mean test loss already stored in the accumulator.

Every metric is computed separately for each classifier and each arm relative to the reference arm.

## 6.1 Compatibility validation

Before any cross-arm calculation, validate that every arm has:

- a non-empty sample-size grid;
- positive integer sample sizes;
- unique sample sizes;
- strictly increasing sample sizes;
- the same sample-size values as the reference arm;
- the same classifier identities;
- the same subset identities;
- no missing required accumulator cell;
- finite mean loss in every required cell.

Subset order may differ between arms only if it can be normalized safely through tuple identity.
A missing or extra subset is an error.
Do not silently intersect incompatible arms.

Return clear errors that identify the arm and mismatch.

The current `MonteCarloConfig` does not guarantee every one of these cross-arm invariants, so validation belongs in the structural layer.

## 6.2 Ranking Structural Fidelity

For each arm, classifier, and sample size, create a mean-loss vector in the same canonical subset order.
Convert losses to ranks using average-rank tie handling:

```python
scipy.stats.rankdata(losses, method="average")
```

Do not use standard error, subset label, or subset order to break equal-loss ties.

Define:

\[
\rho_{\mathrm{rank}}^{(a,c)}(N_t)
=
\rho_{\mathrm{Spearman}}
\left(
R^{(a_0,c)}(N_t),
R^{(a,c)}(N_t)
\right).
\]

Requirements:

- use all non-empty subsets;
- compute from \(N_1\);
- self-comparison is stored as exactly `1.0`;
- if a non-self comparison has a constant rank vector in either arm, return `None` with status `constant_ranking`;
- never serialize SciPy `NaN`;
- persist the number of subsets used.

Recommended row:

```json
{
  "classifier": "linear_svm",
  "arm": "single_gaussian_to_real",
  "n_per_class": 32,
  "rho_rank": 0.91,
  "n_subsets": 31,
  "status": "ok"
}
```

Allowed core statuses:

```text
ok
self
constant_ranking
```

A self-comparison with a constant loss vector remains `1.0`, but may preserve an auxiliary diagnostic such as `constant_ranking=true`.

## 6.3 Directed winner matrix

For one arm, classifier, and sample size, define:

```text
+1  row subset has lower mean loss than column subset
-1  row subset has higher mean loss than column subset
 0  exact mean-loss tie
NA  diagonal or unavailable cell
```

Use exact floating-point equality for tie detection in the first implementation.
Do not add an undocumented tolerance.

The matrix is antisymmetric away from ties:

\[
W_{ij}=-W_{ji}.
\]

The diagonal is unavailable, not zero.

## 6.4 Winner Agreement

Winner Agreement compares unordered subset pairs only.
For every pair \(\{S_i,S_j\}\), use one orientation, for example `i < j`.

Skip a pair if either arm has an exact tie for that pair.
On all remaining pairs:

\[
A_W^{(a,c)}(N_t)
=
\frac{
\#\{\text{valid pairs with the same winner}\}
}{
\#\{\text{valid pairs}\}
}.
\]

Requirements:

- use all non-empty subsets;
- compute from \(N_1\);
- exclude diagonal cells;
- exclude mirrored duplicates;
- self-comparison is exactly `1.0` when at least one valid pair exists;
- if no valid pair exists, return `None` with status `no_valid_pairs`;
- persist total, valid, matching, and skipped-tie counts.

Recommended row:

```json
{
  "classifier": "linear_svm",
  "arm": "gmm_to_real",
  "n_per_class": 64,
  "winner_agreement": 0.87,
  "n_pairs_total": 465,
  "n_pairs_valid": 463,
  "n_pairs_matching": 403,
  "n_pairs_skipped_tie": 2,
  "status": "ok"
}
```

## 6.5 Directed crossing event

For each ordered subset pair \((S_i,S_j)\), define:

\[
\Delta_{ij}(N_t)=L(S_i,N_t)-L(S_j,N_t).
\]

A directed event \(S_i\rightarrow S_j\) occurs at observed grid point \(N_k\), for \(k\ge2\), when the row subset becomes strictly better than the column subset:

\[
\Delta_{ij}(N_{k-1})\ge0
\quad\text{and}\quad
\Delta_{ij}(N_k)<0.
\]

Interpretation:

```text
S_i → S_j means row subset S_i has just become the lower-loss winner over S_j.
```

Rules:

- use observed grid points only;
- do not interpolate;
- do not create a crossing at \(N_1\);
- a row subset already winning at \(N_1\) has not crossed;
- ties at the previous grid point are eligible because the previous condition is `>= 0`;
- multiple alternations are valid;
- opposite directions are distinct events.

## 6.6 Progressive directed \(N^*(\leq N)\)

For every prefix ending at \(N_t\), define:

\[
N^*_{ij}(\leq N_t)
=
\max\{N_k\le N_t:\;S_i\rightarrow S_j\text{ occurred at }N_k\},
\]

or missing if no event in that direction has occurred.

Requirements:

- the first progressive matrix is defined at \(N_2\), not \(N_1\);
- every stored value must satisfy `n_star <= n_prefix`;
- a later event in the same direction replaces the earlier displayed progressive value;
- opposite-direction events do not erase this direction's historical latest event;
- diagonal cells are unavailable;
- missing cells are `None`/JSON `null`, never `0`, `N1`, or another sentinel;
- persist crossing events sparsely and reconstruct progressive matrices from events.

Recommended sparse event:

```json
{
  "row": 7,
  "col": 12,
  "n_crossing": 64
}
```

The subset indices refer to one persisted canonical subset catalog.

## 6.7 Progressive \(N^*\)-Similarity

For each prefix \(N_t\), \(t\ge2\), define the directed non-missing crossing-cell sets:

\[
C^a_{\le t}
=
\{(i,j):N^{*,a}_{ij}(\le N_t)\text{ is available}\}.
\]

Crossing-set similarity:

\[
J_C(\le N_t)
=
\frac{|C^{a}_{\le t}\cap C^{a_0}_{\le t}|}
{|C^{a}_{\le t}\cup C^{a_0}_{\le t}|}.
\]

For shared cells \(C^\cap_{\le t}\), define mean log-grid timing distance:

\[
d_{\mathrm{timing}}(\le N_t)
=
\frac{1}{|C^\cap_{\le t}|}
\sum_{(i,j)\in C^\cap_{\le t}}
\left|
\log_2 N^{*,a}_{ij}(\le N_t)
-
\log_2 N^{*,a_0}_{ij}(\le N_t)
\right|.
\]

Normalize by the current prefix span:

\[
\widetilde d_{\mathrm{timing}}(\le N_t)
=
\frac{d_{\mathrm{timing}}(\le N_t)}
{\log_2N_t-\log_2N_1}.
\]

Timing similarity:

\[
S_{\mathrm{timing}}(\le N_t)
=
\min\left(1,\max\left(0,1-\widetilde d_{\mathrm{timing}}(\le N_t)\right)\right).
\]

Final progressive N-star similarity:

\[
S_{N^*}(\le N_t)
=
J_C(\le N_t)\,S_{\mathrm{timing}}(\le N_t).
\]

Required edge cases:

### First grid point

At \(N_1\):

```text
nstar_similarity = None
status = unavailable_first_prefix
```

Do not plot it as zero.

### Neither arm has any crossing cell

```text
nstar_similarity = 1.0
crossing_jaccard = 1.0
timing_similarity = 1.0
status = no_crossings_in_either
```

The report must display the status because a score of 1.0 alone may be misleading.

### At least one arm has crossings, but none are shared

```text
nstar_similarity = 0.0
crossing_jaccard = 0.0
timing_similarity = None
status = no_shared_crossings
```

### Shared cells exist

Compute the full formula:

```text
status = ok
```

### Self-comparison

Store `1.0` where defined.
Preserve the informative crossing status:

- `no_crossings_in_either` if both self structures are empty;
- `ok` if crossings exist.

Recommended row:

```json
{
  "classifier": "linear_svm",
  "arm": "single_gaussian_to_real",
  "n_prefix": 128,
  "n_reference_crossings": 42,
  "n_arm_crossings": 39,
  "n_shared_crossings": 35,
  "n_union_crossings": 46,
  "crossing_jaccard": 0.7608695652,
  "timing_similarity": 0.94,
  "nstar_similarity": 0.7152173913,
  "status": "ok"
}
```

## 6.8 Numerical data versus display reduction

All three metrics must use all non-empty subsets.

Matrix plots may be reduced for readability, but the reduction must not affect metric calculations or persisted full structural data.

### Scenario matrix display selection

For each classifier:

1. use the reference arm at \(N_{\max}\);
2. group subsets by cardinality;
3. select the lowest-mean-loss subset in each cardinality;
4. break an exact display-selection tie lexicographically by subset tuple;
5. freeze these subset identities across every arm, every sample size, and every prefix.

This produces one fixed reference-selected subset per available cardinality.

### Arm matrix display selection

For an individual arm's own `Structural dynamics` section:

1. use that arm at \(N_{\max}\);
2. select one best subset per cardinality;
3. use lexicographic tie-breaking;
4. freeze the selected identities across that arm's sample sizes and prefixes.

Label this explicitly as an arm-local display selection.

---

## 7. Report requirements

## 7.1 Scenario report: final section order

After all blocks, `src/coinfosim/reports/occupancy_scenario.py` must render:

```text
1. Scientific question
2. Scenario summary
3. Experimental protocol
4. Data visualization
5. Best subset comparison at largest N
6. Top-ranked subsets
7. Structural fidelity metrics
8. N-star availability
9. Interpretation notes
```

The existing scientific content must remain.
Only presentation compaction and additive structural content are authorized.

## 7.2 Scenario report: tab hierarchies

All controls below must be tab buttons, not dropdowns.

### Section 4: Data visualization

Replace the automatic carousel with nested tabs:

```text
outer tabs: training source/model
  Real data
  Single Gaussian
  GMM

inner tabs: projection dimension
  1D
  2D
  3D
```

Defaults:

```text
Real data
1D
```

Requirements:

- preserve all nine existing images;
- remove autoplay and the Play/Pause toggle;
- show one image at a time;
- preserve metadata above the tabs;
- preserve offline behavior.

### Section 5: Best subset comparison graphs

Keep the current cross-arm summary table visible.
Compact the current three arm graphs with tabs:

```text
arm tabs:
  Real → Real
  Single Gaussian → Real
  GMM → Real
```

Default:

```text
Real → Real
```

Do not change the graph meaning in Block 1.
Each graph continues to show the current best subset per classifier for that arm.

### Section 6: Top-ranked subsets

Use nested tabs:

```text
outer tabs: classifier
  Linear SVM
  Logistic Regression
  Gaussian Naive Bayes

inner tabs: arm
  Real → Real
  Single Gaussian → Real
  GMM → Real
```

Each final panel contains the current table and its associated graph.

Defaults:

```text
Linear SVM
Real → Real
```

### Existing N-star availability section

Use three tab levels:

```text
outer tabs: classifier
middle tabs: reference cardinality
inner tabs: arm
```

For the current five-channel scenario, the middle tabs are:

```text
Best 1-channel reference
Best 2-channel reference
Best 3-channel reference
Best 4-channel reference
Best 5-channel reference
```

Each final panel contains:

- the reference-subset annotation;
- the existing N-star table;
- the associated graph.

Defaults:

```text
Linear SVM
Best 1-channel reference
Real → Real
```

Do not change:

- competitor selection;
- Real-based reference selection;
- multiple-crossing detection;
- exclusion of left-censored values from the reported N-star;
- last-genuine-crossing reporting;
- interpolated marker behavior;
- table columns.

Only the layout changes in Block 1.

## 7.3 New scenario structural-fidelity section

Section 7 must contain:

1. a concise explanation of the three separate metrics;
2. an explicit statement that no composite index is reported;
3. a final summary table at \(N_{\max}\);
4. tabbed structural metric curves;
5. tabbed winner matrices;
6. tabbed progressive N-star matrices;
7. visible status explanations.

### Summary table

Use one row per classifier and non-reference comparison arm.
Do not visually emphasize trivial self-comparison rows.

Recommended columns:

```text
Classifier
Comparison arm
rho_rank(Nmax)
Ranking status
A_W(Nmax)
Valid winner pairs
Skipped tie pairs
Winner status
S_N*(<=Nmax)
Reference crossings
Arm crossings
Shared crossings
Crossing Jaccard
Timing similarity
N-star status
```

### Metric curves

Tabs:

```text
outer tabs: classifier
inner tabs: metric
  Ranking fidelity
  Winner agreement
  Progressive N-star similarity
```

Each panel plots all non-reference comparison arms as separate lines.

Axis rules:

```text
x-axis: evaluated n_per_class, logarithmic base 2
rho_rank y-axis: [-1, 1]
winner agreement y-axis: [0, 1]
N-star similarity y-axis: [0, 1]
```

The N-star similarity curve begins at \(N_2\).
Do not insert a zero point at \(N_1\).

### Winner matrices

Tabs:

```text
outer tabs: classifier
middle tabs: arm
inner tabs: n_per_class
```

Include the reference arm because its matrix is needed for visual comparison.
Use fixed reference-selected display subsets by cardinality.

### Progressive N-star matrices

Tabs:

```text
outer tabs: classifier
middle tabs: arm
inner tabs: prefix n_per_class
```

The prefix tabs begin at \(N_2\).
Use the same fixed reference-selected display subsets as the winner matrices.

### Status presentation

The section must explain and visibly distinguish:

```text
constant_ranking
no_valid_pairs
unavailable_first_prefix
no_crossings_in_either
no_shared_crossings
ok
```

Do not rely on tooltips alone.

## 7.4 Structured arm reports: new section and final numbering

Target only the current structured Occupancy arm reports produced by `generate_structured_monte_carlo_report(...)`.
Do not redesign the legacy report generator.

Insert a new section after the current Section 12 and before the current N-star diagnostics.

Final arm-report order:

```text
1. Arm summary
2. Scientific role of this arm
3. Dataset provenance summary
4. Run configuration
5. Training/evaluation protocol
6. Arm-specific data and model description
7. Reproducibility controls
8. Monte Carlo stopping rule
9. Monte Carlo precision diagnostics
10. Loss curves
11. Best subset comparison
12. Subset ranking by sample size
13. Structural dynamics
14. N-star diagnostics
15. Robustness notes
16. Validity and limitations for this arm
17. Technical appendix / exported tables
```

The new `Structural dynamics` section is not a cross-arm fidelity section.
It displays the arm's own:

- directed winner matrices over sample size;
- progressive directed N-star matrices over prefixes.

Tab hierarchy:

```text
outer tabs: classifier
middle tabs: structure
  Winner matrix
  Progressive N-star matrix
inner tabs:
  n_per_class for winner matrices
  prefix n_per_class, beginning at N2, for N-star matrices
```

Use arm-local best-by-cardinality display subsets.
Persist and compute structural data using all subsets.

Do not modify the scientific meaning of the current Section 13 N-star diagnostics when renumbering it to Section 14.

---

## 8. Additive persistence requirements

## 8.1 Simulation-level report-ready data

Extend:

```python
simulation_report_data(result)
```

in:

```text
src/coinfosim/runs/report_data.py
```

Add:

```text
result_data["structural_dynamics"]
```

Recommended schema:

```json
{
  "schema_version": 1,
  "subset_catalog": [[0], [1], [0, 1]],
  "sample_sizes": [2, 4, 8],
  "classifiers": {
    "linear_svm": {
      "winner_pairs_by_n": {
        "2": [
          {"i": 0, "j": 1, "outcome": 1}
        ]
      },
      "directed_crossing_events": [
        {"row": 0, "col": 1, "n_crossing": 8}
      ]
    }
  }
}
```

Rules:

- persist the canonical subset catalog once;
- persist winner outcomes only once for `i < j`;
- use `+1`, `-1`, and `0` for those unordered outcomes;
- persist directed crossing events sparsely;
- omit missing crossing cells;
- never use zero as a missing N-star;
- do not persist full dense progressive matrices when they can be reconstructed;
- do not add derived structures to `result_data_*.json.gz`.

## 8.2 Scenario-level report-ready data

Extend the scenario report data additively with:

```text
report_data["structural_fidelity"]
```

Recommended contents:

```text
schema_version
reference_arm
arm_labels
sample_sizes
subset_catalog
ranking_fidelity_series
winner_agreement_series
nstar_similarity_series
final_summary
reference_display_subsets_by_classifier
```

Self-comparison rows may be persisted for validation, but primary report tables and curves should focus on non-reference arms.

Existing keys must remain unchanged:

```text
channel_names
sample_sizes
arms
visualization
graphs
```

## 8.3 JSON cleaning

All unavailable values must be standard JSON `null`.
Do not serialize:

```text
NaN
Infinity
-Infinity
```

Reuse or extend the current `_clean(...)` behavior in `report_data.py`.

## 8.4 Regeneration backfill

Extend:

```python
regenerate_from_scenario_run(...)
```

in:

```text
scripts/run_occupancy_scenario.py
```

The regeneration path must:

1. load full results from the existing gzip files;
2. compute all new derived structures in memory;
3. regenerate the three arm reports;
4. regenerate the scenario report and all structural PNGs;
5. update each simulation record's additive `structural_dynamics` field;
6. update the scenario record's additive `structural_fidelity` field;
7. refresh `graphs` and `graph_images` metadata when structural graphs are generated;
8. atomically replace each standalone JSON file;
9. update registry records through existing `update_run(...)` methods;
10. preserve run IDs, timestamps that identify the original execution, result-data paths, and source gzip files.

Implementation discipline:

- compute and render everything before mutating JSON records;
- no file may be partially written;
- individual file writes must use temp-file plus replace;
- an exact multi-file transaction is not required, but avoid partial logical updates by finalizing only after all calculations and report rendering succeed.

Do not call `complete_run(...)` during regeneration because the original run is already completed.
Use additive `update_run(...)` operations.

---

## 9. Work blocks and review gates

# Block 1 — Reuse the current tab system and compact the scenario report

## Objective

Port the existing structured-arm tab design into the scenario report and compact all existing multi-panel scenario visualizations before adding any new metric.

## Files to create

```text
src/coinfosim/reports/html_tabs.py
tests/test_report_tabs.py
```

## Files to modify

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_scenario.py
tests/test_occupancy_scenario_report.py
```

Modify `src/coinfosim/reports/scenario_visualization.py` only if graph descriptors need minimal metadata changes. Do not redesign its plotting functions in this block.

## Required work

1. Extract the existing arm-report tab builder, CSS, and JavaScript into `html_tabs.py`.
2. Keep arm-report tab rendering visually unchanged after extraction.
3. Include shared tab CSS and JavaScript once in the scenario report.
4. Replace the Section 4 automatic carousel with nested tabs:
   - source/model;
   - projection dimension.
5. Compact Section 5's three graphs with arm tabs.
6. Compact Section 6 with classifier tabs containing nested arm tabs.
7. Compact the current Section 7 N-star availability output with classifier → reference cardinality → arm tabs.
8. Preserve every existing graph key in `graphs_out` unless a key is demonstrably invalid.
9. Preserve all tables, captions, reference annotations, and numerical behavior.
10. Keep scenario section numbering at 1–8 in this block.
11. Do not implement any new metric in this block.

## Defaults

Use:

```text
Linear SVM
Real → Real
1D
Best 1-channel reference
```

as applicable.

## Lightweight tests

`tests/test_report_tabs.py` must test at least:

- default tab is active;
- non-default panels are hidden;
- nested groups use independent IDs;
- labels and keys are escaped;
- one shared JavaScript payload controls nested groups;
- no `<select>` element is generated.

A no-Monte-Carlo persisted-report test may load the full gzip results for runs 6–8 and render with `generate_graphs=False` to inspect tab structure.
It must not call the simulator.

## Commands allowed in this block

```bash
pytest -q tests/test_report_tabs.py
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
```

Then inspect the regenerated full scenario HTML for:

```text
source/model tabs
projection tabs
best-comparison arm tabs
top-ranked classifier and arm tabs
N-star classifier, cardinality, and arm tabs
absence of the old autoplay toggle
absence of <select>
```

Do not run another test command without user approval.

## Review deliverable

Report:

- exact changed files;
- exact commands executed;
- test result;
- regenerated scenario-report path;
- tab hierarchy inventory;
- confirmation that all prior graphs/tables remain present;
- confirmation that arm-report tabs still work;
- confirmation that Monte Carlo was not run.

Then stop.
Do not begin Block 2.

---

# Block 2 — Generic compatibility, ranking fidelity, and winner agreement

## Objective

Implement the generic all-subset structural core for ranking and pairwise winners, then add the first part of the scenario structural-fidelity section.

## Files to create

```text
src/coinfosim/results/structural.py
src/coinfosim/reports/structural_visualization.py
tests/test_structural_metrics.py
```

## Files to modify

```text
src/coinfosim/runs/report_data.py
src/coinfosim/reports/occupancy_scenario.py
tests/test_occupancy_scenario_report.py
```

## Required work

1. Implement strict cross-arm compatibility validation.
2. Implement canonical subset normalization.
3. Implement average-rank vectors.
4. Implement Ranking Structural Fidelity and statuses.
5. Implement directed winner matrices.
6. Implement Winner Agreement and denominator diagnostics.
7. Implement generic mapping-based scenario structural analysis.
8. Select fixed reference-arm display subsets by cardinality at `Nmax`.
9. Add new scenario Section 7 and renumber existing sections:
   - old Section 7 becomes Section 8;
   - old Section 8 becomes Section 9.
10. In Section 7, add:
    - metric explanations;
    - no-composite-index statement;
    - partial final summary table;
    - Ranking Fidelity curves;
    - Winner Agreement curves;
    - winner matrix visualizations.
11. Use tabs for every multi-panel collection.
12. Add in-memory additive report-data fields for the two metrics without removing existing keys.
13. Do not implement progressive N-star yet.

## Minimum pure tests

Use handcrafted fake results or accumulators to cover:

- compatible arms;
- mismatched sample grids;
- duplicate or non-increasing grid rejection;
- mismatched classifiers;
- missing and extra subsets;
- non-finite required loss rejection;
- identical ranking gives `1.0`;
- reversed ranking gives `-1.0` when no ties exist;
- average-rank ties;
- non-self constant ranking returns `None` and `constant_ranking`;
- winner matrix antisymmetry;
- winner matrix exact tie encoding;
- Winner Agreement with no ties;
- exact-tie pair skipping;
- no-valid-pairs status;
- self-comparison.

## Commands allowed in this block

```bash
pytest -q tests/test_structural_metrics.py \
  -k "compatibility or ranking or winner"
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
```

Do not run full report test files.

## Review deliverable

Report:

- exact changed files;
- exact commands and results;
- regenerated scenario-report path;
- final `rho_rank(Nmax)` by classifier and comparison arm;
- final `A_W(Nmax)` and valid/skipped pair counts by classifier and comparison arm;
- fixed reference-selected display subsets by classifier and cardinality;
- confirmation that metrics use all subsets;
- confirmation that matrices use only the fixed display subsets;
- confirmation that no composite index was added;
- confirmation that Monte Carlo was not run.

Then stop.
Do not begin Block 3.

---

# Block 3 — Progressive directed N-star and full structural-fidelity section

## Objective

Implement the observed-grid progressive crossing structure and complete the scenario structural-fidelity section.

## Files to modify

```text
src/coinfosim/results/structural.py
src/coinfosim/reports/structural_visualization.py
src/coinfosim/runs/report_data.py
src/coinfosim/reports/occupancy_scenario.py
tests/test_structural_metrics.py
tests/test_occupancy_scenario_report.py
```

## Required work

1. Extract sparse directed crossing events for all ordered subset pairs.
2. Reconstruct progressive latest-same-direction matrices for every prefix from `N2` through `NT`.
3. Implement crossing-set Jaccard similarity.
4. Implement log2 timing similarity normalized by the current prefix span.
5. Implement the final product and all required statuses.
6. Complete the Section 7 final summary table.
7. Add Progressive N-star Similarity metric curves beginning at `N2`.
8. Add progressive N-star matrix visualizations.
9. Add explicit status explanations and status counts.
10. Preserve current Section 8 N-star availability semantics unchanged.
11. Preserve current Section 9 interpretation content, adding only a short interpretation paragraph for the three new metrics.
12. Extend in-memory scenario report data with N-star similarity series and summary fields.
13. Do not yet modify the structured arm reports or durable regeneration backfill.

## Minimum pure tests

Cover:

- no progressive object at `N1`;
- one row-becomes-better crossing;
- opposite-direction crossing;
- tie at the previous point followed by a strict win;
- a row already winning at `N1` does not create an event;
- multiple alternations;
- latest event in the same direction is retained;
- progressive values never exceed their prefix;
- no crossings in either arm;
- no shared crossings;
- shared crossing Jaccard;
- log2 timing distance;
- current-prefix normalization rather than final-grid normalization;
- clipping to `[0, 1]`;
- self-comparison;
- JSON-safe `None` handling.

## Commands allowed in this block

```bash
pytest -q tests/test_structural_metrics.py \
  -k "crossing or progressive or nstar or similarity"
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
```

## Review deliverable

Report:

- exact changed files;
- exact commands and results;
- regenerated scenario-report path;
- final N-star similarity decomposition by classifier and comparison arm;
- every `no_crossings_in_either` case;
- every `no_shared_crossings` case;
- confirmation that curves begin at `N2`;
- confirmation that matrix values are observed grid values, not interpolations;
- confirmation that current Section 8 N-star availability remains scientifically unchanged;
- confirmation that Monte Carlo was not run.

Then stop.
Do not begin Block 4.

---

# Block 4 — Arm structural dynamics and durable no-Monte-Carlo backfill

## Objective

Add arm-local structural matrix views and make all new report-ready data durable through the current regeneration workflow.

## Files to modify

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_monte_carlo.py
src/coinfosim/reports/structural_visualization.py
src/coinfosim/results/structural.py
src/coinfosim/runs/report_data.py
scripts/run_occupancy_scenario.py
src/coinfosim/runs/registry.py            # only if existing update_run is insufficient
tests/test_structural_metrics.py
tests/test_occupancy_monte_carlo_reports.py
tests/test_occupancy_run_tracking.py
```

## Required work

1. Add arm-report Section 13 `Structural dynamics`.
2. Renumber the current structured-arm Sections 13–16 to 14–17.
3. Add tabbed arm-local winner matrices.
4. Add tabbed arm-local progressive N-star matrices.
5. Use arm-local best-by-cardinality display subsets.
6. Preserve all current structured arm-report sections and tab behavior.
7. Finalize `simulation_report_data(...)["structural_dynamics"]`.
8. Finalize `scenario_report_data(...)["structural_fidelity"]`.
9. Extend regeneration to backfill:
   - simulation JSON for runs 6, 7, and 8;
   - simulation registry records;
   - scenario JSON for run 2;
   - scenario registry record;
   - structural graph artifacts.
10. Use atomic replacement for standalone JSON files.
11. Use existing registry `update_run(...)` unless a minimal batch helper is necessary.
12. Do not alter the gzip result payloads.
13. Do not allocate new run IDs.
14. Ensure repeated regeneration is deterministic.

## Lightweight tests

Add pure serialization tests to `tests/test_structural_metrics.py` for:

- sparse winner encoding;
- sparse directed crossing events;
- canonical subset catalog;
- JSON `null` for unavailable values;
- no NaN leakage;
- deterministic record ordering.

Update existing report/run tests for future compatibility, but do not run their full files in this block.

## Commands allowed in this block

```bash
pytest -q tests/test_structural_metrics.py \
  -k "serialization or report_data or json or deterministic"
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
```

After regeneration, inspect:

```text
output/reports/scenarios/000002_occupancy_baseline_full/scenario.json
output/reports/simulations/000006_occupancy_real_data_full/simulation.json
output/reports/simulations/000007_occupancy_single_gaussian_to_real_full/simulation.json
output/reports/simulations/000008_occupancy_gmm_to_real_full/simulation.json
output/reports/scenario_runs.json
output/reports/simulation_runs.json
```

Also inspect all four regenerated HTML reports.

## Review deliverable

Report:

- exact changed files;
- exact commands and results;
- regenerated scenario and arm-report paths;
- final section order of each arm report;
- concise schema examples for `structural_dynamics` and `structural_fidelity`;
- confirmation that unavailable values are `null`;
- confirmation that missing crossings are not zero;
- confirmation that registry and standalone records agree;
- confirmation that gzip files were not modified;
- confirmation that run IDs remain 2 and 6–8;
- confirmation that Monte Carlo was not run.

Then stop.
Do not begin Block 5.

---

# Block 5 — Focused regression audit and final review package

## Objective

Perform a narrow audit of only the behavior touched by Blocks 1–4, fix defects discovered by that audit, and prepare the final review package.

This block is not permission for a broad refactor or full test-suite run.

## Files that may be modified

Only files already touched in Blocks 1–4 and their focused tests.

## Required audit

Verify:

1. scenario section numbering is 1–9;
2. structured arm-report numbering is 1–17;
3. old arm-report tabs still use the same layout;
4. every scenario multi-panel collection uses tabs;
5. no `<select>` controls were introduced;
6. the old automatic carousel is gone;
7. every tab group has one active panel;
8. nested tab groups are independent;
9. all expected graph files exist;
10. all expected `graphs_out` keys and artifact paths are registered;
11. Ranking Fidelity uses average ranks and all subsets;
12. Winner Agreement uses unordered valid pairs and exact tie exclusion;
13. progressive N-star starts at `N2` and uses observed grid values;
14. N-star similarity uses the current prefix span;
15. matrix display reduction does not alter metric calculations;
16. self-comparisons are persisted but not emphasized in primary plots;
17. no composite index exists;
18. existing N-star availability behavior is unchanged;
19. existing arm N-star diagnostics behavior is unchanged;
20. old threshold functions are unchanged;
21. JSON contains no NaN or Infinity;
22. regeneration does not change run IDs;
23. regeneration does not rewrite gzip result payloads;
24. two consecutive regenerations produce identical derived numerical data.

## Commands allowed in this block

```bash
pytest -q tests/test_report_tabs.py tests/test_structural_metrics.py
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
python scripts/run_occupancy_scenario.py \
  --report-from-scenario-run 2 \
  --no-color
```

Optionally run only these existing pure or near-pure node IDs if a regression specifically requires them:

```bash
pytest -q tests/test_occupancy_scenario_report.py::test_nstar_multiple_crossings_stores_all_reports_last
pytest -q tests/test_sprint1_analysis.py -k "threshold"
```

Do not run any other existing test without explicit user approval.

## Final review deliverable

Provide:

- final changed-file list;
- concise diff summary;
- every command executed in Block 5;
- test results;
- final scenario-report path;
- final three arm-report paths;
- final metric/status summary;
- final persistence-schema summary;
- tab hierarchy inventory;
- idempotence result;
- confirmation that no Monte Carlo was run;
- confirmation that the complete test suite was not run;
- residual risks that would require broader testing.

Then stop for final user review.

---

## 10. Acceptance criteria

The implementation is accepted only when all of the following are true:

- the agent began from the corrected current remote baseline;
- current structured arm-report improvements were preserved;
- scenario selectors use the same tab layout as arm reports;
- no dropdown or select-box UI was introduced;
- existing scenario multi-panel output is compacted with tabs;
- the three new metrics are implemented separately;
- no composite metric exists;
- numerical metrics use all non-empty subsets;
- ranking ties use average ranks;
- standard error never breaks metric-ranking ties;
- Winner Agreement excludes exact-tie pairs and exposes its denominator;
- winner matrices use `+1`, `-1`, `0`, and unavailable diagonal cells;
- directed crossings use observed grid values and begin no earlier than `N2`;
- a subset already winning at `N1` is not treated as a crossing;
- progressive matrices retain the latest event in the same direction;
- N-star similarity reports crossing-set and timing components;
- edge-case statuses are explicit;
- unavailable values are JSON `null`, never zero or NaN;
- generic numerical code supports arbitrary compatible arms, subsets, classifiers, and grids;
- scenario matrix plots use fixed reference-selected subsets by cardinality;
- arm matrix plots use fixed arm-local subsets by cardinality;
- display reduction never changes metric calculations;
- the scenario report has nine final sections;
- structured arm reports have seventeen final sections;
- existing scenario N-star availability remains scientifically unchanged;
- existing arm N-star diagnostics remain scientifically unchanged;
- report-ready persistence is additive;
- source gzip payloads remain unchanged;
- report regeneration uses scenario run 2 and simulation runs 6–8;
- no new run IDs are allocated;
- Monte Carlo is not rerun;
- only focused tests are executed;
- the agent stops after every block for user review.
