# CoInfoSim — Predictive-Cooperation Terminology, Semantic Schema, and Provenance-Ready Refactor
## Professional Implementation Plan

**Repository:** `paulorenatoaz/coinfosim`  
**Primary recovery source:** remote branch `gh-pages`  
**Scientific objective:** complete the migration from the retired cooperation-structure / `N*` framework to a canonical predictive-cooperation-profile framework and establish stable machine-readable terminology for later ontology and provenance artifacts.  
**Computation policy:** no Monte Carlo rerun; all ordinary tests must remain short; exactly one multi-minute validation command is permitted, at the end.

---

# 1. Why this refactor is required

The codebase, published reports, persisted run data, package metadata, and academic documents have historically used several overlapping vocabularies:

- cooperative advantage;
- cooperation structure;
- structural fidelity;
- directed crossing;
- `N*` / `N-star`;
- last crossing;
- timing similarity;
- progressive `N*` similarity;
- composite similarity.

The approved scientific formulation now uses:

- predictive cooperation;
- predictive cooperation profile;
- pairwise winner relation;
- winner reversal;
- winner matrix \(\mathbf W\);
- reversal matrix \(\mathbf R\);
- ranking fidelity;
- Winner Agreement;
- reversal existence agreement;
- mean log2 reversal distance;
- reversal sample-size similarity.

This terminology must be consistent not only in prose but also in:

- Python module names;
- function names;
- persisted JSON keys;
- schema versions;
- report sections and labels;
- publication metadata;
- semantic identifiers;
- future ontology classes and properties;
- provenance entities and activities.

A superficial search-and-replace is not sufficient. The refactor must preserve scientific semantics, legacy readability, and reproducibility while making the canonical vocabulary explicit and machine-readable.

---

# 2. External semantic standards

The semantic foundation should be designed to interoperate with:

- **JSON-LD 1.1** for machine-readable linked-data contexts;
- **PROV-O** for provenance entities, activities, agents, and derivation relations;
- **OWL 2** as the likely later ontology formalization layer.

This implementation does not need a reasoner or an RDF database. It should provide stable identifiers, a JSON-LD context, provenance-ready records, and a documented mapping so that a later ontology phase does not have to reinterpret Python field names.

Recommended references:

- W3C JSON-LD 1.1 Recommendation;
- W3C PROV-O Recommendation;
- W3C OWL 2 Recommendation.

---

# 3. Hard execution constraints

## 3.1 No expensive experiments

Never execute:

```text
coinfosim scenario run
```

Never execute any simulation in:

- smoke;
- fast;
- full;
- full-scale;
- strict;

as part of this refactor.

The recovery and validation workflow must use existing persisted `result_data_*.json.gz` files.

## 3.2 Test-time budget

All ordinary tests must satisfy both conditions:

- targeted by explicit file and preferably explicit test selection;
- expected wall time below 90 seconds.

Use a timeout for every ordinary test command:

```bash
timeout 90s pytest ...
```

There may be **exactly one command expected to take several minutes**, and it must be the final end-to-end validation command.

The final command must:

- have an explicit timeout, recommended `timeout 12m`;
- run no Monte Carlo;
- regenerate reports only from persisted results;
- validate only the five scientifically relevant full scenarios;
- run only the selected regression tests listed in this plan.

Do not run:

```bash
pytest
pytest -m "not slow"
pytest tests/
```

Do not run the full suite.

## 3.3 Credit and navigation economy

The implementation agent must:

1. inspect only files named in the active block;
2. use symbol-targeted `rg` searches;
3. avoid repeated repository-wide searches;
4. never reread large generated HTML files unless a specific assertion fails;
5. avoid rendering every report or every LaTeX page during code refactoring;
6. reuse one generated inventory across later blocks;
7. commit each block but continue automatically unless a real blocker exists;
8. produce one consolidated final report rather than requesting approval after every block.

## 3.4 Branch and history safety

- Do not merge `gh-pages` into the development branch.
- Treat `gh-pages` as an immutable artifact source.
- Do not force-push or rewrite history.
- Do not modify the remote `gh-pages` branch in this task.
- Do not commit the recovered multi-gigabyte output tree unless project policy explicitly requires it.
- Commit only compact recovery manifests, checksums, schemas, source code, tests, and documentation.

---

# 4. Canonical scientific vocabulary

## 4.1 Central concepts

| Stable semantic ID | English label | Portuguese label | Role |
|---|---|---|---|
| `coinfosim:PredictiveCooperation` | predictive cooperation | cooperação preditiva | umbrella phenomenon |
| `coinfosim:PredictiveComplementarity` | predictive complementarity | complementaridade preditiva | distinct contributions improve joint performance |
| `coinfosim:PredictiveRedundancy` | predictive redundancy | redundância preditiva | combination adds little or no predictive gain |
| `coinfosim:PredictiveCooperationProfile` | predictive cooperation profile | perfil de cooperação preditiva | quantitative object over sample size |
| `coinfosim:PredictiveCooperationPattern` | predictive cooperation pattern | padrão de cooperação preditiva | interpreted regularity |
| `coinfosim:Attribute` | attribute | atributo | formal model-input variable |
| `coinfosim:InformationChannel` | information channel | canal de informação | conceptual or operational source |
| `coinfosim:EstimatedMeanTestLoss` | estimated mean test loss | perda média estimada no conjunto de teste | base comparison quantity |
| `coinfosim:PairwiseWinnerRelation` | pairwise winner relation | relação de vencedor par a par | effective winner state |
| `coinfosim:WinnerMatrix` | winner matrix | matriz de vencedores | \(\mathbf W(n)\) |
| `coinfosim:WinnerReversal` | winner reversal | inversão de vencedor | valid change of effective winner |
| `coinfosim:ReversalMatrix` | reversal matrix | matriz de inversões | \(\mathbf R(n_t)\) |
| `coinfosim:RankingFidelity` | ranking fidelity | fidelidade de ranking | global ordering agreement |
| `coinfosim:WinnerAgreement` | Winner Agreement | concordância de vencedores | effective pairwise agreement |
| `coinfosim:ReversalExistenceAgreement` | reversal existence agreement | concordância de existência de inversões | support-set Jaccard |
| `coinfosim:MeanLog2ReversalDistance` | mean log2 reversal distance | distância média das inversões em escala log2 | unnormalized sample-size displacement |
| `coinfosim:ReversalSampleSizeSimilarity` | reversal sample-size similarity | similaridade dos tamanhos amostrais das inversões | normalized conditional agreement |

## 4.2 Required terminology distinctions

### Attribute versus information channel

- `attribute` is the formal variable consumed by a classifier;
- `information channel` is the conceptual or operational source represented by an attribute or explicitly defined attribute group.

The implementation may store both labels but must not treat them as automatically identical ontology classes.

### Profile versus pattern

- `predictive cooperation profile` is a computed quantitative artifact;
- `predictive cooperation pattern` is an interpretation derived from one or more profiles.

A report JSON is allowed to contain a profile. It must not claim to contain a pattern unless an explicit interpretation activity created one.

### Reversal versus crossing

A valid event is defined by a change in effective pairwise winner, not by geometric interpolation of continuous curves.

Use:

- `winner_reversal`;
- `reversal_sample_size`;
- `last_observed_reversal_sample_size`.

Do not use as canonical field names:

- `crossing`;
- `last_crossing`;
- `nstar`;
- `timing`.

## 4.3 Deprecated active terms

The following may remain only in compatibility adapters, migration notes, or explicitly historical artifacts:

- `structural_fidelity`;
- `structural_dynamics`;
- `nstar_similarity`;
- `timing_similarity`;
- `crossing_jaccard`;
- `directed_crossing_events`;
- `progressive_directed_nstar`;
- `cooperative_threshold`;
- `last_crossing`.

New outputs must not emit them as canonical keys.

---

# 5. Canonical mathematical contract

## 5.1 Estimated mean test loss

\[
\widehat L_{S_i,f}^{(g)}(n)
=
\frac{1}{R_n}
\sum_{r=1}^{R_n}
\widehat L_{S_i,f,r}^{(g)}(n).
\]

## 5.2 Observed pairwise outcome

For \(i<j\):

\[
U_{ij}(n_k)=
\begin{cases}
+1, & \widehat L_i(n_k)<\widehat L_j(n_k),\\
-1, & \widehat L_i(n_k)>\widehat L_j(n_k),\\
0, & \widehat L_i(n_k)=\widehat L_j(n_k).
\end{cases}
\]

## 5.3 Effective winner relation

\[
W_{ij}(n_k)=
\begin{cases}
U_{ij}(n_k), & U_{ij}(n_k)\neq 0,\\
W_{ij}(n_{k-1}), &
U_{ij}(n_k)=0\text{ and a previous winner exists},\\
0, &
U_{ij}(n_k)=0\text{ and no previous winner exists}.
\end{cases}
\]

Rules:

- initial ties remain unresolved;
- the first strict winner initializes the relation;
- a later tie carries the previous winner;
- the first winner after initial ties is not a reversal;
- a tie neither creates nor erases a reversal.

## 5.4 Valid winner reversal

A reversal occurs at \(n_k\) only when:

\[
W_{ij}(n_{k-1})\neq 0,\qquad
W_{ij}(n_k)\neq 0,\qquad
W_{ij}(n_k)\neq W_{ij}(n_{k-1}).
\]

## 5.5 Reversal matrix

\[
R_{ij}(n_t)=
\begin{cases}
\max\mathcal E_{ij}(n_t), &
\mathcal E_{ij}(n_t)\neq\varnothing,\\
\varnothing, &
\mathcal E_{ij}(n_t)=\varnothing.
\end{cases}
\]

`R_ij(n_t)` exists if and only if at least one valid winner reversal occurred for the unordered pair in the evaluated prefix.

Representation:

- upper triangle only;
- one cell per unordered pair;
- no duplicated direction;
- current winner obtained from `W`;
- cell value is latest observed reversal sample size.

## 5.6 Separate preservation indicators

- ranking fidelity;
- Winner Agreement;
- reversal existence agreement;
- mean log2 reversal distance;
- reversal sample-size similarity.

Do not calculate a composite product.

---

# 6. Recovery source and scientific scenario inventory

## 6.1 Source branch

Use:

```text
origin/gh-pages
```

Do not rely on the homepage count alone.

The published index lists eight scenario reports, but the branch also contains scenario `000008`, which must be discovered by filesystem inventory.

## 6.2 Expected scenario directories

The recovery inventory should recognize:

| Scenario ID | Dataset/mode | Role |
|---|---|---|
| `000000` | Occupancy smoke | cheap validation fixture |
| `000001` | Occupancy fast | historical |
| `000002` | Occupancy full | academic result |
| `000003` | Air Quality smoke | cheap/historical |
| `000004` | Air Quality smoke | cheap/historical |
| `000005` | Air Quality full | academic result |
| `000006` | Air Quality full-scale | academic result |
| `000007` | SUPPORT2 full | academic result |
| `000008` | SUPPORT2 full with Random Forest | academic result, not reliably indexed on homepage |

## 6.3 Scientifically relevant full scenarios

The final end-to-end validation must use exactly:

```text
000002
000005
000006
000007
000008
```

## 6.4 Required recovered artifacts

For each selected scenario:

- scenario directory;
- `scenario.json`;
- dataset/split/preprocessing metadata;
- scenario HTML and referenced images;
- linked simulation directories;
- each `simulation.json`;
- each `summary_*.json`;
- each `result_data_*.json.gz`;
- relevant CSV tables.

The compressed `result_data` files are the critical scientific source for recomputing the new profile metrics without rerunning Monte Carlo.

---

# 7. Target architecture

## 7.1 Canonical analysis module

Preferred new module:

```text
src/coinfosim/results/predictive_profile.py
```

Canonical functions:

```python
observed_pairwise_outcome(...)
effective_winner_matrices(...)
winner_agreement_series(...)
winner_reversal_events(...)
progressive_reversal_matrices(...)
ranking_fidelity_series(...)
progressive_reversal_agreement(...)
simulation_pairwise_profile_dynamics(...)
scenario_predictive_profile_agreement(...)
```

Compatibility module:

```text
src/coinfosim/results/structural.py
```

It may re-export canonical functions and legacy aliases for one compatibility cycle, but active code must import from `predictive_profile.py`.

Do not emit runtime deprecation warnings during report generation unless tests establish that warnings are safe. Prefer documented aliases.

## 7.2 Canonical visualization module

Preferred:

```text
src/coinfosim/reports/predictive_profile_visualization.py
```

Canonical figures:

```python
profile_metric_series_figure(...)
winner_matrix_figure(...)
reversal_matrix_figure(...)
paired_winner_reversal_figure(...)
```

Compatibility shim:

```text
src/coinfosim/reports/structural_visualization.py
```

Active report code must use the canonical module.

## 7.3 Canonical persisted structures

Recommended schema version:

```text
predictive_profile_schema_version = 3
```

Use version 3 only if the current feature branch already implemented the W/R schema as version 2. If the branch has not implemented version 2, preserve the same logical migration but document actual version progression.

Canonical scenario payload:

```json
{
  "predictive_cooperation_profile": {
    "schema_version": 3,
    "semantic_vocabulary_version": "1.0.0",
    "semantic_type": "coinfosim:PredictiveCooperationProfile",
    "ranking_fidelity_series": [],
    "winner_agreement_series": [],
    "reversal_agreement_series": [],
    "final_summary": {},
    "reference_display_subsets_by_classifier": {}
  }
}
```

Canonical simulation payload:

```json
{
  "pairwise_profile_dynamics": {
    "schema_version": 3,
    "semantic_vocabulary_version": "1.0.0",
    "semantic_type": "coinfosim:PairwiseProfileDynamics",
    "subset_catalog": [],
    "sample_sizes": [],
    "classifiers": {
      "<classifier>": {
        "effective_winner_relations_by_n": [],
        "winner_reversal_events": [],
        "reversal_matrices_by_prefix": []
      }
    }
  }
}
```

Canonical metric row:

```json
{
  "classifier": "linear_svm",
  "arm": "single_gaussian_to_real",
  "n_prefix": 512,
  "reference_reversal_pair_count": 0,
  "arm_reversal_pair_count": 0,
  "shared_reversal_pair_count": 0,
  "union_reversal_pair_count": 0,
  "reversal_existence_agreement": 1.0,
  "mean_log2_reversal_distance": null,
  "reversal_sample_size_similarity": null,
  "status": "no_reversals_in_either"
}
```

Do not emit:

```text
nstar_similarity
timing_similarity
crossing_jaccard
structural_fidelity
structural_dynamics
```

in new canonical output.

## 7.4 Compatibility readers

Provide explicit adapters:

```python
load_predictive_profile_payload(...)
upgrade_predictive_profile_payload(...)
```

Rules:

- schema 1 may be read for inventory and historical display;
- old composite values must not be converted into new metrics;
- schema 2 W/R fields may be renamed losslessly to schema 3;
- new writes emit canonical schema only;
- old HTML remains historical and immutable;
- a migration report must identify fields that cannot be semantically upgraded.

---

# 8. Semantic and provenance-ready resources

## 8.1 Machine-readable vocabulary

Create:

```text
src/coinfosim/resources/scientific_vocabulary.json
```

Required fields:

```json
{
  "vocabulary_version": "1.0.0",
  "namespace": "https://paulorenatoaz.github.io/coinfosim/ns#",
  "preferred_language": "en",
  "concepts": {}
}
```

Each concept must contain:

- stable ID;
- English preferred label;
- Portuguese preferred label;
- English definition;
- Portuguese definition;
- symbol when applicable;
- broader concept when applicable;
- deprecated aliases;
- canonical Python key when applicable.

## 8.2 JSON-LD context

Create:

```text
src/coinfosim/resources/coinfosim-context.jsonld
```

It must map canonical persisted keys to stable semantic identifiers and include:

- `coinfosim`;
- `prov`;
- `rdf`;
- `rdfs`;
- `xsd`.

Do not map deprecated keys as preferred terms. Legacy aliases may be documented separately.

## 8.3 Vocabulary loader

Create:

```text
src/coinfosim/semantics/vocabulary.py
```

Responsibilities:

- load packaged vocabulary;
- validate uniqueness of IDs and canonical keys;
- return labels and definitions;
- expose semantic IDs;
- reject unknown canonical concept IDs;
- work from an installed wheel.

Update package data so `.jsonld` files ship with the package.

## 8.4 Documentation

Create:

```text
docs/semantics/predictive_cooperation_vocabulary.md
docs/semantics/provenance_mapping.md
```

The provenance mapping should define future/initial mappings:

| CoInfoSim object | PROV-O role |
|---|---|
| dataset snapshot | `prov:Entity` |
| fixed train/test split | `prov:Entity` |
| generator fit | `prov:Activity` |
| fitted generator | `prov:Entity` |
| Monte Carlo simulation run | `prov:Activity` |
| persisted simulation result | `prov:Entity` |
| predictive cooperation analysis | `prov:Activity` |
| predictive cooperation profile | `prov:Entity` |
| HTML/PDF report generation | `prov:Activity` |
| report artifact | `prov:Entity` |
| CoInfoSim commit/version | `prov:SoftwareAgent` or associated software entity |

## 8.5 Per-run semantic manifest

Generate a compact:

```text
semantic_manifest.json
```

for each newly regenerated scenario.

Required content:

- vocabulary version;
- context path;
- semantic type of the scenario output;
- canonical metric IDs;
- dataset ID;
- classifier IDs;
- generator/training-condition IDs;
- sample-size grid;
- fixed real test-set statement;
- source simulation run IDs;
- source result-data paths;
- code commit SHA;
- recovered-source `gh-pages` commit SHA;
- artifact hashes.

## 8.6 Initial provenance JSON-LD

Generate:

```text
provenance.jsonld
```

for each newly regenerated scenario.

Minimum graph:

- entities:
  - recovered result-data files;
  - dataset/split metadata;
  - predictive cooperation profile;
  - generated scenario report;
- activities:
  - original simulation;
  - profile recomputation;
  - report regeneration;
- agent:
  - CoInfoSim software/commit;
- relations:
  - `prov:used`;
  - `prov:wasGeneratedBy`;
  - `prov:wasDerivedFrom`;
  - `prov:wasAssociatedWith`.

This is a provenance-ready minimum, not a complete domain ontology.

---

# 9. Implementation blocks

---

## Block 0 — Repository audit and immutable `gh-pages` recovery
**Complexity:** medium I/O, negligible CPU  
**Expected duration:** under 90 seconds excluding network fetch  
**Tests:** no pytest

### Files to inspect

- `.gitignore`
- `src/coinfosim/runs/registry.py`
- `src/coinfosim/runs/report_data.py`
- `src/coinfosim/scenarios/dataset_anchored_runner.py`
- `src/coinfosim/publish/publisher.py`
- `src/coinfosim/publish/site.py`

### State commands

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log --oneline --decorate -12
git fetch origin main gh-pages
git rev-parse origin/gh-pages
```

If the worktree is dirty, stop.

Determine whether an existing local feature branch already contains the W/R refactor. Do not discard it. Continue from the most advanced safe branch.

### Recovery procedure

Use `git archive`, not a merge:

```bash
GH_PAGES_SHA="$(git rev-parse origin/gh-pages)"
SNAPSHOT="output/recovered_gh_pages/${GH_PAGES_SHA}"

mkdir -p "${SNAPSHOT}"
git archive origin/gh-pages reports | tar -x -C "${SNAPSHOT}"

mkdir -p output/reports
rsync -a "${SNAPSHOT}/reports/" output/reports/
```

Do not use `rsync --delete`.

### Inventory script

Create:

```text
scripts/inventory_recovered_reports.py
```

It must:

- scan directories, not homepage cards;
- enumerate scenarios and linked simulations;
- identify `000000` through `000008`;
- identify all `result_data_*.json.gz`;
- record sizes and SHA-256 hashes;
- validate gzip integrity;
- validate that referenced files exist;
- flag missing registry entries;
- write compact tracked output:

```text
docs/provenance/gh_pages_recovery_manifest.json
```

The manifest must include:

- recovery timestamp;
- `gh-pages` commit SHA;
- scenario inventory;
- simulation inventory;
- hashes of critical persisted inputs;
- missing/inconsistent registry references;
- academic-scenario flag.

### Cheap validation

```bash
timeout 90s python scripts/inventory_recovered_reports.py \
  --reports-dir output/reports \
  --source-ref origin/gh-pages \
  --manifest docs/provenance/gh_pages_recovery_manifest.json \
  --verify-gzip
```

### Acceptance criteria

- original branch remains unchanged;
- `gh-pages` is not merged;
- local immutable snapshot exists;
- working `output/reports` exists;
- scenarios `000000`–`000008` are inventoried by directory;
- critical compressed results pass gzip validation;
- inconsistencies are reported, not silently repaired.

### Commit

```text
chore: add reproducible gh-pages artifact recovery inventory
```

---

## Block 1 — Canonical vocabulary and semantic identifiers
**Complexity:** medium  
**Expected test duration:** under 30 seconds

### Files to create

- `src/coinfosim/resources/scientific_vocabulary.json`
- `src/coinfosim/resources/coinfosim-context.jsonld`
- `src/coinfosim/semantics/__init__.py`
- `src/coinfosim/semantics/vocabulary.py`
- `docs/semantics/predictive_cooperation_vocabulary.md`
- `docs/semantics/provenance_mapping.md`
- `tests/test_semantic_vocabulary.py`

### Files to modify

- `pyproject.toml`

### Requirements

- implement the terminology ledger in Section 4;
- assign stable IDs;
- include deprecated aliases only as metadata;
- package JSON and JSON-LD resources;
- validate duplicate labels/IDs/keys;
- support wheel-installed resource loading;
- define no scientific calculation here.

### Cheap test

```bash
timeout 90s pytest -q tests/test_semantic_vocabulary.py
```

### Acceptance criteria

- vocabulary loads from source and installed-resource mechanism;
- every canonical persisted key maps to one semantic ID;
- deprecated terms are not preferred labels;
- JSON-LD context parses as JSON;
- package data includes `.jsonld`.

### Commit

```text
feat: define predictive cooperation semantic vocabulary
```

---

## Block 2 — Canonical predictive-profile analysis module
**Complexity:** medium-high  
**Expected test duration:** under 90 seconds

### Primary files

- create `src/coinfosim/results/predictive_profile.py`
- modify `src/coinfosim/results/structural.py`
- inspect `src/coinfosim/results/analysis.py`
- modify `tests/test_structural_metrics.py`
- optionally rename/add `tests/test_predictive_profile_metrics.py`

### Adaptive rule

If the current branch already implements effective winners and reversal metrics:

- preserve correct code;
- move/rename it to the canonical module;
- update names and docstrings;
- avoid rewriting algorithms without evidence.

If it does not:

- implement the mathematical contract in Section 5.

### Canonical functions

Implement or expose:

```python
observed_pairwise_outcome
effective_winner_matrices
winner_agreement_series
winner_reversal_events
progressive_reversal_matrices
ranking_fidelity_series
progressive_reversal_agreement
simulation_pairwise_profile_dynamics
scenario_predictive_profile_agreement
```

### Compatibility policy

`structural.py` may expose:

```python
from .predictive_profile import ...
```

Legacy names may remain as aliases only when required by external compatibility.

Active modules and tests must use canonical names.

### Required focused cases

- initial ties then first winner: no reversal;
- winner, tie, same winner: no reversal;
- winner, tie, opposite winner: reversal;
- multiple reversals: `R` stores latest;
- unresolved pair remains unavailable;
- upper triangle only;
- empty/empty reversal support;
- disjoint supports;
- shared identical sample sizes;
- shared displaced sample sizes;
- no composite metric.

### Cheap test

Use only the focused metric file:

```bash
timeout 90s pytest -q \
  tests/test_predictive_profile_metrics.py
```

If the repository retains the old filename:

```bash
timeout 90s pytest -q \
  tests/test_structural_metrics.py \
  -k "winner or reversal or ranking or semantic"
```

Run one command, not both, unless the first path does not exist.

### Acceptance criteria

- canonical module contains active implementation;
- active code no longer imports legacy analysis names;
- compatibility aliases are explicit;
- no interpolation;
- no composite metric;
- focused tests pass within budget.

### Commit

```text
refactor: canonicalize predictive cooperation profile analysis
```

---

## Block 3 — Persisted schema migration and legacy adapters
**Complexity:** medium-high  
**Expected test duration:** under 90 seconds

### Primary files

- `src/coinfosim/runs/report_data.py`
- create `src/coinfosim/results/profile_schema.py`
- `src/coinfosim/runs/registry.py`
- `tests/test_predictive_profile_schema.py`
- targeted assertions in:
  - `tests/test_occupancy_run_tracking.py`
  - `tests/test_parallel_scientific_equivalence.py`

### Requirements

- emit canonical schema;
- add semantic vocabulary version;
- add semantic type IDs;
- implement schema 1/2 readers;
- prohibit conversion of old composite values into new metrics;
- preserve strict JSON:
  `json.dumps(..., allow_nan=False)`;
- preserve deterministic ordering;
- use `null` for unavailable metrics.

### Registry terminology

Do not rename stable run IDs or filesystem layout.

Add optional compact fields without breaking old records:

```text
semantic_schema_version
semantic_manifest_path
provenance_path
scientific_object_type
```

`from_dict` must continue ignoring unknown future keys.

### Cheap tests

```bash
timeout 90s pytest -q \
  tests/test_predictive_profile_schema.py \
  tests/test_occupancy_run_tracking.py \
  -k "schema or report_data or semantic or registry"
```

Do not run parallel equivalence here unless a changed assertion directly requires it.

### Acceptance criteria

- new outputs contain canonical keys only;
- old W/R schema upgrades losslessly;
- old composite schema loads as historical but cannot masquerade as new metrics;
- registries remain backward-readable;
- strict JSON passes.

### Commit

```text
refactor: version predictive profile report schema
```

---

## Block 4 — Semantic manifest and PROV-O-compatible provenance export
**Complexity:** medium  
**Expected test duration:** under 45 seconds

### Files to create

- `src/coinfosim/provenance/__init__.py`
- `src/coinfosim/provenance/semantic_manifest.py`
- `src/coinfosim/provenance/jsonld.py`
- `tests/test_provenance_export.py`

### Files to modify

- `src/coinfosim/scenarios/dataset_anchored_runner.py`
- `src/coinfosim/runs/report_data.py`

### Requirements

Generate for regenerated scenarios:

```text
semantic_manifest.json
provenance.jsonld
```

Use existing metadata only:

- run IDs;
- dataset metadata;
- split manifest;
- preprocessing metadata;
- generator/classifier metadata;
- result-data paths;
- code commit;
- source `gh-pages` commit;
- generated artifact paths and hashes.

Do not create claims that are not evidenced by persisted metadata.

### JSON-LD design

Minimum context:

```json
{
  "@context": {
    "coinfosim": "https://paulorenatoaz.github.io/coinfosim/ns#",
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#"
  }
}
```

Use deterministic IDs based on:

- scenario run ID;
- simulation run ID;
- artifact hash;
- relative path.

Never use absolute user-local paths as semantic identifiers.

### Cheap test

```bash
timeout 90s pytest -q tests/test_provenance_export.py
```

### Acceptance criteria

- JSON-LD parses as strict JSON;
- all referenced local artifacts exist in fixture;
- graph includes entity/activity/agent;
- source and generated commits are distinguishable;
- hashes deterministic;
- no deprecated scientific term appears as preferred semantic type.

### Commit

```text
feat: emit semantic and provenance manifests
```

---

## Block 5 — Reports, publication, CLI text, and active documentation
**Complexity:** medium-high  
**Expected test duration:** under 90 seconds

### Primary source files

- `src/coinfosim/reports/predictive_profile_visualization.py`
- `src/coinfosim/reports/structural_visualization.py`
- `src/coinfosim/reports/occupancy_scenario.py`
- `src/coinfosim/reports/dataset_anchored_scenario.py`
- `src/coinfosim/reports/monte_carlo.py`
- dataset-specific wrappers only when a targeted assertion fails
- `src/coinfosim/publish/site.py`
- `src/coinfosim/cli/app.py`
- `README.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `pyproject.toml`
- `src/coinfosim/__init__.py`

### Canonical code renames

Preferred:

```text
_structural_fidelity_section
    -> _predictive_cooperation_profile_section

_structural_dynamics_panel
    -> _pairwise_profile_dynamics_panel

metric_series_figure
    -> profile_metric_series_figure

progressive_nstar_matrix_figure
    -> reversal_matrix_figure
```

Legacy private helpers may be deleted after direct-import confirmation.

### Active report structure

Use separate sections/curves:

1. ranking fidelity;
2. Winner Agreement;
3. reversal existence agreement;
4. reversal sample-size similarity.

Show `W` and `R` together.

Remove active:

- N-star diagnostics;
- interpolation;
- directed crossing matrices;
- timing similarity;
- composite product.

### Publication page

Update:

- project tagline;
- scenario questions;
- report-card scientific descriptions;
- machine-readable artifact links.

The publication code should detect and link:

```text
semantic_manifest.json
provenance.jsonld
```

when present.

Do not edit historical HTML recovered from `gh-pages`.

### Bounded stale-term search

```bash
rg -n \
  "structural_fidelity|structural_dynamics|nstar_similarity|timing_similarity|crossing_jaccard|Progressive N-star|Last Crossing|cooperative structure" \
  src/coinfosim/results \
  src/coinfosim/runs \
  src/coinfosim/reports \
  src/coinfosim/publish \
  src/coinfosim/cli \
  README.md CHANGELOG.md CITATION.cff pyproject.toml
```

Classify every hit:

- active and must change;
- compatibility alias;
- historical changelog entry.

### Cheap tests

Run only:

```bash
timeout 90s pytest -q \
  tests/test_occupancy_scenario_report.py \
  tests/test_support2_scenario_report.py \
  tests/test_publish_site.py \
  -k "profile or reversal or semantic or provenance or report"
```

Air Quality should use the shared renderer and does not require a separate routine test unless a dataset-specific file changed.

### Acceptance criteria

- active UI uses canonical terminology;
- tabs remain;
- publication detects semantic/provenance artifacts;
- historical outputs remain untouched;
- targeted tests pass within budget.

### Commit

```text
refactor: align reports and publication with predictive profiles
```

---

## Block 6 — Cheap recovered-data smoke regeneration
**Complexity:** medium I/O  
**Expected duration:** under 90 seconds  
**No Monte Carlo**

### Scenario

Use only:

```text
000000 Occupancy smoke
```

### Output

Generate into a separate directory:

```text
output/predictive_profile_smoke_validation
```

Do not overwrite:

- recovered snapshot;
- `output/reports`;
- published historical reports.

### Command

Use the recovered registry and existing CLI regeneration command. The exact command may depend on current CLI syntax, but must be equivalent to:

```bash
timeout 90s coinfosim scenario regenerate occupancy \
  --run-id 0 \
  --output-dir output/predictive_profile_smoke_validation
```

### Audit

Check:

- canonical report section labels;
- schema version;
- semantic manifest;
- provenance JSON-LD;
- no old active terms;
- no composite field;
- links resolve;
- source `gh-pages` SHA recorded.

### No commit

Do not commit generated smoke output unless project policy tracks equivalent fixtures.

Commit only test/fixture changes required to make this reproducible:

```text
test: add recovered-data predictive profile smoke validation
```

---

## Block 7 — Documentation of migration and ontology readiness
**Complexity:** low-medium  
**Expected duration:** under 30 seconds

### Files to create/update

- `docs/migration-predictive-profile-schema.md`
- `docs/semantics/predictive_cooperation_vocabulary.md`
- `docs/semantics/provenance_mapping.md`
- `README.md`
- `CHANGELOG.md`

### Required migration table

Document:

| Deprecated | Canonical |
|---|---|
| cooperation structure | predictive cooperation profile |
| structural fidelity | profile agreement / specific metric name |
| directed crossing | winner reversal |
| N-star matrix | reversal matrix |
| crossing Jaccard | reversal existence agreement |
| timing similarity | reversal sample-size similarity |
| `structural_fidelity` JSON | `predictive_cooperation_profile` |
| `structural_dynamics` JSON | `pairwise_profile_dynamics` |

State explicitly:

- historical artifacts are not rewritten;
- old composite values cannot be decomposed;
- recovered `gh-pages` results are source entities;
- regenerated outputs are derived entities;
- ontology/provenance identifiers are stable even when labels evolve.

### Cheap checks

```bash
timeout 30s python -m compileall -q src/coinfosim
```

```bash
timeout 30s python -m json.tool \
  src/coinfosim/resources/scientific_vocabulary.json >/dev/null
```

```bash
timeout 30s python -m json.tool \
  src/coinfosim/resources/coinfosim-context.jsonld >/dev/null
```

### Commit

```text
docs: document predictive profile semantic migration
```

---

## Block 8 — Single final multi-minute validation
**Complexity:** high I/O, no simulation  
**This is the only permitted multi-minute command**

### Validation script

Create:

```text
scripts/validate_predictive_profile_refactor.py
```

It must perform, in one command:

1. verify recovery manifest and critical hashes;
2. validate gzip files for academic scenarios;
3. load each recovered simulation result;
4. regenerate scenarios:
   - `000002`;
   - `000005`;
   - `000006`;
   - `000007`;
   - `000008`;
5. write to:
   `output/predictive_profile_regenerated`;
6. validate canonical schema;
7. validate semantic manifests;
8. validate provenance JSON-LD;
9. validate no composite fields;
10. audit active report text;
11. run only the selected targeted tests:
    - semantic vocabulary;
    - predictive profile metrics;
    - profile schema;
    - provenance export;
    - one shared scenario report test;
    - publication-site test;
    - parallel scientific equivalence only if payload serialization changed.

### Single final command

```bash
timeout 12m python scripts/validate_predictive_profile_refactor.py \
  --reports-dir output/reports \
  --recovery-manifest docs/provenance/gh_pages_recovery_manifest.json \
  --scenario-ids 000002,000005,000006,000007,000008 \
  --output-dir output/predictive_profile_regenerated \
  --run-targeted-tests
```

This is the only command allowed to exceed 90 seconds.

### Prohibitions

The script must fail if it detects an attempt to:

- call scenario execution;
- fit a generator anew;
- execute Monte Carlo;
- write into recovered snapshot;
- write into `gh-pages`;
- overwrite `output/reports`;
- publish remotely.

### Final acceptance criteria

For all five academic scenarios:

- persisted results load;
- new profile metrics compute;
- canonical schema writes;
- `W` and `R` render;
- `A_R`, `D_R`, and `S_R` are separate;
- semantic manifest exists;
- provenance JSON-LD exists;
- source hashes and commits are recorded;
- report HTML has no active old terminology;
- no composite field exists;
- targeted tests pass;
- command completes within final timeout.

### Commit

Only when tracked validation code or fixtures changed:

```text
test: validate recovered predictive profile artifacts
```

---

# 10. File-by-file bounded implementation map

## Core analysis

```text
src/coinfosim/results/structural.py
src/coinfosim/results/predictive_profile.py
src/coinfosim/results/analysis.py
src/coinfosim/results/profile_schema.py
```

## Persistence and run tracking

```text
src/coinfosim/runs/report_data.py
src/coinfosim/runs/registry.py
src/coinfosim/scenarios/dataset_anchored_runner.py
```

## Reports

```text
src/coinfosim/reports/structural_visualization.py
src/coinfosim/reports/predictive_profile_visualization.py
src/coinfosim/reports/occupancy_scenario.py
src/coinfosim/reports/dataset_anchored_scenario.py
src/coinfosim/reports/monte_carlo.py
```

## Publication and CLI text

```text
src/coinfosim/publish/site.py
src/coinfosim/publish/publisher.py
src/coinfosim/cli/app.py
```

## Semantic and provenance foundation

```text
src/coinfosim/resources/scientific_vocabulary.json
src/coinfosim/resources/coinfosim-context.jsonld
src/coinfosim/semantics/vocabulary.py
src/coinfosim/provenance/semantic_manifest.py
src/coinfosim/provenance/jsonld.py
```

## Recovery and validation

```text
scripts/inventory_recovered_reports.py
scripts/validate_predictive_profile_refactor.py
docs/provenance/gh_pages_recovery_manifest.json
```

## Documentation

```text
README.md
CHANGELOG.md
CITATION.cff
pyproject.toml
docs/migration-predictive-profile-schema.md
docs/semantics/predictive_cooperation_vocabulary.md
docs/semantics/provenance_mapping.md
```

## Focused tests

```text
tests/test_semantic_vocabulary.py
tests/test_predictive_profile_metrics.py
tests/test_predictive_profile_schema.py
tests/test_provenance_export.py
tests/test_occupancy_scenario_report.py
tests/test_support2_scenario_report.py
tests/test_publish_site.py
tests/test_occupancy_run_tracking.py
tests/test_parallel_scientific_equivalence.py
```

Do not inspect unrelated test files unless a direct import failure identifies them.

---

# 11. Provenance integrity rules

## 11.1 Source versus derived artifacts

Recovered `gh-pages` artifacts are immutable sources.

New artifacts must record:

```text
prov:wasDerivedFrom
```

relations to:

- exact `result_data` hashes;
- scenario/simulation metadata;
- recovery commit SHA.

## 11.2 Commit distinction

Record separately:

- original scientific simulation commit, when persisted;
- `gh-pages` publication commit;
- current refactor commit;
- report-regeneration commit.

Do not collapse them into one “scientific commit.”

## 11.3 Paths

Persist repository-relative paths.

Do not persist:

- `/home/...`;
- user names;
- temporary worktree paths;
- machine-specific mount points.

## 11.4 Determinism

Semantic and provenance manifests must:

- use sorted keys;
- use stable ordering;
- normalize relative paths;
- use SHA-256;
- avoid nondeterministic timestamps in identity fields.

A generated timestamp may appear as metadata but not as the sole artifact identity.

---

# 12. Non-goals

This refactor must not:

- rerun scientific experiments;
- change datasets, targets, splits, features, classifiers, generators, or stopping rules;
- change 0–1 loss;
- introduce costs or synergy calculations;
- implement mutual information or Fisher information;
- build a complete domain ontology;
- require RDF/OWL dependencies;
- deploy a triple store;
- rewrite historical `gh-pages` reports;
- publish new reports remotely;
- merge `gh-pages` into development;
- update final report/presentation LaTeX in this code-refactor task;
- bump the package version;
- run the full test suite.

The output of this task is the canonical semantic and computational foundation used by the later ontology/provenance and academic-document phases.

---

# 13. Completion definition

The refactor is complete when:

1. `gh-pages` artifacts are recovered reproducibly and inventoried;
2. scenarios `000000`–`000008` are discoverable by directory;
3. academic scenarios `000002`, `000005`, `000006`, `000007`, and `000008` have valid persisted inputs;
4. the active analysis module uses predictive-profile terminology;
5. effective winner and reversal semantics match the approved mathematics;
6. persisted output uses canonical schema keys;
7. old schemas remain readable through explicit adapters;
8. old composite values are never reinterpreted;
9. reports and publication use canonical labels;
10. a machine-readable vocabulary exists;
11. a JSON-LD context exists;
12. regenerated scenarios emit semantic manifests;
13. regenerated scenarios emit PROV-O-compatible provenance JSON-LD;
14. recovered source hashes and commits are recorded;
15. ordinary tests remain below 90 seconds;
16. only one multi-minute validation command is run;
17. no Monte Carlo experiment is executed;
18. all five academic scenarios regenerate from persisted results;
19. no active composite metric or `N*` terminology remains in new output.
