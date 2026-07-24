# CoInfoSim: A Simulator for Cooperative Classification from Multiple Information Channels

**Author:** Paulo Renato Azevedo — Programa de Pós-Graduação em Informática (PPGI), UFRJ
**Research advisors:** Prof. Don Towsley (University of Massachusetts Amherst) · Prof. Daniel Sadoc Menasché (UFRJ)
**Course:** Ciência de Dados — UFRJ
**Course professors:** Prof. Jorge Zavaleta · Prof. Sérgio Serra

The research advisors guided the underlying research program; they are not repository authors. The course professors are listed because this revision was prepared for their Data Science course and are not research advisors.

CoInfoSim is a research simulator for evaluating **predictive cooperation among information channels** in supervised classification tasks. In plain terms: it asks whether training a classifier on *synthetic* data — sampled from a fitted Gaussian model — reproduces the same cooperative behavior among input-channel subsets observed when training on *real* data, evaluated on one shared, held-out real test set. It studies the **predictive cooperation profile** of a channel subset — how its predictive performance relative to other subsets evolves as the number of labeled training samples per class grows — comparing real versus synthetic training-condition arms evaluated on the same fixed real test set.

## Project artifacts

| Artifact | Where to find it |
|---|---|
| Source code | this repository, [`src/coinfosim/`](src/coinfosim/) |
| Scientific report (PDF) | [`coinfosim-report-latex/main.pdf`](coinfosim-report-latex/main.pdf) |
| Academic presentation (PDF, Beamer) | [`coinfosim-presentation/presentation.pdf`](coinfosim-presentation/presentation.pdf) |
| Experimental reports (published HTML) | [GitHub Pages project site](https://paulorenatoaz.github.io/coinfosim/) |
| Project data | published dataset mirror linked from the [project home page](https://paulorenatoaz.github.io/coinfosim/); provenance/licensing per dataset in [`data/raw/`](data/raw/) |
| Result figures and visualizations | published scenario reports on [GitHub Pages](https://paulorenatoaz.github.io/coinfosim/) |
| Readable provenance diagrams | [`provenance/`](provenance/) (editorial projections; see [`provenance/README.md`](provenance/README.md)) |
| Canonical PROV artifacts (audit layer) | persisted next to each scenario run (`semantic_manifest.json`, `provenance.provjson`/`.provn`/`.ttl`); manifest published alongside each scenario report — see [Report regeneration](#report-regeneration) below |
| Ontology / semantic layer | [`ontology/coinfosim.owl.ttl`](ontology/coinfosim.owl.ttl), [`docs/semantics/`](docs/semantics/) |
| Citation | [`CITATION.cff`](CITATION.cff) |
| License | [`LICENSE`](LICENSE) |
| Generative-AI disclosure | [below](#generative-ai-disclosure) |

## Architecture at a glance

```text
real dataset
  → dataset-specific preparation
  → fixed training reservoir + fixed real test set
  → real / Single-Gaussian / GMM training arms
  → classifiers over channel subsets and sample-size grid
  → persisted results
  → predictive cooperation profile
  → reports / visualizations / provenance
```

See [Repository structure](#repository-structure) below for the code layout, and the [scientific report](coinfosim-report-latex/main.pdf) for the full experimental design.

> **Core research question:** When does cooperation among information channels improve supervised classification?

> **Status:** Active research project. The installable CLI runs three reproducible, dataset-anchored scenarios — Occupancy Detection, UCI Air Quality, and SUPPORT2 180-day mortality — each comparing real, single-Gaussian synthetic, and class-conditional GMM synthetic training on one fixed real evaluation set.

## Installation

```bash
python -m pip install coinfosim
```

Python 3.10 or later is required. Installing from PyPI does **not** require cloning this repository or manually locating any dataset: the CLI downloads and hash-verifies the datasets it needs the first time you run a scenario.

## 60-second quick start

```bash
coinfosim scenario list
coinfosim scenario run occupancy --mode smoke
```

The first run automatically downloads and verifies the Occupancy Detection dataset from the CoInfoSim GitHub Pages site into your platform's user data directory, then runs the smoke-scale three-arm protocol and writes an HTML report. Nothing is installed into your project directory or current working directory.

## Three built-in scenarios

| Scenario | Slug | Aliases | Dataset |
|---|---|---|---|
| Occupancy Detection | `occupancy` | `occupancy-detection` | UCI Occupancy Detection (CC BY 4.0) |
| UCI Air Quality | `air-quality` | `air_quality`, `airquality` | UCI Air Quality (CC BY 4.0) |
| SUPPORT2 180-day mortality | `support2` | `support-2` | SUPPORT2 (acknowledgment required; see [Datasets](#dataset-download-and-cache-behavior)) |

```bash
coinfosim scenario run occupancy --mode smoke
coinfosim scenario run air-quality --mode smoke
coinfosim scenario run support2 --mode smoke
```

Each command resolves the dataset, downloads and hash-verifies any missing files, runs the same three-arm scientific protocol (real, single-Gaussian synthetic, GMM synthetic — see [Scientific modes](#scientific-modes-and-computational-cost)), and writes the dataset report, three simulation-arm reports, consolidated scenario report, run registries, and persisted result data. See scientific detail for each dataset below, and full option reference with `coinfosim scenario run --help`.

Numeric scenario identifiers are not used — always refer to scenarios by slug or alias.

## Generated outputs

Every `coinfosim scenario run` (default `--output-dir output/reports`) writes an immutable, ID-addressed set of directories with two shared registries:

```text
output/reports/scenario_runs.json
output/reports/simulation_runs.json
output/reports/scenarios/<scenario-run-id>_<scenario-slug>_<mode>/
output/reports/simulations/<simulation-run-id>_<simulation-slug>_<mode>/
```

The scenario directory contains the dataset report, the consolidated scenario report, `scenario.json`, (unless `--no-visualizations`) projection images and analytical graphs, and the scenario's provenance artifacts: `semantic_manifest.json`, `provenance.provjson`, `provenance.provn`, `provenance.ttl`, and (when Graphviz is available) `provenance.png`/`provenance.pdf` — see [Report regeneration](#report-regeneration) below. Each of the three simulation directories contains its arm's Monte Carlo report, a compressed result payload, a summary, and `simulation.json`. Re-running a scenario never overwrites a previous run — every run gets a new, monotonically increasing ID.

Inspect run registries directly:

```bash
coinfosim runs scenarios
coinfosim runs simulations
coinfosim runs scenario <run-id>
coinfosim runs simulation <run-id>
```

## Dataset download and cache behavior

Dataset files are mirrored, byte-for-byte, on the CoInfoSim GitHub Pages site (`https://paulorenatoaz.github.io/coinfosim/datasets/`) — never downloaded directly from UCI or HBiostat at runtime. Every file's SHA-256 hash is pinned in the installed package (`src/coinfosim/resources/datasets.json`) and verified before it is ever used; a present-but-invalid file is never used silently.

Resolution order for `coinfosim scenario run` (and `dataset path`/`dataset fetch`/`dataset verify`):

1. an explicit `--data-dir`;
2. a dataset-specific path from your loaded CoInfoSim configuration;
3. `COINFOSIM_DATA_DIR`, a root containing dataset subdirectories;
4. the verified platform cache (via [`platformdirs`](https://pypi.org/project/platformdirs/) — e.g. `~/.local/share/coinfosim/datasets` on Linux);
5. automatic download into the platform cache, unless `--no-download` is passed;
6. a compatibility fallback to `data/raw/<dataset>` when running from a source checkout.

```bash
coinfosim dataset list
coinfosim dataset show support2
coinfosim dataset status support2
coinfosim dataset fetch support2
coinfosim dataset verify support2 --data-dir /path/to/existing/copy
coinfosim dataset path support2
```

Provenance, licenses, and hashes for each dataset:

- **Occupancy Detection** — UCI dataset 357, DOI `10.24432/C5X01N`, Creative Commons Attribution 4.0 International (CC BY 4.0).
- **UCI Air Quality** — UCI dataset 360, DOI `10.24432/C59K5F`, displayed by UCI as CC BY 4.0; please cite the associated publication (De Vito et al., *Sensors and Actuators B: Chemical*, 2008).
- **SUPPORT2** — DOI `10.3886/ICPSR02957.v2`. **No explicit redistribution license was identified.** It is a public research dataset; follow the acknowledgment policy of the original HBiostat dataset site (<https://hbiostat.org/data/>). Do not treat SUPPORT2 as openly licensed.

Full per-file hashes, sizes, and direct download links are published on the [project home page](https://paulorenatoaz.github.io/coinfosim/) and in [`datasets/manifest.json`](https://paulorenatoaz.github.io/coinfosim/datasets/manifest.json), and are queryable with `coinfosim dataset show <dataset>`.

## Scientific modes and computational cost

| Mode | `n_per_class` values |
|---|---|
| `smoke` | `2, 4, 8, 16, 32` |
| `fast` | `2, 4, 8, 16, 32, 64, 128` |
| `full` | `2, 4, 8, 16, 32, 64, 128, 256, 512` |
| `full-scale` | Powers of two from 2 through the largest power of two not exceeding the training minority-class count; same Monte Carlo precision/budget as `full` |
| `strict` | `2, 4, 8, 16, 32, 64, 128, 256, 512` |

`--mode smoke` is the default and validates the full pipeline end to end. `fast`, `full`, `full-scale`, and `strict` are explicit, deliberate research-budget choices — none of them ever start implicitly, and `full-scale` in particular can be substantially more expensive than `full`. Choose them only after reviewing the real training reservoir's minority-class capacity.

## Multiprocessing

```bash
coinfosim scenario run air-quality \
  --mode full \
  --backend process \
  --workers 8 \
  --worker-threads 1 \
  --start-method auto
```

`--backend sequential` (the default) always uses exactly one worker; `--workers > 1` requires `--backend process`. `--start-method auto` resolves to a safe, platform-appropriate multiprocessing start method — `spawn` on Windows and macOS, `forkserver` (falling back to `spawn`) on Linux — and never selects `fork` automatically. The resolved execution metadata (backend, worker counts, start method, logical CPU count) is persisted in every simulation's output.

## Report regeneration

Regenerate the full report hierarchy for a completed scenario run from its persisted results, without rerunning any Monte Carlo simulation:

```bash
coinfosim scenario regenerate air-quality --run-id 6 --output-dir output/reports
```

Both normal scenario execution and regeneration (re)write `semantic_manifest.json` and the canonical W3C PROV provenance artifacts next to the scenario report: `provenance.provjson` (PROV-JSON), `provenance.provn` (PROV-N), and `provenance.ttl` (PROV-O/Turtle) are always written; `provenance.png`/`provenance.pdf` are additionally rendered when the Graphviz `dot` executable is available. All of these are derived from one canonical `prov.model.ProvDocument` per scenario run — dataset preparation, generator fitting, the three simulation runs (all sharing the same fixed real test set), predictive-cooperation-profile computation, and report generation are `prov:Activity` nodes; datasets, results, and the report are `prov:Entity` nodes; CoInfoSim is the sole `prov:SoftwareAgent`. The legacy `provenance.jsonld` format is no longer produced by new runs but remains a publisher fallback for older, already-published scenario runs. The `coinfosim:*` types used throughout ([`coinfosim:PredictiveCooperationProfile`](ontology/coinfosim.owl.ttl), [`coinfosim:RealSimulationRun`](ontology/coinfosim.owl.ttl), etc.) are formally declared by [`ontology/coinfosim.owl.ttl`](ontology/coinfosim.owl.ttl), a small OWL 2 ontology that specializes PROV-O and is published alongside the reports at `ontology/coinfosim.owl.ttl` on GitHub Pages. See [`docs/semantics/predictive_cooperation_vocabulary.md`](docs/semantics/predictive_cooperation_vocabulary.md) and [`docs/semantics/provenance_mapping.md`](docs/semantics/provenance_mapping.md).

## Development installation

```bash
git clone https://github.com/paulorenatoaz/coinfosim.git
cd coinfosim
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This installs CoInfoSim in editable mode with the `dev` extra (pytest, ruff, mypy, `build`, `twine`). Run the test suite with `pytest` (add `-m "not slow"` to skip the real-mode wheel-install and end-to-end integration tests). See [`docs/cli.md`](docs/cli.md) for the full command reference, [`docs/datasets.md`](docs/datasets.md) for dataset internals, [`docs/installation.md`](docs/installation.md) for detailed install paths, [`docs/releasing.md`](docs/releasing.md) for the release process, and [`docs/migration-cli-0.2.md`](docs/migration-cli-0.2.md) if you previously ran the `scripts/run_*.py` reference scripts directly.

## Citation

If you use CoInfoSim in published research, please cite this repository. See [CITATION.cff](CITATION.cff).

## License

CoInfoSim is distributed under the GNU General Public License v3.0 (GPL-3.0), inherited from the SLACGS project from which it is derived. See the [LICENSE](LICENSE) file for the full text. Note that the GPL-3.0 license applies to the CoInfoSim *software*; dataset licenses/acknowledgment requirements are tracked separately (see [Dataset download and cache behavior](#dataset-download-and-cache-behavior)).

## Generative AI disclosure

This project used the generative-AI models GPT-5.6 and Claude Sonnet 5 as supporting tools during research and development. Methodological decisions, experimental validation, scientific interpretation, and responsibility for the final content remained with the author.

---

## Scientific framework

This section documents the modeling framework, research plan, and reporting structure in depth. It assumes you have already read the practical usage sections above.

### Modeling framework

Let

$$
X = (X_1, \ldots, X_d)
$$

be a **standardized** input vector, where each component $X_j$ is an **information channel**, and let $Y \in \{1, \ldots, K\}$ be the class label. An information channel may be a sensor reading, a sensor-derived variable, a laboratory or contextual measurement, an engineered feature, or any other standardized observable variable available to a classifier. Multi-channel sensing systems are an important *motivating* application, but the formulation is intentionally broader.

In the parametric Gaussian mode, each class is described by its own class-conditional distribution:

$$
X_c \sim \mathcal{N}(\mu_c, \Sigma_c), \qquad c = 1, \ldots, K.
$$

The simulator receives the class centers and covariance matrices directly, as

$$
\{(\mu_c, \Sigma_c)\}_{c=1}^{K}.
$$

Unlike the previous restricted formulation, CoInfoSim does **not** require covariance matrices to be equal across classes; it allows

$$
\Sigma_0 \neq \Sigma_1,
$$

so differences in location, dispersion, correlation, or distributional shape may all affect classification.

A **channel subset** is denoted $A \subseteq \{1, \ldots, d\}$, and the classifier observes only the restricted vector $X_A$. For $d = 3$, the simulator evaluates all $2^3 - 1 = 7$ non-empty subsets — three isolated channels, three pairs, and the full three-channel set.

#### Balanced sampling and standardization

The initial protocol uses **class-balanced** sampling: $n$ always denotes the number of labeled training samples *per class*. Synthetic simulations generate $n$ samples per class; real-data simulations draw balanced subsets from the training reservoir. When sampling without replacement from real data, the feasible maximum $n$ is limited by the smallest class in the training reservoir. CoInfoSim operates on **standardized** channels by default, which matters for scale-sensitive classifiers such as Linear SVM and Logistic Regression. In dataset-anchored mode, the estimated parameters $\hat{\mu}_c$ and $\hat{\Sigma}_c$ are computed from standardized data.

### Core empirical object

The central quantity is the **Monte Carlo average classification loss**

$$
\overline{L}_{A,f}(n),
$$

the average loss for channel subset $A$, classifier $f$, and $n$ labeled training samples per class. For replication $r$,

$$
\widehat{L}_{A,f,r}(n) = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\{\widehat{Y}^{(A,f)}_{i,r}(n) \neq Y_i\},
\qquad
\overline{L}_{A,f}(n) = \frac{1}{R_n} \sum_{r=1}^{R_n} \widehat{L}_{A,f,r}(n),
$$

where the number of repetitions $R_n$ may depend on $n$ (more repetitions for smaller samples, fewer once estimates stabilize).

Two derived quantities are central. The **best subset** for a classifier is

$$
A_f^\star(n) = \arg\min_A \overline{L}_{A,f}(n),
$$

For subsets $A$ and $B$, the **pairwise winner relation** $W_{A,B}(n)$ records which of the two subsets currently has the lower average loss at $n$; an exact tie after a pair's first strict winner carries the previous winner forward instead of erasing it. A **valid winner reversal** occurs at $n$ when the effective winner is defined at both the previous and the current evaluated sample size and the winner differs between the two. The **reversal matrix** $R_{A,B}$ stores the sample size of the last observed valid reversal through the evaluated prefix, and a cell $R_{A,B}$ exists **only** for pairs that have undergone at least one such reversal — not merely because one subset currently wins.

Two separate metrics compare $R$ across training-condition arms (real, single-Gaussian synthetic, GMM synthetic — every arm evaluated on the same fixed real test set): **reversal existence agreement**, the Jaccard agreement over which unordered pairs have a defined reversal, and **reversal sample-size similarity**, the normalized agreement between the reversal sample sizes for the pairs shared by both arms. These two metrics are always reported separately; no composite/product metric is computed.

### Initial classifiers

The initial classifier set is intentionally small and interpretable:

- **Linear SVM** — preserves continuity with the previous framework and provides a margin-based linear baseline.
- **Logistic Regression** — a lightweight probabilistic linear baseline.
- **Gaussian Naive Bayes** — a probabilistic baseline whose conditional-independence assumption helps distinguish cooperative gains that arise from marginal evidence accumulation from gains that depend on multivariate dependence.

These three remain the historical default set. The SUPPORT2 protocol explicitly opts into a separately calibrated Random Forest; that opt-in does not change the defaults for synthetic, Occupancy, or Air Quality runs. Future classifier sets may include RBF SVM, kNN, gradient boosting, or neural networks, added once their scientific protocols are approved.

### Research plan

CoInfoSim is organized into three phases.

#### Phase 1 — Idealized Synthetic Multi-Channel Scenarios

Controlled experiments using manually specified Gaussian simulation models, each defined by class centers and covariance matrices. The initial focus is binary classification with three standardized channels. This phase studies additive gains from weak but complementary channels, redundancy among strong channels, channel-subset ranking as $n$ grows, predictive cooperation profiles and pairwise winner reversals, and differences among the initial classifiers.

#### Phase 2 — Dataset-Anchored Multi-Channel Simulation

Real datasets are used to build dataset-anchored scenarios. Each implemented dataset provides an explicit scientific protocol, prepared training/test data, and dataset-specific reporting text. The standard three-arm design compares balanced training from the real standardized reservoir, a training-fitted single Gaussian per class, and a training-fitted class-conditional GMM. Every arm is evaluated on the same fixed real test set. The dataset-anchored scenario report compares

$$
\overline{L}^{\,real}_{A,f}(n), \quad
\overline{L}^{\,Gaussian}_{A,f}(n), \quad \text{and} \quad
\overline{L}^{\,GMM}_{A,f}(n),
$$

assessing whether fitted synthetic training distributions preserve the cooperative patterns observed under fixed real-data evaluation.

#### Phase 3 — Cost-Aware Channel Selection

A generic **channel cost** $C_X(A)$ and a **label/reference cost** $C_Y(n)$ are introduced, with total

$$
C(A, n) = C_X(A) + C_Y(n).
$$

Channel cost is treated generically and may include acquisition, deployment, calibration, maintenance, computation, latency, energy, or logistical burden. This supports constrained decisions such as minimizing $C(A,n)$ subject to $\overline{L}_{A,f}(n) \le L_{\max}$, or penalized objectives $\overline{L}_{A,f}(n) + \lambda C_X(A) + \gamma C_Y(n)$.

### Key concepts and vocabulary

| Term | Meaning |
|---|---|
| Information channel | A component $X_j$ of the input vector $X$ |
| Channel subset $A$ | A subset $A \subseteq \{1,\ldots,d\}$ observed by the classifier |
| Simulation model | The object holding the class-conditional parameters $\{(\mu_c, \Sigma_c)\}$ |
| Scenario | One or more simulation models grouped around an experimental question |
| Scenario grid | A scenario whose models come from a parameter grid |
| Simulation report | Report for one simulation model |
| Scenario report | Aggregated report for one scenario |
| Dataset report | Report describing a real dataset used in dataset-anchored experiments |

### Reporting structure

The reporting system is layered and dataset-aware. Automated reports display the parameters explicitly defined in each simulation model together with generic loss, ranking, and predictive cooperation profile (winner/reversal) analyses; dataset-specific provenance and interpretation remain explicit rather than being inferred from arbitrary input files. The structure is:

1. **Project index** — published entry point listing scenario, dataset, and simulation reports, result files, generation date, and software/commit version.
2. **Dataset report** — dataset name and source, target $Y$, class distribution, candidate and selected channels, preprocessing and standardization, training reservoir/test set, and visual diagnostics of standardized data.
3. **Simulation report** — model id and source, class centers and covariance matrices, evaluated subsets, classifiers, values of $n$, repetitions/convergence, loss curves $\overline{L}_{A,f}(n)$, subset rankings, thresholds $N^*$, and data-geometry visualizations.
4. **Scenario report** — the scenario question, a table of simulation models, links to simulation reports, aggregate loss curves, threshold summaries, best-subset maps, scenario-grid heatmaps, and animations of geometry across the scenario.
5. **Dataset-anchored scenario report** — links to the dataset report and three arm reports, real-versus-synthetic loss curves and rankings, and separate ranking-fidelity, Winner Agreement, reversal existence agreement, and reversal sample-size similarity metrics.

Implemented dataset-anchored reports include real and synthetic projection panels, loss curves $\overline{L}_{A,f}(n)$, channel-subset rankings, paired winner ($W$) and reversal ($R$) matrices, and separate structural-fidelity curves. Broader grids and animations remain future work.

### Real-world motivating cases

The dataset-anchored phase uses small, interpretable real-data studies with a few selected channels:

- **Occupancy detection** — implemented with Temperature, Humidity, Light, CO₂, and HumidityRatio.
- **Air quality** — implemented with five PT08 metal-oxide sensor responses and a benzene reference used only to construct the target.
- **SUPPORT2 180-day mortality** — implemented with seven baseline physiologic channels and a fixed endpoint derived from `death` and `d.time`.
- **Water potability** — channels such as pH, conductivity, and turbidity (future work).
- **Hydraulic systems / condition monitoring** — multiple physical or operational measurements for fault classification (future work).

The first three pipelines are implemented; the remaining examples are motivations, not plug-and-play dataset support. CoInfoSim does not claim arbitrary CSV plug-and-play support — a new dataset requires an explicit validated loader, execution spec, model builders, and dataset-specific report text.

### Relationship to SLACGS and CoSenSim

CoInfoSim evolves from SLACGS, which studied cooperative gains in synthetic Gaussian settings and was organized primarily around dimensionality comparisons (lower- versus higher-dimensional models). CoInfoSim generalizes that idea: the object of comparison is no longer dimension $d$ but the channel subset $A \subseteq \{1,\ldots,d\}$, and the framing moves from sensor networks to multi-channel classification. The intermediate CoSenSim stage carried this toward real-data-anchored sensor studies; CoInfoSim completes the generalization to information channels. The reproducible Monte Carlo machinery, nested sample growth, adaptive repetition, and scenario-level reporting from SLACGS are retained.

### Repository structure

```text
docs/          # Documentation, design notes, and planned-architecture descriptions
data/          # Tracked raw dataset files and provenance (mirrored on GitHub Pages at runtime)
src/coinfosim/ # Installable package: cli/, datasets/, scenarios/, publish/, simulation/, reports/, ...
scripts/       # Deprecated compatibility wrappers around the scenario service; prefer the CLI
tests/         # Unit, integration, and CLI test suite
```

### SUPPORT2 Random Forest calibration (maintainer operation)

The SUPPORT2 scenario's Random Forest configuration is calibrated once using only the real training reservoir and frozen at `config/calibration/support2_random_forest.json`, then reused unchanged across arms, sample sizes, subsets, and replications. Calibration never runs as part of a normal scenario command, and there is no fallback when the artifact is missing, invalid, stale, or incompatible.

Recalibrating the approved protocol is a maintainer-only operation, separate from the public CLI:

```bash
python scripts/calibrate_support2_random_forest.py \
  --raw-dir data/raw/support2 \
  --output config/calibration/support2_random_forest.json \
  --calibration-seed 0
```

Existing artifacts require `--force` to overwrite. Validation checks the raw file SHA-256, training-partition fingerprint, target, channel order, split seed, estimator parameters, enforced `n_jobs=1`, and scikit-learn compatibility. During Monte Carlo execution, Random Forest uses the versioned `classifier_seed_v1` policy: its estimator seed depends only on the base seed, a stable classifier namespace, and replication ID, so the same replication seed is used across every arm, sample size, and feature subset. Linear SVM retains its historical `random_state=0`.

### Extending dataset-anchored scenarios

Future datasets require an explicit implementation. The extension workflow is:

1. Implement a validated loader and prepared-data object with fixed train/test semantics.
2. Provide the dataset/scenario execution specification and model builders in `src/coinfosim/scenarios/definitions/`.
3. Add the pinned file hashes to `src/coinfosim/resources/datasets.json` and provenance to `data/raw/<dataset>/README.md`.
4. Provide dataset-specific report text, provenance, diagnostics, limitations, and interpretation.
5. Reuse the generic dataset-anchored runner and report cores for execution, persistence, structural analysis, and regeneration.

### Follow-up

Cost-aware optimization, additional explicitly implemented datasets, broader reviewed experiment grids, and publication automation remain future work.
