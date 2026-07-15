# CoInfoSim: A Simulator for Cooperative Classification from Multiple Information Channels

CoInfoSim is a research simulator for evaluating **cooperative advantage among information channels** in supervised classification tasks. It studies when a *subset* of channels provides a measurable advantage over isolated channels, redundant pairs, or simpler subsets, and how many labeled samples are needed before that advantage appears.

CoInfoSim is a conceptual evolution of the earlier **SLACGS** and **CoSenSim** lines of work. It preserves their incremental Monte Carlo protocol, reproducible sample generation, adaptive repetition logic, and scenario-based reporting structure, while reformulating the scientific object from sensor-network dimensionality to multi-channel classification.

> **Status:** Active research project. The repository supports idealized synthetic experiments and reproducible dataset-anchored scenarios for Occupancy Detection, UCI Air Quality, and SUPPORT2 180-day mortality. Dataset scenarios compare real, single-Gaussian synthetic, and class-conditional GMM synthetic training on one fixed real evaluation set.

## Core research question

> When does cooperation among information channels improve supervised classification?

Rather than asking only whether a single channel is informative, CoInfoSim evaluates when combining channels yields lower classification loss, when an apparently useful channel is redundant, and when a weaker channel is valuable because it complements stronger ones.

## Modeling framework

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

### Balanced sampling and standardization

The initial protocol uses **class-balanced** sampling: $n$ always denotes the number of labeled training samples *per class*. Synthetic simulations generate $n$ samples per class; real-data simulations draw balanced subsets from the training reservoir. When sampling without replacement from real data, the feasible maximum $n$ is limited by the smallest class in the training reservoir. CoInfoSim operates on **standardized** channels by default, which matters for scale-sensitive classifiers such as Linear SVM and Logistic Regression. In dataset-anchored mode, the estimated parameters $\hat{\mu}_c$ and $\hat{\Sigma}_c$ are computed from standardized data.

## Core empirical object

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

and the **cooperative advantage threshold** between subsets $A$ and $B$ is

$$
N^*(A, B; f) = \min\{n : \overline{L}_{B,f}(n) < \overline{L}_{A,f}(n)\},
$$

the smallest sample size per class at which subset $B$ first achieves lower average loss than subset $A$.

## Initial classifiers

The initial classifier set is intentionally small and interpretable:

- **Linear SVM** — preserves continuity with the previous framework and provides a margin-based linear baseline.
- **Logistic Regression** — a lightweight probabilistic linear baseline.
- **Gaussian Naive Bayes** — a probabilistic baseline whose conditional-independence assumption helps distinguish cooperative gains that arise from marginal evidence accumulation from gains that depend on multivariate dependence.

Future classifier sets may include RBF SVM, kNN, Random Forest, gradient boosting, or neural networks, added once the Monte Carlo and reporting protocols are stable.

## Research plan

CoInfoSim is organized into three phases.

### Phase 1 — Idealized Synthetic Multi-Channel Scenarios

Controlled experiments using manually specified Gaussian simulation models, each defined by class centers and covariance matrices. The initial focus is binary classification with three standardized channels. This phase studies additive gains from weak but complementary channels, redundancy among strong channels, channel-subset ranking as $n$ grows, cooperative advantage thresholds $N^*$, and differences among the initial classifiers.

### Phase 2 — Dataset-Anchored Multi-Channel Simulation

Real datasets are used to build dataset-anchored scenarios. Each implemented dataset provides an explicit scientific protocol, prepared training/test data, and dataset-specific reporting text. The standard three-arm design compares balanced training from the real standardized reservoir, a training-fitted single Gaussian per class, and a training-fitted class-conditional GMM. Every arm is evaluated on the same fixed real test set. The dataset-anchored scenario report compares

$$
\overline{L}^{\,real}_{A,f}(n), \quad
\overline{L}^{\,Gaussian}_{A,f}(n), \quad \text{and} \quad
\overline{L}^{\,GMM}_{A,f}(n),
$$

assessing whether fitted synthetic training distributions preserve the cooperative patterns observed under fixed real-data evaluation.

### Phase 3 — Cost-Aware Channel Selection

A generic **channel cost** $C_X(A)$ and a **label/reference cost** $C_Y(n)$ are introduced, with total

$$
C(A, n) = C_X(A) + C_Y(n).
$$

Channel cost is treated generically and may include acquisition, deployment, calibration, maintenance, computation, latency, energy, or logistical burden. This supports constrained decisions such as minimizing $C(A,n)$ subject to $\overline{L}_{A,f}(n) \le L_{\max}$, or penalized objectives $\overline{L}_{A,f}(n) + \lambda C_X(A) + \gamma C_Y(n)$.

## Key concepts and vocabulary

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

## Reporting structure

The reporting system is layered and dataset-aware. Automated reports display the parameters explicitly defined in each simulation model together with generic loss, ranking, N-star, and structural-fidelity analyses; dataset-specific provenance and interpretation remain explicit rather than being inferred from arbitrary input files. The structure is:

1. **Project index** — published entry point listing scenario, dataset, and simulation reports, result files, generation date, and software/commit version.
2. **Dataset report** — dataset name and source, target $Y$, class distribution, candidate and selected channels, preprocessing and standardization, training reservoir/test set, and visual diagnostics of standardized data.
3. **Simulation report** — model id and source, class centers and covariance matrices, evaluated subsets, classifiers, values of $n$, repetitions/convergence, loss curves $\overline{L}_{A,f}(n)$, subset rankings, thresholds $N^*$, and data-geometry visualizations.
4. **Scenario report** — the scenario question, a table of simulation models, links to simulation reports, aggregate loss curves, threshold summaries, best-subset maps, scenario-grid heatmaps, and animations of geometry across the scenario.
5. **Dataset-anchored scenario report** — links to the dataset report and three arm reports, real-versus-synthetic loss curves and rankings, N-star comparisons, and separate ranking-fidelity, winner-agreement, and progressive N-star-similarity metrics.

Implemented dataset-anchored reports include real and synthetic projection panels, loss curves $\overline{L}_{A,f}(n)$, channel-subset rankings, cooperative advantage thresholds $N^*$, winner matrices, progressive N-star matrices, and separate structural-fidelity curves. Broader grids and animations remain future work.

## Real-world motivating cases

The dataset-anchored phase uses small, interpretable real-data studies with a few selected channels:

- **Occupancy detection** — implemented with Temperature, Humidity, Light, CO₂, and HumidityRatio.
- **Air quality** — implemented with five PT08 metal-oxide sensor responses and a benzene reference used only to construct the target.
- **SUPPORT2 180-day mortality** — implemented with seven baseline physiologic channels and a fixed endpoint derived from `death` and `d.time`.
- **Water potability** — channels such as pH, conductivity, and turbidity.
- **Hydraulic systems / condition monitoring** — multiple physical or operational measurements for fault classification (future work).

The first three pipelines are implemented; the remaining examples are motivations, not plug-and-play dataset support.

## Relationship to SLACGS and CoSenSim

CoInfoSim evolves from SLACGS, which studied cooperative gains in synthetic Gaussian settings and was organized primarily around dimensionality comparisons (lower- versus higher-dimensional models). CoInfoSim generalizes that idea: the object of comparison is no longer dimension $d$ but the channel subset $A \subseteq \{1,\ldots,d\}$, and the framing moves from sensor networks to multi-channel classification. The intermediate CoSenSim stage carried this toward real-data-anchored sensor studies; CoInfoSim completes the generalization to information channels. The reproducible Monte Carlo machinery, nested sample growth, adaptive repetition, and scenario-level reporting from SLACGS are retained.

## Repository structure

```
docs/          # Documentation, design notes, and planned-architecture descriptions
data/          # Raw dataset files and provenance; Air Quality and SUPPORT2 CSVs are committed
experiments/   # Experiment manifests and scripts (planned)
src/coinfosim/ # Main package (inherited functional code; reformulation in progress)
tests/         # Smoke and unit tests
```

The package namespace and CLI are `coinfosim`.

## Installation (developer quick-start)

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Basic usage

The CLI entry point is `coinfosim`:

```bash
coinfosim --help
```

Dataset-anchored scenarios are run through their dataset-specific scripts. The supported sample-size presets are:

| Mode | `n_per_class` values |
|---|---|
| `smoke` | `2, 4, 8, 16, 32` |
| `fast` | `2, 4, 8, 16, 32, 64, 128` |
| `full` | `2, 4, 8, 16, 32, 64, 128, 256, 512` |
| `full-scale` | Powers of two from 2 through the largest power of two not exceeding the training minority-class count; same Monte Carlo precision/budget as `full` |
| `strict` | `2, 4, 8, 16, 32, 64, 128, 256, 512` |

Only `smoke` is used by the documented validation workflow. `fast`, `full`, `full-scale`, and `strict` are never run automatically; use them only after explicit scientific review and after checking the real training reservoir's minority-class capacity. `full-scale` may be substantially more expensive than `full`, and all three dataset-anchored arms share its single resolved sample-size grid.

## SUPPORT2 180-day mortality scenario

The SUPPORT2 scenario predicts death within 180 days after study entry. Its
fixed derived target is:

```python
death_180d = ((death == 1) & (d_time <= 180)).astype(int)
```

where the raw source field is `d.time` and day 180 is inclusive. `hospdead` is
neither the primary target nor a predictor. Complete cases are required for
`meanbp`, `hrt`, `resp`, `temp`, `wblc`, `crea`, `sod`, `death`, `d.time`, and
`dzgroup`, yielding 8,873 patients. A fixed 80/20 split uses
`random_state=0`, joint `death_180d × dzgroup` stratification, and ascending ID
ordering within partitions. Z-score parameters use training rows only with
`ddof=0`.

All 127 non-empty subsets of the seven channels are evaluated with Linear SVM,
Logistic Regression, and Gaussian Naive Bayes under Real → Real, Single
Gaussian → Real, and GMM → Real. Every arm reuses the same fixed real test set.

Run the approved smoke workflow:

```bash
.venv/bin/python scripts/run_support2_scenario.py --mode smoke
```

Regenerate the dataset-linked three-arm report hierarchy from persisted results
without rerunning Monte Carlo:

```bash
.venv/bin/python scripts/run_support2_scenario.py \
  --report-from-scenario-run <SCENARIO_RUN_ID>
```

The scenario directory contains the SUPPORT2 dataset report, consolidated
scenario report, exact split manifest, target metadata, preprocessing metadata,
and `scenario.json`. Each simulation directory contains its arm report,
compressed result payload, summary, and `simulation.json`. Production/full-mode
experiments are intentionally reserved for the repository owner after smoke
validation.

## UCI Air Quality scenario

The Air Quality scenario tests whether synthetic training distributions preserve cooperative channel-subset structure when every arm is evaluated on the same fixed future real observations. It uses the official UCI Air Quality dataset, DOI `10.24432/C59K5F`. The original `AirQualityUCI.csv` is committed at `data/raw/air_quality/`; provenance and its SHA-256 are recorded in that directory's README.

Classifier input is restricted to these five sensor-response channels:

- `PT08.S1(CO)`
- `PT08.S2(NMHC)`
- `PT08.S3(NOx)`
- `PT08.S4(NO2)`
- `PT08.S5(O3)`

`C6H6(GT)` is a benzene reference used only to define the binary target and is excluded from classifier input. After complete-case filtering, rows are split chronologically: the first 80% form the training reservoir and the last 20% form the fixed future real test set. The target threshold is the training-only 75th percentile of `C6H6(GT)`, with positive label `C6H6(GT) >= threshold`. Z-score parameters are fitted on training only with `ddof=0`.

The scenario evaluates all 31 non-empty channel subsets and the same three classifiers under:

- Real → Real
- Single Gaussian → Real
- GMM → Real

Run the approved smoke scenario:

```bash
.venv/bin/python scripts/run_air_quality_scenario.py --mode smoke
```

Regenerate reports and graphs from a persisted scenario without rerunning Monte Carlo:

```bash
.venv/bin/python scripts/run_air_quality_scenario.py \
  --report-from-scenario-run <SCENARIO_RUN_ID>
```

Run outputs are immutable, ID-addressed directories with shared registries:

```text
output/reports/scenario_runs.json
output/reports/simulation_runs.json
output/reports/scenarios/<scenario-id>_air_quality_baseline_<mode>/
output/reports/simulations/<simulation-id>_air_quality_<arm>_<mode>/
```

The scenario directory contains the dataset report, scenario report, `scenario.json`, projection images, and analytical graphs. Each of the three simulation directories contains its arm report, compressed result payload, summary, `simulation.json`, and exported diagnostic tables.

## Occupancy Detection scenario

Occupancy remains fully supported. Place its UCI raw files at:

```text
data/raw/occupancy/datatraining.txt
data/raw/occupancy/datatest.txt
data/raw/occupancy/datatest2.txt
```

Run the smoke scenario with:

```bash
.venv/bin/python scripts/run_occupancy_scenario.py --mode smoke
```

Smoke mode uses the current `2, 4, 8, 16, 32` preset and evaluates all 31 non-empty subsets of Temperature, Humidity, Light, CO2, and HumidityRatio with Linear SVM, Logistic Regression, and Gaussian Naive Bayes. Its runner uses the same generic three-arm execution, persistence, registry, and scenario-report cores as Air Quality while preserving the Occupancy public wrappers.

## Extending dataset-anchored scenarios

Future datasets require an explicit implementation; CoInfoSim does not claim arbitrary CSV plug-and-play support. The extension workflow is:

1. Implement a validated loader and prepared-data object with fixed train/test semantics.
2. Provide the dataset/scenario execution specification and model builders.
3. Provide dataset-specific report text, provenance, diagnostics, limitations, and interpretation.
4. Reuse the generic dataset-anchored runner and report cores for execution, persistence, structural analysis, and regeneration.

## Citation

If you use CoInfoSim in published research, please cite this repository. See [CITATION.cff](CITATION.cff).

## License

CoInfoSim is distributed under the GNU General Public License v3.0 (GPL-3.0), inherited from the SLACGS project from which it is derived. See the [LICENSE](LICENSE) file for the full text.

## Contributing / development notes

- This is an academic research project under active development.
- Keep pull requests small and focused; propose larger refactors in an issue first.
- Preserve reproducibility: experiments should record random seeds and environment details.

## Follow-up

Cost-aware optimization, additional explicitly implemented datasets, broader reviewed experiment grids, and publication automation remain future work.
