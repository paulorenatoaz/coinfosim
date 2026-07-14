# CoInfoSim report improvement tasks

Repository:

```text
https://github.com/paulorenatoaz/coinfosim
```

Working goal: redesign and regenerate the CoInfoSim Occupancy reports using already persisted simulation/scenario data. This is a report-generation project, not a Monte Carlo simulation project.

Expected final report family:

1. `Occupancy Dataset Report`
2. `Occupancy Scenario Report`
3. `Occupancy Real → Real Monte Carlo Report`
4. `Occupancy Single Gaussian → Real Monte Carlo Report`
5. `Occupancy GMM → Real Monte Carlo Report`

The reports should become a coherent academic report family. The scenario report is the main comparative scientific report. The dataset report documents provenance and preprocessing. The three Monte Carlo reports are arm-specific technical appendices.

---

## 0. Non-negotiable constraints

### 0.1 Do not rerun simulations for report-design work

Do **not** run commands such as:

```bash
python scripts/run_occupancy_scenario.py --mode smoke
python scripts/run_occupancy_scenario.py --mode fast
python scripts/run_occupancy_scenario.py --mode full
python scripts/run_occupancy_scenario.py --mode strict
```

unless the user explicitly authorizes a new simulation run.

The report-design loop should use existing completed scenario/simulation runs and regenerate HTML reports from persisted `result_data_*.json.gz` files.

Primary validation command:

```bash
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

This command is already intended to reload persisted simulation result data and re-render the reports without rerunning Monte Carlo.

If report regeneration is broken or incomplete, fix the report-regeneration path. Do not solve report-generation problems by rerunning simulations.

### 0.2 Ask before running any new simulation

If you believe a structural change cannot be validated without a new Monte Carlo run, stop and ask the user first.

Use a message like:

```text
I can validate this with existing persisted results. A fresh smoke simulation is not necessary.

or

I found a structural issue that may require a fresh smoke simulation to validate. Do you authorize running `python scripts/run_occupancy_scenario.py --mode smoke`?
```

Default assumption: fresh simulations are not needed.

### 0.3 Keep tests narrow

Avoid broad test runs that consume unnecessary time/credits. Prefer:

```bash
python -m compileall src/coinfosim/reports scripts/run_occupancy_scenario.py
python scripts/run_occupancy_scenario.py --list-scenario-runs
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

Optionally run only targeted report-related tests if they already exist, for example:

```bash
pytest -q -k "report or regeneration or occupancy"
```

Do **not** run the full test suite unless needed and authorized.

### 0.4 Stop after each block

Execute exactly one block at a time.

At the end of each block:

1. regenerate the relevant reports;
2. list the generated report paths;
3. summarize what changed;
4. ask the user for approval before continuing.

Do not proceed to the next block without explicit user confirmation.

### 0.5 Do not commit or push

Do not run:

```bash
git add
git commit
git push
git tag
```

The user will handle version control manually.

---

## 1. Current repo anchors

Start from these files. Do not spend much time rediscovering repository context.

### Report generation

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_monte_carlo.py
src/coinfosim/reports/occupancy_scenario.py
src/coinfosim/reports/occupancy_dataset.py
src/coinfosim/reports/scenario_visualization.py
```

### Data loading and provenance

```text
src/coinfosim/datasets/occupancy.py
```

Important current behavior:

- `datatraining.txt` is the finite training pool.
- `datatest.txt` and `datatest2.txt` are concatenated into one fixed real test set.
- Standardization parameters are estimated from the training pool only and then applied to both train and test.
- Occupancy channels are:
  - `Temperature`
  - `Humidity`
  - `Light`
  - `CO2`
  - `HumidityRatio`
- Target variable:
  - `Occupancy`

### Monte Carlo and persisted results

```text
src/coinfosim/simulation/monte_carlo.py
src/coinfosim/results/summary.py
src/coinfosim/results/analysis.py
src/coinfosim/results/persistence.py
src/coinfosim/runs/report_data.py
scripts/run_occupancy_scenario.py
```

Important current behavior:

- `scripts/run_occupancy_scenario.py --report-from-scenario-run <ID>` loads persisted `result_data_*.json.gz` for the three arms and re-renders reports.
- The current Monte Carlo report renderer is generic and debug-like.
- The current scenario report is already closer to the desired academic layout.
- The dataset report exists but needs stronger provenance/reproducibility sections.

### Current three arms

Use these scientific names consistently:

```text
Real → Real
Single Gaussian → Real
GMM → Real
```

Current code slugs/families may include:

```text
occupancy_real_data
occupancy_single_gaussian_to_real
occupancy_gmm_to_real
```

---

## 2. External dataset provenance to encode

The Occupancy data should be described as:

```text
Dataset name: Occupancy Detection
Repository: UCI Machine Learning Repository
UCI dataset ID: 357
DOI: 10.24432/C5X01N
Donor/creator: Luis Candanedo
Donation date: 2016-02-28
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Associated paper:
  L. Candanedo and V. Feldheim.
  "Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models."
  Energy and Buildings, Volume 112, 2016.
UCI URL:
  https://archive.ics.uci.edu/dataset/357/occupancy+detection
```

The UCI page describes the dataset as experimental data for binary room-occupancy classification using temperature, humidity, light, and CO2, with ground-truth occupancy obtained from timestamped pictures taken every minute.

The report should also mention the three distributed files:

```text
datatraining.txt
datatest.txt
datatest2.txt
```

In this project:

```text
datatraining.txt              -> training pool
datatest.txt + datatest2.txt  -> fixed real evaluation split
```

---

## 3. Global report-design principles

### 3.1 Preserve original loss scale

Do not statistically normalize `test_loss`. It is directly interpretable as empirical misclassification rate.

Use either:

```text
mean_loss ± SE
```

or:

```text
mean_loss ± CI95
```

with clear labels.

The thing to normalize is the visual structure, not the data scale.

### 3.2 SVM first

The reports should be oriented primarily toward `Linear SVM`.

Classifier ordering everywhere:

```text
1. Linear SVM
2. Logistic Regression
3. Gaussian Naive Bayes
```

The default selected classifier in any carousel/tab/selector should be `Linear SVM`.

Other classifiers are secondary but fully available.

### 3.3 Classifier should be the last selector layer

Whenever several dimensions are nested, `classifier` should be the last selector layer.

Examples:

```text
Top-ranked subsets:
  arm > classifier

Best subset graphs:
  arm

Within-arm N-star:
  best k-channel subset > classifier

Simulation loss curves:
  subset dimension group > classifier
```

### 3.4 Do not repeat structural variables as table columns

If a table repeats the same structural variable many times, turn that variable into report structure instead.

Examples:

```text
arm              -> section / tab / carousel layer
classifier       -> final selector layer
subset_size      -> selector / group
best k-subset    -> selector / group
```

Avoid long repeated columns such as:

```text
Temperature+Humidity+Light+CO2+HumidityRatio
```

Use compact subset notation:

```text
{X1}
{X1, X3}
{X1, X2, X3, X4, X5}
```

Keep a channel legend:

```text
X1 = Temperature
X2 = Humidity
X3 = Light
X4 = CO2
X5 = HumidityRatio
```

For display, render `CO2` as `CO₂` and `HumidityRatio` as `Humidity Ratio` where appropriate.

---

## 4. Expected final report structures

### 4.1 Dataset Report

Purpose: provenance, dataset structure, split protocol, preprocessing, reproducibility.

Expected sections:

```text
1. Dataset identity and public provenance
2. Citation and external source
3. Local source files and file hashes
4. Role of each file in the experiment
5. Train/test protocol
6. Feature dictionary and target variable
7. Row counts and class distribution
8. Raw channel statistics
9. Standardization protocol
10. Standardized channel statistics
11. Leakage-control notes
12. Reproducibility metadata
```

Must include SHA-256 hashes for:

```text
datatraining.txt
datatest.txt
datatest2.txt
```

### 4.2 Scenario Report

Purpose: main comparative scientific report.

Expected sections:

```text
1. Scientific question
2. Scenario summary
3. Dataset provenance summary
4. Methodological controls of the comparative experiment
5. Experimental protocol
6. Data and training-distribution visualizations
7. Best subset comparison
8. Top-ranked subsets
9. N-star diagnostics
10. Monte Carlo precision diagnostics
11. Robustness analysis
12. Validity discussion
13. Methodological limitations
14. Interpretation notes
15. Technical appendices / exported tables
```

The scenario report must describe the experiment as a comparative experiment among:

```text
Real → Real:
  empirical baseline

Single Gaussian → Real:
  simple class-conditional parametric approximation

GMM → Real:
  more flexible class-conditional parametric approximation,
  capable of representing multimodality
```

All three arms must be described as evaluated on the same fixed real test split.

### 4.3 Real → Real Monte Carlo Report

Purpose: technical appendix for the empirical baseline.

Expected sections:

```text
1. Arm summary
2. Scientific role of this arm
3. Dataset provenance summary
4. Real-data source description
5. Run configuration
6. Training/evaluation protocol
7. Reproducibility controls
8. Monte Carlo stopping rule
9. Monte Carlo precision diagnostics
10. Loss curves
11. Best subset by sample size
12. Final ranking at largest N
13. Within-arm N-star diagnostics
14. Robustness notes
15. Validity and limitations for this arm
16. Technical appendix / exported tables
```

Arm-specific description:

```text
Training source:
  balanced samples drawn from the standardized datatraining.txt pool

Evaluation source:
  fixed real test split = standardized datatest.txt + datatest2.txt

Randomness:
  deterministic balanced sampling by base_seed, class_label and replication_id

Role:
  empirical reference for observed cooperative structure
```

### 4.4 Single Gaussian → Real Monte Carlo Report

Purpose: technical appendix for the unimodal synthetic arm.

Expected sections:

```text
1. Arm summary
2. Scientific role of this arm
3. Dataset provenance summary
4. Run configuration
5. Training/evaluation protocol
6. Single-Gaussian training model
7. Reproducibility controls
8. Monte Carlo stopping rule
9. Monte Carlo precision diagnostics
10. Loss curves
11. Best subset by sample size
12. Final ranking at largest N
13. Within-arm N-star diagnostics
14. Robustness notes
15. Validity and limitations for this arm
16. Technical appendix / exported tables
```

Subsections for section 6:

```text
6.1 Estimation protocol
6.2 Class-conditional means
6.3 Class-conditional covariance matrices
6.4 Positive-definiteness / ridge notes, if applicable
```

Scientific role:

```text
This arm tests whether a simple unimodal class-conditional Gaussian approximation
is sufficient to preserve the cooperative structure observed under real evaluation.
```

### 4.5 GMM → Real Monte Carlo Report

Purpose: technical appendix for the multimodal synthetic arm.

Expected sections:

```text
1. Arm summary
2. Scientific role of this arm
3. Dataset provenance summary
4. Run configuration
5. Training/evaluation protocol
6. GMM training model
7. Reproducibility controls
8. Monte Carlo stopping rule
9. Monte Carlo precision diagnostics
10. Loss curves
11. Best subset by sample size
12. Final ranking at largest N
13. Within-arm N-star diagnostics
14. Robustness notes
15. Validity and limitations for this arm
16. Technical appendix / exported tables
```

Subsections for section 6:

```text
6.1 Model-selection protocol
6.2 Candidate component counts
6.3 Selected components by class
6.4 BIC/AIC tables
6.5 Mixture weights
6.6 Component means
6.7 Component covariance matrices
```

Use collapsible `<details><summary>...</summary>...</details>` blocks for large covariance matrices.

Scientific role:

```text
This arm tests whether a more flexible multimodal class-conditional approximation
better preserves real-data cooperative behavior than the single-Gaussian approximation.
```

---

## 5. Methodological-controls section for the scenario report

Add an explicit section:

```text
Methodological controls of the comparative experiment
```

This section must treat the scenario as a complete comparative experiment.

Required content:

### 5.1 Provenance and traceability

Explain:

- public dataset origin;
- local raw files;
- role of each file;
- row counts;
- class distribution;
- channels;
- target;
- standardization learned only from training pool;
- raw-file hashes where available.

### 5.2 Reproducibility controls

Register:

- dataset and source files;
- preprocessing rule;
- channel definitions;
- target variable;
- subset list;
- classifiers;
- metric;
- sample sizes;
- base seed;
- min/max replications;
- Monte Carlo batch size;
- stopping rule;
- arm-specific generator metadata:
  - real balanced sampler;
  - Gaussian class means/covariances;
  - GMM selected components and mixture parameters.

### 5.3 Traceability of estimates

Every reported loss estimate should be traceable by:

```text
scenario_id
arm
n_per_class
classifier
subset_id
subset_size
subset_label
replications
mean_loss
standard_error
ci95_low
ci95_high
ci95_half_width
stopping_reason
```

### 5.4 Experimental control

Distinguish:

```text
Fixed factors:
  fixed real test split
  empirical test loss
  classifier set
  channel set
  subset list
  Monte Carlo stopping rule

Manipulated factors:
  training arm
  n_per_class
  classifier
  channel subset

Sources of randomness:
  balanced real training sampling
  synthetic training generation
  Monte Carlo replications
  GMM fitting initialization, controlled by random_state
```

### 5.5 Quantification of uncertainty

Rename:

```text
Monte Carlo uncertainty summary
```

to:

```text
Monte Carlo precision diagnostics
```

or in Portuguese:

```text
Diagnóstico de precisão Monte Carlo
```

The section must report Monte Carlo precision, not general model uncertainty.

### 5.6 Robustness

Discuss:

- stability of best subsets over `n_per_class`;
- consistency across classifiers;
- consistency across arms;
- whether largest-N conclusions already appear at smaller N;
- whether synthetic arms preserve the same cooperative patterns as Real → Real.

### 5.7 Validity

Discuss:

```text
Internal validity:
  no test leakage, train-only standardization, fixed test set, controlled seeds.

Construct validity:
  empirical test loss and subset contrasts as operational measures of channel cooperation.

External validity:
  results are anchored in the Occupancy Detection dataset and should not be generalized automatically.

Statistical validity:
  SE, CI95, replication count and stopping status support interpretation.
```

### 5.8 Limitations

Mention:

- single dataset;
- fixed real evaluation split;
- original class imbalance;
- balanced training samples;
- synthetic models fitted from training pool only;
- strong normality assumption in Single Gaussian;
- BIC and component cap in GMM;
- only three classifiers;
- possible dependency on software versions / numerical environment.

---

## 6. Technical tables and exports

Create reusable helpers to produce two technical tables.

### 6.1 `full_loss_table`

Long clean format:

```text
scenario_id
arm
n_per_class
classifier
classifier_label
subset_id
subset_size
subset_label
mean_loss
std_loss
se_loss
ci95_low
ci95_high
ci95_half_width
replications
stopping_reason
```

Notes:

- preserve original `test_loss` scale;
- use `subset_label` in `{X1, X3}` notation;
- include `classifier_label`;
- include `scenario_id` and `arm` whenever available;
- use existing accumulator information;
- do not recompute simulations.

### 6.2 `subset_metadata_table`

Format:

```text
subset_id
subset_label
subset_size
has_temperature
has_humidity
has_light
has_co2
has_humidity_ratio
```

Optional display columns:

```text
sensor_label
x_label
```

Use this table to avoid repeating long sensor combinations in report bodies.

### 6.3 Export policy

For each report regeneration, save clean technical data next to the HTML when feasible:

```text
full_loss_table_<arm>.csv
subset_metadata_table.csv
precision_diagnostics_<arm>.csv
```

Do not make the HTML body a dump of all rows. Put full tables in appendices or separate CSV files.

---

## 7. Implementation blocks

Each block should end with regenerated reports and a stop for user review.

Use an existing scenario run ID selected from:

```bash
python scripts/run_occupancy_scenario.py --list-scenario-runs
```

Then regenerate using:

```bash
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

If there is no completed scenario run available locally, stop and ask the user what existing run directory or persisted result files should be used. Do not start a new simulation.

---

# Block 1 — Report-only workflow and shared report data helpers

## Goal

Make the report-only workflow reliable and create reusable data helpers for all later blocks.

## Files to edit

Likely files:

```text
scripts/run_occupancy_scenario.py
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_scenario.py
src/coinfosim/runs/report_data.py
src/coinfosim/results/summary.py
```

Optional new files:

```text
src/coinfosim/reports/report_tables.py
src/coinfosim/reports/html_components.py
```

## Tasks

1. Verify that `--report-from-scenario-run <ID>` regenerates:
   - Real → Real Monte Carlo report;
   - Single Gaussian → Real Monte Carlo report;
   - GMM → Real Monte Carlo report;
   - Scenario report.

2. If regeneration fails, fix it without rerunning Monte Carlo.

3. Add reusable helpers for:
   - subset notation `{X1, X2}`;
   - subset metadata table;
   - full loss table;
   - CI95 columns;
   - compact precision diagnostics by `n_per_class`;
   - stable classifier ordering with Linear SVM first.

4. Avoid changing simulator logic.

5. Avoid redesigning the visual layout deeply in this block. This block is infrastructure.

## Validation

Run:

```bash
python -m compileall src/coinfosim/reports scripts/run_occupancy_scenario.py
python scripts/run_occupancy_scenario.py --list-scenario-runs
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

## End-of-block deliverable

Report paths plus a concise note:

```text
Block 1 complete.
Report-only regeneration works.
Shared report tables/helpers added.
No Monte Carlo simulation was rerun.
Please review before I continue to Block 2.
```

Stop.

---

# Block 2 — Dataset Report provenance and reproducibility

## Goal

Rewrite the dataset report into a provenance/reproducibility report.

## Files to edit

Likely files:

```text
src/coinfosim/datasets/occupancy.py
src/coinfosim/reports/occupancy_dataset.py
scripts/run_occupancy_scenario.py
```

Optional:

```text
src/coinfosim/reports/html_components.py
```

## Tasks

1. Add raw-file SHA-256 hashing for:
   - `datatraining.txt`;
   - `datatest.txt`;
   - `datatest2.txt`.

2. Add explicit public provenance:
   - UCI Machine Learning Repository;
   - dataset ID 357;
   - DOI `10.24432/C5X01N`;
   - Luis Candanedo;
   - CC BY 4.0;
   - Candanedo & Feldheim paper.

3. Preserve and improve current content:
   - source files;
   - train/test protocol;
   - row counts;
   - class distribution;
   - raw channel statistics;
   - standardization parameters;
   - standardized channel statistics;
   - training-pool correlation matrix.

4. Add leakage-control notes:
   - train-only standardization;
   - fixed test split;
   - no test information used for Gaussian/GMM fitting.

5. Add reproducibility metadata:
   - raw paths;
   - hashes;
   - row counts;
   - channel list;
   - target;
   - preprocessing rule.

6. Add a report-only way to regenerate the dataset report without running Monte Carlo if needed. Options:
   - a lightweight CLI flag such as `--dataset-report-only`;
   - or a documented local Python command using `load_occupancy_data()` and `generate_occupancy_dataset_report()`.

Do not run a full scenario just to regenerate the dataset report.

## Validation

Run a report-only dataset regeneration command. Do not run Monte Carlo.

Also run:

```bash
python -m compileall src/coinfosim/datasets src/coinfosim/reports scripts/run_occupancy_scenario.py
```

## End-of-block deliverable

Report path plus note:

```text
Block 2 complete.
Dataset Report rewritten with provenance, hashes, leakage controls and reproducibility metadata.
No Monte Carlo simulation was rerun.
Please review before I continue to Block 3.
```

Stop.

---

# Block 3 — Monte Carlo report backbone and arm-specific methodology

## Goal

Rewrite the three Monte Carlo reports into coherent arm-specific technical appendices, without advanced carousel visualizations yet.

## Files to edit

Likely files:

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_monte_carlo.py
src/coinfosim/reports/report_tables.py
src/coinfosim/reports/html_components.py
```

## Tasks

1. Introduce a structured Monte Carlo report spec if useful, for example:

```python
SimulationReportSpec(
    title=...,
    arm_id=...,
    arm_label=...,
    scientific_role=...,
    train_source=...,
    test_source=...,
    protocol_html=...,
    model_section_html=...,
    limitations_html=...,
)
```

2. Rewrite all three Monte Carlo reports with this shared backbone:

```text
1. Arm summary
2. Scientific role of this arm
3. Dataset provenance summary
4. Run configuration
5. Training/evaluation protocol
6. Arm-specific data/model description
7. Reproducibility controls
8. Monte Carlo stopping rule
9. Monte Carlo precision diagnostics
10. Loss curves
11. Best subset by sample size
12. Final ranking at largest N
13. Within-arm N-star diagnostics
14. Robustness notes
15. Validity and limitations for this arm
16. Technical appendix / exported tables
```

3. Arm-specific section 6:
   - Real → Real: real-data source description.
   - Single Gaussian → Real: class-conditional means/covariances.
   - GMM → Real: selected components, BIC/AIC, mixture weights, means, covariances.

4. Rename `Monte Carlo uncertainty summary` to `Monte Carlo precision diagnostics`.

5. Add compact precision diagnostics by `n_per_class`:
   - `n_per_class`;
   - replications;
   - max SE;
   - max CI95 half-width;
   - target CI95 half-width;
   - stopping reason.

6. Keep the existing generic loss curves if needed, but make them secondary. The more advanced carousel-based loss curves are Block 4.

7. Use `{X1, X2}` notation in main report tables.

8. Keep full long tables available as appendix/export.

## Validation

Use only report regeneration:

```bash
python -m compileall src/coinfosim/reports scripts/run_occupancy_scenario.py
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

Open/review generated HTML paths.

## End-of-block deliverable

Report paths plus note:

```text
Block 3 complete.
The three Monte Carlo reports were rewritten as arm-specific technical appendices.
No Monte Carlo simulation was rerun.
Please review before I continue to Block 4.
```

Stop.

---

# Block 4 — Monte Carlo visual organization, carousels and nested tables

## Goal

Improve the three Monte Carlo reports with better visual organization and nested selectors/carousels.

## Files to edit

Likely files:

```text
src/coinfosim/reports/monte_carlo.py
src/coinfosim/reports/occupancy_monte_carlo.py
src/coinfosim/reports/scenario_visualization.py
src/coinfosim/reports/html_components.py
src/coinfosim/reports/report_tables.py
```

## Tasks

1. Add loss curves for subset-dimension groups plus full-channel subset.

For each classifier, construct curves for:

```text
best 1-channel subset
best 2-channel subset
best 3-channel subset
best 4-channel subset
full 5-channel subset
```

Selection of best k-channel subset should be based on the largest evaluated `n_per_class` within the current arm and classifier.

2. Present these graphs using a carousel/tab-like UI.

Selector nesting:

```text
subset dimension group > classifier
```

Classifier must be the final selector. Default classifier: Linear SVM.

3. Add carousel/tab organization for `Best subset by sample size`.

Selector:

```text
classifier
```

Default: Linear SVM.

4. Add carousel/tab organization for `Final ranking at largest N`.

Selector:

```text
classifier
```

Default: Linear SVM.

5. Add within-arm N-star diagnostics using:

```text
best k-channel subset > classifier
```

Default classifier: Linear SVM.

6. Use the same visual idiom as the scenario report carousels where practical:
   - clear buttons/tabs;
   - active state;
   - no heavy external JS dependency;
   - self-contained HTML.

7. Do not show all 31 subsets at once in the main body unless explicitly in an appendix.

8. Keep complete data available as CSV or appendix table.

## Validation

Use only report regeneration:

```bash
python -m compileall src/coinfosim/reports scripts/run_occupancy_scenario.py
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

Open the three Monte Carlo reports and verify:

- SVM is first and default.
- Classifier is final selector layer.
- Loss curves show best k-channel subsets plus full-channel subset.
- Full 31-subset tables are not dumped into the main body.

## End-of-block deliverable

Report paths plus note:

```text
Block 4 complete.
Monte Carlo visual organization, carousels and nested tables were added.
No Monte Carlo simulation was rerun.
Please review before I continue to Block 5.
```

Stop.

---

# Block 5 — Scenario Report scientific controls, nesting and robustness

## Goal

Upgrade the main Scenario Report into a full comparative scientific report, including methodological controls, traceability, precision diagnostics, robustness, validity and limitations.

## Files to edit

Likely files:

```text
src/coinfosim/reports/occupancy_scenario.py
src/coinfosim/runs/report_data.py
src/coinfosim/reports/report_tables.py
src/coinfosim/reports/html_components.py
```

## Tasks

1. Add the full `Methodological controls of the comparative experiment` section.

Subsections:

```text
Provenance and traceability
Reproducibility controls
Traceability of estimates
Experimental control
Monte Carlo precision control
Robustness checks
Validity considerations
Methodological limitations
```

2. Explicitly state the scientific objective:

```text
Evaluate which training distribution best preserves the cooperative structure
observed under real-data evaluation.
```

3. Explicitly describe arms as comparable experimental conditions:

```text
Real → Real:
  empirical baseline

Single Gaussian → Real:
  simple class-conditional parametric approximation

GMM → Real:
  flexible class-conditional parametric approximation capable of multimodality
```

4. Make clear all arms use the same fixed real evaluation split.

5. Apply nested visual/table organization to the Scenario Report too.

Required nesting:

```text
Top-ranked subsets:
  arm > classifier

Best subset graphs:
  arm

N-star:
  best k-channel subset > classifier

Monte Carlo precision diagnostics:
  arm > n_per_class
```

Classifier remains the last selector layer wherever classifier appears. SVM first and default.

6. Add robustness analysis:
   - best subset stability across `n_per_class`;
   - agreement/divergence among arms;
   - agreement/divergence among classifiers;
   - whether synthetic arms preserve Real → Real cooperative patterns;
   - whether largest-N behavior appears at smaller N.

7. Add validity discussion:
   - internal validity;
   - construct validity;
   - external validity;
   - statistical validity.

8. Add methodological limitations:
   - single dataset;
   - fixed test split;
   - original class imbalance;
   - balanced training samples;
   - synthetic models fitted from training pool;
   - Single Gaussian normality assumption;
   - GMM BIC/component cap;
   - only three classifiers;
   - software/numerical environment.

9. Keep the report concise. Do not turn it into a massive raw table dump.

## Validation

Use only report regeneration:

```bash
python -m compileall src/coinfosim/reports scripts/run_occupancy_scenario.py
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

Open the Scenario Report and verify:

- method controls section exists;
- arms are treated as comparable experimental conditions;
- SVM is default;
- classifier is last selector layer;
- repeated table variables became structure instead of repeated columns.

## End-of-block deliverable

Report path plus note:

```text
Block 5 complete.
Scenario Report now includes methodological controls, nested views, robustness, validity and limitations.
No Monte Carlo simulation was rerun.
Please review before I continue to Block 6.
```

Stop.

---

# Block 6 — Final consistency, polish and minimal targeted validation

## Goal

Make the five reports consistent as a family and run only minimal necessary validation.

## Files to edit

Likely files:

```text
src/coinfosim/reports/*.py
src/coinfosim/runs/report_data.py
scripts/run_occupancy_scenario.py
README.md
```

## Tasks

1. Ensure consistent naming:
   - Real → Real
   - Single Gaussian → Real
   - GMM → Real
   - Monte Carlo precision diagnostics
   - empirical test loss
   - fixed real evaluation split

2. Ensure consistent channel notation:
   - `{X1}`
   - `{X1, X2}`
   - full subset as `{X1, X2, X3, X4, X5}`

3. Ensure channel legend appears where needed.

4. Ensure SVM is always first and default.

5. Ensure classifier is always the final selector layer.

6. Ensure exported CSV/technical tables exist where intended.

7. Ensure report regeneration does not mutate simulation result data.

8. Update README or a small report-regeneration note only if useful:
   - explain how to list scenario runs;
   - explain how to regenerate reports without Monte Carlo;
   - warn not to rerun simulations for report layout work.

9. Run minimal validation only.

## Validation

Run:

```bash
python -m compileall src/coinfosim/reports scripts/run_occupancy_scenario.py
python scripts/run_occupancy_scenario.py --list-scenario-runs
python scripts/run_occupancy_scenario.py --report-from-scenario-run <SCENARIO_RUN_ID>
```

Optional targeted test only if needed:

```bash
pytest -q -k "report or regeneration or occupancy"
```

Do not run full simulations.

Do not run full test suite unless the user authorizes it.

## End-of-block deliverable

Final regenerated report paths plus note:

```text
Block 6 complete.
All five reports are consistent as a report family.
Report regeneration works from persisted scenario/simulation data.
No Monte Carlo simulation was rerun.
Please review the final generated reports.
```

Stop.

---

## 8. Final acceptance criteria

The project is complete when:

1. Dataset Report includes:
   - public provenance;
   - local file roles;
   - SHA-256 hashes;
   - row counts;
   - class distribution;
   - channel dictionary;
   - target;
   - train-only standardization;
   - leakage-control notes.

2. Scenario Report includes:
   - comparative scientific framing;
   - three arms as experimental conditions;
   - methodological controls;
   - traceability;
   - reproducibility;
   - precision diagnostics;
   - robustness;
   - validity;
   - limitations.

3. All three Monte Carlo reports include:
   - arm summary;
   - scientific role;
   - dataset provenance summary;
   - training/evaluation protocol;
   - arm-specific data/model section;
   - reproducibility controls;
   - Monte Carlo precision diagnostics;
   - loss curves organized by subset dimension group + classifier;
   - best-subset/ranking tables organized by classifier;
   - within-arm N-star organized by best k-channel subset + classifier;
   - technical appendix/exported tables.

4. All reports:
   - use compact `{X_i}` subset notation;
   - keep channel names in legends/reference text;
   - show Linear SVM first and by default;
   - keep classifier as final selector layer;
   - avoid repeated structural variables as table columns.

5. Regeneration:
   - works with `--report-from-scenario-run <ID>`;
   - does not rerun Monte Carlo;
   - reuses persisted `result_data_*.json.gz`.

6. No new simulations were run without user authorization.

7. No commit or push was performed.
