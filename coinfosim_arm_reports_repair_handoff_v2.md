# CoInfoSim Occupancy — Individual Arm Reports Repair Handoff v2

## 0. Why this document exists

This document supersedes the previous broad task description for the individual arm reports. It is written for a new coding agent who should not spend time rediscovering the design intent. The previous agent already executed work through Block 3, but the implementation is only partially satisfactory.

The goal now is a focused repair pass. Do not redesign the whole reporting system. Do not infer unspecified behavior from vague references. Follow the explicit section-by-section instructions below.

The user may need to switch agents because the prior GitHub Copilot workflow has run out of available credits. Therefore, this document is the source of truth for the next implementation pass.

---

## 1. Exact target reports

All instructions in this document apply to these three individual arm reports unless a subsection explicitly says otherwise:

### Report A — Real → Real

File:

`occupancy_real_data_monte_carlo_report_full_000006.html`

Arm:

`Real → Real`

Relevant current sections:

- `10. Loss curves`
- `11. Best subset comparison`
- `12. Final ranking at largest N`
- `13. N-star diagnostics`
- `16. Technical appendix / exported tables`

### Report B — Single Gaussian → Real

File:

`occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`

Arm:

`Single Gaussian → Real`

Relevant current sections:

- `6. Arm-specific data and model description`
- `10. Loss curves`
- `11. Best subset comparison`
- `12. Final ranking at largest N`
- `13. N-star diagnostics`
- `16. Technical appendix / exported tables`

### Report C — GMM → Real

File:

`occupancy_gmm_to_real_monte_carlo_report_full_000008.html`

Arm:

`GMM → Real`

Relevant current sections:

- `6. Arm-specific data and model description`
- `10. Loss curves`
- `11. Best subset comparison`
- `12. Final ranking at largest N`
- `13. N-star diagnostics`
- `16. Technical appendix / exported tables`

Report C has one additional mandatory repair: the GMM model matrices and component parameters must be compacted with selectors/tabs.

---

## 2. Files that provide context, not tasks

The scenario report exists and has already been reviewed externally. Do not make the new agent infer the desired structure by reading it. The relevant structure has been extracted and described in this document.

Context file:

`occupancy_baseline_scenario_report_full_000002.html`

Use it only as optional validation if needed. The implementation should follow this document, not reinterpret the scenario report.

Dataset context file:

`occupancy_dataset_report_full_000002.html`

Use only if necessary for labels/provenance. Do not spend time redesigning dataset-provenance sections unless they are broken.

---

## 3. Scientific scope of the three arm reports

These reports are three views of the same experiment:

1. **Real → Real**: empirical baseline. Balanced real training samples are drawn from the standardized `datatraining.txt` pool and evaluated on the fixed real test split.
2. **Single Gaussian → Real**: one class-conditional Gaussian is fitted per class; balanced synthetic training samples are generated; evaluation is on the same fixed real test split.
3. **GMM → Real**: one class-conditional Gaussian mixture model is fitted per class; balanced synthetic training samples are generated; evaluation is on the same fixed real test split.

The reports should let the reader inspect channel-subset performance, cardinality effects, full-reference comparisons, subset rankings by sample size, and N-star diagnostics.

---

## 4. Review of Blocks 1–3 already executed

The prior agent has already produced HTML reports that include some useful scaffolding. Preserve what is correct, but repair the incomplete analytical structure.

### Block 1 — Scientific framing and provenance

Applies to:

- Report A: `occupancy_real_data_monte_carlo_report_full_000006.html`
- Report B: `occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`
- Report C: `occupancy_gmm_to_real_monte_carlo_report_full_000008.html`

Status:

**Mostly accepted. Preserve.**

Preserve these sections unless there is a concrete bug:

- `1. Arm summary`
- `2. Scientific role of this arm`
- `3. Dataset provenance summary`
- `4. Run configuration`
- `5. Training/evaluation protocol`
- `7. Reproducibility controls`
- `8. Monte Carlo stopping rule`
- `9. Monte Carlo precision diagnostics`

Notes:

- The Monte Carlo target must be reported as `0.01` on the 95% CI half-width scale.
- Do not reintroduce the earlier ambiguous `0.01960` target wording.
- Empirical test loss must remain on its original scale. Do not normalize or standardize the loss for display.

### Block 2 — Sticky legend, compact notation, and visual scaffolding

Applies to:

- Report A
- Report B
- Report C

Status:

**Partially accepted. Preserve and audit.**

The current reports now contain a sticky channel legend and compact notation. Preserve this direction.

Required channel mapping:

- `X₁ = Temperature`
- `X₂ = Humidity`
- `X₃ = Light`
- `X₄ = CO₂`
- `X₅ = Humidity Ratio`

Main result views must use compact subset notation, for example:

- `{X₁}`
- `{X₁, X₃}`
- `{X₁, X₂, X₃}`
- `{X₁, X₂, X₃, X₄, X₅}`

Full channel names may appear only in:

- the sticky legend;
- metadata;
- tooltips;
- captions when necessary;
- technical appendices;
- model-parameter sections when unavoidable.

The sticky legend must appear once near the top of the report and should not be repeated before every plot.

### Block 3 — Initial selectors and yellow reference highlighting

Applies to:

- Report A, sections `10`, `11`, `12`, `13`;
- Report B, sections `10`, `11`, `12`, `13`;
- Report C, sections `10`, `11`, `12`, `13`, plus section `6` for GMM matrices.

Status:

**Incomplete. Repair required.**

What the prior agent did correctly:

- added classifier tabs in some result sections;
- made Linear SVM the default in at least some places;
- added a yellow reference style;
- introduced compact subset notation in several main views.

What remains wrong or incomplete:

- Section `10. Loss curves` in all three target reports lacks nested cardinality views.
- Section `12. Final ranking at largest N` in all three target reports is too narrow; it must become a ranking-by-sample-size section with an `n_per_class` selector.
- Section `13. N-star diagnostics` in all three target reports is too shallow; it must contain the full N-star structure described in Section 9 of this handoff document.
- Report C still prints too many GMM component means and covariance matrices sequentially; this must be compacted using selectors.

---

## 5. Shared report conventions for future blocks

### 5.1 Result row model

Before implementing the repaired UI sections, verify that the report generator has access to a clean result representation with at least:

- `arm`
- `classifier`
- `n_per_class`
- `subset_id`
- `subset_size`
- `subset_label_var`, for example `{X₁, X₃}`
- `subset_label_full`, for example `Temperature + Light`
- `mean_loss`
- `se_loss`
- `ci95_half_width`, if available
- `rank_within_classifier_n`
- `is_full_reference`

### 5.2 Full-reference subset

For this five-channel experiment, the full max-channel reference subset is always:

`{X₁, X₂, X₃, X₄, X₅}`

Use this exact compact label in main views.

When the full subset is used as a reference in plots or tables, use the existing yellow reference style consistently.

### 5.3 Classifier order and defaults

Classifier order:

1. `Linear SVM`
2. `Logistic Regression`
3. `Gaussian Naive Bayes`

Default classifier everywhere:

`Linear SVM`

### 5.4 Evaluated sample sizes

All reports use the same sample-size grid:

`2, 4, 8, 16, 32, 64, 128, 256, 512`

The largest sample size is:

`n_per_class = 512`

Use `512` as a default table view when a default sample size is needed. Do not treat it as the only valid table.

---

## 6. Future Block 4 — Repair Section 10 in each target report

This block applies explicitly to:

- Report A: `occupancy_real_data_monte_carlo_report_full_000006.html`, section `10. Loss curves`.
- Report B: `occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`, section `10. Loss curves`.
- Report C: `occupancy_gmm_to_real_monte_carlo_report_full_000008.html`, section `10. Loss curves`.

### 6.1 Current problem

The current Section 10 mainly shows loss curves for the best k-channel subset per cardinality plus the full max-channel subset. That answers only this question:

> What is the best subset at each cardinality?

It does not answer this second required question:

> How does the full max-channel reference compare against all subsets of a fixed smaller cardinality?

### 6.2 Required Section 10 structure

Section `10. Loss curves` must contain two complementary views.

#### View 10A — Best-by-cardinality loss curves

This view already partly exists. Preserve and repair it.

For the selected classifier, show loss vs. `n_per_class` with these curves:

- best 1-channel subset;
- best 2-channel subset;
- best 3-channel subset;
- best 4-channel subset;
- full 5-channel subset `{X₁, X₂, X₃, X₄, X₅}`.

The full 5-channel subset should be the yellow reference curve.

Use compact labels only.

Example legend labels:

- `Best-1: {X₃}`
- `Best-2: {X₁, X₃}`
- `Best-3: {X₁, X₃, X₄}`
- `Best-4: {X₁, X₂, X₃, X₄}`
- `Full-5 reference: {X₁, X₂, X₃, X₄, X₅}`

Actual subset values must come from the computed results.

#### View 10B — Nested cardinality loss curves

Add a second static selector for nested cardinality views.

For a five-channel experiment, expose exactly these nested views:

1. `1-channel subsets + full 5-channel reference`
2. `2-channel subsets + full 5-channel reference`
3. `3-channel subsets + full 5-channel reference`
4. `4-channel subsets + full 5-channel reference`

Each selected nested view must show:

- all subsets of the selected cardinality;
- the full 5-channel subset `{X₁, X₂, X₃, X₄, X₅}` as the yellow reference curve.

Expected subset counts:

- 1-channel view: 5 single-channel subset curves + 1 full-reference curve = 6 curves.
- 2-channel view: 10 pair subset curves + 1 full-reference curve = 11 curves.
- 3-channel view: 10 triple subset curves + 1 full-reference curve = 11 curves.
- 4-channel view: 5 four-channel subset curves + 1 full-reference curve = 6 curves.

The full 5-channel subset must not be duplicated if it already appears in a selected view. It is included only as the reference curve.

### 6.3 Section 10 selector hierarchy

Use static HTML controls, not an automatic carousel.

Required hierarchy:

1. classifier selector;
2. loss-curve view selector:
   - `Best by cardinality`
   - `Nested cardinality`
3. nested-cardinality selector, used only for the nested-cardinality view:
   - `1-channel + full reference`
   - `2-channel + full reference`
   - `3-channel + full reference`
   - `4-channel + full reference`

Defaults:

- classifier: `Linear SVM`;
- loss-curve view: `Best by cardinality`;
- nested-cardinality view, if opened: `1-channel + full reference`.

### 6.4 Section 10 acceptance criteria

For Report A, Report B, and Report C:

- Section 10 contains View 10A and View 10B.
- Linear SVM is the default classifier.
- View 10B has the four nested cardinality options listed above.
- Full 5-channel subset is the yellow reference curve in both views when shown.
- Plot labels use `{Xᵢ}` notation, not long channel names.
- The section text explicitly explains the difference between the two views.

---

## 7. Future Block 5 — Audit Section 11 in each target report

This block applies explicitly to:

- Report A: section `11. Best subset comparison`.
- Report B: section `11. Best subset comparison`.
- Report C: section `11. Best subset comparison`.

### 7.1 Status

Section 11 is mostly acceptable. Do not spend major effort redesigning it unless concrete inconsistencies are found.

### 7.2 Required Section 11 behavior

For the selected classifier, the section should compare:

- best 1-channel subset;
- best 2-channel subset;
- best 3-channel subset;
- best 4-channel subset;
- full 5-channel subset `{X₁, X₂, X₃, X₄, X₅}`.

The full 5-channel subset must be highlighted as the full-reference row.

### 7.3 Section 11 acceptance criteria

For Report A, Report B, and Report C:

- Linear SVM is the default classifier.
- Compact variable notation is used.
- Full 5-channel reference row is highlighted.
- Table columns are readable and not polluted by repeated metadata that could be in a title or selector.
- If `classifier` is repeated as a table column despite a classifier selector, remove that redundant column from the visible table.

---

## 8. Future Block 6 — Repair Section 12 in each target report

This block applies explicitly to:

- Report A: `occupancy_real_data_monte_carlo_report_full_000006.html`, current section `12. Final ranking at largest N`.
- Report B: `occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`, current section `12. Final ranking at largest N`.
- Report C: `occupancy_gmm_to_real_monte_carlo_report_full_000008.html`, current section `12. Final ranking at largest N`.

### 8.1 Current problem

The current title and structure imply that only the largest sample size matters. This is wrong for the intended report.

The largest N is the default view, not the only available ranking.

### 8.2 Required title

Rename the section in all three target reports to:

`12. Subset ranking by sample size`

### 8.3 Required Section 12 selector hierarchy

Use static HTML controls with this hierarchy:

1. classifier selector;
2. `n_per_class` selector;
3. ranking table for the selected classifier and selected `n_per_class`.

Default state:

- classifier: `Linear SVM`;
- `n_per_class`: `512`.

The `n_per_class` selector must expose all evaluated values:

`2, 4, 8, 16, 32, 64, 128, 256, 512`

### 8.4 Required ranking table columns

Each selected table should show all 31 non-empty channel subsets ranked by empirical test loss.

Visible columns:

- `Rank`
- `Subset`
- `k`
- `Mean loss`
- `SE`
- `CI95 half-width`, if available
- `Reference`, or a compact marker indicating whether the row is the full 5-channel reference

Do not show redundant columns such as `classifier` or `n_per_class` inside the table if those values are already selected by controls and displayed in the title/subtitle.

### 8.5 Required explanatory text

Use text equivalent to:

“Subsets are ranked by empirical test loss for the selected classifier and selected sample size. The largest evaluated sample size is shown by default, but rankings for all evaluated `n_per_class` values are available through the selector.”

### 8.6 Section 12 acceptance criteria

For Report A, Report B, and Report C:

- Section title is `12. Subset ranking by sample size`.
- Rankings are available for all nine evaluated sample sizes.
- `n_per_class = 512` is default, not the only ranking.
- Linear SVM is default classifier.
- Each selected table ranks all 31 subsets.
- Full 5-channel reference row is highlighted in yellow or clearly marked.
- Compact subset notation is used.

---

## 9. Future Block 7 — Repair Section 13 in each target report: explicit N-star structure

This block applies explicitly to:

- Report A: `occupancy_real_data_monte_carlo_report_full_000006.html`, section `13. N-star diagnostics` for the `Real → Real` arm only.
- Report B: `occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`, section `13. N-star diagnostics` for the `Single Gaussian → Real` arm only.
- Report C: `occupancy_gmm_to_real_monte_carlo_report_full_000008.html`, section `13. N-star diagnostics` for the `GMM → Real` arm only.

Do not ask the agent to “copy the scenario report”. The N-star structure required here is described below.

### 9.1 N-star conceptual structure extracted from the scenario report

The scenario report organizes N-star as follows:

1. First level: classifier.
   - `Linear SVM`
   - `Logistic Regression`
   - `Gaussian Naive Bayes`

2. Second level: reference cardinality.
   - `Best 1-channel subset`
   - `Best 2-channel subset`
   - `Best 3-channel subset`
   - `Best 4-channel subset`
   - `Best 5-channel subset`

3. For each classifier and reference cardinality, the reference subset is the best subset of that cardinality at the largest evaluated sample size.

4. In the original scenario report, reference and competitor subsets are selected using the `Real → Real` arm so that Real, Single Gaussian, and GMM are compared on the same subsets.

5. For the individual arm reports, preserve that comparability rule:
   - select reference and competitor subsets using the `Real → Real` arm at `n_per_class = 512`;
   - then compute/display N-star values using the current report’s own arm data.

This means:

- Report A displays Real → Real N-star values using Real-selected references and competitors.
- Report B displays Single Gaussian → Real N-star values using the same Real-selected references and competitors.
- Report C displays GMM → Real N-star values using the same Real-selected references and competitors.

Do not independently reselect the reference subsets inside the Single Gaussian or GMM reports unless the scientific design is intentionally changed.

### 9.2 N-star competitor selection rule

For each classifier and reference cardinality `k`, create one reference subset:

`Reference = Best-k-ChSub selected at n_per_class = 512 on Real → Real`

Then compare the reference subset against:

1. the best subset of every other cardinality;
2. the second-best subset of the same cardinality, when such a subset exists.

For five channels, this means:

#### If the reference is `Best 1-channel subset`

Rows should compare against:

- `2-Best-1-ChSub`
- `Best-2-ChSub`
- `Best-3-ChSub`
- `Best-4-ChSub`
- `Best-5-ChSub`

#### If the reference is `Best 2-channel subset`

Rows should compare against:

- `Best-1-ChSub`
- `2-Best-2-ChSub`
- `Best-3-ChSub`
- `Best-4-ChSub`
- `Best-5-ChSub`

#### If the reference is `Best 3-channel subset`

Rows should compare against:

- `Best-1-ChSub`
- `Best-2-ChSub`
- `2-Best-3-ChSub`
- `Best-4-ChSub`
- `Best-5-ChSub`

#### If the reference is `Best 4-channel subset`

Rows should compare against:

- `Best-1-ChSub`
- `Best-2-ChSub`
- `Best-3-ChSub`
- `2-Best-4-ChSub`
- `Best-5-ChSub`

#### If the reference is `Best 5-channel subset`

Rows should compare against:

- `Best-1-ChSub`
- `Best-2-ChSub`
- `Best-3-ChSub`
- `Best-4-ChSub`

There is no `2-Best-5-ChSub` in a five-channel experiment because there is only one 5-channel subset.

### 9.3 Required N-star table columns

For each selected classifier/reference-cardinality state, show exactly these columns:

- `VS`
- `N*`
- `Interpolated N*`
- `Winner`

Column meanings:

- `VS`: competitor subset label, for example `Best-2-ChSub: {X₁, X₃}` or `2-Best-3-ChSub: {X₂, X₃, X₅}`.
- `N*`: the reported grid sample size associated with the last crossing detected inside the evaluated sample-size grid.
- `Interpolated N*`: linear interpolation of the crossing location when available.
- `Winner`: which curve has lower empirical test loss at the largest evaluated sample size; values should be `Reference` or `VS`.

Dash convention:

- Use `—` when no crossing is detected inside the evaluated sample-size grid.
- When `N*` is `—`, the `Winner` column must still report which curve has lower loss at the largest evaluated sample size.

Crossing convention:

- If multiple crossings are detected, preserve all crossings in underlying structured data if available, but report and highlight the last `N*` in the HTML table.

### 9.4 Required N-star graph

For each selected classifier/reference-cardinality state, show a corresponding graph.

The graph must contain:

- x-axis: `n_per_class`;
- y-axis: empirical test loss;
- reference subset curve;
- competitor subset curves listed in the table;
- dashed vertical lines for reported `Interpolated N*` values when available;
- clear visual distinction for the reference subset, preferably using the existing yellow reference style.

Use compact notation in labels.

### 9.5 Required Section 13 selector hierarchy

Use static HTML controls with this hierarchy:

1. classifier selector:
   - `Linear SVM`
   - `Logistic Regression`
   - `Gaussian Naive Bayes`
2. reference-cardinality selector:
   - `Best 1-channel reference`
   - `Best 2-channel reference`
   - `Best 3-channel reference`
   - `Best 4-channel reference`
   - `Best 5-channel reference`
3. content panel containing:
   - reference subset statement;
   - N-star table;
   - N-star graph.

Defaults:

- classifier: `Linear SVM`;
- reference-cardinality: `Best 1-channel reference`.

### 9.6 Required reference subset statement

Each selected panel should begin with a statement like:

`Reference subset: {X₃} = {Light}`

or:

`Reference subset: {X₁, X₃} = {Temperature, Light}`

Main label should use compact notation first. Full names may follow after `=`.

### 9.7 Section 13 acceptance criteria

For Report A, Report B, and Report C:

- Section 13 has classifier selector and reference-cardinality selector.
- Linear SVM is default.
- Best 1-channel reference is default.
- Every classifier/reference-cardinality state shows one table and one graph.
- Tables use exactly `VS`, `N*`, `Interpolated N*`, `Winner`.
- The reference and competitor subsets follow the rules in Section 9.2 above.
- The individual report displays only its own arm, not all three arms.
- Reference/competitor subset definitions are selected from the Real → Real arm at `n_per_class = 512` for comparability.
- Graphs include reference curve, competitor curves, and vertical crossing markers when available.
- No instruction in this section should require the agent to inspect the scenario report to infer structure.

---

## 10. Future Block 8 — GMM-specific repair for Report C section 6

This block applies only to:

`occupancy_gmm_to_real_monte_carlo_report_full_000008.html`

Current section:

`6. Arm-specific data and model description`

### 10.1 Current problem

The GMM report currently prints long sequences of:

- class headers;
- selected component counts;
- BIC/AIC tables;
- mixture weights;
- component means;
- component covariance matrices.

This is scientifically useful but visually overwhelming. It should not be dumped sequentially in the main reading path.

### 10.2 Required compact GMM structure

Use selectors/tabs to compact the model description.

Required selector hierarchy:

1. top-level GMM content selector:
   - `Model-selection summary`
   - `Component details`
2. if `Model-selection summary` is selected, show:
   - model-selection configuration;
   - selected component count by class;
   - BIC/AIC table for Class 0;
   - BIC/AIC table for Class 1;
   - mixture weights summary for Class 0;
   - mixture weights summary for Class 1.
3. if `Component details` is selected, show:
   - class selector: `Class 0`, `Class 1`;
   - component selector: `Component 0`, `Component 1`, `Component 2`, `Component 3`, `Component 4`;
   - detail selector if needed: `Mean vector`, `Covariance matrix`, or both together.

### 10.3 Matrix notation

Matrix row/column labels should use:

- `X₁`
- `X₂`
- `X₃`
- `X₄`
- `X₅`

The sticky legend already maps these variables to full channel names.

### 10.4 Report B note

Report B, Single Gaussian → Real, also contains class means and covariance matrices, but it has only one Gaussian per class. Do not spend major time compacting Report B unless it is straightforward. The mandatory matrix-compaction repair is for Report C, GMM → Real.

### 10.5 GMM acceptance criteria

For Report C:

- GMM matrices are no longer printed as one long sequential block.
- The default section-6 view is the model-selection summary.
- Component-level details are accessible through class/component selectors.
- Matrix labels use `X₁`–`X₅`.
- No scientific information is deleted merely to shorten the report.

---

## 11. Future Block 9 — Final QA pass across all target reports

Apply these checks to:

- Report A: `occupancy_real_data_monte_carlo_report_full_000006.html`
- Report B: `occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html`
- Report C: `occupancy_gmm_to_real_monte_carlo_report_full_000008.html`

### 11.1 Required QA checklist

1. Linear SVM is the default classifier everywhere classifier selection exists.
2. Full 5-channel subset `{X₁, X₂, X₃, X₄, X₅}` is consistently treated as full reference where applicable.
3. Yellow reference styling is visible and explained wherever used.
4. Compact variable notation is used in main result views.
5. The sticky channel legend appears once and remains readable.
6. Tables are not raw dumps when a selector is more appropriate.
7. Section numbers and titles are consistent across the three reports.
8. Monte Carlo CI target is consistently reported as `0.01` on the 95% half-width scale.
9. Reports remain self-contained HTML files.
10. Report A, Report B, and Report C have parallel structure unless an arm-specific section is scientifically necessary.
11. Section 10 in all three reports has both View 10A and View 10B.
12. Section 12 in all three reports supports all evaluated `n_per_class` values.
13. Section 13 in all three reports follows the explicit N-star structure from Section 9 of this document.
14. Report C section 6 has compact GMM selectors.

---

## 12. What not to do

Do not spend time on unrelated cosmetic redesigns.

Do not reintroduce long channel names into main tables and plot labels.

Do not replace static selectors with an automatic carousel.

Do not normalize empirical test loss.

Do not remove the methodological sections already added in Blocks 1–3.

Do not make the GMM section shorter by deleting scientific information. Make it compact by using selectors.

Do not treat `n_per_class = 512` as the only meaningful ranking. It is the default ranking, not the only ranking.

Do not make Section 13 a small summary only. It must contain the explicit N-star selector/table/graph structure described above.

Do not tell the user that the report now follows the scenario report unless the explicit acceptance criteria above are satisfied.

---

## 13. Final expected result

After this repair, each individual arm report should allow the reader to answer these questions quickly:

1. What is the scientific role of this arm?
2. What data source and evaluation split are used?
3. What are the Monte Carlo precision diagnostics?
4. Which subset is best at each cardinality?
5. How does every k-channel subset compare with the full 5-channel reference?
6. How do subset rankings change across sample sizes?
7. What are the N-star thresholds for each classifier and reference cardinality?
8. For GMM, what class/component/model parameters were fitted, without overwhelming the report body?

The final reports should be navigable scientific HTML reports, not raw sequences of tables, plots, and matrices.


