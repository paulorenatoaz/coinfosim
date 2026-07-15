"""Scientific dataset report for the SUPPORT2 180-day mortality scenario."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from coinfosim.datasets.support2 import (
    SUPPORT2_CHANNELS,
    SUPPORT2_COMPLETE_CASE_COLUMNS,
    SUPPORT2_TARGET,
    Support2Data,
)
from coinfosim.reports.dataset_common import (
    class_distribution_image,
    correlation_heatmap_image,
    dataframe_html,
    file_hash_table_html,
    standardization_table_html,
)

_STYLE = """
body{font-family:Arial,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#20242a;line-height:1.48}
h1,h2{color:#17365d} code{background:#f2f4f7;padding:.12rem .28rem;border-radius:3px}
table.data{border-collapse:collapse;width:100%;margin:.8rem 0 1.4rem;font-size:.9rem}
table.data th,table.data td{border:1px solid #c8d0da;padding:.38rem;text-align:right}
table.data th:first-child,table.data td:first-child{text-align:left}.note{background:#eef5ff;border-left:4px solid #2867a8;padding:.8rem}
.figure img{max-width:100%;height:auto}.warning{background:#fff6df;border-left:4px solid #b57a00;padding:.8rem}
"""

_CHANNEL_DESCRIPTIONS = {
    "meanbp": "Mean arterial blood pressure at study entry",
    "hrt": "Heart rate at study entry",
    "resp": "Respiratory rate at study entry",
    "temp": "Body temperature at study entry",
    "wblc": "White blood cell count at study entry",
    "crea": "Serum creatinine at study entry",
    "sod": "Serum sodium at study entry",
}


def generate_support2_dataset_report(
    data: Support2Data,
    output_dir: Path | str = "output/reports",
    filename: str = "support2_dataset_report.html",
) -> Path:
    """Render an independently reproducible endpoint/cohort/split audit."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    counts = data.row_counts()
    classes = data.class_counts()
    fingerprints = data.id_fingerprints()

    missing = pd.DataFrame(
        [
            {
                "field": column,
                "role": data.column_roles[column],
                "raw_missing": int(data.raw_frame[column].isna().sum()),
            }
            for column in SUPPORT2_COMPLETE_CASE_COLUMNS
        ]
    )
    missing_by_target = (
        data.raw_frame.groupby(SUPPORT2_TARGET, observed=True)[list(SUPPORT2_CHANNELS)]
        .apply(lambda frame: frame.isna().sum())
        .reset_index()
        .rename(columns={SUPPORT2_TARGET: "death_180d"})
    )
    disease = pd.crosstab(
        data.cohort_frame["dzgroup"], data.cohort_frame[SUPPORT2_TARGET]
    ).reset_index()
    disease.columns = ["dzgroup", "death_180d=0", "death_180d=1"]
    disease["total"] = disease["death_180d=0"] + disease["death_180d=1"]
    disease["180_day_mortality_prevalence"] = disease["death_180d=1"] / disease["total"]

    class_conditional = (
        data.cohort_frame.groupby(SUPPORT2_TARGET, observed=True)[list(SUPPORT2_CHANNELS)]
        .agg(["mean", "std", "median", "min", "max"])
    )
    class_conditional.columns = [f"{channel}_{stat}" for channel, stat in class_conditional.columns]
    class_conditional = class_conditional.reset_index()
    marginal = data.cohort_frame.loc[:, SUPPORT2_CHANNELS].describe().T.reset_index()
    marginal = marginal.rename(columns={"index": "channel"})
    zero_extreme = pd.DataFrame(
        [
            {
                "channel": channel,
                "zeros": int((data.cohort_frame[channel] == 0).sum()),
                "minimum": float(data.cohort_frame[channel].min()),
                "maximum": float(data.cohort_frame[channel].max()),
                "policy": "preserved unchanged",
            }
            for channel in SUPPORT2_CHANNELS
        ]
    )
    split_table = pd.DataFrame(
        [
            {
                "partition": name,
                "rows": len(frame),
                "death_180d=0": int((frame[SUPPORT2_TARGET] == 0).sum()),
                "death_180d=1": int((frame[SUPPORT2_TARGET] == 1).sum()),
                "id_sha256": fingerprints[name],
            }
            for name, frame in (
                ("cohort", data.cohort_frame),
                ("train", data.raw_train),
                ("test", data.raw_test),
            )
        ]
    )
    target_relation = pd.DataFrame(
        [
            {"death": 0, "death_180d": 0, "count": 2_904, "interpretation": "survived beyond validated follow-up"},
            {"death": 1, "death_180d": 0, "count": 1_936, "interpretation": "death occurred after day 180"},
            {"death": 1, "death_180d": 1, "count": 4_265, "interpretation": "death occurred on or before day 180"},
        ]
    )
    channel_rows = "".join(
        f"<tr><td>X{index}</td><td><code>{channel}</code></td><td>{html.escape(_CHANNEL_DESCRIPTIONS[channel])}</td></tr>"
        for index, channel in enumerate(SUPPORT2_CHANNELS, start=1)
    )
    class_image = class_distribution_image(
        {name: values for name, values in classes.items() if name in {"raw", "cohort", "train", "test"}},
        data.class_labels,
        title="SUPPORT2 death_180d class counts",
    )
    corr_image = correlation_heatmap_image(
        data.train_correlation(), title="Standardized training-channel correlations"
    )
    corr = data.train_correlation().reset_index().rename(columns={"index": "channel"})

    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CoInfoSim — SUPPORT2 180-Day Mortality Dataset Report</title><style>{_STYLE}</style></head><body>
<h1>CoInfoSim — SUPPORT2 180-Day Mortality Dataset Report</h1>
<p class="note"><strong>Prediction target:</strong> death within 180 days after SUPPORT2 study entry. This implementation follows the original SUPPORT six-month prognostic objective.</p>

<h2>1. Dataset overview and provenance</h2>
<p>SUPPORT2 contains 9,105 seriously ill hospitalized adults from five United States medical centers. The canonical source is the Vanderbilt University Department of Biostatistics repository, mirrored by the UCI Machine Learning Repository. DOI: <code>10.3886/ICPSR02957.v2</code>.</p>
{file_hash_table_html(data.file_hashes)}

<h2>2. Raw structural anomaly and reconstructed patient ID</h2>
<p>The source header advertises 47 names while every data row has 48 fields. The unnamed leading field is reconstructed explicitly as the integral, unique patient <code>id</code> running from 1 through 9,105; no implicit pandas index behavior is used.</p>

<h2>3. Original prognostic objective and fixed endpoint</h2>
<p>The original SUPPORT work sought prognostic estimates over a six-month (180-day) period. CoInfoSim therefore uses a fixed, interpretable horizon instead of eventual death:</p>
<p><code>death_180d = ((death == 1) &amp; (d.time &lt;= 180)).astype(int)</code></p>
<p>Day 180 is inclusive: a recorded death on day 180 is positive; a death after day 180 is negative for this endpoint. <code>death_180d</code> is derived after schema validation and before complete-case filtering or splitting; it is not a raw CSV column.</p>

<h2>4. Relationship among death, d.time, and death_180d</h2>
{dataframe_html(target_relation)}
<p><code>death</code> is the eventual death indicator and <code>d.time</code> is follow-up/event time. All 2,904 patients with <code>death == 0</code> have defined follow-up beyond the horizon; the minimum survivor <code>d.time</code> is 344 days and zero survivors have <code>d.time &lt;= 180</code>. This survivor-follow-up validation is required before assigning them class 0.</p>

<h2>5. Why hospdead is not the target</h2>
<p><strong>hospdead is not used as the primary target or as a predictor.</strong> It describes in-hospital death, which is a different estimand from the approved fixed 180-day outcome and does not implement the original six-month prognostic objective.</p>

<h2>6. Selected channels and measurement timing</h2>
<table class="data"><thead><tr><th>Index</th><th>Column</th><th>Definition</th></tr></thead><tbody>{channel_rows}</tbody></table>
<p>Channels are baseline physiologic measurements in the fixed order shown. All 127 non-empty subsets use this same seven-channel cohort.</p>

<h2>7. Excluded variables and leakage controls</h2>
<p><code>death</code>, <code>d.time</code>, <code>death_180d</code>, <code>hospdead</code>, <code>surv2m</code>, <code>surv6m</code>, <code>id</code>, and <code>dzgroup</code> never enter predictor data. <code>dzgroup</code> is used only for stratification and reporting. No target-source, alternative-outcome, post-baseline survival estimate, identifier, or disease-group field enters standardization, Gaussian/GMM fitting, subset enumeration, or classifiers.</p>

<h2>8. Raw target counts and survivor audit</h2>
<p>Raw rows: <strong>9,105</strong>. Raw classes: <strong>4,840</strong> negative and <strong>4,265</strong> positive. Survivor count: 2,904; minimum survivor follow-up: <strong>344</strong> days.</p>
<div class="figure"><img src="{class_image}" alt="death_180d class distribution"/></div>

<h2>9. Complete-case cohort construction</h2>
<p>Complete cases are required for {html.escape(', '.join(SUPPORT2_COMPLETE_CASE_COLUMNS))}. No imputation is performed. The resulting cohort has <strong>{counts['complete_case_cohort']:,}</strong> rows: <strong>{classes['cohort'][0]:,}</strong> class 0 and <strong>{classes['cohort'][1]:,}</strong> class 1; {counts['discarded_incomplete']} rows are excluded.</p>

<h2>10. Raw and selected-field missingness</h2>
{dataframe_html(missing)}

<h2>11. Missingness by death_180d</h2>
{dataframe_html(missing_by_target)}

<h2>12. Disease-group composition and mortality prevalence</h2>
{dataframe_html(disease, float_cols={"180_day_mortality_prevalence": ".6f"})}

<h2>13. Fixed split protocol and fingerprints</h2>
<p>The complete cohort is split 80/20 with <code>random_state=0</code>, stratified jointly by <code>death_180d × dzgroup</code>, then each partition is sorted by ascending patient ID. All 16 observed joint strata occur in both partitions.</p>
{dataframe_html(split_table)}
<p>Training has <strong>7,098</strong> rows (3,768 class 0; 3,330 class 1). Test has <strong>1,775</strong> rows (943 class 0; 832 class 1). The same fixed real test set is used by all three scenario arms.</p>

<h2>14. Training-only preprocessing</h2>
<p>Z-score means and population standard deviations are fitted only on the 7,098-row training reservoir with <code>ddof=0</code>, then applied unchanged to test rows. Imputation, clipping, winsorization, transformation, outlier removal, target-dependent preprocessing, and dimensionality reduction are absent.</p>
{standardization_table_html(data.standardization)}

<h2>15. Marginal channel diagnostics</h2>
{dataframe_html(marginal, float_cols={column: ".6f" for column in marginal.columns if column != "channel"})}

<h2>16. Class-conditional channel diagnostics</h2>
{dataframe_html(class_conditional, float_cols={column: ".6f" for column in class_conditional.columns if column != SUPPORT2_TARGET})}

<h2>17. Correlation diagnostics</h2>
{dataframe_html(corr, float_cols={channel: ".4f" for channel in SUPPORT2_CHANNELS})}
<div class="figure"><img src="{corr_image}" alt="training correlation heatmap"/></div>

<h2>18. Zero and extreme-value policy</h2>
{dataframe_html(zero_extreme, float_cols={"minimum": ".6f", "maximum": ".6f"})}
<p>Observed zeros and extremes are preserved; there is no clipping, winsorization, transformation, or outlier removal.</p>

<h2>19. Limitations</h2>
<p>This is an observational cohort of seriously ill hospitalized adults from a specific historical clinical setting. Complete-case restriction can alter cohort composition. The endpoint collapses time-to-event information to a single horizon, and smoke-mode simulations are pipeline validation rather than stable inferential evidence. Disease group is retained for stratification but intentionally excluded from predictors.</p>

<h2>20. Citation, acknowledgment, and license</h2>
<p>The SUPPORT Principal Investigators (1995), “A controlled trial to improve care for seriously ill hospitalized patients,” <em>JAMA</em> 274(20):1591–1598. Dataset citation: Harrell, F. (1995), SUPPORT2, DOI <code>10.3886/ICPSR02957.v2</code>. Data obtained from <a href="https://hbiostat.org/data/">hbiostat.org/data</a> courtesy of the Vanderbilt University Department of Biostatistics; use is subject to the source acknowledgment policy.</p>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
