"""Scientific dataset report for the UCI Air Quality scenario."""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from coinfosim.datasets.air_quality import AirQualityData
from coinfosim.reports.dataset_common import (
    class_distribution_image,
    correlation_heatmap_image,
    dataframe_html,
    figure_to_data_uri,
    file_hash_table_html,
    standardization_table_html,
    standardized_mean_comparison_image,
)


def _temporal_image(data: AirQualityData) -> str:
    frame = data.cohort_frame
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.plot(
        frame["timestamp"],
        frame[data.reference_name],
        linewidth=0.7,
        color="#356aa0",
        label="C6H6(GT)",
    )
    axis.axhline(
        data.threshold_value,
        color="#c44e52",
        linestyle="--",
        linewidth=1.2,
        label=f"training Q0.75 = {data.threshold_value:g}",
    )
    axis.axvline(
        data.cutoff_timestamp,
        color="#222",
        linestyle=":",
        linewidth=1.5,
        label="chronological split",
    )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("C6H6(GT)")
    axis.set_title("Benzene reference series and fixed chronological split")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    return figure_to_data_uri(figure)


def generate_air_quality_dataset_report(
    data: AirQualityData,
    output_dir: Path | str = "output/reports",
    filename: str = "air_quality_dataset_report.html",
) -> Path:
    """Generate the 19-section Air Quality provenance and protocol report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    counts = data.row_counts()
    missing = data.missing_counts()
    class_counts = data.class_counts()
    raw_summary = data.raw_channel_summary()
    standardized_summary = data.standardized_channel_summary()
    correlation = data.train_correlation(standardized=True)
    reference_correlations = (
        data.train_sensor_reference_correlations()
        .rename_axis("channel")
        .reset_index()
    )
    reference_correlations.columns = ["channel", "correlation_with_C6H6(GT)"]
    schema = pd.DataFrame(
        {
            "column": data.raw_frame.columns,
            "dtype_after_parsing": [str(data.raw_frame[column].dtype) for column in data.raw_frame],
            "missing_after_sentinel_replacement": [
                int(data.raw_frame[column].isna().sum()) for column in data.raw_frame
            ],
        }
    )
    missing_frame = pd.DataFrame(
        [{"required_field": field, "missing_rows": value} for field, value in missing.items()]
    )
    cohort_frame = pd.DataFrame(
        [
            {"stage": "Raw non-empty rows", "rows": counts["raw_non_empty"]},
            {"stage": "Complete-case cohort", "rows": counts["complete_case_cohort"]},
            {"stage": "Discarded incomplete rows", "rows": counts["discarded_incomplete"]},
            {"stage": "Chronological training reservoir", "rows": counts["train"]},
            {"stage": "Fixed future real test", "rows": counts["test"]},
        ]
    )
    class_rows = []
    for split, split_counts in class_counts.items():
        total = sum(split_counts.values())
        majority_error = min(split_counts.values()) / total
        class_rows.append(
            {
                "split": split,
                "class_0": split_counts[0],
                "class_1": split_counts[1],
                "class_1_prevalence": split_counts[1] / total,
                "majority_class_error_baseline": majority_error,
            }
        )
    class_frame = pd.DataFrame(class_rows)
    shift = (
        standardized_summary.pivot(index="channel", columns="split", values="mean")
        .reset_index()
    )
    shift["test_minus_train_standardized_mean"] = (
        shift["fixed_test"] - shift["train_pool"]
    )

    class_distribution = class_distribution_image(
        class_counts, data.class_labels, title="Class distribution by chronological split"
    )
    correlation_image = correlation_heatmap_image(
        correlation, title="Training attribute correlation matrix"
    )
    standardized_means_image = standardized_mean_comparison_image(
        standardized_summary,
        data.channel_names,
        title="Standardized train/test attribute means",
    )
    temporal_image = _temporal_image(data)

    channel_descriptions = {
        "PT08.S1(CO)": "tin-oxide sensor response, nominally CO targeted",
        "PT08.S2(NMHC)": "titania sensor response, nominally NMHC targeted",
        "PT08.S3(NOx)": "tungsten-oxide sensor response, nominally NOx targeted",
        "PT08.S4(NO2)": "tungsten-oxide sensor response, nominally NO2 targeted",
        "PT08.S5(O3)": "indium-oxide sensor response, nominally O3 targeted",
    }
    channel_rows = "".join(
        f"<tr><td><code>X{index}</code></td><td><code>{html.escape(channel)}</code></td>"
        f"<td>{html.escape(channel_descriptions[channel])}</td></tr>"
        for index, channel in enumerate(data.channel_names, 1)
    )
    float_summary = {"mean": ".6f", "std": ".6f", "min": ".6f", "max": ".6f"}
    strict_order = data.raw_train["timestamp"].max() < data.raw_test["timestamp"].min()

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>CoInfoSim — UCI Air Quality Dataset Report</title>
<style>
body {{ font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif; max-width:1050px;
margin:0 auto; padding:2rem; line-height:1.5; color:#222; }}
h1 {{ border-bottom:3px solid #356aa0; padding-bottom:.4rem; }}
h2 {{ color:#1f3b66; border-bottom:1px solid #ddd; margin-top:2rem; }}
table.data {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.88rem; }}
table.data th, table.data td {{ border:1px solid #ccc; padding:.35rem .55rem; text-align:center; }}
table.data th {{ background:#f0f4f8; }}
.notice {{ background:#fff8e1; border-left:4px solid #e1ad01; padding:.8rem 1rem; }}
.figure {{ text-align:center; margin:1.2rem 0; }}
.figure img {{ max-width:100%; border:1px solid #eee; }}
code {{ background:#f5f5f5; padding:.1rem .3rem; }}
</style></head><body>
<h1>CoInfoSim — UCI Air Quality Dataset Report</h1>
<div class="notice">This report documents the fixed dataset preparation protocol.
The five PT08 variables are metal-oxide sensor responses, not measured gas
concentrations. The binary threshold is experimental and is not a health or
regulatory limit.</div>

<h2>1. Dataset identity and public provenance</h2>
<p><strong>Air Quality</strong>, UCI Machine Learning Repository dataset 360,
DOI <code>10.24432/C59K5F</code>. Official source:
<a href="https://archive.ics.uci.edu/dataset/360/air+quality">UCI Air Quality</a>.
UCI credits Saverio Vito and identifies the dataset as CC BY 4.0.</p>

<h2>2. Citation and associated study</h2>
<p>Vito, S. (2008). <em>Air Quality</em> [Dataset]. UCI Machine Learning Repository.
Associated study: S. De Vito, E. Massera, M. Piga, L. Martinotto and G. Francia,
“On field calibration of an electronic nose for benzene estimation in an urban
pollution monitoring scenario,” Sensors and Actuators B: Chemical 129(2), 2008.</p>

<h2>3. Local source file and SHA-256</h2>
<p>Committed original filename: <code>AirQualityUCI.csv</code>; local path:
<code>{html.escape(str(data.source_files['AirQualityUCI.csv']))}</code>.</p>
{file_hash_table_html(data.file_hashes)}

<h2>4. Raw schema and parsing rules</h2>
<p>Semicolon delimiter, decimal comma, day-first <code>Date</code>, dot-separated
<code>Time</code>, stable chronological ordering, and <code>-200</code> missing-value
sentinel replacement. Fully empty trailing rows and unnamed trailing columns are removed.</p>
{dataframe_html(schema)}

<h2>5. Sensor-response channel dictionary</h2>
<table class='data'><thead><tr><th>Index</th><th>Column</th><th>Interpretation</th></tr></thead>
<tbody>{channel_rows}</tbody></table>
<p>Parenthetical gases denote nominal sensor targets. Cross-sensitivity means these
responses must not be interpreted as five independent gas concentration measurements.</p>

<h2>6. Benzene reference and binary target definition</h2>
<p><code>C6H6(GT)</code> is the certified-analyzer benzene reference and is excluded
from classifier input <code>X</code>. The training-only linear 75th percentile is
<strong>{data.threshold_value:.6g}</strong>. Class 1 is defined by
<code>C6H6(GT) &gt;= {data.threshold_value:.6g}</code>; class 0 is below that threshold.</p>

<h2>7. Missing-value audit and complete-case cohort</h2>
{dataframe_html(missing_frame)}
{dataframe_html(cohort_frame)}
<p>Complete cases require all five sensor responses plus <code>C6H6(GT)</code>.
No imputation is performed, and all 31 subsets use this same cohort.</p>

<h2>8. Chronological train/test protocol</h2>
<p><code>split_index = floor(0.80 × {counts['complete_case_cohort']}) =
{data.split_index}</code>. Training: {data.train_first_timestamp} through
{data.train_last_timestamp}. Fixed future real test: {data.test_first_timestamp}
through {data.test_last_timestamp}. Cutoff timestamp: {data.cutoff_timestamp}.
All training timestamps precede all test timestamps: <strong>{strict_order}</strong>.</p>

<h2>9. Target threshold and class distributions</h2>
{dataframe_html(class_frame, float_cols={'class_1_prevalence': '.6f', 'majority_class_error_baseline': '.6f'})}
<p>Empirical test loss is affected by class prevalence; the majority-class error
baseline is therefore shown for each split.</p>
<div class="figure"><img src="{class_distribution}" alt="class distribution"/></div>

<h2>10. Raw attribute statistics</h2>
{dataframe_html(raw_summary, float_cols=float_summary)}

<h2>11. Standardization parameters</h2>
<p>Z-score means and standard deviations are fitted on the chronological training
reservoir only with <code>ddof=0</code>, then applied unchanged to the fixed test.</p>
{standardization_table_html(data.standardization)}

<h2>12. Standardized attribute statistics</h2>
{dataframe_html(standardized_summary, float_cols=float_summary)}
<div class="figure"><img src="{standardized_means_image}" alt="standardized means"/></div>

<h2>13. Training attribute correlation matrix</h2>
{dataframe_html(correlation.reset_index().rename(columns={'index': 'channel'}), float_cols={channel: '.4f' for channel in data.channel_names})}
<div class="figure"><img src="{correlation_image}" alt="correlation heatmap"/></div>

<h2>14. Training sensor-to-reference correlations</h2>
{dataframe_html(reference_correlations, float_cols={'correlation_with_C6H6(GT)': '.6f'})}

<h2>15. Temporal diagnostics and split visualization</h2>
<div class="figure"><img src="{temporal_image}" alt="benzene reference time series and split"/></div>

<h2>16. Train-to-test distribution shift summary</h2>
{dataframe_html(shift, float_cols={'train_pool': '.6f', 'fixed_test': '.6f', 'test_minus_train_standardized_mean': '.6f'})}

<h2>17. Leakage-control notes</h2>
<ul><li>Threshold estimation uses training reference values only.</li>
<li>Standardization and Gaussian/GMM fitting use training rows only.</li>
<li><code>C6H6(GT)</code>, all other GT columns, T, RH, AH, Date, Time and timestamp
are excluded from classifier inputs.</li><li>The fixed future real test is reused
unchanged by Real → Real, Single Gaussian → Real and GMM → Real.</li></ul>

<h2>18. Limitations</h2>
<p>The sensor array exhibits cross-sensitivity, sensor drift and concept drift.
The chronological shift can affect empirical loss. Converting a continuous benzene
reference into a binary target is an artificial binarization that discards magnitude.
The threshold is relative to this training period and supports no compliance claim.</p>

<h2>19. Reproducibility metadata</h2>
<ul><li>Dataset DOI: <code>10.24432/C59K5F</code></li>
<li>File: <code>AirQualityUCI.csv</code></li>
<li>Attributes in order: {html.escape(', '.join(data.channel_names))}</li>
<li>Reference: <code>{data.reference_name}</code>; quantile: {data.threshold_quantile};
threshold: {data.threshold_value:.12g}</li><li>Split index: {data.split_index};
training fraction: {data.train_fraction}</li><li>Standardization: training-only z-score,
<code>ddof=0</code></li></ul>
</body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
