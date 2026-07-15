"""HTML dataset provenance and reproducibility report for the Occupancy scenario.

Covers public provenance, local file roles, SHA-256 hashes, row counts,
class distribution, channel dictionary, standardization protocol,
leakage-control notes, and reproducibility metadata.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd  # noqa: E402

from coinfosim.datasets.occupancy import OccupancyData
from coinfosim.reports.dataset_common import (
    class_distribution_image,
    correlation_heatmap_image,
    dataframe_html as _dataframe_html,
    standardized_mean_comparison_image,
)


# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #

def _class_distribution_image(data: OccupancyData) -> str:
    return class_distribution_image(
        data.class_counts_by_file(),
        sorted(data.class_labels),
        title="Class distribution by source file",
    )


def _correlation_heatmap_image(data: OccupancyData) -> str:
    return correlation_heatmap_image(
        data.train_correlation(standardized=True),
        title="Training-pool channel correlation",
    )


def _standardized_summary_image(data: OccupancyData) -> str:
    return standardized_mean_comparison_image(
        data.standardized_channel_summary(),
        data.channel_names,
        title="Standardized channel means",
    )


def generate_occupancy_dataset_report(
    data: OccupancyData,
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_dataset_report.html",
) -> Path:
    """Generate the Occupancy Dataset provenance and reproducibility HTML report.

    The report includes: public provenance, citation, local file roles,
    SHA-256 hashes, train/test protocol, feature dictionary, row counts,
    class distribution, raw channel statistics, standardization parameters,
    standardized channel statistics, training-pool correlation matrix,
    leakage-control notes, and reproducibility metadata.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    # --- Build dataframes -------------------------------------------------- #
    row_counts_df = pd.DataFrame(
        [{"source_file": name, "rows": rows} for name, rows in data.row_counts().items()]
    )
    class_rows = []
    for name, counts in data.class_counts_by_file().items():
        row = {"source_file": name}
        for label in data.class_labels:
            row[f"class_{label}"] = counts.get(label, 0)
        class_rows.append(row)
    class_counts_df = pd.DataFrame(class_rows)
    standardization_df = data.standardization.as_dataframe().reset_index()
    standardization_df = standardization_df.rename(columns={"index": "channel"})
    raw_summary = data.raw_channel_summary()
    standardized_summary = data.standardized_channel_summary()
    corr_df = data.train_correlation(standardized=True).reset_index()
    corr_df = corr_df.rename(columns={"index": "channel"})

    float_fmt = {"mean": ".6f", "std": ".6f", "min": ".6f", "max": ".6f"}
    standardization_fmt = {"mean": ".6f", "std": ".6f"}

    # --- Build visualizations ---------------------------------------------- #
    class_distribution = _class_distribution_image(data)
    corr_heatmap = _correlation_heatmap_image(data)
    standardized_summary_plot = _standardized_summary_image(data)

    # --- Named section helpers --------------------------------------------- #
    total_train = data.raw_train.shape[0]
    total_test = data.raw_test.shape[0]
    channels = ", ".join(data.channel_names)

    hashes = data.file_hashes
    hashes_rows_html = "".join(
        f"<tr><td><code>{html.escape(fname)}</code></td>"
        f"<td style='font-family:monospace;font-size:.82em'>"
        f"{html.escape(h or '(not found)')}</td></tr>"
        for fname, h in hashes.items()
    ) if hashes else ""

    file_roles = {
        "datatraining.txt": "Training pool — balanced sampling source for all three arms",
        "datatest.txt": "Fixed real evaluation split (part 1) — concatenated with datatest2.txt",
        "datatest2.txt": "Fixed real evaluation split (part 2) — concatenated with datatest.txt",
    }
    file_roles_rows_html = "".join(
        f"<tr><td><code>{html.escape(fname)}</code></td>"
        f"<td>{html.escape(str(path))}</td>"
        f"<td>{html.escape(file_roles.get(fname, ''))}</td></tr>"
        for fname, path in data.source_files.items()
    )
    display_names = {
        "Temperature":   "Temperature (°C)",
        "Humidity":      "Humidity (%)",
        "Light":         "Light (lux)",
        "CO2":           "CO₂ (ppm)",
        "HumidityRatio": "Humidity Ratio (kg water/kg air)",
    }
    channel_dict_rows = "".join(
        f"<tr><td><code>X{i+1}</code></td><td><code>{html.escape(ch)}</code></td>"
        f"<td>{html.escape(display_names.get(ch, ch))}</td></tr>"
        for i, ch in enumerate(data.channel_names)
    )
    hash_dl_items = "".join(
        f"  <dt>{html.escape(fname)}</dt><dd><code style='font-size:.82em'>"
        f"{html.escape(h or '(not found)')}</code></dd>"
        for fname, h in hashes.items()
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>CoInfoSim — Occupancy Dataset Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1000px; padding: 2rem; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.8rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 2rem; color: #1f3b66; border-bottom: 1px solid #ddd;
        padding-bottom: .2rem; }}
  .notice {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem; margin: 1rem 0; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
  table.data th, table.data td {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: center; }}
  table.data th {{ background: #f0f4f8; }}
  table.data td:first-child {{ text-align: left; }}
  .figure {{ text-align: center; margin: 1.2rem 0; }}
  .figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  dl.meta {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; margin: .8rem 0; }}
  dl.meta dt {{ font-weight: 600; color: #444; }}
  code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
  a {{ color: #1f77b4; }}
  ul li {{ margin: .3rem 0; }}
</style>
</head>
<body>

<h1>CoInfoSim — Occupancy Dataset Report</h1>
<p><strong>Purpose:</strong> Dataset provenance, preprocessing protocol, and reproducibility
record for the Occupancy Detection scenario.</p>
<div class="notice"><strong>Scope.</strong> This report documents the raw data source,
local file roles, SHA-256 hashes, preprocessing steps, and leakage controls used in the
CoInfoSim Occupancy Detection scenario. Monte Carlo simulation results are in the
arm-specific reports.</div>

<h2>1. Dataset identity and public provenance</h2>
<dl class="meta">
  <dt>Dataset name</dt><dd>Occupancy Detection</dd>
  <dt>Repository</dt><dd>UCI Machine Learning Repository</dd>
  <dt>UCI dataset ID</dt><dd>357</dd>
  <dt>DOI</dt><dd><code>10.24432/C5X01N</code></dd>
  <dt>Donor / creator</dt><dd>Luis Candanedo</dd>
  <dt>Donation date</dt><dd>2016-02-28</dd>
  <dt>License</dt><dd>Creative Commons Attribution 4.0 International (CC BY 4.0)</dd>
  <dt>UCI URL</dt>
  <dd><a href="https://archive.ics.uci.edu/dataset/357/occupancy+detection"
      rel="noopener noreferrer" target="_blank">
      https://archive.ics.uci.edu/dataset/357/occupancy+detection</a></dd>
</dl>

<h2>2. Citation and external source</h2>
<p>
  L. Candanedo and V. Feldheim.<br/>
  <em>"Accurate occupancy detection of an office room from light, temperature, humidity
  and CO₂ measurements using statistical learning models."</em><br/>
  Energy and Buildings, Volume 112, 2016.
</p>
<p>Ground-truth occupancy was obtained from timestamped photographs taken every minute.
The dataset supports binary room-occupancy classification using temperature, humidity,
light and CO₂ sensors.</p>

<h2>3. Local source files and file roles</h2>
<table class='data'>
  <thead><tr><th>Filename</th><th>Local path</th><th>Role in experiment</th></tr></thead>
  <tbody>{file_roles_rows_html}</tbody>
</table>
<dl class="meta">
  <dt>Training pool total rows</dt><dd>{total_train:,}</dd>
  <dt>Fixed real test split rows</dt>
  <dd>{total_test:,} (datatest.txt + datatest2.txt, concatenated)</dd>
</dl>

<h2>4. Raw file SHA-256 hashes</h2>
<p>Hashes confirm the local raw files match the expected originals.
Recompute with <code>sha256sum data/raw/occupancy/*.txt</code>.</p>
<table class='data'>
  <thead><tr><th>Filename</th><th>SHA-256</th></tr></thead>
  <tbody>{hashes_rows_html}</tbody>
</table>

<h2>5. Train/test protocol</h2>
<dl class="meta">
  <dt>Training pool</dt>
  <dd><code>datatraining.txt</code> — {total_train:,} rows</dd>
  <dt>Fixed real evaluation split</dt>
  <dd><code>datatest.txt</code> + <code>datatest2.txt</code>
      concatenated — {total_test:,} rows</dd>
  <dt>Channels</dt><dd>{html.escape(channels)}</dd>
  <dt>Target variable</dt><dd><code>{html.escape(data.target_name)}</code></dd>
  <dt>Class labels</dt><dd>{list(data.class_labels)} (0 = unoccupied, 1 = occupied)</dd>
  <dt>Standardization</dt>
  <dd>Fit on training pool only (ddof=0); applied to both training pool and fixed test split</dd>
</dl>

<h2>6. Feature dictionary and target variable</h2>
<table class='data'>
  <thead><tr><th>Index</th><th>Column name</th><th>Description</th></tr></thead>
  <tbody>
    {channel_dict_rows}
    <tr><td>—</td><td><code>{html.escape(data.target_name)}</code></td>
    <td>Binary room occupancy label (0 = unoccupied, 1 = occupied)</td></tr>
  </tbody>
</table>
<p>Compact notation: <code>{{X1}}</code> = Temperature,
<code>{{X1, X3}}</code> = Temperature + Light,
<code>{{X1, X2, X3, X4, X5}}</code> = full five-channel feature set.</p>

<h2>7. Row counts</h2>
{_dataframe_html(row_counts_df)}

<h2>8. Class distribution</h2>
{_dataframe_html(class_counts_df)}
<div class="figure"><img src="{class_distribution}" alt="class distribution"/></div>

<h2>9. Raw channel statistics</h2>
<p>Statistics computed on raw (unstandardized) data, split by role.</p>
{_dataframe_html(raw_summary, float_cols=float_fmt)}

<h2>10. Standardization parameters</h2>
<p>Means and standard deviations estimated from <code>datatraining.txt</code> only (ddof=0).
Applied to both the training pool and the fixed test split.</p>
{_dataframe_html(standardization_df, float_cols=standardization_fmt)}

<h2>Standardized channel statistics</h2>
<p>After applying training-pool standardization parameters to both splits.</p>
{_dataframe_html(standardized_summary, float_cols=float_fmt)}
<div class="figure"><img src="{standardized_summary_plot}" alt="standardized channel means"/></div>

<h2>Training-pool correlation matrix</h2>
<p>Pearson correlation coefficients on standardized training-pool channels.</p>
{_dataframe_html(corr_df, float_cols={channel: ".3f" for channel in data.channel_names})}
<div class="figure"><img src="{corr_heatmap}" alt="correlation heatmap"/></div>

<h2>11. Leakage-control notes</h2>
<ul>
  <li><strong>Train-only standardization:</strong> channel means and standard deviations
    are estimated exclusively from <code>datatraining.txt</code>. The same parameters
    are applied to both the training pool and the fixed test split.</li>
  <li><strong>Fixed test split:</strong> <code>datatest.txt</code> and
    <code>datatest2.txt</code> are concatenated into a single fixed evaluation split that
    is never used for fitting, standardization, or model selection.</li>
  <li><strong>No test information in Gaussian/GMM fitting:</strong> the single-Gaussian
    and GMM class-conditional models are fitted exclusively from the standardized
    training pool. No test rows participate in parameter estimation.</li>
  <li><strong>Balanced training sampling:</strong> Monte Carlo replications draw balanced
    samples from the training pool using a deterministic seed, independently of the test
    split.</li>
</ul>

<h2>12. Reproducibility metadata</h2>
<dl class="meta">
  <dt>Dataset</dt><dd>UCI Occupancy Detection (ID 357, DOI 10.24432/C5X01N)</dd>
  <dt>Raw files</dt><dd>datatraining.txt, datatest.txt, datatest2.txt</dd>
  {hash_dl_items}
  <dt>Training pool rows</dt><dd>{total_train:,}</dd>
  <dt>Fixed test split rows</dt><dd>{total_test:,}</dd>
  <dt>Channels</dt><dd>{html.escape(channels)}</dd>
  <dt>Target</dt><dd>{html.escape(data.target_name)}</dd>
  <dt>Standardization rule</dt>
  <dd>z-score, parameters fitted from training pool only (ddof=0)</dd>
  <dt>Class labels</dt><dd>{list(data.class_labels)}</dd>
</dl>

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
