"""Reusable HTML reports for cooperative Monte Carlo results."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coinfosim.classifiers.registry import classifier_label
from coinfosim.reports.html_tabs import (
    TAB_CSS as _TAB_CSS,
    TAB_JS as _TAB_JS,
    tab_group as _tab_group,
)
from coinfosim.reports.predictive_profile_visualization import (
    figure_to_data_uri,
    reversal_matrix_figure,
    winner_matrix_figure,
)
from coinfosim.reports.report_tables import (
    compact_precision_diagnostics,
    full_loss_table,
    stable_classifier_order,
    subset_metadata_table,
)
from coinfosim.results.analysis import (
    best_subset_rankings,
    standard_threshold_comparisons,
)
from coinfosim.results.summary import summary_dataframe
from coinfosim.results.predictive_profile import (
    effective_winner_matrices,
    progressive_reversal_matrices,
    select_display_subsets_by_cardinality,
)
from coinfosim.simulation.monte_carlo import SimulationResult


Subset = Tuple[int, ...]


def _execution_configuration_html(result: SimulationResult) -> str:
    """Return concise execution audit fields for a report configuration list."""

    execution = result.metadata.get("execution")
    if not isinstance(execution, Mapping):
        return ""

    def value(key: str) -> str:
        return html.escape(str(execution.get(key, "unknown")))

    return f"""
  <dt>Execution backend</dt><dd><code>{value("backend")}</code></dd>
  <dt>Workers requested / effective</dt>
  <dd>{value("requested_workers")} / {value("effective_workers")}</dd>
  <dt>Numeric threads per worker</dt><dd>{value("worker_inner_threads")}</dd>
  <dt>Multiprocessing start method</dt><dd>{value("start_method")}</dd>
  <dt>Detected logical CPUs</dt><dd>{value("logical_cpus")}</dd>
  <dt>Fixed-test cache per worker</dt>
    <dd>{value("fixed_test_cache_bytes_per_worker")} bytes</dd>"""


def _classifier_configuration_html(result: SimulationResult) -> str:
    """Render resolved estimator parameters, seed policy, and calibration."""

    configurations = result.metadata.get("classifier_configurations")
    if not isinstance(configurations, Mapping) or not configurations:
        return (
            "<h3>Classifier configuration</h3>"
            "<p>Detailed classifier configuration was not recorded for this "
            "historical result.</p>"
        )
    rows = []
    for key in result.classifier_names:
        configuration = configurations.get(key, {})
        parameters = configuration.get("parameters", {})
        seed_policy = configuration.get("seed_policy", {})
        calibration = configuration.get("calibration", {})
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(key)}</code></td>"
            f"<td>{html.escape(classifier_label(key))}</td>"
            f"<td><code>{html.escape(str(configuration.get('estimator', 'unknown')))}</code></td>"
            f"<td><code>{html.escape(json.dumps(parameters, sort_keys=True))}</code></td>"
            f"<td><code>{html.escape(json.dumps(seed_policy, sort_keys=True))}</code></td>"
            f"<td>{html.escape(str(calibration.get('artifact_path', '—')))}</td>"
            f"<td><code>{html.escape(str(calibration.get('artifact_sha256', '—')))}</code></td>"
            f"<td>{html.escape(str(parameters.get('n_jobs', '—')))}</td>"
            "</tr>"
        )
    return (
        "<h3>Classifier configuration</h3>"
        "<table class='data'><thead><tr><th>Key</th><th>Display label</th>"
        "<th>Estimator class</th><th>Resolved parameters</th><th>Seed policy</th>"
        "<th>Calibration artifact</th><th>Calibration SHA-256</th>"
        "<th>Internal n_jobs</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def subset_display_label(
    subset: Sequence[int],
    channel_names: Optional[Sequence[str]] = None,
) -> str:
    """Return a readable display label for a zero-based attribute subset."""

    indices = tuple(int(i) for i in subset)
    if not indices:
        raise ValueError("subset must be non-empty")
    if channel_names is None:
        return "+".join(f"X{i + 1}" for i in indices)
    return "+".join(str(channel_names[i]) for i in indices)


def generate_monte_carlo_report(
    result: SimulationResult,
    output_dir: Path | str,
    filename: str,
    title: str,
    experiment_arm: str,
    description: str,
    channel_names: Optional[Sequence[str]] = None,
    fixed_test_description: str = "fixed test set",
    extra_sections: Optional[Mapping[str, str]] = None,
) -> Path:
    """Generate a self-contained Monte Carlo HTML report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    channel_names = tuple(
        channel_names or result.metadata.get("channel_names", []) or []
    ) or None

    loss_curves = {
        clf: _loss_curve_image(result, clf, channel_names)
        for clf in result.classifier_names
    }

    rankings_df = _with_subset_labels(
        best_subset_rankings(
            result.accumulator,
            result.classifier_names,
            result.sample_sizes,
            result.subsets,
        ),
        channel_names,
        source_col="best_subset",
        target_col="best_subset_label",
    )
    thresholds_df = _threshold_comparisons_with_labels(result, channel_names)
    summary_df = _summary_with_labels(result, channel_names)
    final_ranking_df = _final_ranking(result, channel_names)

    rankings_display = rankings_df[
        ["classifier_label", "n_per_class", "best_subset_label", "mean_loss"]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "n_per_class": "n_per_class",
            "best_subset_label": "Best subset",
            "mean_loss": "Mean test loss",
        }
    )
    thresholds_display = thresholds_df[
        [
            "classifier_label",
            "comparison",
            "subset_a_label",
            "subset_b_label",
            "n_star_grid",
            "n_star_interpolated",
            "n_before",
            "n_after",
            "delta_before",
            "delta_after",
            "threshold_status",
        ]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "comparison": "Comparison",
            "subset_a_label": "A",
            "subset_b_label": "B",
            "n_star_grid": "N* grid",
            "n_star_interpolated": "N* interpolated",
            "n_before": "n before",
            "n_after": "n after",
            "delta_before": "Delta before",
            "delta_after": "Delta after",
            "threshold_status": "Status",
        }
    )
    summary_display = summary_df[
        [
            "n_per_class",
            "subset_label",
            "classifier_label",
            "mean_loss",
            "standard_error",
            "replications",
        ]
    ].rename(
        columns={
            "n_per_class": "n_per_class",
            "subset_label": "Subset",
            "classifier_label": "Classifier",
            "mean_loss": "Mean test loss",
            "standard_error": "Std. error",
            "replications": "Reps",
        }
    )
    final_display = final_ranking_df[
        ["classifier_label", "rank", "subset_label", "mean_loss", "standard_error"]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "rank": "Rank",
            "subset_label": "Subset",
            "mean_loss": "Mean test loss",
            "standard_error": "Std. error",
        }
    )

    classifiers = ", ".join(classifier_label(c) for c in result.classifier_names)
    subsets = ", ".join(
        subset_display_label(subset, channel_names) for subset in result.subsets
    )
    curves_html = "".join(
        f"<div class='figure'><img src='{src}' alt='loss curve "
        f"{html.escape(classifier_label(clf))}'/></div>"
        for clf, src in loss_curves.items()
    )
    extra_html = "".join(
        f"<h2>{html.escape(section_title)}</h2>{section_html}"
        for section_title, section_html in (extra_sections or {}).items()
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1100px; padding: 2rem; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.8rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 2rem; color: #1f3b66; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
  .notice {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem; margin: 1rem 0; }}
  .question {{ font-style: italic; background: #eef5fb; padding: .8rem 1rem; border-left: 4px solid #1f77b4; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .86rem; }}
  table.data th, table.data td {{ border: 1px solid #ccc; padding: .32rem .5rem; text-align: center; }}
  table.data th {{ background: #f0f4f8; }}
  table.matrix {{ border-collapse: collapse; display: inline-block; margin: .3rem 0; }}
  table.matrix td {{ border: 1px solid #bbb; padding: .25rem .55rem; text-align: right; font-family: monospace; }}
  .figure {{ text-align: center; margin: 1.2rem 0; }}
  .figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  dl.meta {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }}
  dl.meta dt {{ font-weight: 600; color: #444; }}
  code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
</style>
</head>
<body>

<h1>{html.escape(title)}</h1>
<p><strong>Experiment arm:</strong> {html.escape(experiment_arm)}</p>
<p class="question">{html.escape(description)}</p>

<div class="notice"><strong>Metric.</strong> This report uses empirical test loss only,
defined as the misclassification rate on the fixed test set.</div>

<h2>Run configuration</h2>
<dl class="meta">
  <dt>Execution mode</dt><dd><code>{html.escape(str(result.config.mode))}</code></dd>
  <dt>Sample sizes</dt><dd>{list(result.config.sample_sizes)}</dd>
  <dt>Classifiers</dt><dd>{html.escape(classifiers)}</dd>
  <dt>Number of classifiers</dt><dd>{len(result.classifier_names)}</dd>
  <dt>Attribute subsets</dt><dd>{html.escape(subsets)}</dd>
  <dt>Number of attribute subsets</dt><dd>{len(result.subsets)}</dd>
  <dt>Fixed test set</dt><dd>{html.escape(fixed_test_description)} ({result.metadata.get("fixed_test_size", "unknown")} rows)</dd>
  <dt>Monte Carlo stopping rule</dt><dd>Standard-error rule at replication batch boundaries</dd>
  <dt>Target CI half-width</dt><dd>{result.config.ci_half_width_target}</dd>
  <dt>Min / max replications</dt><dd>{result.config.min_replications} / {result.config.max_replications}</dd>
  <dt>Replication batch size</dt><dd>{result.config.replication_batch_size}</dd>
  <dt>Base seed</dt><dd>{result.config.base_seed}</dd>
{_execution_configuration_html(result)}
  <dt>Runtime</dt><dd>{result.runtime_seconds:.2f} s</dd>
</dl>
{_classifier_configuration_html(result)}

{extra_html}

<h2>Monte Carlo stopping and replications</h2>
{_stopping_table_html(result)}

<h2>Loss curves</h2>
<p>One panel per classifier; error bars show standard error across replications.</p>
{curves_html}

<h2>Best subset by sample size</h2>
{_dataframe_html(rankings_display, float_cols={"Mean test loss": ".4f"})}

<h2>Final ranking at largest sample size</h2>
{_dataframe_html(final_display, float_cols={"Mean test loss": ".4f", "Std. error": ".4f"})}

<h2>Interpolated N-star</h2>
<p><code>Delta(n) = L_A(n) - L_B(n)</code>. Positive values mean subset B has lower empirical test loss than subset A.</p>
{_dataframe_html(thresholds_display, float_cols={"N* interpolated": ".2f", "Delta before": ".4f", "Delta after": ".4f"})}

<h2>Monte Carlo uncertainty summary</h2>
{_dataframe_html(summary_display, float_cols={"Mean test loss": ".4f", "Std. error": ".4f"})}

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path


def gaussian_parameters_section(model, channel_names: Sequence[str]) -> str:
    """Return HTML for Gaussian class-conditional parameter estimates."""

    rows = []
    for label in model.class_labels:
        rows.append(
            "<h3>Class "
            + html.escape(str(label))
            + "</h3>"
            + "<p><strong>Mean:</strong> "
            + html.escape(_vector_text(model.mean(label), channel_names))
            + "</p>"
            + _matrix_html(model.covariance(label), channel_names)
        )
    return "".join(rows)


def gmm_parameters_section(model, channel_names: Sequence[str]) -> str:
    """Return HTML for class-conditional GMM parameter estimates and selection.

    Includes the model-selection configuration, per-class selected number of
    components, a per-class BIC/AIC table over candidate component counts, and,
    per class, the mixture weights, component means and component covariance
    matrices.
    """
    config_html = ""
    class_tabs: List[Tuple[str, str, str]] = []

    # Model-selection configuration (shared; read from the first class if any).
    any_selection = next(
        (model.model_selection(label) for label in model.class_labels
         if model.model_selection(label)),
        {},
    )
    if any_selection:
        cfg_rows = [
            ("Criterion", any_selection.get("criterion")),
            ("Covariance type", any_selection.get("covariance_type")),
            ("reg_covar", any_selection.get("reg_covar")),
            ("n_init", any_selection.get("n_init")),
        ]
        cfg_html = "".join(
            f"<tr><th>{html.escape(str(k))}</th>"
            f"<td>{html.escape(str(v))}</td></tr>"
            for k, v in cfg_rows
            if v is not None
        )
        config_html = (
            "<h4>Model-selection configuration</h4>"
            f"<table class='data'><tbody>{cfg_html}</tbody></table>"
        )

    for label in model.class_labels:
        selection = model.model_selection(label)
        weights = model.component_weights(label)
        means = model.component_means(label)
        covs = model.component_covariances(label)
        k = model.selected_components(label)

        summary = (
            f"<p><strong>Selected components:</strong> {k}"
            + (
                f" (of candidates {selection.get('candidate_components')})"
                if selection
                else ""
            )
            + (
                f" &middot; fitted on {selection.get('n_samples')} samples"
                if selection
                else ""
            )
            + "</p>"
        )

        scores = (selection or {}).get("scores") or {}
        if scores:
            header = "<tr><th>K</th><th>BIC</th><th>AIC</th></tr>"
            body = "".join(
                f"<tr><td>{html.escape(str(kk))}</td>"
                f"<td>{float(sc.get('bic', float('nan'))):.2f}</td>"
                f"<td>{float(sc.get('aic', float('nan'))):.2f}</td></tr>"
                for kk, sc in sorted(scores.items(), key=lambda kv: int(kv[0]))
            )
            summary += (
                "<p><strong>Model-selection scores:</strong></p>"
                f"<table class='data'><thead>{header}</thead>"
                f"<tbody>{body}</tbody></table>"
            )

        weight_text = ", ".join(f"{float(w):.4f}" for w in weights)
        summary += f"<p><strong>Mixture weights:</strong> [{weight_text}]</p>"

        comp_tabs = []
        for j in range(k):
            comp_tabs.append((
                f"c{label}_{j}",
                f"Component {j}",
                f"<p><strong>Component {j} mean:</strong> "
                + html.escape(_vector_text(means[j], channel_names))
                + "</p>"
                + f"<p><strong>Component {j} covariance:</strong></p>"
                + _matrix_html(covs[j], channel_names)
            ))
        class_panel = (
            "<h4>Model-selection summary</h4>"
            + summary
            + "<h4>Component details</h4>"
            + _tab_group(f"gmm-components-class-{label}", comp_tabs, comp_tabs[0][0])
        )
        class_tabs.append((str(label), f"Class {label}", class_panel))
    class_tabs_html = (
        _tab_group("gmm-class-details", class_tabs, class_tabs[0][0])
        if class_tabs
        else ""
    )
    return (
        "<div class='selector-caption'>Use the selectors to inspect model-selection "
        "summary versus per-class component mean vectors and covariance matrices.</div>"
        + config_html
        + class_tabs_html
    )


def _loss_curve_image(
    result: SimulationResult,
    classifier_name: str,
    channel_names: Optional[Sequence[str]],
) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_sizes = result.sample_sizes
    cmap = plt.get_cmap("tab20")
    for idx, subset in enumerate(result.subsets):
        means = [
            result.accumulator.mean_loss(n, subset, classifier_name)
            for n in sample_sizes
        ]
        errs = [
            result.accumulator.standard_error(n, subset, classifier_name)
            for n in sample_sizes
        ]
        ax.errorbar(
            sample_sizes,
            means,
            yerr=errs,
            marker="o",
            markersize=3,
            linewidth=1,
            capsize=2,
            color=cmap(idx % cmap.N),
            alpha=0.85,
            label=subset_display_label(subset, channel_names),
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_per_class")
    ax.set_ylabel("Empirical test loss")
    ax.set_title(f"Empirical test loss - {classifier_label(classifier_name)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Subset", fontsize=6, ncol=3)
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _summary_with_labels(
    result: SimulationResult,
    channel_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    df = summary_dataframe(
        result.accumulator,
        result.sample_sizes,
        result.subsets,
        result.classifier_names,
    )
    return _with_subset_labels(df, channel_names)


def _threshold_comparisons_with_labels(
    result: SimulationResult,
    channel_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    df = standard_threshold_comparisons(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )
    df = _with_subset_labels(df, channel_names, "subset_a", "subset_a_label")
    df = _with_subset_labels(df, channel_names, "subset_b", "subset_b_label")
    if channel_names is not None:
        full_subset = tuple(sorted(max(result.subsets, key=len)))
        full_label = subset_display_label(full_subset, channel_names)
        df["comparison"] = df["comparison"].replace(
            {
                "X1+X3 vs X1": (
                    f"{subset_display_label((0, 2), channel_names)} vs "
                    f"{subset_display_label((0,), channel_names)}"
                ),
                "X1+X2+X3 vs X1+X2": (
                    f"{subset_display_label((0, 1, 2), channel_names)} vs "
                    f"{subset_display_label((0, 1), channel_names)}"
                ),
                "full subset vs best pair": f"{full_label} vs best pair",
            }
        )
    return df


def _with_subset_labels(
    df: pd.DataFrame,
    channel_names: Optional[Sequence[str]],
    source_col: str = "subset",
    target_col: str = "subset_label",
) -> pd.DataFrame:
    if channel_names is None or source_col not in df.columns:
        return df
    out = df.copy()
    out[target_col] = [
        subset_display_label(subset, channel_names) for subset in out[source_col]
    ]
    return out


def _final_ranking(
    result: SimulationResult,
    channel_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    n = max(result.sample_sizes)
    summary = _summary_with_labels(result, channel_names)
    summary = summary[summary["n_per_class"] == n].copy()
    summary = summary.sort_values(
        ["classifier", "mean_loss", "standard_error", "subset_label"]
    )
    summary["rank"] = summary.groupby("classifier").cumcount() + 1
    return summary


def _dataframe_html(df: pd.DataFrame, float_cols: Optional[Dict[str, str]] = None) -> str:
    float_cols = float_cols or {}
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            value = row[col]
            if col in float_cols and value is not None and not _is_nan(value):
                cells.append(f"<td>{float(value):{float_cols[col]}}</td>")
            elif value is None or _is_nan(value):
                cells.append("<td>&mdash;</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<table class='data'><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _is_nan(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _stopping_table_html(result: SimulationResult) -> str:
    rows = []
    for n in result.sample_sizes:
        info = result.stopping_info[n]
        rows.append(
            f"<tr><td>{n}</td><td>{info.replications}</td>"
            f"<td>{html.escape(info.reason)}</td>"
            f"<td>{info.max_ci_half_width:.4f}</td></tr>"
        )
    return (
        "<table class='data'><thead><tr>"
        "<th>n_per_class</th><th>replications</th>"
        "<th>stopping reason</th><th>max CI half-width</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _vector_text(vector: np.ndarray, channel_names: Sequence[str]) -> str:
    pairs = [f"{name}={value:.4f}" for name, value in zip(channel_names, vector)]
    return "[" + ", ".join(pairs) + "]"


def _matrix_html(matrix: np.ndarray, channel_names: Sequence[str]) -> str:
    header = "<tr><th></th>" + "".join(
        f"<th>{html.escape(name)}</th>" for name in channel_names
    ) + "</tr>"
    rows = []
    for name, row in zip(channel_names, matrix):
        cells = "".join(f"<td>{float(value):.4f}</td>" for value in row)
        rows.append(f"<tr><th>{html.escape(name)}</th>{cells}</tr>")
    return f"<table class='matrix'><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


# --------------------------------------------------------------------------- #
# Structured dataset-anchored Monte Carlo report
# --------------------------------------------------------------------------- #

_STRUCTURED_CSS = """
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1100px; padding: 2rem; color: #222; line-height: 1.5; }
  h1 { font-size: 1.8rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }
  h2 { font-size: 1.3rem; margin-top: 2rem; color: #1f3b66; border-bottom: 1px solid #ddd;
       padding-bottom: .2rem; }
  h3 { font-size: 1.05rem; margin-top: 1.2rem; color: #444; }
  h4 { font-size: .96rem; margin-top: 1rem; color: #444; }
  .notice { background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem;
            margin: 1rem 0; }
  .role { background: #eef5fb; border-left: 4px solid #1f77b4; padding: .8rem 1rem;
          margin: 1rem 0; font-style: italic; }
  table.data { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .86rem; }
  table.data th, table.data td { border: 1px solid #ccc; padding: .32rem .5rem;
                                  text-align: center; }
  table.data th { background: #f0f4f8; }
  table.matrix { border-collapse: collapse; display: inline-block; margin: .3rem 0; }
  table.matrix td { border: 1px solid #bbb; padding: .25rem .55rem; text-align: right;
                    font-family: monospace; }
  .figure { text-align: center; margin: 1.2rem 0; }
  .figure img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
  dl.meta { display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }
  dl.meta dt { font-weight: 600; color: #444; }
  code { background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }
  details summary { cursor: pointer; font-weight: 600; margin: .5rem 0; color: #1f3b66; }
  a { color: #1f77b4; }
  ul li { margin: .3rem 0; }

  /* Sticky attribute legend (single, persistent, opaque). */
  .attribute-legend { position: sticky; top: 0; z-index: 100; background: #fffbe6;
    border: 1px solid #e6d97a; border-bottom: 2px solid #d9c65a;
    border-radius: 0 0 6px 6px; padding: .55rem 1rem; font-size: .9rem;
    margin: 0 0 1.2rem 0; box-shadow: 0 2px 6px rgba(0,0,0,.10); }
  .attribute-legend .leg-item { margin-right: 1.2rem; white-space: nowrap; }

  /* Yellow reference highlight for the full-subset row / baseline. */
  tr.ref-row td { background: #fff2b8; font-weight: 600; }
  .ref-note { color: #8a6d00; font-size: .85rem; }

  .wr-pair { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center;
             align-items: flex-start; }
  .wr-pair .figure { margin: .5rem; }
"""

_DATASET_PROVENANCE_DEFAULT = """
<p>Dataset: <strong>UCI Occupancy Detection</strong> (ID 357, DOI 10.24432/C5X01N,
Luis Candanedo, CC BY 4.0).<br/>
Training pool: <code>datatraining.txt</code>.
Fixed real evaluation split: <code>datatest.txt + datatest2.txt</code>.
Standardization parameters are estimated from the training pool only.</p>
"""

# Subscript digits for compact variable notation, e.g. X₁, X₃.
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _compact_sub(subset: Sequence[int]) -> str:
    """Compact subscript variable notation for a zero-based subset: ``{X₁, X₃}``."""
    return "{" + ", ".join("X" + str(int(i) + 1).translate(_SUBSCRIPTS) for i in subset) + "}"


def _sticky_legend_html(channel_names: Sequence[str]) -> str:
    """Return the single sticky attribute legend (variable → attribute name)."""
    display = {"CO2": "CO₂", "HumidityRatio": "Humidity Ratio"}
    items = "".join(
        f"<span class='leg-item'><b>X{str(i + 1).translate(_SUBSCRIPTS)}</b> = "
        f"{html.escape(display.get(ch, ch))}</span>"
        for i, ch in enumerate(channel_names)
    )
    return (
        "<div class='attribute-legend' id='sticky-legend'>"
        "<strong>Attributes:</strong> " + items + "</div>"
    )


# --------------------------------------------------------------------------- #
# Per-classifier analysis helpers (compact subscript notation)
# --------------------------------------------------------------------------- #

def _best_k_subsets(result: SimulationResult, classifier: str, n: int) -> Dict[int, Subset]:
    """Return {k: best subset of size k} at sample size ``n`` for ``classifier``."""
    d = max(len(s) for s in result.subsets)
    out: Dict[int, Subset] = {}
    for k in range(1, d + 1):
        cands = [tuple(s) for s in result.subsets if len(s) == k]
        if not cands:
            continue
        out[k] = min(cands, key=lambda s: result.accumulator.mean_loss(n, s, classifier))
    return out


def _loss_curve_by_dimension_image(
    result: SimulationResult,
    classifier: str,
    channel_names: Optional[Sequence[str]],
) -> str:
    """Loss vs N for the best k-attribute subset per dimension plus the full subset.

    The full (largest-dimension) subset is drawn as a thick yellow reference
    line. Best k-attribute subsets are selected at the largest evaluated N.
    """
    sizes = result.sample_sizes
    n_max = max(sizes)
    best_k = _best_k_subsets(result, classifier, n_max)
    d = max(best_k) if best_k else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    ks = sorted(best_k)
    non_full = [k for k in ks if k != d]
    for idx, k in enumerate(ks):
        subset = best_k[k]
        means = [result.accumulator.mean_loss(n, subset, classifier) for n in sizes]
        errs = [result.accumulator.standard_error(n, subset, classifier) for n in sizes]
        if k == d:
            # Yellow reference line for the full-dimension subset.
            ax.plot(
                sizes, means, color="#e6b800", linewidth=4.0, zorder=2, alpha=0.95,
                label=f"{_compact_sub(subset)} — full (reference)",
            )
            ax.errorbar(
                sizes, means, yerr=errs, fmt="o", color="#a8830a", markersize=4,
                capsize=2, zorder=3, linewidth=0,
            )
        else:
            frac = (non_full.index(k) / max(1, len(non_full) - 1)) * 0.82 if len(non_full) > 1 else 0.4
            ax.errorbar(
                sizes, means, yerr=errs, marker="o", markersize=4, linewidth=1.4,
                capsize=2, color=cmap(frac), alpha=0.9,
                label=f"{_compact_sub(subset)} — best {k}-attr",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_per_class")
    ax.set_ylabel("Empirical test loss")
    ax.set_title(f"Best k-attribute subsets — {classifier_label(classifier)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Subset (best per dimension)", fontsize=8)
    return _fig_to_base64(fig)


def _loss_curve_nested_cardinality_image(
    result: SimulationResult,
    classifier: str,
    cardinality: int,
    full_reference_label: Optional[str] = None,
) -> str:
    """Loss vs N for every subset of one cardinality plus the full reference."""
    sizes = result.sample_sizes
    d = max(len(s) for s in result.subsets)
    full_subset = tuple(range(d))
    reference_label = full_reference_label or f"Full-{d} reference"
    subsets = sorted(tuple(s) for s in result.subsets if len(s) == cardinality)
    plot_subsets = [s for s in subsets if s != full_subset]

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab20")
    for idx, subset in enumerate(plot_subsets):
        means = [result.accumulator.mean_loss(n, subset, classifier) for n in sizes]
        errs = [result.accumulator.standard_error(n, subset, classifier) for n in sizes]
        ax.errorbar(
            sizes, means, yerr=errs, marker="o", markersize=3, linewidth=1.2,
            capsize=2, color=cmap(idx % cmap.N), alpha=0.82,
            label=_compact_sub(subset),
        )

    means = [result.accumulator.mean_loss(n, full_subset, classifier) for n in sizes]
    errs = [result.accumulator.standard_error(n, full_subset, classifier) for n in sizes]
    ax.plot(
        sizes, means, color="#e6b800", linewidth=4.0, zorder=3, alpha=0.95,
        label=f"{reference_label}: {_compact_sub(full_subset)}",
    )
    ax.errorbar(
        sizes, means, yerr=errs, fmt="o", color="#a8830a", markersize=4,
        capsize=2, zorder=4, linewidth=0,
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_per_class")
    ax.set_ylabel("Empirical test loss")
    ax.set_title(
        f"{cardinality}-attribute subsets + full reference — {classifier_label(classifier)}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Subset", fontsize=7, ncol=2)
    return _fig_to_base64(fig)


def _loss_curves_panel(
    result: SimulationResult,
    classifier: str,
    channel_names: Optional[Sequence[str]],
    full_reference_label: Optional[str] = None,
) -> str:
    best_panel = (
        "<div class='figure'><img src='"
        + _loss_curve_by_dimension_image(result, classifier, channel_names)
        + f"' alt='Best-by-cardinality loss curves {html.escape(classifier_label(classifier))}'/></div>"
    )
    nested_tabs = []
    for k in range(1, max(len(s) for s in result.subsets)):
        nested_tabs.append((
            f"k{k}",
            f"{k}-attribute + full reference",
            "<div class='figure'><img src='"
            + _loss_curve_nested_cardinality_image(
                result, classifier, k, full_reference_label
            )
            + f"' alt='{k}-attribute nested loss curves {html.escape(classifier_label(classifier))}'/></div>",
        ))
    nested_panel = _tab_group(f"losscurve-nested-{classifier}", nested_tabs, "k1")
    view_tabs = [
        ("best", "Best by cardinality", best_panel),
        ("nested", "Nested cardinality", nested_panel),
    ]
    return _tab_group(f"losscurve-view-{classifier}", view_tabs, "best")


_DIM_NAMES = {
    1: "Best single attribute",
    2: "Best pair",
    3: "Best triple",
    4: "Best four-attribute",
    5: "Best five-attribute",
}


def _best_by_dimension_table(result: SimulationResult, classifier: str) -> str:
    """Comparative table: best subset per dimension at the largest N.

    The full-dimension row is highlighted yellow as the reference subset.
    """
    n = max(result.sample_sizes)
    best_k = _best_k_subsets(result, classifier, n)
    d = max(best_k) if best_k else 0
    rows_html = ""
    for k in sorted(best_k):
        subset = best_k[k]
        mean = result.accumulator.mean_loss(n, subset, classifier)
        se = result.accumulator.standard_error(n, subset, classifier)
        hw = 1.96 * se
        if k == d:
            label = f"Full {d}-attribute (reference)"
            rowcls = " class='ref-row'"
        else:
            label = _DIM_NAMES.get(k, f"Best {k}-attribute")
            rowcls = ""
        rows_html += (
            f"<tr{rowcls}><td>{html.escape(label)}</td>"
            f"<td>{_compact_sub(subset)}</td>"
            f"<td>{mean:.4f}</td><td>{se:.4f}</td><td>{hw:.4f}</td></tr>"
        )
    head = (
        "<th>Comparison</th><th>Subset</th><th>Mean test loss</th>"
        "<th>SE</th><th>CI95 half-width</th>"
    )
    return (
        f"<table class='data'><thead><tr>{head}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
        "<p class='ref-note'>The highlighted row is the full-attribute reference subset.</p>"
    )


def _best_by_n_table(result: SimulationResult, classifier: str) -> str:
    """Best overall subset per sample size (compact notation)."""
    rows_html = ""
    for n in result.sample_sizes:
        best = min(
            (tuple(s) for s in result.subsets),
            key=lambda s: result.accumulator.mean_loss(n, s, classifier),
        )
        mean = result.accumulator.mean_loss(n, best, classifier)
        se = result.accumulator.standard_error(n, best, classifier)
        rows_html += (
            f"<tr><td>{n}</td><td>{_compact_sub(best)}</td>"
            f"<td>{mean:.4f}</td><td>{se:.4f}</td></tr>"
        )
    head = "<th>n_per_class</th><th>Best subset</th><th>Mean test loss</th><th>SE</th>"
    return (
        f"<table class='data'><thead><tr>{head}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _ranking_table(result: SimulationResult, classifier: str, n: int) -> str:
    """Complete subset ranking for one classifier and sample size."""
    d = max(len(s) for s in result.subsets)
    entries = [
        (
            tuple(s),
            result.accumulator.mean_loss(n, tuple(s), classifier),
            result.accumulator.standard_error(n, tuple(s), classifier),
        )
        for s in result.subsets
    ]
    entries.sort(key=lambda e: (e[1], e[2]))
    rows_html = ""
    for rank, (s, mean, se) in enumerate(entries, 1):
        hw = 1.96 * se
        rowcls = " class='ref-row'" if len(s) == d else ""
        ref = "Full reference" if len(s) == d else ""
        rows_html += (
            f"<tr{rowcls}><td>{rank}</td><td>{_compact_sub(s)}</td>"
            f"<td>{len(s)}</td><td>{mean:.4f}</td><td>{se:.4f}</td>"
            f"<td>{hw:.4f}</td><td>{html.escape(ref)}</td></tr>"
        )
    head = (
        "<th>Rank</th><th>Subset</th><th>k</th><th>Mean loss</th>"
        "<th>SE</th><th>CI95 half-width</th><th>Reference</th>"
    )
    return (
        f"<table class='data'><thead><tr>{head}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )


def _ranking_by_sample_size_panel(result: SimulationResult, classifier: str) -> str:
    tabs = [
        (str(n), str(n), _ranking_table(result, classifier, int(n)))
        for n in result.sample_sizes
    ]
    return _tab_group(f"rank-n-{classifier}", tabs, str(max(result.sample_sizes)))


def _pairwise_profile_dynamics_panel(
    result: SimulationResult,
    classifier: str,
) -> str:
    """Render one classifier's arm-local paired winner/reversal (W/R) panel."""

    display_subsets = select_display_subsets_by_cardinality(
        result, [classifier]
    )[classifier]
    labels = [_compact_sub(subset) for subset in display_subsets]
    sample_sizes = list(result.sample_sizes)

    winner_matrices = {
        int(item["n_per_class"]): item["matrix"]
        for item in effective_winner_matrices(result, classifier, display_subsets)
    }
    reversal_matrices = {
        int(item["n_prefix"]): item["matrix"]
        for item in progressive_reversal_matrices(
            result, classifier, display_subsets
        )
    }

    n_tabs = []
    for n in sample_sizes:
        winner_uri = figure_to_data_uri(
            winner_matrix_figure(
                winner_matrices[n],
                labels,
                f"{classifier_label(classifier)} — arm-local Winner matrix — N={n}",
            )
        )
        reversal_uri = figure_to_data_uri(
            reversal_matrix_figure(
                reversal_matrices[n],
                labels,
                f"{classifier_label(classifier)} — arm-local Reversal matrix "
                f"— N={n}",
            )
        )
        note = (
            "<p class='ref-note'>N is the first evaluated sample size, so no "
            "reversal can yet exist: the Reversal matrix is empty.</p>"
            if n == sample_sizes[0]
            else ""
        )
        n_tabs.append(
            (
                str(n),
                f"N = {n}",
                "<div class='wr-pair'>"
                f"<div class='figure'><img src='{winner_uri}' "
                f"alt='winner matrix {html.escape(classifier_label(classifier))} "
                f"N={n}'/></div>"
                f"<div class='figure'><img src='{reversal_uri}' "
                f"alt='reversal matrix {html.escape(classifier_label(classifier))} "
                f"N={n}'/></div>"
                "</div>" + note,
            )
        )

    selection = ", ".join(labels)
    return (
        "<p class='ref-note'><strong>Arm-local display selection at Nmax:</strong> "
        f"{html.escape(selection)}. Numerical structures use all subsets.</p>"
        "<p>The Winner matrix <code>W</code> shows who currently wins each "
        "unordered subset pair; the Reversal matrix <code>R</code> shows when "
        "that pair last changed winner through the selected prefix. An exact "
        "tie after a pair's first strict winner carries the previous winner "
        "forward in <code>W</code> rather than resetting it to unresolved. An "
        "<code>R</code> cell exists only for a pair with at least one valid "
        "winner reversal: a defined effective winner at both the previous and "
        "the current sample size that differs between the two.</p>"
        + _tab_group(f"pairwise-profile-dynamics-{classifier}-n", n_tabs, str(sample_sizes[0]))
    )


def generate_structured_monte_carlo_report(
    result: SimulationResult,
    output_dir: Path | str,
    filename: str,
    title: str,
    arm_id: str,
    arm_label: str,
    arm_summary: str,
    scientific_role: str,
    channel_names: Optional[Sequence[str]] = None,
    fixed_test_description: str = "fixed real Occupancy evaluation split",
    dataset_provenance_html: str = _DATASET_PROVENANCE_DEFAULT,
    dataset_name: str = "UCI Occupancy Detection",
    target_description: str = "binary Occupancy label",
    full_reference_label: Optional[str] = None,
    training_protocol_html: str = "",
    model_section_html: str = "",
    reproducibility_html: str = "",
    robustness_html: str = "",
    validity_html: str = "",
    nstar_selection_result: Optional[SimulationResult] = None,
    export_csvs: bool = True,
) -> Path:
    """Generate a structured, navigable 16-section Monte Carlo HTML report.

    Used for the standard dataset-anchored arms (Real → Real, Single Gaussian
    → Real, GMM → Real). The report uses a single sticky
    attribute legend, compact subscript variable notation (``{X₁, X₃}``), static
    per-classifier tab selectors (Linear SVM default), a yellow full-subset
    reference highlight, and expandable full tables. The old
    :func:`generate_monte_carlo_report` is kept for Sprint 1 backward
    compatibility.

    CSV files are exported alongside the HTML when *export_csvs* is ``True``:
    ``full_loss_table_{arm_id}.csv``, ``subset_metadata_table.csv``,
    ``precision_diagnostics_{arm_id}.csv``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    ch: Optional[Sequence[str]] = tuple(
        channel_names or result.metadata.get("channel_names", []) or []
    ) or None
    dimension = max(len(subset) for subset in result.subsets)
    reference_label = full_reference_label or f"Full-{dimension} reference"

    if export_csvs:
        _export_csv_tables(result, output_dir, arm_id, ch)

    # Classifier ordering (Linear SVM first, present only).
    clf_order = stable_classifier_order(result.classifier_names)
    present = [c for c in clf_order if c in result.classifier_names]
    default_clf = present[0] if present else ""

    # --- Sticky legend + config summary ----------------------------------- #
    legend_html = _sticky_legend_html(ch) if ch else ""
    classifiers_str = ", ".join(classifier_label(c) for c in present)
    subsets_compact = ", ".join(_compact_sub(s) for s in result.subsets)

    # --- Section 9: precision diagnostics (fixed target column) ----------- #
    prec_df = compact_precision_diagnostics(result)
    prec_display = prec_df.rename(
        columns={
            "n_per_class": "n_per_class",
            "replications": "Replications",
            "max_se": "Max SE",
            "max_ci95_half_width": "Max CI95 half-width",
            "target_ci95_half_width": "Target CI95 half-width",
            "stopping_reason": "Stopping reason",
        }
    )
    prec_html = _dataframe_html(
        prec_display,
        float_cols={
            "Max SE": ".5f",
            "Max CI95 half-width": ".5f",
            "Target CI95 half-width": ".5f",
        },
    )

    # --- Section 10: loss curves by dimension, tabs by classifier --------- #
    losscurve_tabs = [
        (
            c,
            classifier_label(c),
            _loss_curves_panel(result, c, ch, reference_label),
        )
        for c in present
    ]
    losscurve_html = _tab_group("losscurve", losscurve_tabs, default_clf)

    # --- Section 11: best subset comparisons, tabs by classifier ---------- #
    bestsub_tabs = [
        (
            c,
            classifier_label(c),
            "<h3>Best subset by dimension (at largest N)</h3>"
            + _best_by_dimension_table(result, c)
            + "<h3>Best overall subset by sample size</h3>"
            + _best_by_n_table(result, c),
        )
        for c in present
    ]
    bestsub_html = _tab_group("bestsub", bestsub_tabs, default_clf)

    # --- Section 12: final ranking (full table in <details>) -------------- #
    finalrank_tabs = [
        (c, classifier_label(c), _ranking_by_sample_size_panel(result, c))
        for c in present
    ]
    finalrank_html = _tab_group("finalrank", finalrank_tabs, default_clf)

    # --- Section 13: arm-local pairwise winner and reversal dynamics ------ #
    structural_tabs = [
        (c, classifier_label(c), _pairwise_profile_dynamics_panel(result, c))
        for c in present
    ]
    structural_html = _tab_group("structural", structural_tabs, default_clf)

    csv_links = _csv_links_html(output_dir, arm_id) if export_csvs else ""
    full_table_details = _full_table_details(result, ch, arm_id)

    def _section(num: int, title_text: str, body: str) -> str:
        if not body.strip():
            body = (
                "<p class='selector-caption'>No additional arm-specific "
                "content is configured for this section.</p>"
            )
        return f"<h2>{num}. {html.escape(title_text)}</h2>\n{body}\n"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>{_STRUCTURED_CSS}{_TAB_CSS}</style>
</head>
<body>

<h1>{html.escape(title)}</h1>
{legend_html}
<div class="notice"><strong>Metric.</strong> Empirical test loss (misclassification rate
on the fixed real evaluation split). The test-loss scale is preserved as-is.
Attribute subsets use compact variable notation (see sticky legend above);
full sensor names appear in metadata and appendices.</div>

{_section(1, "Arm summary",
    f"<p><strong>Arm:</strong> {html.escape(arm_label)}</p>"
    f"<p>{html.escape(arm_summary)}</p>")}

{_section(2, "Scientific role of this arm",
    f"<div class='role'>{html.escape(scientific_role)}</div>")}

{_section(3, "Dataset provenance summary", dataset_provenance_html)}

<h2>4. Run configuration</h2>
<dl class="meta">
  <dt>Arm</dt><dd>{html.escape(arm_label)}</dd>
  <dt>Dataset</dt><dd>{html.escape(dataset_name)}</dd>
  <dt>Target</dt><dd>{html.escape(target_description)}</dd>
  <dt>Mode</dt><dd><code>{html.escape(str(result.config.mode))}</code></dd>
  <dt>Sample sizes (n_per_class)</dt><dd>{list(result.config.sample_sizes)}</dd>
  <dt>Classifiers</dt><dd>{html.escape(classifiers_str)}</dd>
  <dt>Attribute subsets evaluated</dt><dd>{len(result.subsets)} subsets</dd>
  <dt>Fixed test set</dt>
  <dd>{html.escape(fixed_test_description)}
      ({result.metadata.get("fixed_test_size", "unknown")} rows)</dd>
  <dt>Min / max replications</dt>
  <dd>{result.config.min_replications} / {result.config.max_replications}</dd>
  <dt>Replication batch size</dt><dd>{result.config.replication_batch_size}</dd>
  <dt>Target CI95 half-width</dt><dd>{result.config.ci_half_width_target}</dd>
  <dt>Base seed</dt><dd>{result.config.base_seed}</dd>
{_execution_configuration_html(result)}
  <dt>Runtime</dt><dd>{result.runtime_seconds:.2f} s</dd>
</dl>
{_classifier_configuration_html(result)}
<details><summary>List all evaluated attribute subsets</summary>
<p>{html.escape(subsets_compact)}</p></details>

{_section(5, "Training/evaluation protocol", training_protocol_html)}

{_section(6, "Arm-specific data and model description", model_section_html)}

{_section(7, "Reproducibility controls", reproducibility_html)}

<h2>8. Monte Carlo stopping rule</h2>
<p>Standard-error stopping: replications continue in batches of
{result.config.replication_batch_size} until the maximum 95% CI half-width
(<code>1.96 &times; SE</code>) across all (subset, classifier) cells falls at or
below the target ({result.config.ci_half_width_target}) or the replication budget
({result.config.max_replications}) is exhausted.</p>
{_stopping_table_html(result)}

<h2>9. Monte Carlo precision diagnostics</h2>
<p>Per <code>n_per_class</code>: the maximum standard error and the maximum 95% CI
half-width (<code>1.96 &times; SE</code>) across all (subset, classifier) cells,
compared against the convergence target
(<code>{result.config.ci_half_width_target}</code>, on the same CI95 half-width
scale). All quantities preserve the empirical test-loss scale.</p>
{prec_html}

<h2>10. Loss curves</h2>
<p>Loss curves are available in two complementary views. <strong>Best by
cardinality</strong> shows the best subset for every available dimension from
1 through {dimension}, including the {html.escape(reference_label)}.
<strong>Nested cardinality</strong> shows all subsets of one selected
cardinality against the same {dimension}-attribute reference. Linear SVM is shown
by default; error bars show &plusmn;1 SE.</p>
{losscurve_html}

<h2>11. Best subset comparison</h2>
<p>Interpretable comparison of the best subset at every available dimension
from 1 through {dimension}, including the full {dimension}-attribute subset.
The full-attribute reference row is highlighted. Select a classifier below
(Linear SVM default).</p>
{bestsub_html}

<h2>12. Subset ranking by sample size</h2>
<p>Subsets are ranked by empirical test loss for the selected classifier and
selected sample size. The largest evaluated sample size is shown by default, but
rankings for all evaluated <code>n_per_class</code> values are available through
the selector. The full-attribute reference row is highlighted.</p>
{finalrank_html}

<h2>13. Pairwise winner and reversal dynamics</h2>
<p>This arm-local view shows the effective winner matrix <code>W</code> and the
progressive reversal matrix <code>R</code> together over sample size. One best
subset per cardinality is selected using this arm at the largest N and frozen
across all displayed panels. This display reduction does not change the
all-subset structural data persisted for the run.</p>
{structural_html}

{_section(14, "Robustness notes", robustness_html)}

{_section(15, "Validity and limitations for this arm", validity_html)}

<h2>16. Technical appendix / exported tables</h2>
{full_table_details}
<p>Clean technical tables are also exported as CSV next to this report:</p>
{csv_links}

{_TAB_JS}
</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path


def _full_table_details(
    result: SimulationResult,
    channel_names: Optional[Sequence[str]],
    arm_id: str,
) -> str:
    """Render the full long-format loss table inside a collapsed <details>."""
    tbl = full_loss_table(result, channel_names=channel_names, arm=arm_id)
    display = tbl[
        [
            "n_per_class", "classifier_label", "x_label", "subset_size",
            "mean_loss", "se_loss", "ci95_half_width", "replications",
        ]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "x_label": "Subset",
            "subset_size": "Size",
            "mean_loss": "Mean test loss",
            "se_loss": "SE",
            "ci95_half_width": "CI95 half-width",
            "replications": "Reps",
        }
    )
    tbl_html = _dataframe_html(
        display,
        float_cols={"Mean test loss": ".4f", "SE": ".4f", "CI95 half-width": ".4f"},
    )
    return (
        f"<details><summary>Show full loss table ({len(tbl)} rows: all subsets "
        f"&times; all N &times; all classifiers)</summary>{tbl_html}</details>"
    )


def _export_csv_tables(
    result: SimulationResult,
    output_dir: Path,
    arm_id: str,
    channel_names: Optional[Sequence[str]],
) -> None:
    """Export full_loss_table, subset_metadata, and precision_diagnostics CSVs."""
    # full loss table
    tbl = full_loss_table(result, channel_names=channel_names, arm=arm_id)
    tbl.to_csv(output_dir / f"full_loss_table_{arm_id}.csv", index=False)
    # subset metadata
    meta = subset_metadata_table(result.subsets, channel_names)
    meta.to_csv(output_dir / "subset_metadata_table.csv", index=False)
    # precision diagnostics
    prec = compact_precision_diagnostics(result)
    prec.to_csv(output_dir / f"precision_diagnostics_{arm_id}.csv", index=False)


def _csv_links_html(output_dir: Path, arm_id: str) -> str:
    files = [
        f"full_loss_table_{arm_id}.csv",
        "subset_metadata_table.csv",
        f"precision_diagnostics_{arm_id}.csv",
    ]
    items = "".join(
        f"<li><a href='{html.escape(f)}'><code>{html.escape(f)}</code></a></li>"
        for f in files
        if (output_dir / f).exists()
    )
    if not items:
        return "<p>(CSV export files not found)</p>"
    return f"<ul>{items}</ul>"
