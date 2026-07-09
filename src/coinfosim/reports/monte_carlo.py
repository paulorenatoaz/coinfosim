"""Reusable HTML reports for cooperative Monte Carlo results."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from coinfosim.classifiers.registry import classifier_label
from coinfosim.results.analysis import (
    best_subset_rankings,
    standard_threshold_comparisons,
)
from coinfosim.results.summary import summary_dataframe
from coinfosim.simulation.monte_carlo import SimulationResult


Subset = Tuple[int, ...]


def subset_display_label(
    subset: Sequence[int],
    channel_names: Optional[Sequence[str]] = None,
) -> str:
    """Return a readable display label for a zero-based channel subset."""

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
  <dt>Channel subsets</dt><dd>{html.escape(subsets)}</dd>
  <dt>Number of channel subsets</dt><dd>{len(result.subsets)}</dd>
  <dt>Fixed test set</dt><dd>{html.escape(fixed_test_description)} ({result.metadata.get("fixed_test_size", "unknown")} rows)</dd>
  <dt>Monte Carlo stopping rule</dt><dd>Standard-error rule at replication batch boundaries</dd>
  <dt>Target CI half-width</dt><dd>{result.config.ci_half_width_target}</dd>
  <dt>Min / max replications</dt><dd>{result.config.min_replications} / {result.config.max_replications}</dd>
  <dt>Replication batch size</dt><dd>{result.config.replication_batch_size}</dd>
  <dt>Base seed</dt><dd>{result.config.base_seed}</dd>
  <dt>Runtime</dt><dd>{result.runtime_seconds:.2f} s</dd>
</dl>

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
    parts: List[str] = []

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
        parts.append(
            "<h3>Model-selection configuration</h3>"
            f"<table class='data'><tbody>{cfg_html}</tbody></table>"
        )

    for label in model.class_labels:
        selection = model.model_selection(label)
        weights = model.component_weights(label)
        means = model.component_means(label)
        covs = model.component_covariances(label)
        k = model.selected_components(label)

        parts.append(f"<h3>Class {html.escape(str(label))}</h3>")
        parts.append(
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
            parts.append(
                "<p><strong>Model-selection scores:</strong></p>"
                f"<table class='data'><thead>{header}</thead>"
                f"<tbody>{body}</tbody></table>"
            )

        weight_text = ", ".join(f"{float(w):.4f}" for w in weights)
        parts.append(f"<p><strong>Mixture weights:</strong> [{weight_text}]</p>")

        for j in range(k):
            parts.append(
                f"<p><strong>Component {j} mean:</strong> "
                + html.escape(_vector_text(means[j], channel_names))
                + "</p>"
                + f"<p><strong>Component {j} covariance:</strong></p>"
                + _matrix_html(covs[j], channel_names)
            )
    return "".join(parts)


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
