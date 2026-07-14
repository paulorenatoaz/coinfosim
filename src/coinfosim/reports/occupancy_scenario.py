"""Occupancy Detection scenario report (academic layout).

Renders a self-contained, academic-style HTML scenario report comparing the
three main Occupancy arms — ``Real → Real`` (real training pool, real evaluation
split), ``Single Gaussian → Real`` (single-Gaussian synthetic training, real
evaluation split) and ``GMM → Real`` (class-conditional GMM synthetic training,
real evaluation split). All three main arms are evaluated on the same fixed real
Occupancy evaluation split. Channels are referred to with mathematical labels
``X_i`` and a sticky channel legend; sensor names are used only in the legend and
in explicit "reference subset" annotations, never as primary table/plot labels.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from coinfosim.classifiers.registry import classifier_label
from coinfosim.reports.html_tabs import TAB_CSS, TAB_JS, tab_group
from coinfosim.reports.scenario_visualization import save_loss_vs_n
from coinfosim.reports.structural_visualization import (
    metric_series_figure,
    progressive_nstar_matrix_figure,
    save_figure,
    winner_matrix_figure,
)
from coinfosim.results.analysis import cooperative_threshold_interpolated
from coinfosim.results.structural import (
    progressive_directed_nstar,
    scenario_structural_fidelity,
    winner_matrix,
)
from coinfosim.simulation.monte_carlo import SimulationResult

Subset = Tuple[int, ...]

# Occupancy channel order: X1..X5. Sensor names appear only in the legend and in
# explicit reference-subset annotations.
_SENSOR_DISPLAY = {"CO2": "CO₂", "HumidityRatio": "Humidity Ratio"}
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sensor(name: str) -> str:
    return _SENSOR_DISPLAY.get(str(name), str(name))


def _xsub(i0: int) -> str:
    """Subscripted math label for a zero-based channel index, e.g. ``X₃``."""
    return "X" + str(i0 + 1).translate(_SUBSCRIPTS)


def _subset_notation(subset: Sequence[int]) -> str:
    """Plain notation for a zero-based subset, e.g. ``{X3, X4}``."""
    return "{" + ", ".join(f"X{i + 1}" for i in subset) + "}"


def _subset_sensors(subset: Sequence[int], channel_names: Sequence[str]) -> str:
    return "{" + ", ".join(_sensor(channel_names[i]) for i in subset) + "}"


# --------------------------------------------------------------------------- #
# Ranking helpers (operate directly on the accumulator)
# --------------------------------------------------------------------------- #
def _ranked(
    result: SimulationResult,
    classifier: str,
    n: int,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Tuple[Subset, float, float]]:
    """Return ``(subset, mean_loss, standard_error)`` sorted by loss then SE."""
    subsets = subsets if subsets is not None else result.subsets
    acc = result.accumulator
    entries = [
        (
            tuple(s),
            acc.mean_loss(n, tuple(s), classifier),
            acc.standard_error(n, tuple(s), classifier),
        )
        for s in subsets
    ]
    entries.sort(key=lambda e: (e[1], e[2]))
    return entries


def _ranked_cardinality(
    result: SimulationResult, classifier: str, n: int, k: int
) -> List[Tuple[Subset, float, float]]:
    subsets = [s for s in result.subsets if len(s) == k]
    return _ranked(result, classifier, n, subsets)


def _finite(value: float) -> bool:
    return value == value  # False for NaN


def _series(
    result: SimulationResult,
    classifier: str,
    subsets: Sequence[Subset],
) -> List[Dict]:
    """Build loss-vs-N series (mean loss + SE per subset) for a classifier."""
    sizes = result.sample_sizes
    return [
        {
            "label": _subset_notation(s),
            "means": [result.accumulator.mean_loss(nn, s, classifier) for nn in sizes],
            "ses": [result.accumulator.standard_error(nn, s, classifier) for nn in sizes],
        }
        for s in subsets
    ]


def _emit_graph(
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    key: str,
    sample_sizes: Sequence[int],
    series: Sequence[Dict],
    title: str,
    markers: Optional[Sequence[Dict]] = None,
    enabled: bool = True,
    nstar_style: bool = False,
) -> str:
    """Render a loss-vs-N graph, record it, and return an embedding <figure>.

    When ``enabled`` is ``False`` nothing is rendered (useful for lightweight
    programmatic/test runs); an empty string is returned.
    """
    if not enabled:
        return ""
    filename = f"{key}_{graph_suffix}.png" if graph_suffix else f"{key}.png"
    save_loss_vs_n(
        sample_sizes, series, title, output_dir, filename,
        markers=markers, nstar_style=nstar_style,
    )
    if graphs_out is not None:
        graphs_out[key] = filename
    return (
        f"<figure class='graph'><img src='{html.escape(filename)}' "
        f"alt='{html.escape(key)}'/></figure>"
    )


# --------------------------------------------------------------------------- #
# HTML primitives
# --------------------------------------------------------------------------- #
def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        f"<table class='data'><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _fmt_loss(value: float) -> str:
    return f"{value:.4f}" if _finite(value) else "&mdash;"


def _fmt_num(value: Optional[float], fmt: str) -> str:
    if value is None or not _finite(float(value)):
        return "&mdash;"
    return f"{float(value):{fmt}}"


def _emit_structural_graph(
    fig,
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    key: str,
    enabled: bool,
) -> str:
    if not enabled:
        return ""
    filename = f"{key}_{graph_suffix}.png" if graph_suffix else f"{key}.png"
    save_figure(fig, output_dir, filename)
    if graphs_out is not None:
        graphs_out[key] = filename
    return (
        f"<figure class='graph'><img src='{html.escape(filename, quote=True)}' "
        f"alt='{html.escape(key, quote=True)}'/></figure>"
    )


# --------------------------------------------------------------------------- #
# Structural fidelity and N-star availability
# --------------------------------------------------------------------------- #
def _structural_fidelity_section(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
    arm_labels: Mapping[str, str],
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    generate_graphs: bool,
) -> str:
    structural = scenario_structural_fidelity(
        arm_results, reference_arm, arm_labels
    )
    sample_sizes = [int(n) for n in structural["sample_sizes"]]
    n_max = sample_sizes[-1]
    ranking = structural["ranking_fidelity_series"]
    agreements = structural["winner_agreement_series"]
    nstar_similarities = structural["nstar_similarity_series"]
    summaries = [
        row for row in structural["final_summary"] if row["arm"] != reference_arm
    ]
    rows = [
        [
            html.escape(classifier_label(str(row["classifier"]))),
            html.escape(arm_labels[str(row["arm"])]),
            _fmt_num(row["rho_rank"], ".4f"),
            html.escape(str(row["ranking_status"])),
            _fmt_num(row["winner_agreement"], ".4f"),
            str(row["n_pairs_valid"]),
            str(row["n_pairs_skipped_tie"]),
            html.escape(str(row["winner_status"])),
            _fmt_num(row["nstar_similarity"], ".4f"),
            str(row["n_reference_crossings"]),
            str(row["n_arm_crossings"]),
            str(row["n_shared_crossings"]),
            _fmt_num(row["crossing_jaccard"], ".4f"),
            _fmt_num(row["timing_similarity"], ".4f"),
            html.escape(str(row["nstar_status"])),
        ]
        for row in summaries
    ]

    classifiers = list(arm_results[reference_arm].classifier_names)
    metric_classifier_tabs = []
    for classifier in classifiers:
        metric_tabs = []
        for metric, label in (
            ("rho_rank", "Ranking fidelity"),
            ("winner_agreement", "Winner agreement"),
            ("nstar_similarity", "Progressive N-star similarity"),
        ):
            source_rows = {
                "rho_rank": ranking,
                "winner_agreement": agreements,
                "nstar_similarity": nstar_similarities,
            }[metric]
            metric_rows = [
                row
                for row in source_rows
                if row["classifier"] == classifier and row["arm"] != reference_arm
            ]
            key = f"graph_structural_metric_{classifier}_{metric}"
            figure_html = ""
            if generate_graphs:
                fig = metric_series_figure(
                    metric_rows,
                    metric,
                    arm_labels,
                    f"{classifier_label(classifier)} — {label}",
                )
                figure_html = _emit_structural_graph(
                    fig,
                    output_dir,
                    graph_suffix,
                    graphs_out,
                    key,
                    True,
                )
            metric_tabs.append((metric, label, figure_html))
        metric_classifier_tabs.append(
            (
                classifier,
                classifier_label(classifier),
                tab_group(
                    f"scenario-structural-metrics-{classifier}",
                    metric_tabs,
                    "rho_rank",
                ),
            )
        )
    default_classifier = (
        "linear_svm" if "linear_svm" in classifiers else classifiers[0]
    )

    display_by_classifier = structural["reference_display_subsets_by_classifier"]
    winner_classifier_tabs = []
    nstar_classifier_tabs = []
    for classifier in classifiers:
        display_subsets = [
            tuple(subset) for subset in display_by_classifier[classifier]
        ]
        subset_labels = [_subset_notation(subset) for subset in display_subsets]
        arm_tabs = []
        for arm, result in arm_results.items():
            n_tabs = []
            for n in sample_sizes:
                key = f"graph_structural_winner_{classifier}_{arm}_n{n}"
                figure_html = ""
                if generate_graphs:
                    matrix = winner_matrix(
                        result, classifier, n, subsets=display_subsets
                    )
                    fig = winner_matrix_figure(
                        matrix,
                        subset_labels,
                        f"{classifier_label(classifier)} — {arm_labels[arm]} — N={n}",
                    )
                    figure_html = _emit_structural_graph(
                        fig,
                        output_dir,
                        graph_suffix,
                        graphs_out,
                        key,
                        True,
                    )
                n_tabs.append((str(n), f"N = {n}", figure_html))
            arm_tabs.append(
                (
                    arm,
                    arm_labels[arm],
                    tab_group(
                        f"scenario-structural-winner-{classifier}-{arm}-n",
                        n_tabs,
                        str(sample_sizes[0]),
                    ),
                )
            )
        selection_text = ", ".join(subset_labels)
        winner_classifier_tabs.append(
            (
                classifier,
                classifier_label(classifier),
                "<p class='refsubset'>Fixed Real → Real display subsets selected "
                f"at Nmax: {html.escape(selection_text)}</p>"
                + tab_group(
                    f"scenario-structural-winner-{classifier}-arm",
                    arm_tabs,
                    reference_arm,
                ),
            )
        )

        nstar_arm_tabs = []
        for arm, result in arm_results.items():
            progressive = progressive_directed_nstar(
                result, classifier, display_subsets
            )
            prefix_tabs = []
            for item in progressive:
                n_prefix = int(item["n_prefix"])
                key = (
                    f"graph_structural_nstar_{classifier}_{arm}_prefix{n_prefix}"
                )
                figure_html = ""
                if generate_graphs:
                    fig = progressive_nstar_matrix_figure(
                        item["matrix"],
                        subset_labels,
                        f"{classifier_label(classifier)} — {arm_labels[arm]} "
                        f"— prefix N={n_prefix}",
                    )
                    figure_html = _emit_structural_graph(
                        fig,
                        output_dir,
                        graph_suffix,
                        graphs_out,
                        key,
                        True,
                    )
                prefix_tabs.append(
                    (str(n_prefix), f"Prefix N = {n_prefix}", figure_html)
                )
            nstar_arm_tabs.append(
                (
                    arm,
                    arm_labels[arm],
                    tab_group(
                        f"scenario-structural-nstar-{classifier}-{arm}-prefix",
                        prefix_tabs,
                        str(sample_sizes[1]),
                    ),
                )
            )
        nstar_classifier_tabs.append(
            (
                classifier,
                classifier_label(classifier),
                "<p class='refsubset'>The same fixed Real → Real display "
                f"subsets are used: {html.escape(selection_text)}</p>"
                + tab_group(
                    f"scenario-structural-nstar-{classifier}-arm",
                    nstar_arm_tabs,
                    reference_arm,
                ),
            )
        )

    return (
        "<h2>7. Structural fidelity metrics</h2>"
        "<p>Ranking Structural Fidelity compares complete all-subset loss "
        "rankings using Spearman correlation with average-rank tie handling. "
        "Winner Agreement compares exact pairwise winners after excluding pairs "
        "tied in either arm. Progressive N-star Similarity compares directed "
        "crossing-cell sets and their latest observed-grid timing through the "
        "current sample-size prefix. These metrics remain separate; no composite "
        "index is reported.</p>"
        "<p class='muted'><code>constant_ranking</code> means a non-reference "
        "ranking has no variation; <code>no_valid_pairs</code> means every "
        "unordered pair was tied in at least one arm; "
        "<code>unavailable_first_prefix</code> marks N1; "
        "<code>no_crossings_in_either</code> and "
        "<code>no_shared_crossings</code> distinguish empty crossing cases; "
        "<code>ok</code> means the corresponding metric is available.</p>"
        f"<h3>7.1 Summary at N = {n_max}</h3>"
        + _table(
            [
                "Classifier",
                "Comparison arm",
                "rho_rank(Nmax)",
                "Ranking status",
                "A_W(Nmax)",
                "Valid winner pairs",
                "Skipped tie pairs",
                "Winner status",
                "S_N*(<=Nmax)",
                "Reference crossings",
                "Arm crossings",
                "Shared crossings",
                "Crossing Jaccard",
                "Timing similarity",
                "N-star status",
            ],
            rows,
        )
        + "<h3>7.2 Metric curves</h3>"
        + tab_group(
            "scenario-structural-metric-classifier",
            metric_classifier_tabs,
            default_classifier,
        )
        + "<h3>7.3 Directed winner matrices</h3>"
        "<p>Matrix values are +1 when the row subset has lower loss, -1 when "
        "it has higher loss, 0 for an exact tie, and unavailable on the diagonal. "
        "The fixed display reduction does not affect either numerical metric.</p>"
        + tab_group(
            "scenario-structural-winner-classifier",
            winner_classifier_tabs,
            default_classifier,
        )
        + "<h3>7.4 Progressive directed N-star matrices</h3>"
        "<p>Each directed cell shows the latest observed-grid crossing in that "
        "direction through the selected prefix. Missing cells are unavailable, "
        "not zero, and prefix tabs begin at N2.</p>"
        + tab_group(
            "scenario-structural-nstar-classifier",
            nstar_classifier_tabs,
            default_classifier,
        )
    )


# --------------------------------------------------------------------------- #
# N-star availability
# --------------------------------------------------------------------------- #
def _competitor_plan(
    real_result: SimulationResult, classifier: str, n: int, k: int, d: int
) -> Tuple[Optional[Subset], List[Tuple[str, Subset]]]:
    """Return the reference subset and ordered competitor ``(label, subset)`` list.

    Reference and competitor subsets are selected on the ``Real → Real`` arm at
    the largest evaluated ``N`` so that both arms are compared on the same
    subsets.
    """
    ref_ranked = _ranked_cardinality(real_result, classifier, n, k)
    if not ref_ranked:
        return None, []
    reference = ref_ranked[0][0]
    second_same = ref_ranked[1][0] if len(ref_ranked) > 1 else None

    competitors: List[Tuple[str, Subset]] = []
    for k2 in range(1, d + 1):
        if k2 == k:
            if second_same is not None:
                competitors.append((f"2-Best-{k}-ChSub", second_same))
            continue
        other = _ranked_cardinality(real_result, classifier, n, k2)
        if other:
            competitors.append((f"Best-{k2}-ChSub", other[0][0]))
    return reference, competitors


def _all_crossings(
    acc,
    classifier: str,
    subset_a: Subset,
    subset_b: Subset,
    sizes: Sequence[int],
) -> List[Dict]:
    """Return all observed N* crossings where subset_b beats subset_a.

    Scans the full sample-size grid and records every point where the sign of
    ``Loss_a(n) - Loss_b(n)`` flips from non-positive to positive. For each
    crossing an interpolated N* is computed.  Returns a list of dicts with keys
    ``n_star_grid`` and ``n_star_interpolated``.
    """
    import math

    deltas = []
    for n in sizes:
        la = acc.mean_loss(n, subset_a, classifier)
        lb = acc.mean_loss(n, subset_b, classifier)
        deltas.append(float(la - lb))

    crossings: List[Dict] = []
    for i in range(len(sizes)):
        d = deltas[i]
        if d > 0:  # b beats a at this point
            if i == 0:
                # Left-censored: b already wins at the first n
                crossings.append(
                    {"n_star_grid": sizes[0], "n_star_interpolated": float(sizes[0]),
                     "status": "left_censored"}
                )
            else:
                prev_d = deltas[i - 1]
                if prev_d <= 0:
                    # Genuine crossing
                    denom = d - prev_d
                    if math.isclose(denom, 0.0, abs_tol=1e-15):
                        interp = float(sizes[i])
                    else:
                        interp = float(
                            sizes[i - 1]
                            + ((0.0 - prev_d) / denom) * (sizes[i] - sizes[i - 1])
                        )
                    status = "exact_zero" if math.isclose(prev_d, 0.0, abs_tol=1e-15) else "interpolated"
                    crossings.append(
                        {"n_star_grid": sizes[i], "n_star_interpolated": interp,
                         "status": status}
                    )
        # After a crossing: if sign flips back, subsequent re-crossings are
        # caught in the next iteration when delta goes positive again.
    return crossings


def _nstar_analysis(
    result: SimulationResult,
    classifier: str,
    reference: Subset,
    competitors: List[Tuple[str, Subset]],
    n: int,
) -> List[Dict]:
    """Analyze reference vs competitors for one arm.

    All detected N* crossings are stored in ``all_crossings`` for downstream
    persistence. Only the **last** crossing is reported in the table and graph
    (``n_star`` / ``interp``), following the rule that if multiple crossings
    exist only the last N* is highlighted.

    Crossings are only stored for genuine grid crossings (status
    ``interpolated`` or ``exact_zero``). Left-censored cases (the VS curve is
    already winning at the smallest evaluated N) are detected but do not count
    as a meaningful crossing for reporting—a dash is shown instead to avoid
    misleading ``N* = 2`` values.
    """
    acc = result.accumulator
    sizes = result.sample_sizes
    loss_ref = acc.mean_loss(n, reference, classifier)
    analysis: List[Dict] = []
    for label, comp in competitors:
        loss_comp = acc.mean_loss(n, comp, classifier)
        entry: Dict = {
            "label": label,
            "comp": comp,
            "n_star": None,
            "interp": None,
            "winner": None,
            "all_crossings": [],
        }
        if not _finite(loss_ref) or not _finite(loss_comp):
            entry["winner"] = "Unavailable"
            analysis.append(entry)
            continue
        if loss_comp < loss_ref:
            entry["winner"] = "VS"
            crossings = _all_crossings(acc, classifier, reference, comp, sizes)
        elif loss_ref < loss_comp:
            entry["winner"] = "Reference"
            crossings = _all_crossings(acc, classifier, comp, reference, sizes)
        else:
            entry["winner"] = "Tie"
            analysis.append(entry)
            continue
        # Store all genuine crossings (exclude left-censored for reporting).
        genuine = [c for c in crossings if c["status"] != "left_censored"]
        entry["all_crossings"] = crossings  # all, for JSON persistence
        if genuine:
            last = genuine[-1]
            entry["n_star"] = last["n_star_grid"]
            entry["interp"] = last["n_star_interpolated"]
        analysis.append(entry)
    return analysis


def _nstar_rows(analysis: List[Dict]) -> List[List[str]]:
    rows: List[List[str]] = []
    for e in analysis:
        comp = tuple(e["comp"]) if not isinstance(e["comp"], tuple) else e["comp"]
        rows.append(
            [
                f"{e['label']}: {_subset_notation(comp)}",
                str(e["n_star"]) if e["n_star"] is not None else "&mdash;",
                f"{e['interp']:.1f}" if e["interp"] is not None else "&mdash;",
                e["winner"],
            ]
        )
    return rows


def _nstar_section(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    channel_names: Sequence[str],
    n: int,
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    generate_graphs: bool = True,
) -> str:
    d = len(channel_names)
    parts: List[str] = ["<h2>8. N-star availability</h2>"]
    parts.append(
        "<p>For each classifier and channel-subset cardinality, the best "
        "subset at the largest evaluated sample size is used as a reference "
        "and compared against the best subset of every other cardinality and, "
        "when it exists, the second-best subset of the same cardinality. "
        "Reference and competitor subsets are selected on the Real → Real arm "
        "so that all three arms are compared on the same subsets.</p>"
    )
    parts.append(
        "<p class='muted'>A dash indicates that no crossing was detected within "
        "the evaluated sample-size grid. In that case, the Winner column reports "
        "which subset has the lower loss at the largest evaluated N.</p>"
    )
    parts.append(
        "<p class='muted'>If multiple crossings are detected, all of them are "
        "preserved in the underlying structured data, but only the last N* is "
        "reported and highlighted in this report. "
        "Dashed vertical lines on the graphs mark the reported interpolated N* "
        "values.</p>"
    )
    classifier_tabs = []
    for sub, clf in enumerate(real_result.classifier_names, start=1):
        cardinality_tabs = []
        for k in range(1, d + 1):
            reference, competitors = _competitor_plan(real_result, clf, n, k, d)
            if reference is None:
                continue
            arm_tabs = []
            for arm_key, arm_name, arm_result in (
                ("real", "Real → Real", real_result),
                ("sgr", "Single Gaussian → Real", gaussian_result),
                ("gmm", "GMM → Real", gmm_result),
            ):
                analysis = _nstar_analysis(
                    arm_result, clf, reference, competitors, n
                )
                panel = (
                    "<p class='refsubset'>Reference subset: "
                    f"{html.escape(_subset_notation(reference))} = "
                    f"{html.escape(_subset_sensors(reference, channel_names))}</p>"
                    f"<h5>{html.escape(arm_name)}</h5>"
                    + _table(
                        ["VS", "N*", "Interpolated N*", "Winner"],
                        _nstar_rows(analysis),
                    )
                )
                subsets_series = [reference] + [e["comp"] for e in analysis]
                series = _series(arm_result, clf, subsets_series)
                if series:
                    series[0]["label"] = f"Reference {_subset_notation(reference)}"
                markers = [
                    {"x": e["interp"], "label": f"N*={e['interp']:.1f}"}
                    for e in analysis
                    if e["interp"] is not None
                ]
                title = (
                    f"{classifier_label(clf)} — Best {k}-channel "
                    f"{_subset_notation(reference)} — {arm_name} (loss vs N)"
                )
                key = f"graph_nstar_{clf}_k{k}_{arm_key}"
                panel += (
                    _emit_graph(
                        output_dir,
                        graph_suffix,
                        graphs_out,
                        key,
                        arm_result.sample_sizes,
                        series,
                        title,
                        markers=markers,
                        enabled=generate_graphs,
                        nstar_style=True,
                    )
                )
                arm_tabs.append((arm_key, arm_name, panel))
            cardinality_tabs.append(
                (
                    f"k{k}",
                    f"Best {k}-channel reference",
                    f"<h4>Best {k}-channel subset</h4>"
                    + tab_group(
                        f"scenario-nstar-{clf}-k{k}-arm", arm_tabs, "real"
                    ),
                )
            )
        if cardinality_tabs:
            classifier_tabs.append(
                (
                    clf,
                    classifier_label(clf),
                    f"<h3>8.{sub} {html.escape(classifier_label(clf))}</h3>"
                    + tab_group(
                        f"scenario-nstar-{clf}-cardinality",
                        cardinality_tabs,
                        cardinality_tabs[0][0],
                    ),
                )
            )
    if classifier_tabs:
        default_clf = (
            "linear_svm"
            if any(key == "linear_svm" for key, _, _ in classifier_tabs)
            else classifier_tabs[0][0]
        )
        parts.append(
            tab_group("scenario-nstar-classifier", classifier_tabs, default_clf)
        )
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #
def _legend_html(channel_names: Sequence[str]) -> str:
    items = "".join(
        f"<span class='legend-item'><b>{_xsub(i)}</b> = "
        f"{html.escape(_sensor(name))}</span>"
        for i, name in enumerate(channel_names)
    )
    return (
        "<div class='legend'><span class='legend-title'>Channel legend</span>"
        f"{items}</div>"
    )


def _summary_table(
    real_result: SimulationResult,
    scenario_meta: Optional[Dict],
    channel_names: Sequence[str],
    n_max: int,
) -> str:
    meta = scenario_meta or {}
    classifiers = ", ".join(
        classifier_label(c) for c in real_result.classifier_names
    )
    rows = [
        ("Scenario run ID", str(meta.get("scenario_run_id", "&mdash;"))),
        ("Scenario family", html.escape(str(meta.get("scenario_family", "dataset")))),
        ("Scenario mode", html.escape(str(real_result.config.mode))),
        ("Dataset", html.escape(str(meta.get("dataset", "Occupancy Detection")))),
        ("Main arms", "Real → Real; Single Gaussian → Real; GMM → Real"),
        ("Number of channels", str(len(channel_names))),
        ("Non-empty channel subsets", str(len(real_result.subsets))),
        ("Classifiers", html.escape(classifiers)),
        ("Metric", "Empirical test loss (0–1 misclassification)"),
        ("Sample sizes", html.escape(str(list(real_result.sample_sizes)))),
        ("Largest evaluated sample size", f"N = {n_max} samples per class"),
        ("Evaluation split", "Fixed real Occupancy evaluation split"),
    ]
    body = "".join(
        f"<tr><th class='key'>{html.escape(k)}</th><td>{v}</td></tr>"
        for k, v in rows
    )
    return f"<table class='meta'>{body}</table>"


def _protocol_html(real_result: SimulationResult) -> str:
    cfg = real_result.config
    stopping = _table(
        ["Parameter", "Value"],
        [
            ["Minimum replications", str(cfg.min_replications)],
            ["Maximum replications", str(cfg.max_replications)],
            ["Replication batch size", str(cfg.replication_batch_size)],
            ["CI target (95% half-width)", f"{cfg.ci_half_width_target:.4f}"],
            [
                "Stopping criterion",
                "Stop when the maximum 95% CI half-width across all "
                "(subset, classifier) cells falls at or below the CI target, "
                "or when the maximum replications budget is reached.",
            ],
        ],
    )
    return (
        "<h2>3. Experimental protocol</h2>"
        "<h3>3.1 Real → Real arm (real-data baseline)</h3>"
        "<p>The real-data baseline draws balanced training samples from the "
        "standardized Occupancy training pool and evaluates all channel "
        "subsets and classifiers using the fixed real Occupancy evaluation "
        "split.</p>"
        "<h3>3.2 Single Gaussian → Real arm</h3>"
        "<p>The single-Gaussian-to-real arm estimates one class-conditional "
        "Gaussian distribution per Occupancy class from the standardized "
        "training pool. Training samples are generated synthetically from these "
        "Gaussian distributions, while evaluation is performed on the fixed "
        "real Occupancy evaluation split.</p>"
        "<h3>3.3 GMM → Real arm</h3>"
        "<p>The GMM-to-real arm fits one class-conditional Gaussian mixture "
        "model per Occupancy class using the standardized training pool. The "
        "number of components is selected per class by BIC under a conservative "
        "component cap. Training samples are generated synthetically from the "
        "fitted GMMs, while evaluation is performed on the fixed real Occupancy "
        "evaluation split.</p>"
        "<p>All three main arms are evaluated on the same fixed real Occupancy "
        "evaluation split.</p>"
        "<h3>3.4 Monte Carlo stopping rule</h3>"
        f"{stopping}"
    )


def _carousel_html(visualization: Optional[Dict]) -> str:
    if not visualization or not visualization.get("images"):
        return (
            "<h2>4. Data visualization</h2>"
            "<p class='muted'>Visualization panels were not generated for this "
            "run.</p>"
        )
    images = visualization["images"]
    meta = visualization.get("metadata", {})
    meta_rows = [
        ("Visualization sample size", meta.get("visualization_sample_size")),
        ("Class balance", meta.get("class_balance")),
        ("Real-data source", meta.get("real_data_source")),
        (
            "Single Gaussian synthetic training source",
            meta.get("single_gaussian_source") or meta.get("synthetic_source"),
        ),
        ("GMM synthetic training source", meta.get("gmm_source")),
        ("Visualization seed", meta.get("visualization_seed")),
    ]
    meta_html = "".join(
        f"<tr><th class='key'>{html.escape(k)}</th>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in meta_rows
        if v is not None
    )

    def _caption(arm: str, dim: str) -> str:
        who = {
            "real": "Real training sample",
            "gaussian": "Single Gaussian synthetic training sample",
            "gmm": "GMM synthetic training sample",
        }.get(arm, arm)
        return f"{who} — {dim.upper()} projections"

    source_tabs = []
    for arm, arm_label in (
        ("real", "Real data"),
        ("gaussian", "Single Gaussian"),
        ("gmm", "GMM"),
    ):
        projection_tabs = []
        for dim in ("1d", "2d", "3d"):
            caption = _caption(arm, dim)
            src = images.get(f"viz_{dim}_{arm}", "")
            projection_tabs.append(
                (
                    dim,
                    dim.upper(),
                    "<figure class='carousel-stage'>"
                    f"<img src='{html.escape(src, quote=True)}' "
                    f"alt='{html.escape(caption, quote=True)}'/>"
                    f"<figcaption>{html.escape(caption)}</figcaption></figure>",
                )
            )
        source_tabs.append(
            (
                arm,
                arm_label,
                tab_group(
                    f"scenario-visualization-{arm}-projection",
                    projection_tabs,
                    "1d",
                ),
            )
        )

    return (
        "<h2>4. Data visualization</h2>"
        f"<table class='meta'>{meta_html}</table>"
        "<h3>4.1 Projection panels</h3>"
        + tab_group("scenario-visualization-source", source_tabs, "real")
    )


def _best_comparison_html(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    n: int,
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    generate_graphs: bool = True,
) -> str:
    rows = []
    for clf in real_result.classifier_names:
        real_best = _ranked(real_result, clf, n)[0]
        gauss_best = _ranked(gaussian_result, clf, n)[0]
        gmm_best = _ranked(gmm_result, clf, n)[0]
        sg_same = "yes" if real_best[0] == gauss_best[0] else "no"
        gmm_same = "yes" if real_best[0] == gmm_best[0] else "no"
        rows.append(
            [
                html.escape(classifier_label(clf)),
                _subset_notation(real_best[0]),
                _fmt_loss(real_best[1]),
                _subset_notation(gauss_best[0]),
                _fmt_loss(gauss_best[1]),
                _subset_notation(gmm_best[0]),
                _fmt_loss(gmm_best[1]),
                sg_same,
                gmm_same,
            ]
        )
    table = _table(
        [
            "Classifier",
            "Real → Real best subset",
            "Real → Real loss",
            "Single Gaussian → Real best subset",
            "Single Gaussian → Real loss",
            "GMM → Real best subset",
            "GMM → Real loss",
            "SG same as real",
            "GMM same as real",
        ],
        rows,
    )

    graph_tabs = []
    for arm_key, arm_name, arm_result in (
        ("real", "Real → Real", real_result),
        ("sgr", "Single Gaussian → Real", gaussian_result),
        ("gmm", "GMM → Real", gmm_result),
    ):
        sizes = arm_result.sample_sizes
        series = []
        for clf in arm_result.classifier_names:
            best = _ranked(arm_result, clf, n)[0][0]
            series.append(
                {
                    "label": f"{classifier_label(clf)}: {_subset_notation(best)}",
                    "means": [
                        arm_result.accumulator.mean_loss(nn, best, clf)
                        for nn in sizes
                    ],
                    "ses": [
                        arm_result.accumulator.standard_error(nn, best, clf)
                        for nn in sizes
                    ],
                }
            )
        title = f"{arm_name} arm — best subset per classifier (loss vs N)"
        graph_tabs.append(
            (
                arm_key,
                arm_name,
                f"<h4>{html.escape(arm_name)} arm</h4>"
                + _emit_graph(
                output_dir,
                graph_suffix,
                graphs_out,
                f"graph_best_comparison_{arm_key}",
                sizes,
                series,
                title,
                enabled=generate_graphs,
                ),
            )
        )

    return (
        "<h2>5. Best subset comparison at largest N</h2>"
        f"<p>Largest evaluated sample size: <b>N = {n} samples per class</b>.</p>"
        f"{table}"
        "<h3>5.1 Loss vs N for the best subsets</h3>"
        + tab_group("scenario-best-comparison-arm", graph_tabs, "real")
    )


def _top_ranked_html(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    n: int,
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    top_k: int = 5,
    generate_graphs: bool = True,
) -> str:
    parts = [
        "<h2>6. Top-ranked subsets</h2>",
        f"<p>Subsets are ranked by empirical test loss at the largest evaluated "
        f"N = {n} samples per class.</p>",
    ]
    classifier_tabs = []
    for i, clf in enumerate(real_result.classifier_names, start=1):
        arm_tabs = []
        for arm_key, arm_name, arm_result in (
            ("real", "Real → Real", real_result),
            ("sgr", "Single Gaussian → Real", gaussian_result),
            ("gmm", "GMM → Real", gmm_result),
        ):
            ranked = _ranked(arm_result, clf, n)[:top_k]
            rows = [
                [
                    str(rank),
                    _subset_notation(subset),
                    _fmt_loss(loss),
                    _fmt_loss(se),
                ]
                for rank, (subset, loss, se) in enumerate(ranked, start=1)
            ]
            panel = (
                f"<h4>{html.escape(arm_name)}</h4>"
                + _table(["Rank", "Subset", "Loss", "SE"], rows)
            )
            subsets = [subset for subset, _, _ in ranked]
            series = _series(arm_result, clf, subsets)
            title = (
                f"{classifier_label(clf)} — {arm_name}: "
                f"top-{len(subsets)} subsets (loss vs N)"
            )
            panel += (
                _emit_graph(
                    output_dir,
                    graph_suffix,
                    graphs_out,
                    f"graph_topranked_{clf}_{arm_key}",
                    arm_result.sample_sizes,
                    series,
                    title,
                    enabled=generate_graphs,
                )
            )
            arm_tabs.append((arm_key, arm_name, panel))
        classifier_tabs.append(
            (
                clf,
                classifier_label(clf),
                f"<h3>6.{i} {html.escape(classifier_label(clf))}</h3>"
                + tab_group(
                    f"scenario-top-ranked-{clf}-arm", arm_tabs, "real"
                ),
            )
        )
    if classifier_tabs:
        default_clf = (
            "linear_svm"
            if any(key == "linear_svm" for key, _, _ in classifier_tabs)
            else classifier_tabs[0][0]
        )
        parts.append(
            tab_group(
                "scenario-top-ranked-classifier", classifier_tabs, default_clf
            )
        )
    return "".join(parts)


def _interpretation_html() -> str:
    return (
        "<h2>9. Interpretation notes</h2>"
        "<h3>Structural fidelity metrics</h3>"
        "<p>Ranking fidelity, pairwise winner agreement, and progressive "
        "N-star similarity describe distinct aspects of preservation and should "
        "be interpreted separately alongside their availability statuses.</p>"
        "<h3>Agreement among the three arms</h3>"
        "<p>Where the Real → Real, Single Gaussian → Real and GMM → Real arms "
        "select the same best subsets and show similar cooperative thresholds "
        "under real-data evaluation, synthetic training preserves the "
        "cooperative advantages observed when classifiers are evaluated on real "
        "Occupancy data.</p>"
        "<h3>Single Gaussian vs GMM under real-data evaluation</h3>"
        "<p>If the GMM → Real arm aligns more closely with Real → Real than "
        "Single Gaussian → Real, this suggests that multimodal class-conditional "
        "structure is important for preserving cooperative behavior under "
        "real-data evaluation.</p>"
        "<h3>Divergences from the Real → Real baseline</h3>"
        "<p>Differences in best subsets or in N* availability indicate "
        "structure in the real Occupancy data that a given synthetic training "
        "distribution does not reproduce under real-data evaluation.</p>"
        "<h3>Classifier-specific behavior</h3>"
        "<p>Linear SVM, Logistic Regression, and Gaussian Naive Bayes may rank "
        "subsets differently; comparisons should be read per classifier.</p>"
        "<h3>Limitations of the current run mode</h3>"
        "<p>Smoke-mode results should be interpreted as preliminary evidence "
        "and pipeline validation rather than final inferential conclusions.</p>"
    )


_STYLE = """
:root { --accent:#1f77b4; --ink:#1a1a1a; --muted:#666; --line:#dcdcdc; }
* { box-sizing: border-box; }
body { font-family: Georgia, 'Times New Roman', serif; color: var(--ink);
       margin: 0 auto; max-width: 980px; padding: 2.4rem 1.6rem 4rem;
       line-height: 1.6; }
h1 { font-size: 1.9rem; line-height:1.25; border-bottom: 3px solid var(--accent);
     padding-bottom: .5rem; }
h2 { font-size: 1.35rem; margin-top: 2.2rem; color:#123; border-bottom:1px solid var(--line);
     padding-bottom:.25rem; }
h3 { font-size: 1.12rem; margin-top: 1.4rem; color:#1f3b66; }
h4 { font-size: 1.0rem; margin: 1rem 0 .3rem; color:#333; }
h5 { font-size: .95rem; margin: .8rem 0 .3rem; color: var(--muted);
     text-transform: uppercase; letter-spacing:.03em; }
p { margin: .6rem 0; }
.question { font-style: italic; background:#eef5fb; border-left:4px solid var(--accent);
            padding:.9rem 1.1rem; }
.refsubset { background:#f7f9fb; border-left:3px solid #9bb; padding:.4rem .7rem;
             font-family: 'SFMono-Regular', Consolas, monospace; font-style: normal; }
.muted { color: var(--muted); }
table.data { border-collapse: collapse; width: 100%; margin: .7rem 0 1.2rem;
             font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
             font-size: .88rem; }
table.data th, table.data td { border:1px solid var(--line); padding:.35rem .6rem;
             text-align:center; }
table.data th { background:#f0f4f8; }
table.meta { border-collapse: collapse; margin:.6rem 0 1.2rem; font-size:.9rem;
             font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
table.meta th.key { text-align:left; background:#f7f9fb; color:#333; font-weight:600;
             padding:.32rem .8rem; border:1px solid var(--line); white-space:nowrap; }
table.meta td { padding:.32rem .8rem; border:1px solid var(--line); }
figure.viz { margin: .4rem 0 1.2rem; text-align:center; }
figure.viz img { max-width:100%; height:auto; border:1px solid var(--line);
             border-radius:4px; }
figure.graph { margin: .5rem 0 1.4rem; text-align:center; }
figure.graph img { max-width:100%; height:auto; border:1px solid var(--line);
             border-radius:4px; }
.carousel-stage { margin:0; text-align:center; }
.carousel-stage img { max-width:100%; height:auto; border:1px solid var(--line);
            border-radius:4px; }
.carousel-stage figcaption { margin-top:.55rem; color:var(--muted);
            font-size:.9rem;
            font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.related { font-size:.85rem; color:var(--muted);
           font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.related a { color: var(--accent); }
.legend { position: sticky; top: 0; z-index: 50; background:#ffffffee;
          backdrop-filter: blur(3px); border:1px solid var(--line); border-radius:6px;
          padding:.5rem .8rem; margin:1rem 0 1.4rem; display:flex; flex-wrap:wrap;
          gap:.4rem 1.1rem; align-items:center;
          font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
          font-size:.86rem; box-shadow:0 1px 3px rgba(0,0,0,.06); }
.legend-title { font-weight:700; color:#123; margin-right:.4rem; }
.legend-item { white-space:nowrap; }
"""


def generate_occupancy_scenario_report(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    output_dir: Path | str = "output/reports",
    dataset_report: str = "occupancy_dataset_report.html",
    real_report: str = "occupancy_real_monte_carlo_report.html",
    sg_real_report: str = "occupancy_single_gaussian_to_real_monte_carlo_report.html",
    gmm_real_report: str = "occupancy_gmm_to_real_monte_carlo_report.html",
    filename: str = "occupancy_scenario_report.html",
    channel_names: Sequence[str] | None = None,
    visualization: Optional[Dict] = None,
    scenario_meta: Optional[Dict] = None,
    graph_suffix: str = "",
    graphs_out: Optional[Dict] = None,
    generate_graphs: bool = True,
) -> Path:
    """Generate the academic Occupancy scenario report.

    The three main arms are ``Real → Real`` (``real_result``),
    ``Single Gaussian → Real`` (``gaussian_result``) and ``GMM → Real``
    (``gmm_result``); all are evaluated on the fixed real Occupancy evaluation
    split. Loss-vs-N graphs are rendered into ``output_dir`` during generation;
    their filenames are recorded in ``graphs_out`` (if provided) so callers can
    register them in ``scenario.json``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    if graphs_out is None:
        graphs_out = {}

    arm_results = {
        "real_to_real": real_result,
        "single_gaussian_to_real": gaussian_result,
        "gmm_to_real": gmm_result,
    }
    arm_labels = {
        "real_to_real": "Real → Real",
        "single_gaussian_to_real": "Single Gaussian → Real",
        "gmm_to_real": "GMM → Real",
    }

    channel_names = tuple(
        channel_names
        or real_result.metadata.get("channel_names", [])
        or gaussian_result.metadata.get("channel_names", [])
    )
    n_max = max(real_result.sample_sizes)

    question = (
        "Which training distribution best preserves the cooperative structure "
        "observed under real-data evaluation in the Occupancy Detection dataset: "
        "real data, single-Gaussian synthetic data, or GMM synthetic data?"
    )

    related = (
        "<p class='related'>Detailed reports: "
        f"<a href='{html.escape(dataset_report)}'>dataset</a> &middot; "
        f"<a href='{html.escape(real_report)}'>Real → Real Monte Carlo</a> &middot; "
        f"<a href='{html.escape(sg_real_report)}'>Single Gaussian → Real Monte Carlo</a> &middot; "
        f"<a href='{html.escape(gmm_real_report)}'>GMM → Real Monte Carlo</a>"
        "</p>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CoInfoSim — Occupancy Detection Baseline Scenario</title>
<style>{_STYLE}{TAB_CSS}</style>
</head>
<body>

<h1>CoInfoSim — Occupancy Detection Baseline Scenario</h1>
{related}

{_legend_html(channel_names)}

<h2>1. Scientific question</h2>
<p class="question">{html.escape(question)}</p>

<h2>2. Scenario summary</h2>
{_summary_table(real_result, scenario_meta, channel_names, n_max)}

{_protocol_html(real_result)}

{_carousel_html(visualization)}

{_best_comparison_html(real_result, gaussian_result, gmm_result, n_max, output_dir, graph_suffix, graphs_out, generate_graphs)}

{_top_ranked_html(real_result, gaussian_result, gmm_result, n_max, output_dir, graph_suffix, graphs_out, generate_graphs=generate_graphs)}

{_structural_fidelity_section(arm_results, "real_to_real", arm_labels, output_dir, graph_suffix, graphs_out, generate_graphs)}

{_nstar_section(real_result, gaussian_result, gmm_result, channel_names, n_max, output_dir, graph_suffix, graphs_out, generate_graphs)}

{_interpretation_html()}

{TAB_JS}
</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
