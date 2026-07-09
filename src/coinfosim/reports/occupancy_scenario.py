"""Occupancy Detection baseline scenario report (academic layout).

Renders a self-contained, academic-style HTML scenario report comparing the
real-data arm and the Gaussian-anchored arm of the Occupancy Detection baseline
scenario. Channels are referred to with mathematical labels ``X_i`` and a
sticky channel legend; sensor names are used only in the legend and in explicit
"reference subset" annotations, never as primary table/plot labels.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from coinfosim.classifiers.registry import classifier_label
from coinfosim.reports.scenario_visualization import save_loss_vs_n
from coinfosim.results.analysis import cooperative_threshold_interpolated
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


# --------------------------------------------------------------------------- #
# N-star availability
# --------------------------------------------------------------------------- #
def _competitor_plan(
    real_result: SimulationResult, classifier: str, n: int, k: int, d: int
) -> Tuple[Optional[Subset], List[Tuple[str, Subset]]]:
    """Return the reference subset and ordered competitor ``(label, subset)`` list.

    Reference and competitor subsets are selected on the real-data arm at the
    largest evaluated ``N`` so that both arms are compared on the same subsets.
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
    channel_names: Sequence[str],
    n: int,
    output_dir: Path,
    graph_suffix: str,
    graphs_out: Optional[Dict],
    generate_graphs: bool = True,
) -> str:
    d = len(channel_names)
    parts: List[str] = ["<h2>7. N-star availability</h2>"]
    parts.append(
        "<p>For each classifier and channel-subset cardinality, the best "
        "subset at the largest evaluated sample size is used as a reference "
        "and compared against the best subset of every other cardinality and, "
        "when it exists, the second-best subset of the same cardinality. "
        "Reference and competitor subsets are selected on the real-data arm so "
        "that both arms are compared on the same subsets.</p>"
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
    sub = 1
    for clf in real_result.classifier_names:
        parts.append(f"<h3>7.{sub} {html.escape(classifier_label(clf))}</h3>")
        sub += 1
        for k in range(1, d + 1):
            reference, competitors = _competitor_plan(real_result, clf, n, k, d)
            if reference is None:
                continue
            parts.append(f"<h4>Best {k}-channel subset</h4>")
            parts.append(
                "<p class='refsubset'>Reference subset: "
                f"{html.escape(_subset_notation(reference))} = "
                f"{html.escape(_subset_sensors(reference, channel_names))}</p>"
            )
            for arm_key, arm_name, arm_result in (
                ("real", "Real-data", real_result),
                ("gaussian", "Gaussian-anchored", gaussian_result),
            ):
                analysis = _nstar_analysis(
                    arm_result, clf, reference, competitors, n
                )
                parts.append(f"<h5>{arm_name}</h5>")
                parts.append(
                    _table(
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
                parts.append(
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
        ("Number of channels", str(len(channel_names))),
        ("Non-empty channel subsets", str(len(real_result.subsets))),
        ("Classifiers", html.escape(classifiers)),
        ("Metric", "Empirical test loss (0–1 misclassification)"),
        ("Sample sizes", html.escape(str(list(real_result.sample_sizes)))),
        ("Largest evaluated sample size", f"N = {n_max} samples per class"),
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
        "<h3>3.1 Real-data arm</h3>"
        "<p>The real-data arm draws balanced training samples from the "
        "standardized Occupancy training pool and evaluates all channel "
        "subsets and classifiers using the fixed real Occupancy evaluation "
        "split.</p>"
        "<h3>3.2 Gaussian-anchored arm</h3>"
        "<p>The Gaussian-anchored arm estimates class-conditional Gaussian "
        "parameters from the standardized Occupancy training pool. Synthetic "
        "training samples are generated deterministically by replication id "
        "and are prefix-nested across sample sizes. The Gaussian-anchored "
        "evaluation set is fixed within the simulation run and is generated "
        "with the same class counts as the real Occupancy evaluation "
        "split.</p>"
        "<h3>3.3 Monte Carlo stopping rule</h3>"
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
        ("Synthetic source", meta.get("synthetic_source")),
        ("Visualization seed", meta.get("visualization_seed")),
    ]
    meta_html = "".join(
        f"<tr><th class='key'>{html.escape(k)}</th>"
        f"<td>{html.escape(str(v))}</td></tr>"
        for k, v in meta_rows
        if v is not None
    )

    order = [
        ("real", "1d"),
        ("gaussian", "1d"),
        ("real", "2d"),
        ("gaussian", "2d"),
        ("real", "3d"),
        ("gaussian", "3d"),
    ]

    def _caption(arm: str, dim: str) -> str:
        who = (
            "Real-data sample"
            if arm == "real"
            else "Gaussian-anchored synthetic sample"
        )
        return f"{who} — {dim.upper()} projections"

    data = {}
    for arm, dim in order:
        key = f"viz_{dim}_{arm}"
        data[f"{arm}_{dim}"] = {
            "src": images.get(key, ""),
            "caption": _caption(arm, dim),
        }
    order_keys = [f"{a}_{d}" for a, d in order]
    first = data[order_keys[0]]

    controls = (
        "<div class='carousel-controls'>"
        "<div class='ctrl-group'>"
        "<button type='button' data-arm='real' class='active'>Real-data</button>"
        "<button type='button' data-arm='gaussian'>Gaussian-anchored</button>"
        "</div>"
        "<div class='ctrl-group'>"
        "<button type='button' data-dim='1d' class='active'>1D</button>"
        "<button type='button' data-dim='2d'>2D</button>"
        "<button type='button' data-dim='3d'>3D</button>"
        "</div>"
        "<button type='button' id='viz-toggle' class='toggle'>Pause</button>"
        "</div>"
    )
    stage = (
        "<figure class='carousel-stage'>"
        f"<img id='viz-img' src='{html.escape(first['src'])}' "
        "alt='data visualization'/>"
        f"<figcaption id='viz-caption'>{html.escape(first['caption'])}</figcaption>"
        "</figure>"
    )
    # Static JS (no framework); safe in a locally-opened static HTML file.
    script = (
        "<script>(function(){"
        "var DATA=" + json.dumps(data) + ";"
        "var ORDER=" + json.dumps(order_keys) + ";"
        "var idx=0,timer=null,playing=false;"
        "var img=document.getElementById('viz-img');"
        "var cap=document.getElementById('viz-caption');"
        "var toggle=document.getElementById('viz-toggle');"
        "function render(){var k=ORDER[idx];var it=DATA[k];if(!it){return;}"
        "if(it.src){img.setAttribute('src',it.src);}cap.textContent=it.caption;"
        "var p=k.split('_');"
        "document.querySelectorAll('[data-arm]').forEach(function(b){"
        "b.classList.toggle('active',b.getAttribute('data-arm')===p[0]);});"
        "document.querySelectorAll('[data-dim]').forEach(function(b){"
        "b.classList.toggle('active',b.getAttribute('data-dim')===p[1]);});}"
        "function setState(arm,dim){var i=ORDER.indexOf(arm+'_'+dim);"
        "if(i>-1){idx=i;}render();}"
        "function next(){idx=(idx+1)%ORDER.length;render();}"
        "document.querySelectorAll('[data-arm]').forEach(function(b){"
        "b.addEventListener('click',function(){var c=ORDER[idx].split('_');"
        "setState(b.getAttribute('data-arm'),c[1]);});});"
        "document.querySelectorAll('[data-dim]').forEach(function(b){"
        "b.addEventListener('click',function(){var c=ORDER[idx].split('_');"
        "setState(c[0],b.getAttribute('data-dim'));});});"
        "function start(){timer=setInterval(next,3000);playing=true;"
        "toggle.textContent='Pause';}"
        "function stop(){if(timer){clearInterval(timer);}playing=false;"
        "toggle.textContent='Play';}"
        "toggle.addEventListener('click',function(){"
        "if(playing){stop();}else{start();}});"
        "render();start();"
        "})();</script>"
    )

    return (
        "<h2>4. Data visualization</h2>"
        f"<table class='meta'>{meta_html}</table>"
        "<h3>4.1 Projection carousel</h3>"
        "<div class='carousel' id='viz-carousel'>"
        + controls
        + stage
        + "</div>"
        + script
    )


def _best_comparison_html(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
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
        same = "yes" if real_best[0] == gauss_best[0] else "no"
        rows.append(
            [
                html.escape(classifier_label(clf)),
                _subset_notation(real_best[0]),
                _fmt_loss(real_best[1]),
                _subset_notation(gauss_best[0]),
                _fmt_loss(gauss_best[1]),
                same,
            ]
        )
    table = _table(
        [
            "Classifier",
            "Real-data best subset",
            "Real loss",
            "Gaussian-anchored best subset",
            "Gaussian loss",
            "Same subset",
        ],
        rows,
    )

    graphs: List[str] = []
    for arm_key, arm_name, arm_result in (
        ("real", "Real-data", real_result),
        ("gaussian", "Gaussian-anchored", gaussian_result),
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
        graphs.append(f"<h4>{arm_name} arm</h4>")
        graphs.append(
            _emit_graph(
                output_dir,
                graph_suffix,
                graphs_out,
                f"graph_best_comparison_{arm_key}",
                sizes,
                series,
                title,
                enabled=generate_graphs,
            )
        )

    return (
        "<h2>5. Best subset comparison at largest N</h2>"
        f"<p>Largest evaluated sample size: <b>N = {n} samples per class</b>.</p>"
        f"{table}"
        "<h3>5.1 Loss vs N for the best subsets</h3>"
        + "".join(graphs)
    )


def _top_ranked_html(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
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
    for i, clf in enumerate(real_result.classifier_names, start=1):
        parts.append(f"<h3>6.{i} {html.escape(classifier_label(clf))}</h3>")
        for arm_key, arm_name, arm_result in (
            ("real", "Real-data arm", real_result),
            ("gaussian", "Gaussian-anchored arm", gaussian_result),
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
            parts.append(f"<h4>{arm_name}</h4>")
            parts.append(_table(["Rank", "Subset", "Loss", "SE"], rows))
            subsets = [subset for subset, _, _ in ranked]
            series = _series(arm_result, clf, subsets)
            title = (
                f"{classifier_label(clf)} — {arm_name}: "
                f"top-{len(subsets)} subsets (loss vs N)"
            )
            parts.append(
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
    return "".join(parts)


def _interpretation_html() -> str:
    return (
        "<h2>8. Interpretation notes</h2>"
        "<h3>Agreement between real-data and Gaussian-anchored arms</h3>"
        "<p>Where the two arms select the same best subsets and show similar "
        "cooperative thresholds, the Gaussian surrogate captures the "
        "cooperative structure of the real channels.</p>"
        "<h3>Divergences between real-data and Gaussian-anchored arms</h3>"
        "<p>Differences in best subsets or in N* availability indicate "
        "structure in the real data that a class-conditional Gaussian model "
        "does not reproduce.</p>"
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
.carousel { border:1px solid var(--line); border-radius:8px; padding:1rem;
            margin:.5rem 0 1.6rem; }
.carousel-controls { display:flex; flex-wrap:wrap; gap:.5rem 1.1rem;
            align-items:center; margin-bottom:.9rem;
            font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.carousel-controls .ctrl-group { display:flex; gap:.3rem; }
.carousel button { border:1px solid var(--accent); background:#fff;
            color:var(--accent); border-radius:4px; padding:.25rem .75rem;
            cursor:pointer; font-size:.82rem; }
.carousel button.active { background:var(--accent); color:#fff; }
.carousel button.toggle { border-color:var(--muted); color:var(--muted); }
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
    output_dir: Path | str = "output/reports",
    dataset_report: str = "occupancy_dataset_report.html",
    real_report: str = "occupancy_real_monte_carlo_report.html",
    gaussian_report: str = "occupancy_gaussian_anchored_monte_carlo_report.html",
    filename: str = "occupancy_scenario_report.html",
    channel_names: Sequence[str] | None = None,
    visualization: Optional[Dict] = None,
    scenario_meta: Optional[Dict] = None,
    graph_suffix: str = "",
    graphs_out: Optional[Dict] = None,
    generate_graphs: bool = True,
) -> Path:
    """Generate the academic Occupancy baseline scenario report.

    Loss-vs-N graphs are rendered into ``output_dir`` during generation; their
    filenames are recorded in ``graphs_out`` (if provided) so callers can
    register them in ``scenario.json``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    if graphs_out is None:
        graphs_out = {}

    channel_names = tuple(
        channel_names
        or real_result.metadata.get("channel_names", [])
        or gaussian_result.metadata.get("channel_names", [])
    )
    n_max = max(real_result.sample_sizes)

    question = (
        "Does the cooperative advantage among real information channels in the "
        "Occupancy Detection dataset resemble the cooperative advantage "
        "predicted by a Gaussian model parameterized from the same real data?"
    )

    related = (
        "<p class='related'>Detailed reports: "
        f"<a href='{html.escape(dataset_report)}'>dataset</a> &middot; "
        f"<a href='{html.escape(real_report)}'>real-data Monte Carlo</a> &middot; "
        f"<a href='{html.escape(gaussian_report)}'>Gaussian-anchored Monte Carlo</a>"
        "</p>"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CoInfoSim — Occupancy Detection Baseline Scenario</title>
<style>{_STYLE}</style>
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

{_best_comparison_html(real_result, gaussian_result, n_max, output_dir, graph_suffix, graphs_out, generate_graphs)}

{_top_ranked_html(real_result, gaussian_result, n_max, output_dir, graph_suffix, graphs_out, generate_graphs=generate_graphs)}

{_nstar_section(real_result, gaussian_result, channel_names, n_max, output_dir, graph_suffix, graphs_out, generate_graphs)}

{_interpretation_html()}

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
