"""
Scenario data-visualization panels for CoInfoSim.

This module renders academic 1D/2D/3D projection panels for a labeled two-class
sample, in the visual spirit of the legacy ``core.model.Model`` data plots
(class-colored scatter, standard-deviation ellipses/ellipsoids, density curves)
but decoupled from the legacy ``Model`` class. It operates directly on NumPy
arrays / :class:`~coinfosim.samplers.dataset.Dataset` objects produced by the
new pipeline.

Axes use mathematical channel labels ($X_1, X_2, \\ldots$); sensor names are
intentionally not used here (the scenario report carries a sticky channel
legend instead). Ellipses/ellipsoids are estimated empirically from the plotted
sample's per-class mean and covariance, so the same code works for both the
real-data arm and the Gaussian-anchored arm.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

_BLUE = "#1f77b4"
_ORANGE = "#ff7f0e"
_BLUE_FILL = "lightblue"
_ORANGE_FILL = "moccasin"
_SIGMA_SCALES = (1.0, 2.0)


def build_balanced_sample(
    X: np.ndarray,
    y: np.ndarray,
    per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return a balanced, deterministic sub-sample of ``(X, y)``.

    Draws ``min(per_class, min_class_count)`` rows per class using a seeded RNG.
    Returns ``(X_sample, y_sample, actual_per_class)``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    labels = np.unique(y)
    rng = np.random.default_rng(seed)

    available = min(int(np.sum(y == label)) for label in labels)
    take = int(min(per_class, available))

    blocks_X: List[np.ndarray] = []
    blocks_y: List[np.ndarray] = []
    for label in labels:
        idx = np.flatnonzero(y == label)
        chosen = rng.permutation(idx)[:take]
        blocks_X.append(X[chosen])
        blocks_y.append(y[chosen])
    return np.vstack(blocks_X), np.concatenate(blocks_y), take


def _class_splits(X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, str, str]]:
    labels = np.unique(y)
    styles = [(_BLUE, _BLUE_FILL), (_ORANGE, _ORANGE_FILL)]
    out = []
    for i, label in enumerate(labels):
        edge, fill = styles[i % len(styles)]
        out.append((X[y == label], edge, fill))
    return out


def _empirical_ellipse(points2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = points2d.mean(axis=0)
    cov = np.cov(points2d, rowvar=False)
    cov = np.atleast_2d(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    return mean, eigvecs @ np.diag(np.sqrt(eigvals))


def _label(i: int) -> str:
    return f"$X_{{{i + 1}}}$"


# --------------------------------------------------------------------------- #
# 1D
# --------------------------------------------------------------------------- #
def plot_1d(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    components: Optional[Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]] = None,
) -> plt.Figure:
    """Render 1D marginal density panels.

    If ``components`` is provided it must map ``class_label`` to a list of
    ``(weight, mean_d, cov_d)`` tuples drawn from a mixture model. One
    weighted Gaussian density curve is plotted per component. When ``None``
    a single empirical curve is plotted per class as before.
    """
    d = X.shape[1]
    ncols = min(d, 3)
    nrows = int(np.ceil(d / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    labels = sorted(int(v) for v in np.unique(y))
    styles = [(_BLUE, _BLUE_FILL), (_ORANGE, _ORANGE_FILL)]
    splits = _class_splits(X, y)
    x_grid = np.linspace(
        float(np.min(X)) - 1.0, float(np.max(X)) + 1.0, 400
    )
    for i in range(d):
        ax = axes[i]
        for s_idx, (pts, edge, _) in enumerate(splits):
            col = pts[:, i]
            jitter = np.random.default_rng(i).normal(0, 0.02, size=col.shape[0])
            ax.scatter(col, jitter, c=edge, alpha=0.25, s=12, linewidths=0)
            label = labels[s_idx]
            if components is not None and label in components:
                # One density curve per component, scaled by weight.
                for w, m_d, cov_d in components[label]:
                    mu = float(m_d[i])
                    sigma = float(np.sqrt(np.clip(cov_d[i, i], 1e-12, None)))
                    ax.plot(
                        x_grid, w * norm.pdf(x_grid, loc=mu, scale=sigma),
                        color=edge, alpha=0.75, linewidth=1.2,
                    )
                    for scale in _SIGMA_SCALES:
                        ax.axvline(
                            mu - scale * sigma, color=edge,
                            linestyle=":", alpha=0.35, linewidth=0.8,
                        )
                        ax.axvline(
                            mu + scale * sigma, color=edge,
                            linestyle=":", alpha=0.35, linewidth=0.8,
                        )
            else:
                mu, sigma = float(np.mean(col)), float(np.std(col) or 1e-6)
                ax.plot(x_grid, norm.pdf(x_grid, loc=mu, scale=sigma), color=edge)
                for scale in _SIGMA_SCALES:
                    ax.axvline(mu - scale * sigma, color=edge, linestyle="--", alpha=0.5)
                    ax.axvline(mu + scale * sigma, color=edge, linestyle="--", alpha=0.5)
        ax.set_title(_label(i))
        ax.set_xlabel(_label(i))
        ax.set_yticks([])
    for j in range(d, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# 2D
# --------------------------------------------------------------------------- #
def plot_2d(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    components: Optional[Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]] = None,
) -> plt.Figure:
    """Render 2D scatter + ellipse panels.

    If ``components`` is provided one ellipse per mixture component is drawn
    (one per sigma scale), with line thickness proportional to weight. When
    ``None`` a single empirical ellipse per class is drawn as before.
    """
    d = X.shape[1]
    pairs = list(combinations(range(d), 2))
    ncols = min(len(pairs), 3)
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    labels = sorted(int(v) for v in np.unique(y))
    splits = _class_splits(X, y)
    angles = np.linspace(0, 2 * np.pi, 100)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])

    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx]
        for s_idx, (pts, edge, fill) in enumerate(splits):
            sub = pts[:, [a, b]]
            ax.scatter(
                sub[:, 0], sub[:, 1], edgecolors=edge, c=fill,
                linewidths=0.4, s=18, alpha=0.6,
            )
            label = labels[s_idx]
            if components is not None and label in components:
                for w, m_d, cov_d in components[label]:
                    m2 = m_d[[a, b]]
                    c2 = cov_d[np.ix_([a, b], [a, b])]
                    eigvals, eigvecs = np.linalg.eigh(c2)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    transform = eigvecs @ np.diag(np.sqrt(eigvals))
                    lw = max(0.7, 2.0 * float(w))
                    for scale in _SIGMA_SCALES:
                        ell = (unit * scale) @ transform.T + m2
                        ax.plot(
                            ell[:, 0], ell[:, 1],
                            color=edge, linestyle="--",
                            alpha=0.7, linewidth=lw,
                        )
            else:
                mean, transform = _empirical_ellipse(sub)
                for scale in _SIGMA_SCALES:
                    ell = (unit * scale) @ transform.T + mean
                    ax.plot(ell[:, 0], ell[:, 1], color=edge, linestyle="--", alpha=0.8)
        ax.set_xlabel(_label(a))
        ax.set_ylabel(_label(b))
        ax.set_title(f"$(X_{{{a + 1}}}, X_{{{b + 1}}})$")
    for j in range(len(pairs), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# 3D
# --------------------------------------------------------------------------- #
def _unit_sphere(n: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def plot_3d(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    components: Optional[Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]] = None,
) -> plt.Figure:
    """Render 3D scatter + ellipsoid panels.

    If ``components`` is provided one ellipsoid surface per mixture component
    is drawn, with opacity proportional to weight. When ``None`` a single
    empirical ellipsoid per class is drawn as before.
    """
    d = X.shape[1]
    triples = list(combinations(range(d), 3))
    ncols = min(len(triples), 3)
    nrows = int(np.ceil(len(triples) / ncols))
    fig = plt.figure(figsize=(5.4 * ncols, 4.6 * nrows))

    labels = sorted(int(v) for v in np.unique(y))
    splits = _class_splits(X, y)
    sx, sy, sz = _unit_sphere(20)
    sphere = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=0)

    for idx, combo in enumerate(triples):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        for s_idx, (pts, edge, fill) in enumerate(splits):
            sub = pts[:, list(combo)]
            ax.scatter(
                sub[:, 0], sub[:, 1], sub[:, 2],
                edgecolors=edge, c=fill, linewidths=0.3, s=14, alpha=0.55,
            )
            label = labels[s_idx]
            if components is not None and label in components:
                for w, m_d, cov_d in components[label]:
                    m3 = m_d[list(combo)]
                    c3 = cov_d[np.ix_(list(combo), list(combo))]
                    eigvals, eigvecs = np.linalg.eigh(c3)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    transform = eigvecs @ np.diag(np.sqrt(eigvals))
                    ell = (transform @ sphere).T + m3
                    ex = ell[:, 0].reshape(sx.shape)
                    ey = ell[:, 1].reshape(sy.shape)
                    ez = ell[:, 2].reshape(sz.shape)
                    ax.plot_surface(
                        ex, ey, ez, color=edge,
                        alpha=max(0.08, 0.35 * float(w)),
                        linewidth=0, rstride=2, cstride=2,
                    )
            else:
                mean = sub.mean(axis=0)
                cov = np.atleast_2d(np.cov(sub, rowvar=False))
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
                transform = eigvecs @ np.diag(np.sqrt(eigvals))
                ell = (transform @ sphere).T + mean
                ex = ell[:, 0].reshape(sx.shape)
                ey = ell[:, 1].reshape(sy.shape)
                ez = ell[:, 2].reshape(sz.shape)
                ax.plot_surface(
                    ex, ey, ez, color=edge, alpha=0.18, linewidth=0,
                    rstride=2, cstride=2,
                )
        ax.set_xlabel(_label(combo[0]))
        ax.set_ylabel(_label(combo[1]))
        ax.set_zlabel(_label(combo[2]))
        ax.set_title(f"$(X_{{{combo[0]+1}}}, X_{{{combo[1]+1}}}, X_{{{combo[2]+1}}})$")
        ax.view_init(elev=25, azim=-50)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# Loss-vs-N analytical graphs
# --------------------------------------------------------------------------- #

# N-star graph palette: reference = deep blue, VS = orange/amber family.
_REF_COLOR = "#1f4e9e"          # deep blue for the reference curve
_VS_COLORS = [                   # coherent warm/amber gradient for VS curves
    "#e07b00",
    "#c0392b",
    "#8e44ad",
    "#d35400",
    "#922b21",
    "#6c3483",
    "#a93226",
]


def save_loss_vs_n(
    sample_sizes: Sequence[int],
    series: Sequence[Dict],
    title: str,
    output_dir: Path | str,
    filename: str,
    markers: Optional[Sequence[Dict]] = None,
    nstar_style: bool = False,
) -> str:
    """Render a loss-vs-N line plot to ``output_dir/filename`` and return the name.

    ``series`` is a sequence of ``{"label", "means", "ses"}`` dicts. The first
    entry is always plotted as the reference curve when ``nstar_style=True``:
    deep blue, thick, full opacity; remaining VS curves use a coherent warm
    palette with reduced transparency.

    ``markers`` is an optional sequence of ``{"x", "label"}`` dicts drawn as
    dashed vertical lines to indicate interpolated N* values. They are annotated
    in the figure and explained in a legend entry.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = list(sample_sizes)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.6, 4.5))

    for i, s in enumerate(series):
        is_ref = nstar_style and i == 0
        if is_ref:
            color = _REF_COLOR
            lw = 2.6
            alpha = 1.0
            ms = 6
            zorder = 4
        elif nstar_style:
            color = _VS_COLORS[(i - 1) % len(_VS_COLORS)]
            lw = 1.3
            alpha = 0.48
            ms = 4
            zorder = 3
        else:
            color = cmap(i % 10)
            lw = 1.5
            alpha = 0.9
            ms = 4
            zorder = 3

        means = s["means"]
        ses = s.get("ses")
        ax.errorbar(
            sizes,
            means,
            yerr=ses,
            marker="o",
            markersize=ms,
            linewidth=lw,
            capsize=2,
            color=color,
            alpha=alpha,
            label=s["label"],
            zorder=zorder,
        )
        # Final-point emphasis for N-star reference.
        if is_ref and means:
            ax.scatter(
                [sizes[-1]], [means[-1]], s=60, color=color,
                zorder=5, edgecolors="white", linewidths=0.8,
            )

    # Draw dashed N* markers.
    if markers:
        nstar_handle_added = False
        for m in markers:
            xv = float(m["x"])
            color = m.get("color", "#444444")
            lbl = m.get("label")
            kw = dict(
                color=color, linestyle="--", linewidth=1.0, alpha=0.8, zorder=2
            )
            if not nstar_handle_added:
                import matplotlib.lines as mlines
                h = mlines.Line2D(
                    [], [], color="#444444", linestyle="--", linewidth=1.0,
                    label="Interpolated N*"
                )
                ax.add_artist(ax.legend(handles=[h], loc="upper right", fontsize=7,
                                        framealpha=0.9, title="Marker legend"))
                nstar_handle_added = True
            ax.axvline(xv, **kw)
            if lbl:
                ylim = ax.get_ylim()
                y_mid = ylim[0] + (ylim[1] - ylim[0]) * 0.55
                ax.text(
                    xv, y_mid, lbl,
                    rotation=90, va="center", ha="right",
                    fontsize=6, color=color, alpha=0.85,
                )

    if len(sizes) > 1:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=7)
    ax.set_xlabel("N (samples per class)")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=1, framealpha=0.9, loc="upper right"
              if not markers else "lower left")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return filename


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_scenario_visualizations(
    samples: Dict[
        str,
        Tuple[
            np.ndarray,
            np.ndarray,
            str,
            Optional[Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]]],
        ],
    ],
    output_dir: Path | str,
    filename_suffix: str,
) -> Dict[str, str]:
    """Render 1D/2D/3D projection panels for each named sample.

    ``samples`` maps an arm key (e.g. ``"real"``, ``"gaussian"``, ``"gmm"``) to a
    ``(X, y, title_prefix, components)`` tuple. ``components`` is optional
    (pass ``None`` or omit the 4th element to use the legacy empirical behaviour).
    When supplied it must map ``class_label`` to a list of
    ``(weight, mean_d, cov_d)`` tuples from the fitted mixture model; the plot
    functions then draw one density/ellipse/ellipsoid per component instead of
    a single empirical region.

    For each key three panels are rendered with keys ``viz_1d_<key>``,
    ``viz_2d_<key>`` and ``viz_3d_<key>``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dims = (("1d", plot_1d), ("2d", plot_2d), ("3d", plot_3d))
    images: Dict[str, str] = {}
    for arm_key, entry in samples.items():
        X, y, prefix = entry[0], entry[1], entry[2]
        components = entry[3] if len(entry) > 3 else None  # type: ignore[arg-type]
        for dim, fn in dims:
            panel_key = f"viz_{dim}_{arm_key}"
            title = f"{prefix} — {dim.upper()} projections"
            fig = fn(X, y, title, components)
            filename = f"{panel_key}_{filename_suffix}.png"
            fig.savefig(output_dir / filename, dpi=110, bbox_inches="tight")
            plt.close(fig)
            images[panel_key] = filename
    plt.close("all")
    return images
