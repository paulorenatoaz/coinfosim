"""Matplotlib helpers for predictive-cooperation-profile report figures."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402


def profile_metric_series_figure(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    arm_labels: Mapping[str, str],
    title: str,
):
    """Build one bounded predictive-cooperation-profile metric curve figure."""

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    arms = list(dict.fromkeys(str(row["arm"]) for row in rows))
    x_field = (
        "n_prefix"
        if metric in ("reversal_existence_agreement", "reversal_sample_size_similarity")
        else "n_per_class"
    )
    for arm in arms:
        arm_rows = sorted(
            (row for row in rows if row["arm"] == arm),
            key=lambda row: int(row[x_field]),
        )
        points = [row for row in arm_rows if row.get(metric) is not None]
        if points:
            ax.plot(
                [int(row[x_field]) for row in points],
                [float(row[metric]) for row in points],
                marker="o",
                linewidth=1.8,
                label=arm_labels.get(arm, arm),
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_per_class")
    ylabel = {
        "rho_rank": "Ranking fidelity",
        "winner_agreement": "Winner agreement",
        "reversal_existence_agreement": "Reversal existence agreement",
        "reversal_sample_size_similarity": "Reversal sample-size similarity",
    }[metric]
    ax.set_ylabel(ylabel)
    ax.set_ylim((-1.0, 1.0) if metric == "rho_rank" else (0.0, 1.0))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    return fig


def _draw_winner_matrix(ax, matrix: Sequence[Sequence[object]], subset_labels: Sequence[str], title: str) -> None:
    values = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in matrix]
    )
    cmap = ListedColormap(["#d95f5f", "#eeeeee", "#5aa469"])
    cmap.set_bad("#ffffff")
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    image = ax.imshow(values, cmap=cmap, norm=norm)
    ax.set_xticks(range(len(subset_labels)), subset_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(subset_labels)), subset_labels)
    ax.set_xlabel("Column subset")
    ax.set_ylabel("Row subset")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{int(values[i, j]):+d}", ha="center", va="center")
    colorbar = ax.figure.colorbar(image, ax=ax, ticks=[-1, 0, 1], fraction=0.046)
    colorbar.ax.set_yticklabels(["row loses", "unresolved", "row wins"])


def winner_matrix_figure(
    matrix: Sequence[Sequence[object]],
    subset_labels: Sequence[str],
    title: str,
):
    """Build a directed effective-winner-matrix (`W`) heatmap.

    `0` denotes an unresolved effective winner; an exact tie after
    initialization is already carried forward before reaching this figure.
    """

    size = max(5.0, 1.05 * len(subset_labels))
    fig, ax = plt.subplots(figsize=(size, size))
    _draw_winner_matrix(ax, matrix, subset_labels, title)
    fig.tight_layout()
    return fig


def _draw_reversal_matrix(ax, matrix: Sequence[Sequence[object]], subset_labels: Sequence[str], title: str) -> None:
    n = len(subset_labels)
    values = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i][j]
            if value is not None:
                values[i, j] = float(value)
    finite = values[np.isfinite(values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ffffff")
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n), subset_labels, rotation=45, ha="right")
    ax.set_yticks(range(n), subset_labels)
    ax.set_xlabel("Column subset")
    ax.set_ylabel("Row subset")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if np.isfinite(values[i, j]):
                ax.text(
                    j,
                    i,
                    str(int(values[i, j])),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > (vmin + vmax) / 2 else "black",
                )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046)
    colorbar.set_label("Last observed reversal sample size")


def reversal_matrix_figure(
    matrix: Sequence[Sequence[object]],
    subset_labels: Sequence[str],
    title: str,
):
    """Build a triangular reversal-matrix (`R`) heatmap.

    Only upper-triangle cells (`i < j`) are shown; the diagonal and lower
    triangle are always masked, and missing (never-reversed) pairs are
    rendered as visually unavailable rather than zero.
    """

    n = len(subset_labels)
    size = max(5.0, 1.05 * n)
    fig, ax = plt.subplots(figsize=(size, size))
    _draw_reversal_matrix(ax, matrix, subset_labels, title)
    fig.tight_layout()
    return fig


def paired_winner_reversal_figure(
    winner_matrix: Sequence[Sequence[object]],
    reversal_matrix: Sequence[Sequence[object]],
    subset_labels: Sequence[str],
    *,
    winner_title: str = "Winner matrix",
    reversal_title: str = "Reversal matrix",
    suptitle: Optional[str] = None,
):
    """Build one figure showing `W` and `R` side by side for the same prefix.

    `W` says who currently wins each pair; `R` says when the pair last
    changed winner. The current winner is always recoverable from `W`, so
    `R` is never annotated with a direction.
    """

    size = max(5.0, 1.05 * len(subset_labels))
    fig, (ax_winner, ax_reversal) = plt.subplots(1, 2, figsize=(2 * size, size))
    _draw_winner_matrix(ax_winner, winner_matrix, subset_labels, winner_title)
    _draw_reversal_matrix(ax_reversal, reversal_matrix, subset_labels, reversal_title)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def save_figure(fig, output_dir: Path | str, filename: str) -> Path:
    """Save and close a report figure."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_to_data_uri(fig) -> str:
    """Encode and close a figure for a self-contained HTML report."""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
