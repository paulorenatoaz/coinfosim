"""Matplotlib helpers for structural-fidelity report figures."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402


def metric_series_figure(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    arm_labels: Mapping[str, str],
    title: str,
):
    """Build one bounded structural-metric curve figure."""

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    arms = list(dict.fromkeys(str(row["arm"]) for row in rows))
    x_field = "n_prefix" if metric == "nstar_similarity" else "n_per_class"
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
        "nstar_similarity": "Progressive N-star similarity",
    }[metric]
    ax.set_ylabel(ylabel)
    ax.set_ylim((-1.0, 1.0) if metric == "rho_rank" else (0.0, 1.0))
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    return fig


def winner_matrix_figure(
    matrix: Sequence[Sequence[object]],
    subset_labels: Sequence[str],
    title: str,
):
    """Build a directed winner-matrix heatmap."""

    values = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in matrix]
    )
    cmap = ListedColormap(["#d95f5f", "#eeeeee", "#5aa469"])
    cmap.set_bad("#ffffff")
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    size = max(5.0, 1.05 * len(subset_labels))
    fig, ax = plt.subplots(figsize=(size, size))
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
    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1], fraction=0.046)
    colorbar.ax.set_yticklabels(["row loses", "tie", "row wins"])
    fig.tight_layout()
    return fig


def progressive_nstar_matrix_figure(
    matrix: Sequence[Sequence[object]],
    subset_labels: Sequence[str],
    title: str,
):
    """Build a progressive directed observed-grid N-star heatmap."""

    values = np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in matrix]
    )
    finite = values[np.isfinite(values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ffffff")
    size = max(5.0, 1.05 * len(subset_labels))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(subset_labels)), subset_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(subset_labels)), subset_labels)
    ax.set_xlabel("Column subset")
    ax.set_ylabel("Row subset")
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(
                    j,
                    i,
                    str(int(values[i, j])),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > (vmin + vmax) / 2 else "black",
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046)
    colorbar.set_label("Latest observed crossing N*")
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
