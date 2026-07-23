"""Compatibility shim for the canonical predictive-cooperation-profile figures.

The active implementation lives in
:mod:`coinfosim.reports.predictive_profile_visualization`. This module
re-exports the canonical figures under their previous names for one
compatibility cycle, plus the historical (unused in the active report path)
N-star progressive-crossing figure. Active code must import from
:mod:`coinfosim.reports.predictive_profile_visualization` directly; do not
add new callers of this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coinfosim.reports.predictive_profile_visualization import (
    figure_to_data_uri,
    paired_winner_reversal_figure,
    profile_metric_series_figure as metric_series_figure,
    reversal_matrix_figure,
    save_figure,
    winner_matrix_figure,
)

__all__ = [
    "metric_series_figure",
    "winner_matrix_figure",
    "reversal_matrix_figure",
    "paired_winner_reversal_figure",
    "save_figure",
    "figure_to_data_uri",
    "progressive_nstar_matrix_figure",
]


def progressive_nstar_matrix_figure(
    matrix: Sequence[Sequence[object]],
    subset_labels: Sequence[str],
    title: str,
):
    """Build a progressive directed observed-grid N-star heatmap. Deprecated, unused."""

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
