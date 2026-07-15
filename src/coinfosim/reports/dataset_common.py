"""Shared rendering primitives for dataset-specific HTML reports."""

from __future__ import annotations

import base64
import html
import io
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def dataframe_html(
    frame: pd.DataFrame, float_cols: Optional[Dict[str, str]] = None
) -> str:
    float_cols = float_cols or {}
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in float_cols and not pd.isna(value):
                cells.append(f"<td>{float(value):{float_cols[column]}}</td>")
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


def key_value_table_html(rows: Sequence[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    return f"<table class='data key-value'><tbody>{body}</tbody></table>"


def file_hash_table_html(hashes: Mapping[str, str]) -> str:
    frame = pd.DataFrame(
        [{"Filename": filename, "SHA-256": digest} for filename, digest in hashes.items()]
    )
    return dataframe_html(frame)


def standardization_table_html(parameters) -> str:
    frame = parameters.as_dataframe().reset_index().rename(columns={"index": "channel"})
    return dataframe_html(frame, float_cols={"mean": ".6f", "std": ".6f"})


def class_distribution_image(
    counts_by_group: Mapping[str, Mapping[int, int]],
    class_labels: Sequence[int],
    *,
    title: str,
) -> str:
    groups = list(counts_by_group)
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(class_labels))
    fig, axis = plt.subplots(figsize=(7, 4))
    for index, label in enumerate(class_labels):
        offset = (index - (len(class_labels) - 1) / 2) * width
        values = [counts_by_group[group].get(int(label), 0) for group in groups]
        axis.bar(x + offset, values, width, label=f"class {label}")
    axis.set_xticks(x)
    axis.set_xticklabels(groups, rotation=20, ha="right")
    axis.set_ylabel("Rows")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return figure_to_data_uri(fig)


def correlation_heatmap_image(correlation: pd.DataFrame, *, title: str) -> str:
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(np.arange(len(correlation.columns)))
    axis.set_yticks(np.arange(len(correlation.index)))
    axis.set_xticklabels(correlation.columns, rotation=35, ha="right")
    axis.set_yticklabels(correlation.index)
    axis.set_title(title)
    for row in range(correlation.shape[0]):
        for column in range(correlation.shape[1]):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return figure_to_data_uri(fig)


def standardized_mean_comparison_image(
    summary: pd.DataFrame,
    channel_names: Sequence[str],
    *,
    title: str,
) -> str:
    channels = list(channel_names)
    train_means = (
        summary[summary["split"] == "train_pool"]
        .set_index("channel")
        .loc[channels, "mean"]
    )
    test_means = (
        summary[summary["split"] == "fixed_test"]
        .set_index("channel")
        .loc[channels, "mean"]
    )
    x = np.arange(len(channels))
    width = 0.35
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - width / 2, train_means, width, label="train pool")
    axis.bar(x + width / 2, test_means, width, label="fixed test")
    axis.axhline(0, color="#444", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(channels, rotation=25, ha="right")
    axis.set_ylabel("Standardized mean")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return figure_to_data_uri(fig)
