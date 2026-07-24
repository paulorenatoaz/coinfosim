#!/usr/bin/env python3
"""Generate compact metric-vs-n figures for the three main result slides.

All values come from the current ``predictive_cooperation_profile`` payloads
persisted in the full-scenario ``scenario.json`` files.  The script does not
run Monte Carlo simulations or derive values from the retired N-star schema.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from coinfosim.results.persistence import load_simulation_result
from coinfosim.results.predictive_profile import (
    scenario_predictive_profile_agreement,
)


@dataclass(frozen=True)
class FigureSpec:
    scenario_path: str
    classifier: str
    title: str
    output_name: str
    expected_final: dict[str, tuple[float, float, float, float]]


SPECS = (
    FigureSpec(
        scenario_path=(
            "output/reports/scenarios/"
            "000002_occupancy_baseline_full/scenario.json"
        ),
        classifier="linear_svm",
        title="Occupancy · SVM linear · cenário full 000002",
        output_name="occupancy_svm_metricas_vs_n_full.pdf",
        expected_final={
            "single_gaussian_to_real": (0.9528, 0.9204, 0.7706, 0.7185),
            "gmm_to_real": (0.8774, 0.8624, 0.7407, 0.7552),
        },
    ),
    FigureSpec(
        scenario_path=(
            "output/reports/scenarios/"
            "000005_air_quality_baseline_full/scenario.json"
        ),
        classifier="gaussian_nb",
        title="Air Quality · Gaussian NB · cenário full 000005",
        output_name="air_quality_gnb_metricas_vs_n_full.pdf",
        expected_final={
            "single_gaussian_to_real": (1.0000, 1.0000, 0.7816, 0.9689),
            "gmm_to_real": (0.9996, 0.9978, 0.8836, 0.9760),
        },
    ),
    FigureSpec(
        scenario_path=(
            "output/reports/scenarios/"
            "000008_support2_baseline_full/scenario.json"
        ),
        classifier="random_forest",
        title="SUPPORT2 · Random Forest · cenário full 000008",
        output_name="support2_rf_metricas_vs_n_full.pdf",
        expected_final={
            "single_gaussian_to_real": (0.8387, 0.8281, 0.5052, 0.8165),
            "gmm_to_real": (0.8665, 0.8429, 0.4272, 0.8059),
        },
    ),
)

ARMS = (
    ("single_gaussian_to_real", "Gaussiana única → Real"),
    ("gmm_to_real", "GMM → Real"),
)
# Colors are the dark (GMM) reference; the light (Single Gaussian) variant is
# looked up from LIGHT_COLORS. Hue/marker-shape encode the metric; lightness,
# line style, and marker fill encode the synthetic arm (task plan Section 3).
METRICS = (
    (
        "rho_rank",
        r"$\rho_{\mathrm{rank}}$",
        "ranking_fidelity_series",
        "n_per_class",
        "#3F68B0",
        "o",
    ),
    (
        "winner_agreement",
        r"$A_W$",
        "winner_agreement_series",
        "n_per_class",
        "#B98136",
        "s",
    ),
    (
        "reversal_existence_agreement",
        r"$A_R$",
        "reversal_agreement_series",
        "n_prefix",
        "#4E8B66",
        "D",
    ),
    (
        "reversal_sample_size_similarity",
        r"$S_R$",
        "reversal_agreement_series",
        "n_prefix",
        "#8B5A88",
        "^",
    ),
)

LIGHT_COLORS = {
    "rho_rank": "#9FB3D8",
    "winner_agreement": "#DCC49B",
    "reversal_existence_agreement": "#A6C5B2",
    "reversal_sample_size_similarity": "#C5ADC4",
}

ARM_STYLE = {
    "single_gaussian_to_real": {
        "linestyle": "--",
        "filled": False,
        "x_jitter": -0.06,
    },
    "gmm_to_real": {
        "linestyle": "-",
        "filled": True,
        "x_jitter": 0.06,
    },
}

INK = "#3B3E58"
GRAY = "#6C6980"
GRID = "#D9D7E4"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_metric_series(
    path: Path, classifier: str
) -> tuple[
    list[int],
    dict[str, dict[str, list[tuple[int, float | None]]]],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload["report_data"].get("predictive_cooperation_profile")
    if profile is None:
        refs = payload["simulation_refs"]
        result_paths = {
            "real_to_real": next(
                value["result_data_path"]
                for key, value in refs.items()
                if key.endswith("_real_data")
            ),
            "single_gaussian_to_real": next(
                value["result_data_path"]
                for key, value in refs.items()
                if key.endswith("_single_gaussian_to_real")
            ),
            "gmm_to_real": next(
                value["result_data_path"]
                for key, value in refs.items()
                if key.endswith("_gmm_to_real")
            ),
        }
        root = repository_root()
        arm_results = {
            arm: load_simulation_result(root / result_path)
            for arm, result_path in result_paths.items()
        }
        profile = scenario_predictive_profile_agreement(
            arm_results,
            "real_to_real",
            {
                "real_to_real": "Real → Real",
                "single_gaussian_to_real": "Gaussiana única → Real",
                "gmm_to_real": "GMM → Real",
            },
        )
    sample_sizes = [int(value) for value in profile["sample_sizes"]]
    data: dict[str, dict[str, list[tuple[int, float | None]]]] = {}

    for arm, _ in ARMS:
        arm_data: dict[str, list[tuple[int, float | None]]] = {}
        for metric, _, series_name, n_field, _, _ in METRICS:
            rows = [
                row
                for row in profile[series_name]
                if row["classifier"] == classifier and row["arm"] == arm
            ]
            points = sorted(
                (
                    (
                        int(row[n_field]),
                        (
                            float(row[metric])
                            if row.get(metric) is not None
                            else None
                        ),
                    )
                    for row in rows
                ),
                key=lambda item: item[0],
            )
            if not points or not any(value is not None for _, value in points):
                raise ValueError(
                    f"No {metric} values for {classifier}/{arm} in {path}"
                )
            observed_n = [n for n, _ in points]
            if observed_n != sample_sizes:
                raise ValueError(
                    f"Incomplete n grid for {metric}/{classifier}/{arm} in "
                    f"{path}: {observed_n} != {sample_sizes}"
                )
            if any(
                not math.isfinite(value)
                for _, value in points
                if value is not None
            ):
                raise ValueError(
                    f"Non-finite {metric} value for {classifier}/{arm} in {path}"
                )
            arm_data[metric] = points
        data[arm] = arm_data
    return sample_sizes, data


def validate_final_values(
    spec: FigureSpec,
    data: dict[str, dict[str, list[tuple[int, float | None]]]],
) -> None:
    metric_names = tuple(metric for metric, _, _, _, _, _ in METRICS)
    for arm, expected in spec.expected_final.items():
        observed = tuple(data[arm][metric][-1][1] for metric in metric_names)
        if any(value is None for value in observed):
            raise RuntimeError(
                f"Missing final value for {spec.output_name}/{arm}: {observed}"
            )
        if any(abs(actual - target) > 5e-4 for actual, target in zip(observed, expected)):
            raise RuntimeError(
                f"Final values differ for {spec.output_name}/{arm}: "
                f"{observed} != {expected}"
            )


def make_figure(
    spec: FigureSpec,
    sample_sizes: list[int],
    data: dict[str, dict[str, list[tuple[int, float | None]]]],
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 8.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.edgecolor": GRAY,
            "pdf.fonttype": 42,
        }
    )

    finite_values = [
        value
        for arm, _ in ARMS
        for metric, _, _, _, _, _ in METRICS
        for _, value in data[arm][metric]
        if value is not None
    ]
    y_min = -1.02 if min(finite_values) < -0.02 else 0.0
    missing_baselines = [
        y_min + 0.055 + index * 0.045 for index in range(len(METRICS))
    ]

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.15))
    has_missing = False
    for arm, _ in ARMS:
        arm_style = ARM_STYLE[arm]
        for metric_index, (
            metric,
            metric_label,
            _,
            _,
            dark_color,
            marker,
        ) in enumerate(METRICS):
            color = dark_color if arm_style["filled"] else LIGHT_COLORS[metric]
            points = data[arm][metric]
            ax.plot(
                [n for n, _ in points],
                [value for _, value in points],
                color=color,
                marker=marker,
                linestyle=arm_style["linestyle"],
                markersize=4.0,
                markerfacecolor=color if arm_style["filled"] else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                linewidth=1.5,
            )
            missing_n = [n for n, value in points if value is None]
            if missing_n:
                has_missing = True
                jittered_n = [
                    n * (2.0 ** arm_style["x_jitter"]) for n in missing_n
                ]
                ax.scatter(
                    jittered_n,
                    [missing_baselines[metric_index]] * len(missing_n),
                    color=color,
                    marker="x",
                    s=18,
                    linewidths=1.1,
                    zorder=4,
                )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_sizes, [str(value) for value in sample_sizes])
    ax.set_ylim(y_min, 1.02)
    if y_min < 0:
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    else:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, which="major", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_xlabel("$n$ por classe")
    ax.set_ylabel("valor da métrica")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    metric_handles = [
        Line2D(
            [],
            [],
            color=dark_color,
            marker=marker,
            markersize=4.4,
            linewidth=1.6,
            label=metric_label,
        )
        for _, metric_label, _, _, dark_color, marker in METRICS
    ]
    arm_handles = [
        Line2D(
            [],
            [],
            color=INK,
            marker="o",
            linestyle="--",
            dashes=(3.4, 1.8),
            markersize=4.4,
            markerfacecolor="white",
            markeredgecolor=INK,
            linewidth=2.0,
            label="Gaussiana única",
        ),
        Line2D(
            [],
            [],
            color=INK,
            marker="o",
            linestyle="-",
            markersize=4.4,
            markerfacecolor=INK,
            markeredgecolor=INK,
            linewidth=2.0,
            label="GMM",
        ),
    ]
    if has_missing:
        arm_handles.append(
            Line2D(
                [],
                [],
                color=GRAY,
                marker="x",
                markersize=4.4,
                linewidth=0,
                label="indisponível",
            )
        )
    metric_legend = fig.legend(
        handles=metric_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(metric_handles),
        frameon=False,
        fontsize=7.4,
        columnspacing=1.1,
        handletextpad=0.4,
        title="métrica",
        title_fontsize=7.4,
    )
    fig.add_artist(metric_legend)
    fig.legend(
        handles=arm_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.088),
        ncol=len(arm_handles),
        frameon=False,
        fontsize=6.8,
        columnspacing=1.3,
        handletextpad=0.5,
        handlelength=4.2,
        title="arma sintética",
        title_fontsize=6.8,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.97,
        bottom=0.34,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={
            "Title": spec.title,
            "Author": "CoInfoSim",
            "Subject": "Full-scale predictive-cooperation-profile metrics versus n",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> None:
    root = repository_root()
    output_dir = root / "coinfosim-presentation/figures"
    for spec in SPECS:
        sample_sizes, data = load_metric_series(
            root / spec.scenario_path, spec.classifier
        )
        validate_final_values(spec, data)
        output = output_dir / spec.output_name
        make_figure(spec, sample_sizes, data, output)
        print(f"Generated {output}")


if __name__ == "__main__":
    main()
