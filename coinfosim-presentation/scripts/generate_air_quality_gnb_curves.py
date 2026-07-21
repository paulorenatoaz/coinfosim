#!/usr/bin/env python3
"""Gera a figura vetorial das curvas cooperativas do Air Quality.

A figura reproduz a selecao usada nos diagnosticos de N-star do relatorio
academico para Gaussian Naive Bayes:

1. no braco Real -> Real, em n=512, ordenar os subconjuntos de cada
   cardinalidade por (perda media, erro-padrao, tupla do subconjunto);
2. usar o melhor subconjunto singular como referencia;
3. manter os melhores subconjuntos de 2 a 5 canais e o segundo melhor
   subconjunto singular exibido no diagnostico original;
4. congelar essa selecao e tracar as perdas persistidas dos tres bracos para
   n em {2, 4, ..., 512}.

Entradas: os ``full_loss_table_*.csv`` das simulacoes 000015--000017.
Saida: PDF vetorial pronto para inclusao na apresentacao Beamer.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

SAMPLE_SIZES = (2, 4, 8, 16, 32, 64, 128, 256, 512)
CLASSIFIER = "gaussian_nb"

ARM_SPECS = (
    (
        "real_to_real",
        "Real → Real",
        Path(
            "output/reports/simulations/000015_air_quality_real_data_full/"
            "full_loss_table_real_to_real.csv"
        ),
    ),
    (
        "single_gaussian_to_real",
        "Gaussiana única → Real",
        Path(
            "output/reports/simulations/"
            "000016_air_quality_single_gaussian_to_real_full/"
            "full_loss_table_single_gaussian_to_real.csv"
        ),
    ),
    (
        "gmm_to_real",
        "GMM → Real",
        Path(
            "output/reports/simulations/000017_air_quality_gmm_to_real_full/"
            "full_loss_table_gmm_to_real.csv"
        ),
    ),
)

# Selecao registrada na figura do relatorio. A verificacao explicita impede
# que uma alteracao acidental dos artefatos mude silenciosamente a evidencia.
EXPECTED_BEST_BY_SIZE = {
    1: (2,),
    2: (2, 4),
    3: (2, 3, 4),
    4: (1, 2, 4, 5),
    5: (1, 2, 3, 4, 5),
}
EXPECTED_SECOND_SINGLE = (4,)


@dataclass(frozen=True)
class LossPoint:
    """Uma celula da tabela longa de perdas."""

    arm: str
    n_per_class: int
    subset: tuple[int, ...]
    mean_loss: float
    se_loss: float


@dataclass(frozen=True)
class CurveSpec:
    """Identidade visual de uma curva selecionada."""

    subset: tuple[int, ...]
    label: str
    color: str
    linewidth: float
    linestyle: str = "-"
    marker: str = "o"
    zorder: int = 3


def repository_root() -> Path:
    """Retorna a raiz do repositorio a partir da localizacao deste script."""

    return Path(__file__).resolve().parents[2]


def parse_subset(label: str) -> tuple[int, ...]:
    """Converte ``{X2, X4}`` na tupla ``(2, 4)``."""

    values = tuple(int(value) for value in re.findall(r"X(\d+)", label))
    if not values:
        raise ValueError(f"Rotulo de subconjunto invalido: {label!r}")
    return values


def load_arm(
    path: Path, expected_arm: str
) -> dict[tuple[tuple[int, ...], int], LossPoint]:
    """Carrega e valida as trajetorias de Gaussian Naive Bayes de um braco."""

    required = {
        "arm",
        "n_per_class",
        "classifier",
        "subset_size",
        "x_label",
        "mean_loss",
        "se_loss",
    }
    points: dict[tuple[tuple[int, ...], int], LossPoint] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")

        for row in reader:
            if row["classifier"] != CLASSIFIER:
                continue
            if row["arm"] != expected_arm:
                raise ValueError(
                    f"Braco inesperado em {path}: {row['arm']!r}; "
                    f"esperado {expected_arm!r}"
                )
            subset = parse_subset(row["x_label"])
            if len(subset) != int(row["subset_size"]):
                raise ValueError(f"Cardinalidade inconsistente em {path}: {row}")
            point = LossPoint(
                arm=expected_arm,
                n_per_class=int(row["n_per_class"]),
                subset=subset,
                mean_loss=float(row["mean_loss"]),
                se_loss=float(row["se_loss"]),
            )
            if not math.isfinite(point.mean_loss) or not math.isfinite(point.se_loss):
                raise ValueError(f"Perda nao finita em {path}: {row}")
            key = (subset, point.n_per_class)
            if key in points:
                raise ValueError(f"Celula duplicada em {path}: {key}")
            points[key] = point

    observed_sizes = {point.n_per_class for point in points.values()}
    if observed_sizes != set(SAMPLE_SIZES):
        raise ValueError(
            f"Grade amostral inesperada em {path}: {sorted(observed_sizes)}"
        )
    for n in SAMPLE_SIZES:
        count = sum(point.n_per_class == n for point in points.values())
        if count != 31:
            raise ValueError(f"Esperados 31 subconjuntos em n={n}, obtidos {count}")
    return points


def select_report_subsets(
    reference_points: dict[tuple[tuple[int, ...], int], LossPoint],
) -> tuple[dict[int, tuple[int, ...]], tuple[int, ...]]:
    """Aplica a mesma ordenacao do relatorio no braco Real -> Real."""

    at_nmax = [point for point in reference_points.values() if point.n_per_class == 512]
    ranked: dict[int, list[LossPoint]] = {}
    for point in at_nmax:
        ranked.setdefault(len(point.subset), []).append(point)
    for points in ranked.values():
        points.sort(key=lambda point: (point.mean_loss, point.se_loss, point.subset))

    best = {size: ranked[size][0].subset for size in sorted(ranked)}
    second_single = ranked[1][1].subset
    if best != EXPECTED_BEST_BY_SIZE or second_single != EXPECTED_SECOND_SINGLE:
        raise RuntimeError(
            "A selecao derivada diverge da figura auditada do relatorio: "
            f"melhores={best}, segundo singular={second_single}"
        )
    return best, second_single


def subset_math(subset: Iterable[int]) -> str:
    """Formata uma tupla como conjunto matematico com subscritos."""

    items = ", ".join(f"X_{{{index}}}" for index in subset)
    return rf"$\{{{items}\}}$"


def curve_specs(
    best: dict[int, tuple[int, ...]], second_single: tuple[int, ...]
) -> list[CurveSpec]:
    """Define curvas e cores coerentes com a identidade da apresentacao."""

    return [
        CurveSpec(
            best[1],
            f"Referência, 1 canal: {subset_math(best[1])}",
            "#12375F",
            3.2,
            marker="o",
            zorder=6,
        ),
        CurveSpec(
            second_single,
            f"2º melhor, 1 canal: {subset_math(second_single)}",
            "#747474",
            1.8,
            linestyle="--",
            marker="D",
            zorder=2,
        ),
        CurveSpec(
            best[2],
            f"Melhor, 2 canais: {subset_math(best[2])}",
            "#D46A1F",
            2.1,
        ),
        CurveSpec(
            best[3],
            f"Melhor, 3 canais: {subset_math(best[3])}",
            "#2E7D5B",
            2.1,
        ),
        CurveSpec(
            best[4],
            f"Melhor, 4 canais: {subset_math(best[4])}",
            "#B43C3C",
            2.1,
        ),
        CurveSpec(
            best[5],
            f"Melhor, 5 canais: {subset_math(best[5])}",
            "#6B5CA5",
            2.1,
        ),
    ]


def trajectory(
    points: dict[tuple[tuple[int, ...], int], LossPoint],
    subset: tuple[int, ...],
) -> list[float]:
    """Retorna as perdas do subconjunto na grade completa e ordenada."""

    missing = [n for n in SAMPLE_SIZES if (subset, n) not in points]
    if missing:
        raise ValueError(f"Trajetoria incompleta para {subset}: faltam {missing}")
    return [points[(subset, n)].mean_loss for n in SAMPLE_SIZES]


def make_figure(
    arm_points: dict[str, dict[tuple[tuple[int, ...], int], LossPoint]],
    specs: list[CurveSpec],
    output_path: Path,
) -> None:
    """Renderiza e salva o painel triplo em PDF vetorial."""

    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "text.color": "#12375F",
            "axes.labelcolor": "#12375F",
            "axes.titlecolor": "#12375F",
            "xtick.color": "#12375F",
            "ytick.color": "#12375F",
            "axes.edgecolor": "#5F5F5F",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.65), sharex=True, sharey=True)
    arm_labels = {arm_id: label for arm_id, label, _ in ARM_SPECS}

    for ax, (arm_id, _, _) in zip(axes, ARM_SPECS, strict=True):
        points = arm_points[arm_id]
        for spec in specs:
            ax.plot(
                SAMPLE_SIZES,
                trajectory(points, spec.subset),
                label=spec.label,
                color=spec.color,
                linewidth=spec.linewidth,
                linestyle=spec.linestyle,
                marker=spec.marker,
                markersize=5.1 if spec.zorder < 6 else 6.2,
                markeredgewidth=0.7,
                markeredgecolor="white",
                alpha=0.98,
                zorder=spec.zorder,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(SAMPLE_SIZES, [str(n) for n in SAMPLE_SIZES])
        ax.tick_params(axis="x", rotation=35, pad=2)
        ax.grid(True, which="major", color="#B8C6D6", alpha=0.38, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(arm_labels[arm_id], fontweight="bold", pad=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    all_values = [
        value
        for arm_id, _, _ in ARM_SPECS
        for spec in specs
        for value in trajectory(arm_points[arm_id], spec.subset)
    ]
    lower = max(0.0, min(all_values) - 0.015)
    upper = max(all_values) + 0.015
    axes[0].set_ylim(lower, upper)
    axes[0].yaxis.set_major_locator(MaxNLocator(nbins=7))
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.2f}".replace(".", ","))
    )
    axes[0].set_ylabel("Perda 0–1 no teste real", fontweight="bold")

    fig.suptitle(
        "UCI Air Quality + Gaussian Naive Bayes: curvas cooperativas",
        fontsize=19,
        fontweight="bold",
        color="#12375F",
        y=0.988,
    )
    fig.text(
        0.5,
        0.918,
        "Seleção no braço Real → Real em n = 512; "
        "todos os painéis usam o mesmo teste real fixo",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#5F5F5F",
    )
    fig.supxlabel(
        "Amostras de treinamento por classe, n",
        x=0.52,
        y=0.205,
        fontsize=12.5,
        fontweight="bold",
        color="#12375F",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        fontsize=10.2,
        frameon=True,
        fancybox=False,
        columnspacing=1.9,
        handlelength=3.0,
        handletextpad=0.7,
        borderpad=0.65,
    )
    legend.get_frame().set_facecolor("#EDF2F8")
    legend.get_frame().set_edgecolor("#B8C6D6")
    legend.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(left=0.065, right=0.995, top=0.845, bottom=0.285, wspace=0.08)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format="pdf",
        metadata={
            "Title": "Air Quality + Gaussian Naive Bayes: curvas cooperativas",
            "Author": "CoInfoSim",
            "Subject": (
                "Curvas de perda dos subconjuntos selecionados no braco "
                "Real para Real em n=512"
            ),
            # Omitir datas torna o PDF byte a byte reprodutivel entre execucoes.
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = repository_root()
    default_output = (
        root / "coinfosim-presentation/figures/air_quality_gnb_curvas_pt.pdf"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"PDF de saida (padrao: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repository_root()
    arm_points = {
        arm_id: load_arm(root / relative_path, arm_id)
        for arm_id, _, relative_path in ARM_SPECS
    }
    best, second_single = select_report_subsets(arm_points["real_to_real"])
    specs = curve_specs(best, second_single)
    make_figure(arm_points, specs, args.output.resolve())

    print("Selecao auditada no braco Real -> Real em n=512:")
    print(f"  referencia (1 canal): {best[1]}")
    print(f"  segundo melhor (1 canal): {second_single}")
    for size in range(2, 6):
        print(f"  melhor ({size} canais): {best[size]}")
    print(f"PDF gerado: {args.output.resolve()}")


if __name__ == "__main__":
    main()
