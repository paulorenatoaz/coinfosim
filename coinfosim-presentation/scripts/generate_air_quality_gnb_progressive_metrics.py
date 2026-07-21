#!/usr/bin/env python3
"""Gera a figura das metricas estruturais progressivas do Air Quality (GNB).

A figura usa exclusivamente as series persistidas em ``structural_fidelity``
do cenario UCI Air Quality full-scale 000006 (grade n = 2, ..., 1024 por
classe). Nenhuma curva e redesenhada ou aproximada manualmente: os valores
sao lidos do ``scenario.json`` gravado pela execucao.

Layout: dois paineis, um por braco sintetico (Gaussiana unica -> Real e
GMM -> Real). Cada painel exibe as tres metricas progressivas em funcao do
prefixo crescente da grade amostral:

1. correlacao de ranking (Spearman entre as ordenacoes por perda media);
2. Winners Agreement (proporcao de decisoes par a par preservadas);
3. similaridade progressiva de N-star (existencia e posicao dos ultimos
   cruzamentos dirigidos).

Saida: PDF vetorial pronto para inclusao na apresentacao Beamer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCENARIO_JSON = Path(
    "output/reports/scenarios/000006_air_quality_baseline_full-scale/scenario.json"
)
CLASSIFIER = "gaussian_nb"
ARMS = (
    ("single_gaussian_to_real", "Gaussiana única → Real"),
    ("gmm_to_real", "GMM → Real"),
)

# Identidade visual da apresentacao (paleta validada para daltonismo).
INK = "#3B3E58"
GRAY = "#6C6980"
GRID = "#D9D7E4"
COLOR_RHO = "#3F68B0"    # correlacao de ranking
COLOR_WA = "#B98136"     # Winners Agreement
COLOR_NS = "#964F9C"     # similaridade progressiva de N-star

# Valores auditados do resumo final (n = 1024): impedem que uma alteracao
# acidental dos artefatos mude silenciosamente a evidencia exibida.
EXPECTED_FINAL = {
    "single_gaussian_to_real": (1.0, 1.0, 0.7319062181447503),
    "gmm_to_real": (1.0, 1.0, 0.8095238095238095),
}


def repository_root() -> Path:
    """Retorna a raiz do repositorio a partir da localizacao deste script."""

    return Path(__file__).resolve().parents[2]


def load_series(scenario_path: Path) -> tuple[list[int], dict[str, dict[str, list]]]:
    """Le as tres series progressivas do GNB para os dois bracos sinteticos."""

    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    fidelity = payload["report_data"]["structural_fidelity"]
    grid = [int(n) for n in fidelity["sample_sizes"]]

    def pick(series: list[dict], arm: str, n_key: str, value_key: str) -> dict[int, float | None]:
        cells = {
            int(entry[n_key]): entry[value_key]
            for entry in series
            if entry["classifier"] == CLASSIFIER and entry["arm"] == arm
        }
        missing = [n for n in grid if n not in cells]
        if missing:
            raise ValueError(f"Serie incompleta para {arm}: faltam n={missing}")
        return cells

    data: dict[str, dict[str, list]] = {}
    for arm, _ in ARMS:
        rho = pick(fidelity["ranking_fidelity_series"], arm, "n_per_class", "rho_rank")
        wa = pick(fidelity["winner_agreement_series"], arm, "n_per_class", "winner_agreement")
        ns = pick(fidelity["nstar_similarity_series"], arm, "n_prefix", "nstar_similarity")
        for name, cells in (("rho", rho), ("wa", wa)):
            bad = [n for n, v in cells.items() if v is None or not math.isfinite(v)]
            if bad:
                raise ValueError(f"Valores indisponiveis de {name} em {arm}: n={bad}")
        # O primeiro prefixo nao pode conter cruzamento e e persistido como nulo.
        if ns[grid[0]] is not None:
            raise ValueError(f"Prefixo inicial de N-star deveria ser indisponivel em {arm}")
        final = (rho[grid[-1]], wa[grid[-1]], ns[grid[-1]])
        expected = EXPECTED_FINAL[arm]
        if any(abs(a - b) > 1e-12 for a, b in zip(final, expected, strict=True)):
            raise RuntimeError(
                f"Valores finais divergem do resumo auditado em {arm}: {final} != {expected}"
            )
        data[arm] = {
            "rho": [rho[n] for n in grid],
            "wa": [wa[n] for n in grid],
            "ns": [ns[n] for n in grid],
        }
    return grid, data


def make_figure(grid: list[int], data: dict[str, dict[str, list]], output: Path) -> None:
    """Renderiza os dois paineis e salva o PDF vetorial."""

    plt.rcParams.update(
        {
            "font.family": "Lato",
            "font.size": 8.6,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.2,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
            "axes.edgecolor": GRAY,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.12), sharex=True, sharey=True)

    series_style = (
        ("rho", "Correlação de ranking", COLOR_RHO, "-", "o", 4.6, 5),
        ("wa", "Winners Agreement", COLOR_WA, (0, (4.2, 1.6)), "D", 3.8, 6),
        ("ns", "Similaridade progressiva de $N^{\\star}$", COLOR_NS, "-", "s", 4.0, 4),
    )

    for ax, (arm, title) in zip(axes, ARMS, strict=True):
        for key, _, color, linestyle, marker, msize, zorder in series_style:
            values = data[arm][key]
            xs = [n for n, v in zip(grid, values, strict=True) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                marker=marker,
                markersize=msize,
                markeredgewidth=0.7,
                markeredgecolor="white",
                zorder=zorder,
                clip_on=False,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(grid, [str(n) for n in grid])
        ax.tick_params(axis="x", pad=2)
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0,25", "0,50", "0,75", "1"])
        ax.grid(True, which="major", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(title, fontweight="bold", pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0.02)

    axes[0].set_ylabel("valor da métrica")
    fig.supxlabel(
        "prefixo crescente da grade: $n$ por classe",
        x=0.54,
        y=0.163,
        fontsize=9.0,
    )

    handles = [
        Line2D(
            [], [],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            marker=marker,
            markersize=msize,
            markeredgewidth=0.7,
            markeredgecolor="white",
            label=label,
        )
        for _, label, color, linestyle, marker, msize, _ in series_style
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.012),
        ncol=3,
        fontsize=8.0,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.5,
        handletextpad=0.55,
    )

    fig.subplots_adjust(left=0.078, right=0.992, top=0.875, bottom=0.345, wspace=0.09)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="pdf",
        metadata={
            "Title": "Air Quality + Gaussian Naive Bayes: metricas progressivas",
            "Author": "CoInfoSim",
            "Subject": (
                "Series progressivas de correlacao de ranking, Winners Agreement "
                "e similaridade de N-star do cenario full-scale 000006"
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
        root
        / "coinfosim-presentation/figures/air_quality_gnb_metricas_progressivas_000006.pdf"
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
    grid, data = load_series(root / SCENARIO_JSON)
    make_figure(grid, data, args.output.resolve())
    print(f"Grade amostral: {grid}")
    for arm, label in ARMS:
        rho, wa, ns = (data[arm][k][-1] for k in ("rho", "wa", "ns"))
        print(f"{label}: rho={rho:.4f}  WA={wa:.4f}  NS={ns:.4f} em n={grid[-1]}")
    print(f"PDF gerado: {args.output.resolve()}")


if __name__ == "__main__":
    main()
