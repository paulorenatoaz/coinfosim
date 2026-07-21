#!/usr/bin/env python3
"""Baixa as figuras originais selecionadas dos relatórios publicados do CoInfoSim.

O script é opcional. O PDF de prévia compila sem as figuras externas, exibindo
marcadores editoriais. Execute em uma máquina com acesso à internet antes da
versão final e copie/ajuste os nomes no LaTeX conforme necessário.
"""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://paulorenatoaz.github.io/coinfosim/reports/scenarios"
TARGET = Path(__file__).resolve().parents[1] / "figures" / "published"

FILES = {
    # Geometrias
    "occupancy_viz_2d_real.png": f"{BASE}/000002_occupancy_baseline_full/viz_2d_real_full_000002.png",
    "occupancy_viz_2d_gaussian.png": f"{BASE}/000002_occupancy_baseline_full/viz_2d_gaussian_full_000002.png",
    "occupancy_viz_2d_gmm.png": f"{BASE}/000002_occupancy_baseline_full/viz_2d_gmm_full_000002.png",
    "air_viz_2d_real.png": f"{BASE}/000005_air_quality_baseline_full/viz_2d_real_full_000005.png",
    "air_viz_2d_gaussian.png": f"{BASE}/000005_air_quality_baseline_full/viz_2d_gaussian_full_000005.png",
    "air_viz_2d_gmm.png": f"{BASE}/000005_air_quality_baseline_full/viz_2d_gmm_full_000005.png",
    "support2_viz_2d_real.png": f"{BASE}/000008_support2_baseline_full/viz_2d_real_full_000008.png",
    "support2_viz_2d_gaussian.png": f"{BASE}/000008_support2_baseline_full/viz_2d_gaussian_full_000008.png",
    "support2_viz_2d_gmm.png": f"{BASE}/000008_support2_baseline_full/viz_2d_gmm_full_000008.png",
    # Matrizes de vencedores solicitadas
    "occupancy_winner_svm_real_n2.png": f"{BASE}/000002_occupancy_baseline_full/graph_structural_winner_linear_svm_real_to_real_n2_full_000002.png",
    "occupancy_winner_svm_real_n512.png": f"{BASE}/000002_occupancy_baseline_full/graph_structural_winner_linear_svm_real_to_real_n512_full_000002.png",
    # Curva publicada de similaridade de N* do Random Forest
    "support2_rf_nstar_similarity.png": f"{BASE}/000008_support2_baseline_full/graph_structural_metric_random_forest_nstar_similarity_full_000008.png",
}


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name, url in FILES.items():
        output = TARGET / name
        try:
            print(f"Downloading {url}")
            urlretrieve(url, output)
        except Exception as exc:  # noqa: BLE001 - CLI audit output
            failures.append(f"{name}: {exc}")
    if failures:
        raise SystemExit("\n".join(["Some downloads failed:", *failures]))
    print(f"Downloaded {len(FILES)} figures to {TARGET}")


if __name__ == "__main__":
    main()
