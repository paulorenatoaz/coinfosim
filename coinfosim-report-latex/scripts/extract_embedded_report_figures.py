#!/usr/bin/env python3
"""Extrai figuras PNG embutidas (data URI base64) dos relatórios HTML do CoInfoSim.

Os relatórios de simulação embutem as figuras diretamente no HTML como
``data:image/png;base64,...``. Este script localiza a figura desejada pelo
atributo ``alt`` do elemento ``<img>``, decodifica o payload base64, valida a
assinatura PNG e grava o arquivo no destino indicado.

Uso:

    python scripts/extract_embedded_report_figures.py \
        --html output/reports/simulations/000006_occupancy_real_data_full/occupancy_real_data_monte_carlo_report_full_000006.html \
        --alt "N-star graph Linear SVM best 2" \
        --out coinfosim-report-latex/figures/nstar_curves/occupancy/example.png

    # Listar todas as figuras embutidas com seus contextos:
    python scripts/extract_embedded_report_figures.py --html <report.html> --list

O casamento do ``--alt`` é por igualdade exata (padrão) ou por expressão
regular (``--regex``). O script falha ruidosamente quando zero ou mais de uma
figura casa com o critério, para impedir a inserção de uma imagem incorreta.
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# <img ... src='data:image/png;base64,....' ... > com aspas simples ou duplas.
IMG_OR_HEADING = re.compile(
    r"<(h[1-6])[^>]*>(?P<head>.*?)</\1>|"
    r"<img\b(?P<attrs>[^>]*?)src=(?P<q>['\"])data:image/png;base64,(?P<b64>[A-Za-z0-9+/=\s]+?)(?P=q)(?P<attrs2>[^>]*)>",
    re.S | re.I,
)
ALT_RE = re.compile(r"alt=(['\"])(?P<alt>.*?)\1", re.S | re.I)
TAG_STRIP = re.compile(r"<[^>]+>")


@dataclass
class EmbeddedImage:
    index: int
    alt: str
    heading: str
    payload_b64: str

    def decode(self) -> bytes:
        return base64.b64decode(re.sub(r"\s+", "", self.payload_b64))


def scan(html_path: Path) -> list[EmbeddedImage]:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    images: list[EmbeddedImage] = []
    heading = ""
    index = 0
    for match in IMG_OR_HEADING.finditer(html):
        if match.group(1):
            heading = TAG_STRIP.sub("", match.group("head")).strip()
            continue
        attrs = (match.group("attrs") or "") + (match.group("attrs2") or "")
        alt_match = ALT_RE.search(attrs)
        alt = alt_match.group("alt").strip() if alt_match else ""
        images.append(EmbeddedImage(index, alt, heading, match.group("b64")))
        index += 1
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html", required=True, type=Path, help="caminho do relatório HTML")
    parser.add_argument("--alt", help="valor do atributo alt da figura desejada")
    parser.add_argument("--regex", action="store_true", help="interpreta --alt como expressão regular")
    parser.add_argument("--out", type=Path, help="arquivo PNG de destino")
    parser.add_argument("--list", action="store_true", help="lista as figuras embutidas e sai")
    args = parser.parse_args()

    if not args.html.is_file():
        parser.error(f"HTML não encontrado: {args.html}")

    images = scan(args.html)
    if args.list:
        for img in images:
            print(f"[{img.index:3d}] heading={img.heading!r} alt={img.alt!r}")
        print(f"{len(images)} figuras embutidas em {args.html}")
        return 0

    if not args.alt or not args.out:
        parser.error("--alt e --out são obrigatórios fora do modo --list")

    if args.regex:
        pattern = re.compile(args.alt)
        matches = [img for img in images if pattern.search(img.alt)]
    else:
        matches = [img for img in images if img.alt == args.alt]

    if len(matches) == 0:
        sys.stderr.write(
            f"ERRO: nenhuma figura com alt {args.alt!r} em {args.html}.\n"
            "Use --list para inspecionar os valores disponíveis.\n"
        )
        return 1
    if len(matches) > 1:
        sys.stderr.write(f"ERRO: {len(matches)} figuras ambíguas casaram com alt {args.alt!r}:\n")
        for img in matches:
            sys.stderr.write(f"  [{img.index}] heading={img.heading!r} alt={img.alt!r}\n")
        return 1

    img = matches[0]
    data = img.decode()
    if not data.startswith(PNG_SIGNATURE):
        sys.stderr.write("ERRO: payload decodificado não possui assinatura PNG válida.\n")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(
        f"OK: {args.html.name} [{img.index}] heading={img.heading!r} alt={img.alt!r}\n"
        f" -> {args.out} ({len(data)} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
