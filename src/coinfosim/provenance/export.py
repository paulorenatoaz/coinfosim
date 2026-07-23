"""Serialization of the canonical ``ProvDocument`` to on-disk artifact formats.

All formats (PROV-JSON, PROV-N, PROV-O/Turtle, and -- when Graphviz is
available -- PNG/PDF) are derived from the same ``prov.model.ProvDocument``;
no format implements independent graph logic.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prov.model import ProvDocument

logger = logging.getLogger(__name__)

_DOT_RENDER_KWARGS = dict(
    direction="LR",
    use_labels=True,
    show_element_attributes=False,
    show_relation_attributes=False,
    show_nary=False,
)


@dataclass(frozen=True)
class ProvenanceArtifactSet:
    """Repository-relative paths to the artifacts derived from one document."""

    provjson: Path
    provn: Path
    ttl: Path
    png: Optional[Path]
    pdf: Optional[Path]
    graphviz_available: bool


def _write_provjson(document: ProvDocument, path: Path) -> None:
    raw = document.serialize(format="json")
    payload = json.loads(raw)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_provn(document: ProvDocument, path: Path) -> None:
    text = document.serialize(format="provn")
    path.write_text(text, encoding="utf-8")


def _write_turtle(document: ProvDocument, path: Path) -> None:
    text = document.serialize(format="rdf", rdf_format="turtle")
    path.write_text(text, encoding="utf-8")


def _render_graphviz(document: ProvDocument, png_path: Path, pdf_path: Path) -> None:
    from prov.dot import prov_to_dot

    dot = prov_to_dot(document, **_DOT_RENDER_KWARGS)
    dot.write_png(str(png_path))
    dot.write_pdf(str(pdf_path))


def export_provenance_artifacts(
    document: ProvDocument,
    output_dir: Path,
    *,
    stem: str = "provenance",
) -> ProvenanceArtifactSet:
    """Export every canonical provenance artifact format from one document.

    Machine-readable formats (PROV-JSON, PROV-N, PROV-O/Turtle) are always
    written. PNG/PDF rendering additionally requires the external Graphviz
    ``dot`` executable; when it is unavailable, a warning is logged and the
    scenario run is not marked as failed.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provjson_path = output_dir / f"{stem}.provjson"
    provn_path = output_dir / f"{stem}.provn"
    ttl_path = output_dir / f"{stem}.ttl"

    _write_provjson(document, provjson_path)
    _write_provn(document, provn_path)
    _write_turtle(document, ttl_path)

    graphviz_available = shutil.which("dot") is not None
    png_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    if graphviz_available:
        png_candidate = output_dir / f"{stem}.png"
        pdf_candidate = output_dir / f"{stem}.pdf"
        _render_graphviz(document, png_candidate, pdf_candidate)
        png_path = png_candidate
        pdf_path = pdf_candidate
    else:
        logger.warning(
            "Graphviz 'dot' executable not found; skipping PNG/PDF provenance "
            "rendering for %s (machine-readable formats were still written).",
            stem,
        )

    return ProvenanceArtifactSet(
        provjson=provjson_path,
        provn=provn_path,
        ttl=ttl_path,
        png=png_path,
        pdf=pdf_path,
        graphviz_available=graphviz_available,
    )
