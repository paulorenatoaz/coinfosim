"""GitHub Pages home-page and report-index generation.

Builds ``index.html`` with, in order: project introduction, install
instructions, quick start, available datasets, published scenario
reports, machine-readable artifacts, and citation/license. All external
text (scenario titles, dataset metadata) is HTML-escaped, and ordering is
deterministic so repeated generation on unchanged input produces byte-
identical output.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import shutil
from pathlib import Path
from typing import Any

PAGES_HOME_URL = "https://paulorenatoaz.github.io/coinfosim/"


def discover_scenarios(reports_root: Path) -> list[dict[str, Any]]:
    """Discover published scenario reports from the scenario-run registry."""

    registry = reports_root / "scenario_runs.json"
    if registry.exists():
        with registry.open(encoding="utf-8") as fh:
            data = json.load(fh)
        items = []
        for run in data.get("runs", []):
            if run.get("status") != "completed":
                continue
            scenario_path = run.get("artifacts", {}).get("scenario_report")
            if not scenario_path:
                scenario_path = _find_scenario_report_for_run(reports_root, run)
            rel_path = _reports_relative_path(reports_root, scenario_path) if scenario_path else None
            if rel_path is None:
                continue
            items.append(
                {
                    "path": rel_path,
                    "title": _scenario_title(run),
                    "dataset": _dataset_name(run),
                    "question": run.get("question", ""),
                    "semantic_manifest_path": _discover_semantic_artifact(
                        reports_root, run, "semantic_manifest_path", "semantic_manifest.json"
                    ),
                    "provenance_path": _discover_semantic_artifact(
                        reports_root, run, "provenance_path", "provenance.jsonld"
                    ),
                    "provenance_artifacts": _discover_provenance_artifacts(
                        reports_root, run
                    ),
                }
            )
        items.sort(key=lambda item: (item["dataset"], item["title"], item["path"].as_posix()))
        return items

    items = []
    if not reports_root.exists():
        return []
    for p in reports_root.rglob("*.html"):
        name = p.name
        if "scenario_report" in name:
            items.append(
                {
                    "path": p.relative_to(reports_root),
                    "title": p.stem.replace("_", " ").title(),
                    "dataset": "Scenario",
                    "question": "",
                }
            )
    items.sort(key=lambda item: item["path"].as_posix())
    return items


def _find_scenario_report_for_run(reports_root: Path, run: dict):
    run_dir = run.get("run_dir")
    if not run_dir:
        return None
    rel_run_dir = _reports_relative_path(reports_root, run_dir)
    if rel_run_dir is None:
        return None
    reports = sorted((reports_root / rel_run_dir).glob("*scenario_report*.html"))
    return reports[0] if reports else None


def _discover_semantic_artifact(
    reports_root: Path, run: dict, registry_field: str, filename: str
) -> Path | None:
    """Locate a per-run semantic/provenance artifact (registry field first, then a sibling file)."""

    value = run.get(registry_field)
    if value:
        rel = _reports_relative_path(reports_root, value)
        if rel is not None and (reports_root / rel).exists():
            return rel
    run_dir = run.get("run_dir")
    if run_dir:
        rel_run_dir = _reports_relative_path(reports_root, run_dir)
        if rel_run_dir is not None and (reports_root / rel_run_dir / filename).exists():
            return rel_run_dir / filename
    return None


_CANONICAL_PROVENANCE_FILENAMES = {
    "provjson": "provenance.provjson",
    "provn": "provenance.provn",
    "ttl": "provenance.ttl",
    "png": "provenance.png",
    "pdf": "provenance.pdf",
}


def _discover_provenance_artifacts(reports_root: Path, run: dict) -> dict[str, Path]:
    """Locate the canonical PROV artifacts for one scenario run.

    Discovery order (Section 13): the registry's own ``provenance_artifacts``
    map first, then sibling canonical files located next to
    ``provenance_path``/the run directory by filename. An empty dict means
    only the historical ``provenance.jsonld`` sibling fallback is available.
    """

    found: dict[str, Path] = {}

    registry_map = run.get("provenance_artifacts") or {}
    for key, value in registry_map.items():
        if not value:
            continue
        rel = _reports_relative_path(reports_root, value)
        if rel is not None and (reports_root / rel).exists():
            found[key] = rel
    if found:
        return found

    base_dir = None
    provenance_path = run.get("provenance_path")
    if provenance_path:
        rel = _reports_relative_path(reports_root, provenance_path)
        if rel is not None:
            base_dir = rel.parent
    if base_dir is None:
        run_dir = run.get("run_dir")
        if run_dir:
            base_dir = _reports_relative_path(reports_root, run_dir)

    if base_dir is not None:
        for key, filename in _CANONICAL_PROVENANCE_FILENAMES.items():
            if (reports_root / base_dir / filename).exists():
                found[key] = base_dir / filename

    return found


def _reports_relative_path(reports_root: Path, path_value):
    path = Path(path_value)
    marker = Path("output") / "reports"
    parts = path.parts
    for index in range(len(parts) - len(marker.parts) + 1):
        if parts[index : index + len(marker.parts)] == marker.parts:
            return Path(*parts[index + len(marker.parts) :])
    try:
        return path.resolve().relative_to(reports_root.resolve())
    except (OSError, ValueError):
        if not path.is_absolute():
            return path
    return None


def _scenario_title(run: dict) -> str:
    name = run.get("scenario_name") or run.get("scenario_slug") or "Scenario"
    mode = run.get("mode")
    return f"{name} ({mode})" if mode else name


def _dataset_name(run: dict) -> str:
    scenario_slug = run.get("scenario_slug", "")
    if scenario_slug.startswith("occupancy"):
        return "Occupancy Detection"
    if scenario_slug.startswith("air_quality"):
        return "UCI Air Quality"
    if scenario_slug.startswith("support2"):
        return "SUPPORT2"
    return run.get("scenario_family", "Scenario").replace("_", " ").title()


def discover_json(data_root: Path) -> list[Path]:
    files = []
    if not data_root.exists():
        return files
    for p in data_root.rglob("*.json"):
        files.append(p.relative_to(data_root))
    files.sort(key=lambda x: x.name)
    return files


def sync_ontology(repo_root: Path, site_dir: Path) -> Path | None:
    """Copy only ``ontology/coinfosim.owl.ttl`` into the Pages worktree, if present.

    Never copies arbitrary repository files -- this single, fixed path is the
    only thing published from the ontology.
    """

    source = repo_root / "ontology" / "coinfosim.owl.ttl"
    if not source.exists():
        return None
    target = site_dir / "ontology" / "coinfosim.owl.ttl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def sync_reports(output_dir: Path, site_dir: Path, *, mirror: bool = False) -> None:
    """Copy ``output_dir/reports`` and ``output_dir/data`` into the worktree.

    By default this merges into any existing ``reports/``/``data/`` content
    already in ``site_dir`` (safe for publishing a partial local output
    without deleting other previously published scenarios). Pass
    ``mirror=True`` to first remove the existing ``reports/``/``data/``
    trees so the published site exactly matches local output -- use this
    only when the local output directory is known to be complete (e.g.
    after regenerating every scenario), since anything published previously
    but absent locally will be removed.
    """

    reports_source = output_dir / "reports"
    data_source = output_dir / "data"
    reports_target = site_dir / "reports"
    data_target = site_dir / "data"
    if mirror:
        if reports_target.exists():
            shutil.rmtree(reports_target)
        if data_target.exists():
            shutil.rmtree(data_target)
    reports_target.mkdir(parents=True, exist_ok=True)
    data_target.mkdir(parents=True, exist_ok=True)
    if reports_source.is_dir():
        shutil.copytree(reports_source, reports_target, dirs_exist_ok=True)
    if data_source.is_dir():
        shutil.copytree(data_source, data_target, dirs_exist_ok=True)


def _dataset_card_html(dataset) -> str:
    files_html = "".join(
        f"<li><a href=\"{html.escape(file.url)}\">{html.escape(file.filename)}</a> "
        f"<span class=\"hash\">sha256:{html.escape(file.sha256)}</span> "
        f"<span class=\"size\">({file.size_bytes:,} bytes)</span></li>"
        for file in dataset.files
    )
    doi_html = (
        f"<p>DOI: <a href=\"https://doi.org/{html.escape(dataset.doi)}\">{html.escape(dataset.doi)}</a></p>"
        if dataset.doi
        else ""
    )
    license_url_html = (
        f' (<a href="{html.escape(dataset.license.url)}">license text</a>)'
        if dataset.license.url
        else ""
    )
    return (
        "<article class=\"dataset-card\">"
        f"<h3>{html.escape(dataset.display_name)}</h3>"
        f"{doi_html}"
        f"<p class=\"license\">License: {html.escape(dataset.license.name)}{license_url_html}</p>"
        f"<p class=\"notice\">{html.escape(dataset.license.notice)}</p>"
        f"<p class=\"source\">Source: <a href=\"{html.escape(dataset.source_url)}\">{html.escape(dataset.source_url)}</a></p>"
        f"<p class=\"citation\">{html.escape(dataset.citation)}</p>"
        f"<ul class=\"files\">{files_html}</ul>"
        "</article>"
    )


def _scenario_card_html(item: dict[str, Any], reports_rel: Path) -> str:
    href = html.escape(str(reports_rel / item["path"]))
    title_text = html.escape(item["title"])
    dataset = html.escape(item["dataset"])
    path_text = html.escape(item["path"].as_posix())
    question = html.escape(item["question"])
    question_html = f"<p>{question}</p>" if question else ""
    semantic_links = []
    semantic_manifest_path = item.get("semantic_manifest_path")
    if semantic_manifest_path:
        semantic_links.append(
            f'<a href="{html.escape(str(reports_rel / semantic_manifest_path))}">semantic manifest</a>'
        )
    provenance_artifacts = item.get("provenance_artifacts") or {}
    for key, label in (
        ("provjson", "PROV-JSON"),
        ("provn", "PROV-N"),
        ("ttl", "PROV-O/Turtle"),
        ("png", "provenance graph (PNG)"),
        ("pdf", "provenance graph (PDF)"),
    ):
        rel = provenance_artifacts.get(key)
        if rel:
            semantic_links.append(
                f'<a href="{html.escape(str(reports_rel / rel))}">{label}</a>'
            )
    if not provenance_artifacts:
        provenance_path = item.get("provenance_path")
        if provenance_path:
            semantic_links.append(
                f'<a href="{html.escape(str(reports_rel / provenance_path))}">provenance (JSON-LD)</a>'
            )
    semantic_html = (
        f"<p class=\"path\">Machine-readable: {' &middot; '.join(semantic_links)}</p>"
        if semantic_links
        else ""
    )
    return (
        "<article class=\"report-card\">"
        f"<h3>{title_text}</h3>"
        f"<p class=\"dataset-label\">Dataset: {dataset}</p>"
        f"{question_html}"
        f"<p class=\"path\">{path_text}</p>"
        f"{semantic_html}"
        f"<p><a href=\"{href}\">Open scenario report</a></p>"
        "</article>"
    )


def write_index(
    site_dir: Path,
    *,
    reports_rel: Path,
    data_rel: Path,
    scenarios: list[dict[str, Any]],
    json_files: list[Path],
    datasets,
    manifest_rel: str = "datasets/manifest.json",
    ontology_rel: str | None = None,
    title: str = "CoInfoSim",
) -> Path:
    """Write the seven-section CoInfoSim Pages home page."""

    site_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    scenario_cards = "\n".join(_scenario_card_html(item, reports_rel) for item in scenarios)
    dataset_cards = "\n".join(_dataset_card_html(dataset) for dataset in datasets)
    json_list = "\n".join(
        f'<li><a href="{html.escape(str(data_rel / p))}">{html.escape(p.name)}</a></li>'
        for p in json_files
    )
    ontology_li = (
        f'<li><a href="{html.escape(ontology_rel)}">ontology/coinfosim.owl.ttl</a> '
        "&mdash; CoInfoSim OWL 2 ontology (specializes PROV-O)</li>"
        if ontology_rel
        else ""
    )

    index_html = f"""<!doctype html>
<html lang=\"en\"><head>
<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{html.escape(title)}</title>
<style>
:root {{ --accent:#1f77b4; --ink:#1a1a1a; --muted:#666; --line:#dcdcdc; --soft:#f7f9fb; --question:#eef5fb; }}
* {{ box-sizing:border-box; }}
body {{ font-family:Georgia, "Times New Roman", serif; color:var(--ink); background:white; max-width:1040px; margin:0 auto; padding:2.4rem 1.6rem 4rem; line-height:1.6; }}
h1 {{ border-bottom:3px solid var(--accent); padding-bottom:.55rem; line-height:1.2; }}
h2 {{ border-bottom:1px solid var(--line); padding-bottom:.35rem; margin-top:2.4rem; }}
h3 {{ line-height:1.3; }}
a {{ color:var(--accent); }}
pre {{ background:var(--soft); border:1px solid var(--line); padding:1rem; overflow-x:auto; }}
.lede {{ font-size:1.08rem; }}
.report-grid, .dataset-grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1rem; }}
.report-card, .dataset-card {{ border:1px solid var(--line); background:var(--soft); padding:1rem 1.15rem; }}
.report-card h3, .dataset-card h3 {{ margin-top:0; }}
.dataset-label, .license, .notice, .source {{ color:var(--muted); }}
.files {{ font-size:.86rem; }}
.hash, .size, .path, .metadata {{ color:var(--muted); font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:.82rem; overflow-wrap:anywhere; }}
footer {{ border-top:1px solid var(--line); margin-top:3rem; padding-top:1rem; color:var(--muted); font-size:.92rem; }}
@media (max-width:700px) {{ body {{ padding:1.4rem 1rem 3rem; }} .report-grid, .dataset-grid {{ grid-template-columns:1fr; }} }}
</style></head><body><div class=\"wrap\">
<h1>{html.escape(title)}</h1>
<p class=\"lede\">CoInfoSim is a research simulator for evaluating predictive cooperation across attribute subsets in supervised classification.</p>

<h2>Install CoInfoSim</h2>
<pre>pip install coinfosim</pre>

<h2>Quick start</h2>
<pre>coinfosim scenario list
coinfosim scenario run occupancy --mode smoke</pre>

<h2>Available datasets</h2>
<p>Files are mirrored byte-for-byte on this site; hashes are pinned in the installed package and verified before every run. See the <a href="{html.escape(manifest_rel)}">machine-readable manifest</a>.</p>
<div class=\"dataset-grid\">{dataset_cards}</div>

<h2>Published scenario reports</h2>
<p><strong>{len(scenarios)}</strong> published scenario reports.</p>
<div class=\"report-grid\">{scenario_cards}</div>

<h2>Machine-readable artifacts</h2>
<ul>
<li><a href="{html.escape(manifest_rel)}">datasets/manifest.json</a> &mdash; dataset provenance, hashes, and URLs</li>
{ontology_li}
{json_list}
</ul>

<h2>Citation and license</h2>
<p>See the project repository for the CoInfoSim citation file and license. Dataset citations and license/acknowledgment status are listed with each dataset above; SUPPORT2 in particular carries no open redistribution license and requires source acknowledgment rather than attribution under an open license.</p>

<p class=\"metadata\">Updated {ts}</p>
<footer>Served from <code>{html.escape(str(reports_rel))}</code>, <code>{html.escape(str(data_rel))}</code>, and <code>datasets/</code> on gh-pages.</footer>
</div></body></html>"""

    (site_dir / "index.html").write_text(index_html, encoding="utf-8")
    return site_dir / "index.html"
