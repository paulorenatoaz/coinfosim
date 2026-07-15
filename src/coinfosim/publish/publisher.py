import argparse
import datetime as dt
import html
import json
import os
import subprocess
from pathlib import Path


def discover_scenarios(reports_root: Path):
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


def _scenario_title(run: dict):
    name = run.get("scenario_name") or run.get("scenario_slug") or "Scenario"
    mode = run.get("mode")
    return f"{name} ({mode})" if mode else name


def _dataset_name(run: dict):
    scenario_slug = run.get("scenario_slug", "")
    if scenario_slug.startswith("occupancy"):
        return "Occupancy Detection"
    if scenario_slug.startswith("air_quality"):
        return "UCI Air Quality"
    if scenario_slug.startswith("support2"):
        return "SUPPORT2"
    return run.get("scenario_family", "Scenario").replace("_", " ").title()


def discover_json(data_root: Path):
    files = []
    if not data_root.exists():
        return files
    for p in data_root.rglob("*.json"):
        files.append(p.relative_to(data_root))
    files.sort(key=lambda x: x.name)
    return files


def write_index(site_dir: Path, reports_rel: Path, data_rel: Path, scenarios, json_files, title: str):
    site_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    def list_html(items):
        return "\n".join(
            _scenario_card(item)
            for item in items
        )

    def _scenario_card(item):
        href = html.escape(str(reports_rel / item["path"]))
        title_text = html.escape(item["title"])
        dataset = html.escape(item["dataset"])
        path_text = html.escape(item["path"].as_posix())
        question = html.escape(item["question"])
        question_html = f"<p>{question}</p>" if question else ""
        return (
            "<article class=\"report-card\">"
            f"<h3>{title_text}</h3>"
            f"<p class=\"dataset-label\">Dataset: {dataset}</p>"
            f"{question_html}"
            f"<p class=\"path\">{path_text}</p>"
            f"<p><a href=\"{href}\">Open scenario report</a></p>"
            "</article>"
        )

    def list_json(items):
        return "\n".join(
            f'<li><a href="{html.escape(str(data_rel / p))}">{html.escape(p.name)}</a></li>'
            for p in items
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
.lede {{ font-size:1.08rem; }}
.question {{ background:var(--question); border-left:5px solid var(--accent); padding:1rem 1.2rem; margin:1.6rem 0; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:1rem; }}
.report-grid {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1rem; }}
.report-card {{ border:1px solid var(--line); background:var(--soft); padding:1rem 1.15rem; }}
.report-card h3 {{ margin-top:0; }}
.dataset-label {{ color:var(--muted); margin-bottom:.25rem; }}
.path, .metadata {{ color:var(--muted); font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:.86rem; overflow-wrap:anywhere; }}
footer {{ border-top:1px solid var(--line); margin-top:3rem; padding-top:1rem; color:var(--muted); font-size:.92rem; }}
@media (max-width:700px) {{ body {{ padding:1.4rem 1rem 3rem; }} .report-grid {{ grid-template-columns:1fr; }} }}
</style></head><body><div class=\"wrap\">
<h1>{html.escape(title)}</h1>
<p class=\"lede\">CoInfoSim is a research simulator for evaluating cooperative advantage among information channels in supervised classification.</p>
<div class=\"question\"><strong>Scientific question:</strong> When does cooperation among information channels improve supervised classification?</div>
<p class=\"metadata\">Updated {ts}</p>
<div class=\"grid\">
<section><h2>Published Scenario Reports</h2><p><strong>{len(scenarios)}</strong> published scenario reports.</p><div class=\"report-grid\">{list_html(scenarios)}</div></section>
<section><h2>Data (JSON)</h2><ul>{list_json(json_files)}</ul></section>
</div>
<footer>Served from <code>{html.escape(str(reports_rel))}</code> and <code>{html.escape(str(data_rel))}</code> on gh-pages.</footer>
</div></body></html>"""

    (site_dir / "index.html").write_text(index_html, encoding="utf-8")


def publish_to_pages(output_dir=None, title="coinfosim Reports"):
    """
    Convenience function to generate index and publish to reports-pages branch.
    
    Args:
        output_dir: Path to output directory (default: ~/coinfosim/output)
        title: Title for the index page
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.expanduser("~"), 'coinfosim', 'output')
        
        site_dir = Path(output_dir).resolve()
        reports_dir = (site_dir / "reports").resolve()
        data_dir = (site_dir / "data").resolve()
        
        # Generate index
        scenarios = discover_scenarios(reports_dir)
        json_files = discover_json(data_dir)
        write_index(site_dir, Path("reports"), Path("data"), scenarios, json_files, title=title)
        print(f"✓ Generated index with {len(scenarios)} scenario reports and {len(json_files)} data files")
        
        # Locate and run publish script
        # Assume we're in package: coinfosim/publish/publisher.py, so repo root is 2 levels up
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "publish_output_to_pages.sh"
        
        if not script.exists():
            print(f"⚠ Publish script not found at {script}")
            print(f"  You can manually run: ./scripts/publish_output_to_pages.sh")
            return False
        
        print(f"✓ Publishing to reports-pages branch...")
        proc = subprocess.run(["bash", str(script)], cwd=str(repo_root), capture_output=True, text=True)
        
        if proc.returncode != 0:
            print(f"✗ Publish failed with code {proc.returncode}")
            if proc.stderr:
                print(proc.stderr)
            return False
        
        print("✓ Successfully published to reports-pages branch")
        return True
        
    except Exception as e:
        print(f"✗ Error during publish: {e}")
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write index.html linking scenario reports and JSON data.")
    ap.add_argument("--reports-dir", default="reports", help="Path to reports root (on gh-pages)")
    ap.add_argument("--data-dir", default="data", help="Path to JSON root (on gh-pages)")
    ap.add_argument("--site-dir", default="output", help="Where to write index.html (recommend: output)")
    ap.add_argument("--title", default="coinfosim Reports", help="Index title")
    ap.add_argument("--publish", action="store_true", help="After generating index, publish output/ to reports-pages via scripts/publish_output_to_pages.sh")
    args = ap.parse_args(argv)

    site_dir = Path(args.site_dir).resolve()
    reports_dir = (site_dir / args.reports_dir).resolve()
    data_dir = (site_dir / args.data_dir).resolve()

    scenarios = discover_scenarios(reports_dir)
    json_files = discover_json(data_dir)
    write_index(site_dir, Path(args.reports_dir), Path(args.data_dir), scenarios, json_files, title=args.title)
    print(f"Wrote {site_dir / 'index.html'}")

    if args.publish:
        # Locate repo root (publisher.py is at coinfosim/publish/publisher.py)
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "publish_output_to_pages.sh"
        if not script.exists():
            raise SystemExit(f"Publish script not found: {script}")
        print(f"Publishing via {script} …")
        # Ensure we run from repo root so the script sees ./output/
        proc = subprocess.run(["bash", str(script)], cwd=str(repo_root))
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        print("Publish completed.")


if __name__ == "__main__":
    main()
