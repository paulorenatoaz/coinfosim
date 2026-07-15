"""Publish CoInfoSim research reports to an artifact-only GitHub Pages branch."""

from __future__ import annotations

import datetime as dt
import html
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlsplit
import uuid


DEFAULT_SITE_TITLE = "CoInfoSim — Published Research Reports"
DEFAULT_COMMIT_MESSAGE = "chore(publish): update CoInfoSim reports"
REPOSITORY_URL = "https://github.com/paulorenatoaz/coinfosim"
_LEGACY_REPORT_RE = re.compile(r"^scenario_.+_report(?:\[test\])?\.html$")
_PUBLISHED_AT_RE = re.compile(
    r'<meta\s+name="coinfosim-published-at"\s+content="([^"]+)"', re.IGNORECASE
)


class PublishError(RuntimeError):
    """Raised when report validation or GitHub Pages publication fails."""


@dataclass(frozen=True)
class PublishResult:
    """Structured outcome of a publication attempt."""

    branch: str
    remote: str
    report_count: int
    changed: bool
    pushed: bool
    commit_sha: Optional[str]
    reports: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _Report:
    path: Path
    title: str
    dataset: str


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._parts.append(data)

    @property
    def title(self) -> str:
        return " ".join("".join(self._parts).split())


class _LocalResourceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attribute = "href" if tag.lower() == "a" else "src" if tag.lower() == "img" else None
        if attribute is None:
            return
        for name, value in attrs:
            if name.lower() == attribute and value is not None:
                self.references.append(value)


def _run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    ok_returncodes: Tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    """Run one checked Git command and turn failures into actionable errors."""

    command = ["git", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise PublishError(
            f"Could not run {' '.join(command)!r} in {cwd}: {exc}"
        ) from exc
    if completed.returncode not in ok_returncodes:
        details = completed.stderr.strip() or completed.stdout.strip() or "no error output"
        raise PublishError(
            f"Git command failed in {cwd}: {' '.join(command)}\n{details}"
        )
    return completed


def discover_scenario_reports(reports_root: Union[Path, str]) -> List[Path]:
    """Return scenario-report paths relative to ``reports_root`` in URL order."""

    root = Path(reports_root)
    if not root.is_dir():
        return []
    reports = []
    for path in root.rglob("*.html"):
        name = path.name
        if (
            name.endswith("_scenario_report.html")
            or "_scenario_report_" in name
            or _LEGACY_REPORT_RE.fullmatch(name)
        ):
            reports.append(path.relative_to(root))
    return sorted(reports, key=lambda path: path.as_posix())


# Backward-compatible import name for callers that used the original helper.
discover_scenarios = discover_scenario_reports


def _extract_title(report_path: Path, relative_path: Path) -> str:
    parser = _TitleParser()
    try:
        parser.feed(report_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError) as exc:
        raise PublishError(f"Could not read scenario report {relative_path.as_posix()}: {exc}") from exc
    if parser.title:
        return parser.title

    stem = relative_path.stem
    stem = re.sub(r"_scenario_report$", "", stem)
    stem = re.sub(r"^scenario_", "Scenario ", stem)
    stem = re.sub(r"_report(?:\[test\])?$", "", stem)
    readable = " ".join(stem.replace("_", " ").replace("-", " ").split()).title()
    return readable or relative_path.as_posix()


def _infer_dataset(title: str, relative_path: Path) -> str:
    evidence = f"{title} {relative_path.as_posix()}".lower().replace("_", " ").replace("-", " ")
    if "air quality" in evidence or "airquality" in evidence or "pt08" in evidence:
        return "UCI Air Quality"
    if "occupancy" in evidence:
        return "Occupancy Detection"

    filename = relative_path.name
    candidate = re.sub(r"_scenario_report\.html$", "", filename)
    if candidate != filename and candidate:
        return " ".join(candidate.replace("_", " ").replace("-", " ").split()).title()
    return "Other research scenarios"


def _collect_reports(reports_root: Path) -> List[_Report]:
    reports = []
    for relative_path in discover_scenario_reports(reports_root):
        title = _extract_title(reports_root / relative_path, relative_path)
        reports.append(
            _Report(
                path=relative_path,
                title=title,
                dataset=_infer_dataset(title, relative_path),
            )
        )
    return sorted(
        reports,
        key=lambda report: (
            report.dataset.casefold(),
            report.title.casefold(),
            report.path.as_posix(),
        ),
    )


def _utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _render_index(
    reports: Sequence[_Report],
    *,
    title: str,
    published_at: str,
    source_sha: str,
) -> str:
    grouped: List[str] = []
    datasets = sorted({report.dataset for report in reports}, key=str.casefold)
    for dataset in datasets:
        cards = []
        for report in (item for item in reports if item.dataset == dataset):
            path = report.path.as_posix()
            url = "reports/" + path
            cards.append(
                "<article class=\"report-card\">"
                f"<h3>{html.escape(report.title)}</h3>"
                f"<p class=\"dataset-label\">Dataset: {html.escape(report.dataset)}</p>"
                f"<p class=\"path\">{html.escape(path)}</p>"
                f"<p><a href=\"{html.escape(url, quote=True)}\">Open scenario report</a></p>"
                "</article>"
            )
        grouped.append(
            f"<section class=\"report-group\"><h3>{html.escape(dataset)}</h3>"
            f"<div class=\"report-grid\">{''.join(cards)}</div></section>"
        )

    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="coinfosim-published-at" content="{html.escape(published_at, quote=True)}">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --accent: #1f77b4;
      --ink: #1a1a1a;
      --muted: #666;
      --line: #dcdcdc;
      --soft: #f7f9fb;
      --question: #eef5fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: white;
      max-width: 1040px;
      margin: 0 auto;
      padding: 2.4rem 1.6rem 4rem;
      line-height: 1.6;
    }}
    h1 {{ border-bottom: 3px solid var(--accent); padding-bottom: .55rem; line-height: 1.2; }}
    h2 {{ border-bottom: 1px solid var(--line); padding-bottom: .35rem; margin-top: 2.4rem; }}
    h3 {{ line-height: 1.3; }}
    a {{ color: var(--accent); }}
    .lede {{ font-size: 1.08rem; }}
    .question {{ background: var(--question); border-left: 5px solid var(--accent); padding: 1rem 1.2rem; margin: 1.6rem 0; }}
    .dataset-grid, .report-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
    .dataset-card, .report-card {{ border: 1px solid var(--line); background: var(--soft); padding: 1rem 1.15rem; }}
    .dataset-card h3, .report-card h3 {{ margin-top: 0; }}
    .report-group > h3 {{ margin-top: 1.6rem; color: #333; }}
    .dataset-label {{ color: var(--muted); margin-bottom: .25rem; }}
    .path, .metadata {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .86rem; overflow-wrap: anywhere; }}
    footer {{ border-top: 1px solid var(--line); margin-top: 3rem; padding-top: 1rem; color: var(--muted); font-size: .92rem; }}
    @media (max-width: 700px) {{
      body {{ padding: 1.4rem 1rem 3rem; }}
      .dataset-grid, .report-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <p class="lede">CoInfoSim is a research simulator for evaluating cooperative advantage among information channels in supervised classification. It studies when channel subsets outperform isolated or simpler subsets and how many labeled samples are needed before those gains emerge.</p>
    <div class="question"><strong>Scientific question:</strong> When does cooperation among information channels improve supervised classification?</div>
  </header>
  <main>
    <section aria-labelledby="protocol-heading">
      <h2 id="protocol-heading">Dataset-anchored protocol</h2>
      <p>Dataset-anchored scenarios compare <strong>Real → Real</strong>, <strong>Single Gaussian → Real</strong>, and <strong>GMM → Real</strong>. All three arms are evaluated on the same fixed real test set.</p>
      <p>The reports examine classification loss, channel-subset rankings, cooperative thresholds N*, ranking fidelity, winner agreement, and progressive N-star similarity.</p>
    </section>
    <section aria-labelledby="datasets-heading">
      <h2 id="datasets-heading">Implemented datasets</h2>
      <div class="dataset-grid">
        <article class="dataset-card">
          <h3>Occupancy Detection</h3>
          <p>A five-channel room-occupancy classification study using Temperature, Humidity, Light, CO₂, and Humidity Ratio.</p>
        </article>
        <article class="dataset-card">
          <h3>UCI Air Quality</h3>
          <p>A five-channel study based on the UCI Air Quality PT08 metal-oxide sensor responses. The C6H6(GT) benzene reference defines the binary target and is excluded from classifier input.</p>
          <p>Dataset DOI: <a href="https://doi.org/10.24432/C59K5F">10.24432/C59K5F</a></p>
        </article>
      </div>
    </section>
    <section aria-labelledby="reports-heading">
      <h2 id="reports-heading">Published scenario reports</h2>
      <p><strong>{len(reports)}</strong> published scenario report{'s' if len(reports) != 1 else ''}.</p>
      {''.join(grouped)}
    </section>
  </main>
  <footer>
    <p class="metadata">Published (UTC): {html.escape(published_at)} · Source commit: {html.escape(source_sha)}</p>
    <p><a href="{REPOSITORY_URL}">CoInfoSim source repository</a>. These reports are computational artifacts of the research project.</p>
    <p>Smoke-mode results represent pipeline validation and preliminary evidence, not final inferential conclusions.</p>
  </footer>
</body>
</html>
"""


def write_index(
    site_dir: Path,
    reports: Sequence[_Report],
    *,
    title: str = DEFAULT_SITE_TITLE,
    published_at: Optional[str] = None,
    source_sha: str = "unknown",
) -> Path:
    """Write the academic project homepage and return its path."""

    site_dir.mkdir(parents=True, exist_ok=True)
    index_path = site_dir / "index.html"
    index_path.write_text(
        _render_index(
            reports,
            title=title,
            published_at=published_at or _utc_timestamp(),
            source_sha=source_sha,
        ),
        encoding="utf-8",
    )
    return index_path


def _validate_local_links(site_root: Path, reports: Sequence[_Report]) -> None:
    root = site_root.resolve()
    for report in reports:
        source = site_root / "reports" / report.path
        parser = _LocalResourceParser()
        try:
            parser.feed(source.read_text(encoding="utf-8", errors="replace"))
        except (OSError, UnicodeError) as exc:
            raise PublishError(f"Could not validate links in {report.path.as_posix()}: {exc}") from exc

        for attribute_value in parser.references:
            value = attribute_value.strip()
            if not value or value.startswith("#") or value.startswith("//"):
                continue
            parsed = urlsplit(value)
            if parsed.scheme.lower() in {"http", "https", "mailto", "javascript", "data"}:
                continue
            referenced_path = unquote(parsed.path)
            if not referenced_path:
                continue
            expected = (source.parent / referenced_path).resolve()
            try:
                expected.relative_to(root)
            except ValueError as exc:
                raise PublishError(
                    "Local resource escapes the published site root: "
                    f"source report={report.path.as_posix()!r}, "
                    f"attribute value={attribute_value!r}, "
                    f"referenced path={referenced_path!r}, expected resolved path={expected}"
                ) from exc
            if not expected.exists():
                raise PublishError(
                    "Broken local resource: "
                    f"source report={report.path.as_posix()!r}, "
                    f"attribute value={attribute_value!r}, "
                    f"referenced path={referenced_path!r}, expected resolved path={expected}"
                )


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _sync_managed_content(
    output_dir: Path,
    site_dir: Path,
    *,
    include_data: bool,
    prune: bool,
) -> None:
    if prune:
        shutil.rmtree(site_dir / "reports", ignore_errors=True)
        shutil.rmtree(site_dir / "data", ignore_errors=True)
    _copy_tree(output_dir / "reports", site_dir / "reports")
    if include_data:
        _copy_tree(output_dir / "data", site_dir / "data")


def _prepare_site(
    output_dir: Path,
    site_dir: Path,
    *,
    title: str,
    include_data: bool,
    prune: bool,
    source_sha: str,
    git_worktree: bool,
) -> List[_Report]:
    _sync_managed_content(output_dir, site_dir, include_data=include_data, prune=prune)
    nojekyll = site_dir / ".nojekyll"
    if not nojekyll.exists() or nojekyll.read_bytes() != b"":
        nojekyll.write_bytes(b"")

    reports = _collect_reports(site_dir / "reports")
    if not reports:
        raise PublishError("No scenario report exists in the final published reports tree")
    _validate_local_links(site_dir, reports)

    index_path = site_dir / "index.html"
    previous = index_path.read_text(encoding="utf-8") if index_path.is_file() else None
    timestamp_match = _PUBLISHED_AT_RE.search(previous or "")
    previous_timestamp = timestamp_match.group(1) if timestamp_match else _utc_timestamp()
    candidate = _render_index(
        reports,
        title=title,
        published_at=previous_timestamp,
        source_sha=source_sha,
    )
    managed_files_changed = bool(
        _run_git(["status", "--porcelain"], cwd=site_dir).stdout.strip()
    ) if git_worktree else True

    if previous != candidate or managed_files_changed:
        rendered = _render_index(
            reports,
            title=title,
            published_at=_utc_timestamp(),
            source_sha=source_sha,
        )
        if rendered != previous:
            index_path.write_text(rendered, encoding="utf-8")
    return reports


def _validate_inputs(output_dir: Union[Path, str], branch: str, remote: str, title: str) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise PublishError(f"Output directory does not exist or is not a directory: {output}")
    if not (output / "reports").is_dir():
        raise PublishError(f"Required reports directory does not exist: {output / 'reports'}")
    source_reports = discover_scenario_reports(output / "reports")
    if not source_reports:
        raise PublishError(f"No scenario reports found under {output / 'reports'}")
    for label, value in (("branch", branch), ("remote", remote), ("site title", title)):
        if not isinstance(value, str) or not value.strip():
            raise PublishError(f"Publication {label} must be a non-empty string")
    return output


def _ensure_git_identity(repository_root: Path) -> None:
    for key, fallback in (
        ("user.name", "CoInfoSim Publisher"),
        ("user.email", "coinfosim-publisher@users.noreply.github.com"),
    ):
        configured = _run_git(
            ["config", "--get", key], cwd=repository_root, ok_returncodes=(0, 1)
        ).stdout.strip()
        if not configured:
            _run_git(["config", "--local", key, fallback], cwd=repository_root)


def publish_to_pages(
    output_dir: Union[Path, str],
    *,
    branch: str = "gh-pages",
    remote: str = "origin",
    site_title: str = DEFAULT_SITE_TITLE,
    include_data: bool = True,
    prune: bool = False,
    dry_run: bool = False,
) -> PublishResult:
    """Validate output and publish it to a GitHub Pages artifact branch."""

    output = _validate_inputs(output_dir, branch, remote, site_title)
    repository_root = Path(
        _run_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    ).resolve()
    source_sha = _run_git(["rev-parse", "--short", "HEAD"], cwd=repository_root).stdout.strip()

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="coinfosim-publish-dry-run-") as temp:
            reports = _prepare_site(
                output,
                Path(temp),
                title=site_title,
                include_data=include_data,
                prune=prune,
                source_sha=source_sha,
                git_worktree=False,
            )
        return PublishResult(
            branch=branch,
            remote=remote,
            report_count=len(reports),
            changed=True,
            pushed=False,
            commit_sha=None,
            reports=tuple(report.path.as_posix() for report in reports),
        )

    _run_git(["fetch", remote, "--prune"], cwd=repository_root)
    remote_listing = _run_git(
        ["branch", "--remotes", "--list", f"{remote}/{branch}"], cwd=repository_root
    ).stdout.strip()
    branch_exists = bool(remote_listing)
    temporary_branch: Optional[str] = None
    worktree_added = False
    primary_error: Optional[BaseException] = None

    with tempfile.TemporaryDirectory(prefix="coinfosim-publish-") as temp:
        site_dir = Path(temp) / "site"
        try:
            if branch_exists:
                _run_git(
                    ["worktree", "add", "--detach", str(site_dir), f"{remote}/{branch}"],
                    cwd=repository_root,
                )
            else:
                _run_git(["worktree", "add", "--detach", str(site_dir), "HEAD"], cwd=repository_root)
            worktree_added = True

            if not branch_exists:
                temporary_branch = f"coinfosim-publish-{uuid.uuid4().hex[:12]}"
                _run_git(["switch", "--orphan", temporary_branch], cwd=site_dir)
                if _run_git(["ls-files"], cwd=site_dir).stdout.strip():
                    _run_git(["rm", "-rf", "."], cwd=site_dir)
                for child in site_dir.iterdir():
                    if child.name == ".git":
                        continue
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()

            reports = _prepare_site(
                output,
                site_dir,
                title=site_title,
                include_data=include_data,
                prune=prune,
                source_sha=source_sha,
                git_worktree=True,
            )
            status = _run_git(["status", "--porcelain"], cwd=site_dir).stdout.strip()
            if not status:
                return PublishResult(
                    branch=branch,
                    remote=remote,
                    report_count=len(reports),
                    changed=False,
                    pushed=False,
                    commit_sha=None,
                    reports=tuple(report.path.as_posix() for report in reports),
                )

            _ensure_git_identity(repository_root)
            _run_git(["add", "--all"], cwd=site_dir)
            _run_git(["commit", "-m", DEFAULT_COMMIT_MESSAGE], cwd=site_dir)
            commit_sha = _run_git(["rev-parse", "HEAD"], cwd=site_dir).stdout.strip()
            _run_git(["push", remote, f"HEAD:{branch}"], cwd=site_dir)
            return PublishResult(
                branch=branch,
                remote=remote,
                report_count=len(reports),
                changed=True,
                pushed=True,
                commit_sha=commit_sha,
                reports=tuple(report.path.as_posix() for report in reports),
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if worktree_added:
                try:
                    _run_git(["worktree", "remove", "--force", str(site_dir)], cwd=repository_root)
                except PublishError:
                    if primary_error is None:
                        raise
            try:
                _run_git(["worktree", "prune"], cwd=repository_root)
                if temporary_branch:
                    _run_git(["branch", "-D", temporary_branch], cwd=repository_root)
            except PublishError:
                if primary_error is None:
                    raise
