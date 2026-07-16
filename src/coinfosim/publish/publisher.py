"""GitHub Pages publication orchestration.

``publish_pages`` is the single entry point used by ``coinfosim publish
pages`` and the CI workflow. It always operates through a temporary git
worktree, never force-pushes, never rewrites published history, and never
commits or pushes when nothing changed.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class PublishError(RuntimeError):
    """Raised when a publish step fails (git, verification, or I/O)."""


@dataclass(frozen=True)
class PublishResult:
    """Structured summary of one ``publish_pages`` invocation."""

    output_dir: Path
    branch: str
    remote: str
    changed: bool
    committed: bool
    pushed: bool
    dry_run: bool
    scenario_count: int
    dataset_file_count: int
    changed_paths: tuple[str, ...] = field(default_factory=tuple)


def _run_git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check
    )


def _git_repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise PublishError(
            f"not inside a Git repository: {proc.stderr.strip() or 'git rev-parse failed'}"
        )
    return Path(proc.stdout.strip())


def _current_commit(repo_root: Path) -> Optional[str]:
    proc = _run_git(["rev-parse", "HEAD"], cwd=repo_root, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _branch_exists_locally(repo_root: Path, branch: str) -> bool:
    proc = _run_git(
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=repo_root, check=False
    )
    return proc.returncode == 0


def _branch_exists_on_remote(repo_root: Path, remote: str, branch: str) -> bool:
    proc = _run_git(
        ["ls-remote", "--exit-code", "--heads", remote, branch], cwd=repo_root, check=False
    )
    return proc.returncode == 0


def publish_pages(
    output_dir: Path,
    *,
    branch: str = "gh-pages",
    remote: str = "origin",
    push: bool = False,
    dry_run: bool = False,
    init_branch_if_missing: bool = False,
) -> PublishResult:
    """Regenerate and publish the CoInfoSim Pages site.

    Parameters
    ----------
    output_dir:
        Directory containing ``reports/`` and ``data/`` (typically the
        report output root used by ``coinfosim scenario run``).
    branch:
        Target Pages branch. Defaults to ``gh-pages``.
    remote:
        Git remote to fetch from and, if ``push``, push to.
    push:
        If ``True``, push a new commit to ``remote``/``branch`` when there
        are changes. Never force-pushes.
    dry_run:
        If ``True``, regenerate the site content and report what would
        change, but never commit or push.
    init_branch_if_missing:
        If ``True`` and ``branch`` does not exist locally or on ``remote``,
        create it as a new orphan branch. Otherwise a missing branch is a
        hard error.
    """

    from coinfosim.datasets.catalog import list_datasets
    from coinfosim.publish.datasets import sync_dataset_files, write_dataset_manifest
    from coinfosim.publish.site import discover_json, discover_scenarios, sync_reports, write_index

    repo_root = _git_repo_root()
    output_dir = Path(output_dir)
    _run_git(["fetch", remote, "--prune"], cwd=repo_root, check=False)

    branch_exists = _branch_exists_locally(repo_root, branch) or _branch_exists_on_remote(
        repo_root, remote, branch
    )
    if not branch_exists and not init_branch_if_missing:
        raise PublishError(
            f"branch {branch!r} does not exist locally or on remote {remote!r}; "
            "pass init_branch_if_missing=True to create it"
        )

    with tempfile.TemporaryDirectory(prefix=".coinfosim-pages-", dir=str(repo_root)) as tmp:
        site_dir = Path(tmp) / "worktree"
        try:
            if branch_exists:
                _run_git(["worktree", "add", str(site_dir), branch], cwd=repo_root)
            else:
                _run_git(["worktree", "add", "--detach", str(site_dir)], cwd=repo_root)
                _run_git(["checkout", "--orphan", branch], cwd=site_dir)
                _run_git(["rm", "-rf", "--quiet", "."], cwd=site_dir, check=False)

            sync_reports(output_dir, site_dir)
            copied_files = sync_dataset_files(repo_root, site_dir)
            manifest_path = site_dir / "datasets" / "manifest.json"
            write_dataset_manifest(manifest_path, source_commit=_current_commit(repo_root))

            scenarios = discover_scenarios(site_dir / "reports")
            json_files = discover_json(site_dir / "data")
            write_index(
                site_dir,
                reports_rel=Path("reports"),
                data_rel=Path("data"),
                scenarios=scenarios,
                json_files=json_files,
                datasets=list_datasets(),
                manifest_rel="datasets/manifest.json",
                title="CoInfoSim - Published Research Reports and Datasets",
            )
            (site_dir / ".nojekyll").touch()

            status = _run_git(["status", "--porcelain"], cwd=site_dir)
            changed_paths = tuple(
                line[3:] for line in status.stdout.splitlines() if line.strip()
            )
            changed = bool(changed_paths)

            committed = False
            pushed = False
            if changed and not dry_run:
                _run_git(["add", "-A"], cwd=site_dir)
                _run_git(
                    ["commit", "-m", "chore(publish): update reports and datasets [skip ci]"],
                    cwd=site_dir,
                )
                committed = True
                if push:
                    _run_git(["push", remote, f"HEAD:{branch}"], cwd=site_dir)
                    pushed = True

            return PublishResult(
                output_dir=output_dir,
                branch=branch,
                remote=remote,
                changed=changed,
                committed=committed,
                pushed=pushed,
                dry_run=dry_run,
                scenario_count=len(scenarios),
                dataset_file_count=len(copied_files),
                changed_paths=changed_paths,
            )
        except subprocess.CalledProcessError as exc:
            raise PublishError(
                f"git command failed: {' '.join(exc.cmd)}\n{exc.stderr}"
            ) from exc
        finally:
            _run_git(["worktree", "remove", str(site_dir), "--force"], cwd=repo_root, check=False)


def publish_to_pages(output_dir=None, title: str = "coinfosim Reports") -> bool:
    """Legacy convenience wrapper preserved for backward compatibility.

    Prefer :func:`publish_pages`, which supports ``branch``/``remote``/
    ``push``/``dry_run`` and publishes datasets in addition to reports.
    """

    import os

    try:
        resolved_output_dir = Path(
            output_dir or os.path.join(os.path.expanduser("~"), "coinfosim", "output")
        )
        result = publish_pages(resolved_output_dir, push=True)
        print(
            f"Published {result.scenario_count} scenario report(s) and "
            f"{result.dataset_file_count} dataset file(s) to {result.branch}."
        )
        return True
    except PublishError as exc:
        print(f"Publish failed: {exc}")
        return False


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Publish CoInfoSim reports and datasets to GitHub Pages.")
    ap.add_argument("--output-dir", default="output", help="Directory containing reports/ and data/")
    ap.add_argument("--branch", default="gh-pages")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--init-branch-if-missing", action="store_true")
    args = ap.parse_args(argv)

    result = publish_pages(
        Path(args.output_dir),
        branch=args.branch,
        remote=args.remote,
        push=args.push,
        dry_run=args.dry_run,
        init_branch_if_missing=args.init_branch_if_missing,
    )
    print(
        f"changed={result.changed} committed={result.committed} pushed={result.pushed} "
        f"scenarios={result.scenario_count} dataset_files={result.dataset_file_count}"
    )


if __name__ == "__main__":
    main()
