"""Per-run semantic manifest: a compact, provenance-ready summary of one
regenerated scenario's predictive-cooperation-profile output.

Uses only already-persisted metadata (run IDs, dataset/split/preprocessing
metadata, classifier/training-condition identifiers, result-data paths and
hashes, and commit SHAs). Never fabricates a claim that isn't evidenced by
persisted metadata, and never persists absolute or machine-specific paths.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from coinfosim.semantics.vocabulary import canonical_key_to_id, vocabulary_version

CONTEXT_RESOURCE_PATH = "coinfosim/resources/coinfosim-context.jsonld"

CANONICAL_METRIC_KEYS = (
    "ranking_fidelity_series",
    "winner_agreement_series",
    "reversal_existence_agreement",
    "mean_log2_reversal_distance",
    "reversal_sample_size_similarity",
)

FIXED_REAL_TEST_SET_STATEMENT = (
    "All training conditions (real, single-Gaussian synthetic, GMM synthetic) "
    "are evaluated on the same fixed real evaluation split; the predictive "
    "cooperation profile compares training conditions, never test data."
)


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_repo_relative(path: str | Path, repo_root: str | Path) -> str:
    """Normalize a path to a POSIX-style, repo-relative string.

    Never returns an absolute path or a path outside ``repo_root``'s naming;
    if ``path`` cannot be made relative to ``repo_root`` it is returned as a
    plain string with any drive/user-specific prefix left to the caller to
    avoid (callers must not pass absolute machine-specific paths here).
    """

    root = Path(repo_root).resolve()
    candidate = Path(path)
    try:
        candidate = candidate.resolve()
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return Path(path).as_posix()


def build_semantic_manifest(
    *,
    scenario_run_id: int,
    scenario_slug: str,
    dataset_id: str,
    classifier_ids: Sequence[str],
    training_condition_ids: Sequence[str],
    sample_sizes: Sequence[int],
    source_simulation_run_ids: Sequence[int],
    source_result_data: Sequence[Mapping[str, str]],
    code_commit_sha: Optional[str],
    recovered_source_commit_sha: Optional[str] = None,
    original_simulation_commit_sha: Optional[str] = None,
    report_artifact_hashes: Optional[Mapping[str, str]] = None,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical semantic manifest for one regenerated scenario.

    ``source_result_data`` items must already be repo-relative paths with a
    precomputed ``sha256`` (see :func:`sha256_of_file` /
    :func:`to_repo_relative`). ``generated_at`` may appear as metadata but is
    never used as artifact identity.
    """

    manifest: Dict[str, Any] = {
        "semantic_vocabulary_version": vocabulary_version(),
        "jsonld_context_path": CONTEXT_RESOURCE_PATH,
        "semantic_type": canonical_key_to_id("predictive_cooperation_profile"),
        "canonical_metric_ids": sorted(
            {
                canonical_key_to_id(key)
                for key in CANONICAL_METRIC_KEYS
            }
        ),
        "scenario_run_id": f"{int(scenario_run_id):06d}",
        "scenario_slug": scenario_slug,
        "dataset_id": dataset_id,
        "classifier_ids": sorted(str(c) for c in classifier_ids),
        "training_condition_ids": list(training_condition_ids),
        "sample_sizes": [int(n) for n in sample_sizes],
        "fixed_real_test_set_statement": FIXED_REAL_TEST_SET_STATEMENT,
        "source_simulation_run_ids": sorted(int(i) for i in source_simulation_run_ids),
        "source_result_data": [
            {"path": str(item["path"]), "sha256": str(item["sha256"])}
            for item in sorted(source_result_data, key=lambda item: str(item["path"]))
        ],
        "code_commit_sha": code_commit_sha,
        "recovered_source_commit_sha": recovered_source_commit_sha,
        "original_simulation_commit_sha": original_simulation_commit_sha,
        "report_artifact_hashes": dict(sorted((report_artifact_hashes or {}).items())),
    }
    if generated_at is not None:
        manifest["generated_at"] = generated_at
    return manifest


def write_semantic_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
