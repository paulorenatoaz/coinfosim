"""Canonical predictive-cooperation-profile analysis.

This module implements the approved mathematical contract for the
predictive-cooperation-profile framework: the effective pairwise winner
relation `W` with exact-tie carry-forward propagation, the unordered
upper-triangular winner-reversal matrix `R` (defined only after a pair's
first valid winner reversal), ranking fidelity, Winner Agreement, and the two
separate reversal-agreement indicators (reversal existence agreement and
reversal sample-size similarity, derived from the unnormalized mean log2
reversal distance). No composite/product metric is computed anywhere in this
module.

See ``docs/semantics/predictive_cooperation_vocabulary.md`` for the
terminology ledger and ``docs/tasks/predictive_profile_semantic_refactor_plan.md``
Section 5 for the full mathematical contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr

from coinfosim.semantics.vocabulary import canonical_key_to_id, vocabulary_version
from coinfosim.simulation.monte_carlo import SimulationResult

Subset = Tuple[int, ...]

SCHEMA_VERSION = 3


@dataclass(frozen=True)
class PredictiveProfileCompatibility:
    """Canonical coordinates shared by a validated arm mapping."""

    reference_arm: str
    sample_sizes: Tuple[int, ...]
    classifiers: Tuple[str, ...]
    subsets: Tuple[Subset, ...]


def _validated_grid(arm: str, values: Sequence[int]) -> Tuple[int, ...]:
    grid = tuple(values)
    if not grid:
        raise ValueError(f"arm {arm!r} has an empty sample-size grid")
    if any(isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0 for n in grid):
        raise ValueError(f"arm {arm!r} sample sizes must be positive integers")
    normalized = tuple(int(n) for n in grid)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"arm {arm!r} has duplicate sample sizes")
    if any(right <= left for left, right in zip(normalized, normalized[1:])):
        raise ValueError(f"arm {arm!r} sample-size grid is not strictly increasing")
    return normalized


def validate_predictive_profile_compatibility(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> PredictiveProfileCompatibility:
    """Validate cross-arm coordinates and every required mean-loss cell."""

    if not arm_results:
        raise ValueError("arm_results must not be empty")
    if reference_arm not in arm_results:
        raise ValueError(f"reference arm {reference_arm!r} is not present")

    reference = arm_results[reference_arm]
    sample_sizes = _validated_grid(reference_arm, reference.sample_sizes)
    classifiers = tuple(str(c) for c in reference.classifier_names)
    subsets = tuple(tuple(int(i) for i in subset) for subset in reference.subsets)
    if not classifiers or len(classifiers) != len(set(classifiers)):
        raise ValueError(f"arm {reference_arm!r} has invalid classifier identities")
    if not subsets or any(not subset for subset in subsets):
        raise ValueError(f"arm {reference_arm!r} must contain non-empty subsets")
    if len(subsets) != len(set(subsets)):
        raise ValueError(f"arm {reference_arm!r} has duplicate subsets")

    classifier_set = set(classifiers)
    subset_set = set(subsets)
    for arm, result in arm_results.items():
        grid = _validated_grid(arm, result.sample_sizes)
        if grid != sample_sizes:
            raise ValueError(f"arm {arm!r} sample-size grid differs from reference")
        arm_classifiers = tuple(str(c) for c in result.classifier_names)
        if len(arm_classifiers) != len(set(arm_classifiers)):
            raise ValueError(f"arm {arm!r} has duplicate classifier identities")
        if set(arm_classifiers) != classifier_set:
            raise ValueError(f"arm {arm!r} classifier identities differ from reference")
        arm_subsets = tuple(tuple(int(i) for i in subset) for subset in result.subsets)
        if len(arm_subsets) != len(set(arm_subsets)):
            raise ValueError(f"arm {arm!r} has duplicate subsets")
        if set(arm_subsets) != subset_set:
            raise ValueError(f"arm {arm!r} subset identities differ from reference")

        count = getattr(result.accumulator, "count", None)
        for classifier in classifiers:
            for n in sample_sizes:
                for subset in subsets:
                    if callable(count) and count(n, subset, classifier) == 0:
                        raise ValueError(
                            f"arm {arm!r} is missing loss cell "
                            f"(n={n}, subset={subset}, classifier={classifier!r})"
                        )
                    value = float(result.accumulator.mean_loss(n, subset, classifier))
                    if not math.isfinite(value):
                        raise ValueError(
                            f"arm {arm!r} has non-finite mean loss at "
                            f"(n={n}, subset={subset}, classifier={classifier!r})"
                        )
    return PredictiveProfileCompatibility(reference_arm, sample_sizes, classifiers, subsets)


def loss_vector(
    result: SimulationResult,
    classifier: str,
    n_per_class: int,
    subsets: Sequence[Subset],
) -> np.ndarray:
    """Return estimated mean test losses in the supplied canonical subset order."""

    return np.asarray(
        [result.accumulator.mean_loss(n_per_class, subset, classifier) for subset in subsets],
        dtype=float,
    )


def rank_vector(losses: Sequence[float]) -> np.ndarray:
    """Return average ranks, preserving exact loss ties."""

    return np.asarray(rankdata(np.asarray(losses, dtype=float), method="average"))


def observed_pairwise_outcome(loss_i: float, loss_j: float) -> int:
    """Return the raw observed pairwise outcome ``U_ij`` for one comparison.

    ``+1`` if ``loss_i < loss_j`` (``i`` wins), ``-1`` if ``loss_i > loss_j``
    (``j`` wins), ``0`` on an exact tie. This is the un-carried instantaneous
    comparison; see :func:`effective_winner_matrices` for the tie-propagated
    effective winner relation ``W``.
    """

    if loss_i < loss_j:
        return 1
    if loss_i > loss_j:
        return -1
    return 0


def ranking_fidelity_series(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> List[Dict[str, object]]:
    """Compute all-arm Spearman ranking fidelity over the common grid."""

    context = validate_predictive_profile_compatibility(arm_results, reference_arm)
    reference = arm_results[reference_arm]
    rows: List[Dict[str, object]] = []
    for classifier in context.classifiers:
        for arm, result in arm_results.items():
            for n in context.sample_sizes:
                ref_ranks = rank_vector(loss_vector(reference, classifier, n, context.subsets))
                arm_ranks = rank_vector(loss_vector(result, classifier, n, context.subsets))
                constant = bool(np.ptp(ref_ranks) == 0 or np.ptp(arm_ranks) == 0)
                if arm == reference_arm:
                    value: Optional[float] = 1.0
                    status = "self"
                elif constant:
                    value = None
                    status = "constant_ranking"
                else:
                    correlation = float(spearmanr(ref_ranks, arm_ranks).statistic)
                    value = correlation if math.isfinite(correlation) else None
                    status = "ok" if value is not None else "constant_ranking"
                rows.append(
                    {
                        "classifier": classifier,
                        "arm": arm,
                        "n_per_class": n,
                        "rho_rank": value,
                        "n_subsets": len(context.subsets),
                        "status": status,
                        "constant_ranking": constant,
                    }
                )
    return rows


def winner_matrix(
    result: SimulationResult,
    classifier: str,
    n_per_class: int,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[List[Optional[int]]]:
    """Return the instantaneous observed pairwise-outcome matrix ``U(n)``."""

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    losses = loss_vector(result, classifier, n_per_class, catalog)
    matrix: List[List[Optional[int]]] = []
    for i, row_loss in enumerate(losses):
        row: List[Optional[int]] = []
        for j, col_loss in enumerate(losses):
            if i == j:
                row.append(None)
            else:
                row.append(observed_pairwise_outcome(float(row_loss), float(col_loss)))
        matrix.append(row)
    return matrix


def effective_winner_matrices(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, object]]:
    """Return the carry-forward effective pairwise winner relation `W` at each sample size.

    Exact ties after a pair's first strict winner preserve the previous
    winner instead of resetting to unresolved (tie-propagation rule).
    """

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    sample_sizes = tuple(int(n) for n in result.sample_sizes)
    n_subsets = len(catalog)
    state: Dict[Tuple[int, int], int] = {}
    rows: List[Dict[str, object]] = []
    for n in sample_sizes:
        observed = winner_matrix(result, classifier, n, catalog)
        matrix: List[List[Optional[int]]] = [[None] * n_subsets for _ in range(n_subsets)]
        for i in range(n_subsets):
            for j in range(i + 1, n_subsets):
                outcome = observed[i][j]
                if outcome != 0:
                    state[(i, j)] = outcome
                effective = state.get((i, j), 0)
                matrix[i][j] = effective
                matrix[j][i] = -effective if effective != 0 else 0
        rows.append({"n_per_class": int(n), "matrix": matrix})
    return rows


def winner_agreement_series(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> List[Dict[str, object]]:
    """Compute carry-forward effective-winner unordered pairwise Winner Agreement."""

    context = validate_predictive_profile_compatibility(arm_results, reference_arm)
    total = len(context.subsets) * (len(context.subsets) - 1) // 2
    rows: List[Dict[str, object]] = []
    for classifier in context.classifiers:
        effective = {
            arm: {
                int(item["n_per_class"]): item["matrix"]
                for item in effective_winner_matrices(result, classifier, context.subsets)
            }
            for arm, result in arm_results.items()
        }
        for arm, result in arm_results.items():
            for n in context.sample_sizes:
                ref_matrix = effective[reference_arm][n]
                arm_matrix = effective[arm][n]
                valid = matching = 0
                for i in range(len(context.subsets)):
                    for j in range(i + 1, len(context.subsets)):
                        ref_outcome = ref_matrix[i][j]
                        arm_outcome = arm_matrix[i][j]
                        if ref_outcome == 0 or arm_outcome == 0:
                            continue
                        valid += 1
                        matching += ref_outcome == arm_outcome
                if valid == 0:
                    value: Optional[float] = None
                    status = "no_valid_pairs"
                else:
                    value = 1.0 if arm == reference_arm else matching / valid
                    status = "self" if arm == reference_arm else "ok"
                rows.append(
                    {
                        "classifier": classifier,
                        "arm": arm,
                        "n_per_class": n,
                        "winner_agreement": value,
                        "n_pairs_total": total,
                        "n_pairs_valid": valid,
                        "n_pairs_matching": matching,
                        "n_pairs_skipped_tie": total - valid,
                        "status": status,
                    }
                )
    return rows


def winner_reversal_events(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, int]]:
    """Extract unordered valid pairwise winner-reversal events.

    A reversal at `n_k` requires a defined effective winner at both
    `n_{k-1}` and `n_k` that differs between the two sample sizes.
    """

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    n_subsets = len(catalog)
    effective = effective_winner_matrices(result, classifier, catalog)
    events: List[Dict[str, int]] = []
    for previous_item, current_item in zip(effective, effective[1:]):
        previous_matrix = previous_item["matrix"]
        current_matrix = current_item["matrix"]
        n_reversal = int(current_item["n_per_class"])
        for i in range(n_subsets):
            for j in range(i + 1, n_subsets):
                previous_value = previous_matrix[i][j]
                current_value = current_matrix[i][j]
                if previous_value != 0 and current_value != 0 and previous_value != current_value:
                    events.append({"i": i, "j": j, "n_reversal": n_reversal})
    return events


def progressive_reversal_matrices(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, object]]:
    """Reconstruct the progressive triangular reversal matrix `R` per prefix.

    Each defined upper-triangle cell stores the last observed reversal
    sample size through that prefix; the diagonal and lower triangle are
    always `None`. The first prefix has no defined cells, because no
    reversal can yet have occurred.
    """

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    sample_sizes = tuple(int(n) for n in result.sample_sizes)
    n_subsets = len(catalog)
    events = winner_reversal_events(result, classifier, catalog)
    by_n: Dict[int, List[Dict[str, int]]] = {}
    for event in events:
        by_n.setdefault(event["n_reversal"], []).append(event)
    latest: Dict[Tuple[int, int], int] = {}
    rows: List[Dict[str, object]] = []
    for n_prefix in sample_sizes:
        for event in by_n.get(n_prefix, []):
            latest[(event["i"], event["j"])] = event["n_reversal"]
        matrix: List[List[Optional[int]]] = []
        for i in range(n_subsets):
            matrix.append(
                [
                    latest.get((i, j)) if i < j else None
                    for j in range(n_subsets)
                ]
            )
        rows.append({"n_prefix": int(n_prefix), "matrix": matrix})
    return rows


def _reversal_cells(matrix: Sequence[Sequence[Optional[int]]]) -> set:
    return {
        (row, col)
        for row, values in enumerate(matrix)
        for col, value in enumerate(values)
        if value is not None
    }


def progressive_reversal_agreement(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> List[Dict[str, object]]:
    """Compute reversal existence agreement and reversal sample-size similarity.

    These two metrics answer separate questions and are never combined into
    a composite/product metric.
    """

    context = validate_predictive_profile_compatibility(arm_results, reference_arm)
    n1 = context.sample_sizes[0]
    rows: List[Dict[str, object]] = []
    for classifier in context.classifiers:
        progressions = {
            arm: {
                int(item["n_prefix"]): item["matrix"]
                for item in progressive_reversal_matrices(
                    result, classifier, context.subsets
                )
            }
            for arm, result in arm_results.items()
        }
        for arm in arm_results:
            rows.append(
                {
                    "classifier": classifier,
                    "arm": arm,
                    "n_prefix": n1,
                    "reference_reversal_pair_count": 0,
                    "arm_reversal_pair_count": 0,
                    "shared_reversal_pair_count": 0,
                    "union_reversal_pair_count": 0,
                    "reversal_existence_agreement": None,
                    "mean_log2_reversal_distance": None,
                    "reversal_sample_size_similarity": None,
                    "status": "unavailable_first_prefix",
                }
            )
            for n_prefix in context.sample_sizes[1:]:
                reference_matrix = progressions[reference_arm][n_prefix]
                arm_matrix = progressions[arm][n_prefix]
                reference_cells = _reversal_cells(reference_matrix)
                arm_cells = _reversal_cells(arm_matrix)
                shared = reference_cells & arm_cells
                union = reference_cells | arm_cells
                if not union:
                    agreement = 1.0
                    distance = None
                    similarity = None
                    status = "no_reversals_in_either"
                elif not shared:
                    agreement = 0.0
                    distance = None
                    similarity = None
                    status = "no_shared_reversals"
                else:
                    agreement = len(shared) / len(union)
                    distances = [
                        abs(
                            math.log2(float(arm_matrix[i][j]))
                            - math.log2(float(reference_matrix[i][j]))
                        )
                        for i, j in shared
                    ]
                    distance = sum(distances) / len(distances)
                    prefix_span = math.log2(n_prefix) - math.log2(n1)
                    similarity = min(1.0, max(0.0, 1.0 - distance / prefix_span))
                    status = "ok"
                rows.append(
                    {
                        "classifier": classifier,
                        "arm": arm,
                        "n_prefix": n_prefix,
                        "reference_reversal_pair_count": len(reference_cells),
                        "arm_reversal_pair_count": len(arm_cells),
                        "shared_reversal_pair_count": len(shared),
                        "union_reversal_pair_count": len(union),
                        "reversal_existence_agreement": agreement,
                        "mean_log2_reversal_distance": distance,
                        "reversal_sample_size_similarity": similarity,
                        "status": status,
                    }
                )
    return rows


def select_display_subsets_by_cardinality(
    result: SimulationResult,
    classifiers: Optional[Sequence[str]] = None,
) -> Dict[str, List[Subset]]:
    """Select one lexicographically tie-broken Nmax winner per cardinality."""

    n_max = max(result.sample_sizes)
    selected: Dict[str, List[Subset]] = {}
    for classifier in classifiers or result.classifier_names:
        by_cardinality: Dict[int, List[Subset]] = {}
        for subset in result.subsets:
            by_cardinality.setdefault(len(subset), []).append(tuple(subset))
        selected[classifier] = [
            min(
                sorted(candidates),
                key=lambda subset: result.accumulator.mean_loss(
                    n_max, subset, classifier
                ),
            )
            for _, candidates in sorted(by_cardinality.items())
        ]
    return selected


def simulation_pairwise_profile_dynamics(
    result: SimulationResult,
) -> Dict[str, object]:
    """Serialize the canonical per-simulation pairwise profile dynamics payload.

    This is the exact content of the persisted ``pairwise_profile_dynamics``
    report key (schema version 3).
    """

    subsets = tuple(tuple(int(i) for i in subset) for subset in result.subsets)
    classifiers: Dict[str, object] = {}
    for classifier in result.classifier_names:
        effective_winner_relations_by_n: List[Dict[str, object]] = []
        for item in effective_winner_matrices(result, classifier, subsets):
            matrix = item["matrix"]
            effective_winner_relations_by_n.append(
                {
                    "n_per_class": int(item["n_per_class"]),
                    "relations": [
                        {"i": i, "j": j, "outcome": int(matrix[i][j])}
                        for i in range(len(subsets))
                        for j in range(i + 1, len(subsets))
                    ],
                }
            )
        classifiers[str(classifier)] = {
            "effective_winner_relations_by_n": effective_winner_relations_by_n,
            "winner_reversal_events": winner_reversal_events(
                result, classifier, subsets
            ),
            "reversal_matrices_by_prefix": progressive_reversal_matrices(
                result, classifier, subsets
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "semantic_vocabulary_version": vocabulary_version(),
        "semantic_type": canonical_key_to_id("pairwise_profile_dynamics"),
        "subset_catalog": [list(subset) for subset in subsets],
        "sample_sizes": [int(n) for n in result.sample_sizes],
        "classifiers": classifiers,
    }


def scenario_predictive_profile_agreement(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
    arm_labels: Mapping[str, str],
) -> Dict[str, object]:
    """Return the canonical persisted ``predictive_cooperation_profile`` payload.

    This is the exact content of the persisted ``predictive_cooperation_profile``
    report key (schema version 3).
    """

    context = validate_predictive_profile_compatibility(arm_results, reference_arm)
    missing_labels = set(arm_results) - set(arm_labels)
    if missing_labels:
        raise ValueError(f"missing arm labels for {sorted(missing_labels)}")
    ranking = ranking_fidelity_series(arm_results, reference_arm)
    agreement = winner_agreement_series(arm_results, reference_arm)
    reversal = progressive_reversal_agreement(arm_results, reference_arm)
    rank_index = {
        (row["classifier"], row["arm"], row["n_per_class"]): row for row in ranking
    }
    agreement_index = {
        (row["classifier"], row["arm"], row["n_per_class"]): row
        for row in agreement
    }
    reversal_index = {
        (row["classifier"], row["arm"], row["n_prefix"]): row for row in reversal
    }
    n_max = context.sample_sizes[-1]
    summary = []
    for classifier in context.classifiers:
        for arm in arm_results:
            rank_row = rank_index[(classifier, arm, n_max)]
            winner_row = agreement_index[(classifier, arm, n_max)]
            reversal_row = reversal_index[(classifier, arm, n_max)]
            summary.append(
                {
                    "classifier": classifier,
                    "arm": arm,
                    "rho_rank": rank_row["rho_rank"],
                    "ranking_status": rank_row["status"],
                    "winner_agreement": winner_row["winner_agreement"],
                    "n_pairs_total": winner_row["n_pairs_total"],
                    "n_pairs_valid": winner_row["n_pairs_valid"],
                    "n_pairs_matching": winner_row["n_pairs_matching"],
                    "n_pairs_skipped_tie": winner_row["n_pairs_skipped_tie"],
                    "winner_status": winner_row["status"],
                    "reversal_existence_agreement": reversal_row[
                        "reversal_existence_agreement"
                    ],
                    "reference_reversal_pair_count": reversal_row[
                        "reference_reversal_pair_count"
                    ],
                    "arm_reversal_pair_count": reversal_row["arm_reversal_pair_count"],
                    "shared_reversal_pair_count": reversal_row[
                        "shared_reversal_pair_count"
                    ],
                    "union_reversal_pair_count": reversal_row[
                        "union_reversal_pair_count"
                    ],
                    "mean_log2_reversal_distance": reversal_row[
                        "mean_log2_reversal_distance"
                    ],
                    "reversal_sample_size_similarity": reversal_row[
                        "reversal_sample_size_similarity"
                    ],
                    "reversal_status": reversal_row["status"],
                }
            )
    display = select_display_subsets_by_cardinality(
        arm_results[reference_arm], context.classifiers
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "semantic_vocabulary_version": vocabulary_version(),
        "semantic_type": canonical_key_to_id("predictive_cooperation_profile"),
        "reference_arm": reference_arm,
        "arm_labels": dict(arm_labels),
        "sample_sizes": list(context.sample_sizes),
        "subset_catalog": [list(subset) for subset in context.subsets],
        "ranking_fidelity_series": ranking,
        "winner_agreement_series": agreement,
        "reversal_agreement_series": reversal,
        "final_summary": summary,
        "reference_display_subsets_by_classifier": {
            classifier: [list(subset) for subset in subsets]
            for classifier, subsets in display.items()
        },
    }
