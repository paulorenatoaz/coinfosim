"""Generic post-processing for structural cooperation metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr

from coinfosim.simulation.monte_carlo import SimulationResult

Subset = Tuple[int, ...]


@dataclass(frozen=True)
class StructuralCompatibility:
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


def validate_structural_compatibility(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> StructuralCompatibility:
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
    return StructuralCompatibility(reference_arm, sample_sizes, classifiers, subsets)


def loss_vector(
    result: SimulationResult,
    classifier: str,
    n_per_class: int,
    subsets: Sequence[Subset],
) -> np.ndarray:
    """Return mean losses in the supplied canonical subset order."""

    return np.asarray(
        [result.accumulator.mean_loss(n_per_class, subset, classifier) for subset in subsets],
        dtype=float,
    )


def rank_vector(losses: Sequence[float]) -> np.ndarray:
    """Return average ranks, preserving exact loss ties."""

    return np.asarray(rankdata(np.asarray(losses, dtype=float), method="average"))


def ranking_fidelity_series(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> List[Dict[str, object]]:
    """Compute all-arm Spearman ranking fidelity over the common grid."""

    context = validate_structural_compatibility(arm_results, reference_arm)
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
    """Return the directed lower-loss winner matrix for one result cell."""

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    losses = loss_vector(result, classifier, n_per_class, catalog)
    matrix: List[List[Optional[int]]] = []
    for i, row_loss in enumerate(losses):
        row: List[Optional[int]] = []
        for j, col_loss in enumerate(losses):
            if i == j:
                row.append(None)
            elif row_loss < col_loss:
                row.append(1)
            elif row_loss > col_loss:
                row.append(-1)
            else:
                row.append(0)
        matrix.append(row)
    return matrix


def winner_agreement_series(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> List[Dict[str, object]]:
    """Compute carry-forward effective-winner unordered pairwise agreement."""

    context = validate_structural_compatibility(arm_results, reference_arm)
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


def effective_winner_matrices(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, object]]:
    """Return the carry-forward effective winner matrix `W` at each sample size.

    Exact ties after a pair's first strict winner preserve the previous
    winner instead of resetting to unresolved (Section 2.3 tie propagation).
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


def winner_reversal_events(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, int]]:
    """Extract unordered valid pairwise winner reversals (Section 2.4).

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
    always `None`. The first prefix has no defined cells.
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


def directed_crossing_events(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, int]]:
    """Extract observed-grid row-becomes-better events for ordered pairs."""

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    sample_sizes = tuple(int(n) for n in result.sample_sizes)
    losses = {
        n: loss_vector(result, classifier, n, catalog) for n in sample_sizes
    }
    events: List[Dict[str, int]] = []
    for row in range(len(catalog)):
        for col in range(len(catalog)):
            if row == col:
                continue
            for previous_n, current_n in zip(sample_sizes, sample_sizes[1:]):
                previous_delta = losses[previous_n][row] - losses[previous_n][col]
                current_delta = losses[current_n][row] - losses[current_n][col]
                if previous_delta >= 0 and current_delta < 0:
                    events.append(
                        {"row": row, "col": col, "n_crossing": current_n}
                    )
    return events


def progressive_directed_nstar(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, object]]:
    """Reconstruct latest same-direction crossing matrices from sparse events."""

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    sample_sizes = tuple(int(n) for n in result.sample_sizes)
    events = directed_crossing_events(result, classifier, catalog)
    latest: Dict[Tuple[int, int], int] = {}
    by_n: Dict[int, List[Dict[str, int]]] = {}
    for event in events:
        by_n.setdefault(event["n_crossing"], []).append(event)
    progressive: List[Dict[str, object]] = []
    for n_prefix in sample_sizes[1:]:
        for event in by_n.get(n_prefix, []):
            latest[(event["row"], event["col"])] = event["n_crossing"]
        matrix: List[List[Optional[int]]] = []
        for row in range(len(catalog)):
            matrix.append(
                [
                    None if row == col else latest.get((row, col))
                    for col in range(len(catalog))
                ]
            )
        progressive.append({"n_prefix": n_prefix, "matrix": matrix})
    return progressive


def _reversal_cells(matrix: Sequence[Sequence[Optional[int]]]) -> set[Tuple[int, int]]:
    return {
        (row, col)
        for row, values in enumerate(matrix)
        for col, value in enumerate(values)
        if value is not None
    }


def progressive_reversal_fidelity(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
) -> List[Dict[str, object]]:
    """Compute reversal existence agreement and reversal sample-size similarity.

    The two metrics (Section 2.6) answer separate questions and are never
    combined into a composite/product metric (Section 2.7).
    """

    context = validate_structural_compatibility(arm_results, reference_arm)
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
                    "n_reference_reversal_pairs": 0,
                    "n_arm_reversal_pairs": 0,
                    "n_shared_reversal_pairs": 0,
                    "n_union_reversal_pairs": 0,
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
                        "n_reference_reversal_pairs": len(reference_cells),
                        "n_arm_reversal_pairs": len(arm_cells),
                        "n_shared_reversal_pairs": len(shared),
                        "n_union_reversal_pairs": len(union),
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


def simulation_structural_dynamics(
    result: SimulationResult,
) -> Dict[str, object]:
    """Serialize compact all-subset winner outcomes and sparse crossings."""

    subsets = tuple(tuple(int(i) for i in subset) for subset in result.subsets)
    classifiers: Dict[str, object] = {}
    for classifier in result.classifier_names:
        effective_winner_pairs_by_n: Dict[str, List[Dict[str, int]]] = {}
        for item in effective_winner_matrices(result, classifier, subsets):
            matrix = item["matrix"]
            effective_winner_pairs_by_n[str(int(item["n_per_class"]))] = [
                {"i": i, "j": j, "outcome": int(matrix[i][j])}
                for i in range(len(subsets))
                for j in range(i + 1, len(subsets))
            ]
        classifiers[str(classifier)] = {
            "effective_winner_pairs_by_n": effective_winner_pairs_by_n,
            "winner_reversal_events": winner_reversal_events(
                result, classifier, subsets
            ),
        }
    return {
        "schema_version": 2,
        "subset_catalog": [list(subset) for subset in subsets],
        "sample_sizes": [int(n) for n in result.sample_sizes],
        "classifiers": classifiers,
    }


def scenario_structural_fidelity(
    arm_results: Mapping[str, SimulationResult],
    reference_arm: str,
    arm_labels: Mapping[str, str],
) -> Dict[str, object]:
    """Return Block-2 report-ready structural fidelity data."""

    context = validate_structural_compatibility(arm_results, reference_arm)
    missing_labels = set(arm_results) - set(arm_labels)
    if missing_labels:
        raise ValueError(f"missing arm labels for {sorted(missing_labels)}")
    ranking = ranking_fidelity_series(arm_results, reference_arm)
    agreement = winner_agreement_series(arm_results, reference_arm)
    reversal = progressive_reversal_fidelity(arm_results, reference_arm)
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
                    "n_reference_reversal_pairs": reversal_row[
                        "n_reference_reversal_pairs"
                    ],
                    "n_arm_reversal_pairs": reversal_row["n_arm_reversal_pairs"],
                    "n_shared_reversal_pairs": reversal_row[
                        "n_shared_reversal_pairs"
                    ],
                    "n_union_reversal_pairs": reversal_row[
                        "n_union_reversal_pairs"
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
        "schema_version": 2,
        "reference_arm": reference_arm,
        "arm_labels": dict(arm_labels),
        "sample_sizes": list(context.sample_sizes),
        "subset_catalog": [list(subset) for subset in context.subsets],
        "ranking_fidelity_series": ranking,
        "winner_agreement_series": agreement,
        "reversal_fidelity_series": reversal,
        "final_summary": summary,
        "reference_display_subsets_by_classifier": {
            classifier: [list(subset) for subset in subsets]
            for classifier, subsets in display.items()
        },
    }
