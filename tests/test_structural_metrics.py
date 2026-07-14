from dataclasses import dataclass
import json

import numpy as np
import pytest

from coinfosim.results.structural import (
    directed_crossing_events,
    progressive_directed_nstar,
    progressive_nstar_similarity,
    rank_vector,
    ranking_fidelity_series,
    select_display_subsets_by_cardinality,
    simulation_structural_dynamics,
    validate_structural_compatibility,
    winner_agreement_series,
    winner_matrix,
)


class FakeAccumulator:
    def __init__(self, values):
        self.values = values

    def count(self, n, subset, classifier):
        return int((n, tuple(subset), classifier) in self.values)

    def mean_loss(self, n, subset, classifier):
        return self.values.get((n, tuple(subset), classifier), float("nan"))


@dataclass
class FakeResult:
    sample_sizes: list
    subsets: list
    classifier_names: list
    accumulator: FakeAccumulator


def make_result(
    losses,
    *,
    grid=(2, 4),
    subsets=((0,), (1,), (0, 1)),
    classifiers=("clf",),
):
    values = {}
    for classifier in classifiers:
        for n in grid:
            row = losses[classifier][n] if isinstance(losses, dict) else losses
            for subset, value in zip(subsets, row):
                values[(n, tuple(subset), classifier)] = float(value)
    return FakeResult(list(grid), list(subsets), list(classifiers), FakeAccumulator(values))


def metric_row(rows, *, arm="arm", n=2, classifier="clf"):
    return next(
        row
        for row in rows
        if row["arm"] == arm
        and row.get("n_per_class", row.get("n_prefix")) == n
        and row["classifier"] == classifier
    )


def test_compatibility_accepts_reordered_subsets_and_classifiers():
    losses = {
        "a": {2: [0.1, 0.2, 0.3], 4: [0.1, 0.2, 0.3]},
        "b": {2: [0.3, 0.2, 0.1], 4: [0.3, 0.2, 0.1]},
    }
    ref = make_result(losses, classifiers=("a", "b"))
    arm = make_result(
        losses,
        subsets=((0, 1), (1,), (0,)),
        classifiers=("b", "a"),
    )
    context = validate_structural_compatibility({"ref": ref, "arm": arm}, "ref")
    assert context.subsets == ((0,), (1,), (0, 1))
    assert context.classifiers == ("a", "b")


@pytest.mark.parametrize("grid", [(2, 8), (2, 2), (4, 2), (0, 2)])
def test_compatibility_rejects_invalid_or_mismatched_grid(grid):
    ref = make_result([0.1, 0.2, 0.3])
    arm = make_result([0.1, 0.2, 0.3], grid=grid)
    with pytest.raises(ValueError, match="sample"):
        validate_structural_compatibility({"ref": ref, "arm": arm}, "ref")


def test_compatibility_rejects_classifier_mismatch():
    ref = make_result([0.1, 0.2, 0.3])
    arm = make_result([0.1, 0.2, 0.3], classifiers=("other",))
    with pytest.raises(ValueError, match="classifier"):
        validate_structural_compatibility({"ref": ref, "arm": arm}, "ref")


@pytest.mark.parametrize(
    "subsets", [((0,), (1,)), ((0,), (1,), (0, 1), (2,))]
)
def test_compatibility_rejects_missing_or_extra_subsets(subsets):
    ref = make_result([0.1, 0.2, 0.3])
    arm = make_result([0.1] * len(subsets), subsets=subsets)
    with pytest.raises(ValueError, match="subset"):
        validate_structural_compatibility({"ref": ref, "arm": arm}, "ref")


def test_compatibility_rejects_missing_and_nonfinite_cells():
    ref = make_result([0.1, 0.2, 0.3])
    missing = make_result([0.1, 0.2, 0.3])
    missing.accumulator.values.pop((2, (0,), "clf"))
    with pytest.raises(ValueError, match="missing loss cell"):
        validate_structural_compatibility({"ref": ref, "arm": missing}, "ref")

    nonfinite = make_result([0.1, np.inf, 0.3])
    with pytest.raises(ValueError, match="non-finite"):
        validate_structural_compatibility({"ref": ref, "arm": nonfinite}, "ref")


def test_ranking_identical_and_reversed_rankings():
    ref = make_result([0.1, 0.2, 0.3])
    same = make_result([0.1, 0.2, 0.3])
    reverse = make_result([0.3, 0.2, 0.1])
    rows = ranking_fidelity_series(
        {"ref": ref, "same": same, "arm": reverse}, "ref"
    )
    assert metric_row(rows, arm="same")["rho_rank"] == pytest.approx(1.0)
    assert metric_row(rows)["rho_rank"] == pytest.approx(-1.0)
    assert metric_row(rows, arm="ref")["rho_rank"] == 1.0
    assert metric_row(rows, arm="ref")["status"] == "self"


def test_ranking_average_ties_and_constant_status():
    assert rank_vector([0.1, 0.1, 0.3]).tolist() == [1.5, 1.5, 3.0]
    ref = make_result([0.1, 0.2, 0.3])
    constant = make_result([0.2, 0.2, 0.2])
    row = metric_row(ranking_fidelity_series({"ref": ref, "arm": constant}, "ref"))
    assert row["rho_rank"] is None
    assert row["status"] == "constant_ranking"
    assert row["n_subsets"] == 3


def test_ranking_display_selection_uses_lexicographic_exact_tie_break():
    result = make_result(
        [0.3, 0.1, 0.2, 0.2],
        subsets=((0,), (1,), (0, 2), (0, 1)),
    )
    assert select_display_subsets_by_cardinality(result)["clf"] == [
        (1,),
        (0, 1),
    ]


def test_winner_matrix_is_directed_antisymmetric_with_exact_ties():
    result = make_result([0.1, 0.1, 0.3])
    matrix = winner_matrix(result, "clf", 2)
    assert [matrix[i][i] for i in range(3)] == [None, None, None]
    assert matrix[0][1] == matrix[1][0] == 0
    assert matrix[0][2] == 1
    assert matrix[2][0] == -1


def test_winner_agreement_counts_unordered_pairs_without_ties():
    ref = make_result([0.1, 0.2, 0.3])
    arm = make_result([0.1, 0.3, 0.2])
    row = metric_row(winner_agreement_series({"ref": ref, "arm": arm}, "ref"))
    assert row["winner_agreement"] == pytest.approx(2 / 3)
    assert row["n_pairs_total"] == 3
    assert row["n_pairs_valid"] == 3
    assert row["n_pairs_matching"] == 2
    assert row["n_pairs_skipped_tie"] == 0


def test_winner_agreement_skips_exact_ties_and_handles_self():
    ref = make_result([0.1, 0.1, 0.3])
    arm = make_result([0.1, 0.2, 0.3])
    rows = winner_agreement_series({"ref": ref, "arm": arm}, "ref")
    row = metric_row(rows)
    assert row["winner_agreement"] == 1.0
    assert row["n_pairs_valid"] == 2
    assert row["n_pairs_matching"] == 2
    assert row["n_pairs_skipped_tie"] == 1
    assert metric_row(rows, arm="ref")["winner_agreement"] == 1.0
    assert metric_row(rows, arm="ref")["status"] == "self"


def test_winner_agreement_reports_no_valid_pairs():
    ref = make_result([0.2, 0.2, 0.2])
    arm = make_result([0.1, 0.2, 0.3])
    row = metric_row(winner_agreement_series({"ref": ref, "arm": arm}, "ref"))
    assert row["winner_agreement"] is None
    assert row["n_pairs_valid"] == 0
    assert row["n_pairs_skipped_tie"] == 3
    assert row["status"] == "no_valid_pairs"


def crossing_result(rows):
    grid = tuple(rows)
    return make_result(
        {"clf": {n: values for n, values in rows.items()}},
        grid=grid,
        subsets=((0,), (1,)),
    )


def test_crossing_starts_at_n2_and_uses_observed_direction_rule():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    assert directed_crossing_events(result, "clf") == [
        {"row": 0, "col": 1, "n_crossing": 4}
    ]
    progressive = progressive_directed_nstar(result, "clf")
    assert [item["n_prefix"] for item in progressive] == [4]
    assert progressive[0]["matrix"] == [[None, 4], [None, None]]


def test_crossing_from_previous_tie_is_valid_but_initial_win_is_not():
    tie_then_win = crossing_result({2: [0.5, 0.5], 4: [0.4, 0.5]})
    assert directed_crossing_events(tie_then_win, "clf") == [
        {"row": 0, "col": 1, "n_crossing": 4}
    ]
    already_winning = crossing_result({2: [0.4, 0.5], 4: [0.3, 0.5]})
    assert directed_crossing_events(already_winning, "clf") == []


def test_progressive_crossing_keeps_latest_same_direction_across_alternations():
    result = crossing_result(
        {
            2: [0.6, 0.5],
            4: [0.4, 0.5],
            8: [0.6, 0.5],
            16: [0.4, 0.5],
        }
    )
    events = directed_crossing_events(result, "clf")
    assert events == [
        {"row": 0, "col": 1, "n_crossing": 4},
        {"row": 0, "col": 1, "n_crossing": 16},
        {"row": 1, "col": 0, "n_crossing": 8},
    ]
    progressive = progressive_directed_nstar(result, "clf")
    assert progressive[0]["matrix"] == [[None, 4], [None, None]]
    assert progressive[1]["matrix"] == [[None, 4], [8, None]]
    assert progressive[2]["matrix"] == [[None, 16], [8, None]]
    assert all(
        value <= item["n_prefix"]
        for item in progressive
        for row in item["matrix"]
        for value in row
        if value is not None
    )


def test_nstar_similarity_is_unavailable_at_first_prefix_and_json_safe():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    rows = progressive_nstar_similarity({"ref": result, "arm": result}, "ref")
    first = metric_row(rows, n=2)
    assert first["nstar_similarity"] is None
    assert first["status"] == "unavailable_first_prefix"
    assert "null" in json.dumps(first, allow_nan=False)


def test_nstar_similarity_no_crossings_in_either():
    ref = crossing_result({2: [0.4, 0.5], 4: [0.3, 0.5]})
    arm = crossing_result({2: [0.2, 0.5], 4: [0.1, 0.5]})
    row = metric_row(progressive_nstar_similarity({"ref": ref, "arm": arm}, "ref"), n=4)
    assert row["crossing_jaccard"] == 1.0
    assert row["timing_similarity"] == 1.0
    assert row["nstar_similarity"] == 1.0
    assert row["status"] == "no_crossings_in_either"


def test_nstar_similarity_no_shared_crossings():
    ref = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    arm = crossing_result({2: [0.4, 0.5], 4: [0.3, 0.5]})
    row = metric_row(progressive_nstar_similarity({"ref": ref, "arm": arm}, "ref"), n=4)
    assert row["n_reference_crossings"] == 1
    assert row["n_arm_crossings"] == 0
    assert row["crossing_jaccard"] == 0.0
    assert row["timing_similarity"] is None
    assert row["nstar_similarity"] == 0.0
    assert row["status"] == "no_shared_crossings"


def test_nstar_similarity_jaccard_and_shared_timing():
    ref = crossing_result(
        {2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.6, 0.5]}
    )
    arm = crossing_result(
        {2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.3, 0.5]}
    )
    row = metric_row(progressive_nstar_similarity({"ref": ref, "arm": arm}, "ref"), n=8)
    assert row["n_reference_crossings"] == 2
    assert row["n_arm_crossings"] == 1
    assert row["n_shared_crossings"] == 1
    assert row["crossing_jaccard"] == pytest.approx(0.5)
    assert row["timing_similarity"] == 1.0
    assert row["nstar_similarity"] == pytest.approx(0.5)
    assert row["status"] == "ok"


def test_nstar_similarity_uses_current_prefix_log2_span_and_clips_bounds():
    ref = crossing_result(
        {2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.3, 0.5], 16: [0.2, 0.5]}
    )
    arm = crossing_result(
        {2: [0.6, 0.5], 4: [0.6, 0.5], 8: [0.4, 0.5], 16: [0.3, 0.5]}
    )
    rows = progressive_nstar_similarity({"ref": ref, "arm": arm}, "ref")
    at_eight = metric_row(rows, n=8)
    assert at_eight["crossing_jaccard"] == 1.0
    assert at_eight["timing_similarity"] == pytest.approx(0.5)
    assert at_eight["nstar_similarity"] == pytest.approx(0.5)
    assert all(
        0.0 <= row["nstar_similarity"] <= 1.0
        for row in rows
        if row["nstar_similarity"] is not None
    )


def test_nstar_similarity_self_comparison_is_one_with_informative_status():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    row = metric_row(
        progressive_nstar_similarity({"ref": result}, "ref"), arm="ref", n=4
    )
    assert row["nstar_similarity"] == 1.0
    assert row["crossing_jaccard"] == 1.0
    assert row["timing_similarity"] == 1.0
    assert row["status"] == "ok"


def test_serialization_uses_sparse_winners_events_and_canonical_catalog():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    data = simulation_structural_dynamics(result)
    assert data["subset_catalog"] == [[0], [1]]
    assert data["sample_sizes"] == [2, 4]
    classifier = data["classifiers"]["clf"]
    assert classifier["winner_pairs_by_n"] == {
        "2": [{"i": 0, "j": 1, "outcome": -1}],
        "4": [{"i": 0, "j": 1, "outcome": 1}],
    }
    assert classifier["directed_crossing_events"] == [
        {"row": 0, "col": 1, "n_crossing": 4}
    ]


def test_report_data_serialization_is_deterministic_and_strict_json_safe():
    ref = crossing_result({2: [0.1, 0.1], 4: [0.1, 0.1]})
    arm = crossing_result({2: [0.2, 0.2], 4: [0.2, 0.2]})
    first = simulation_structural_dynamics(ref)
    second = simulation_structural_dynamics(ref)
    assert first == second

    rows = ranking_fidelity_series({"ref": ref, "arm": arm}, "ref")
    unavailable = metric_row(rows, n=2)
    assert unavailable["rho_rank"] is None
    encoded = json.dumps(
        {"dynamics": first, "unavailable": unavailable},
        allow_nan=False,
        sort_keys=True,
    )
    assert '"rho_rank": null' in encoded
    assert "NaN" not in encoded
