from dataclasses import dataclass
import json

import numpy as np
import pytest

from coinfosim.results.structural import (
    directed_crossing_events,
    effective_winner_matrices,
    progressive_directed_nstar,
    progressive_reversal_fidelity,
    progressive_reversal_matrices,
    rank_vector,
    ranking_fidelity_series,
    scenario_structural_fidelity,
    select_display_subsets_by_cardinality,
    simulation_structural_dynamics,
    validate_structural_compatibility,
    winner_agreement_series,
    winner_matrix,
    winner_reversal_events,
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


def test_effective_winner_no_reversal_after_initial_tie_then_first_winner():
    result = crossing_result({2: [0.5, 0.5], 4: [0.4, 0.5]})
    effective = effective_winner_matrices(result, "clf")
    assert effective[0]["matrix"] == [[None, 0], [0, None]]
    assert effective[1]["matrix"] == [[None, 1], [-1, None]]
    assert winner_reversal_events(result, "clf") == []


def test_no_reversal_after_repeated_initial_ties_then_first_winner():
    result = crossing_result({2: [0.5, 0.5], 4: [0.5, 0.5], 8: [0.4, 0.5]})
    assert winner_reversal_events(result, "clf") == []


def test_no_reversal_when_winner_repeats_across_intervening_tie():
    result = crossing_result({2: [0.4, 0.5], 4: [0.5, 0.5], 6: [0.3, 0.5]})
    assert winner_reversal_events(result, "clf") == []


def test_single_reversal_after_tie_then_strict_opposite_winner():
    result = crossing_result({2: [0.4, 0.5], 4: [0.5, 0.5], 6: [0.6, 0.5]})
    assert winner_reversal_events(result, "clf") == [{"i": 0, "j": 1, "n_reversal": 6}]


def test_single_reversal_between_two_strict_winners():
    result = crossing_result({2: [0.4, 0.5], 4: [0.6, 0.5]})
    assert winner_reversal_events(result, "clf") == [{"i": 0, "j": 1, "n_reversal": 4}]


def test_multiple_alternations_retain_all_events_and_progressive_r_keeps_latest():
    result = crossing_result(
        {2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.6, 0.5], 16: [0.4, 0.5]}
    )
    events = winner_reversal_events(result, "clf")
    assert events == [
        {"i": 0, "j": 1, "n_reversal": 4},
        {"i": 0, "j": 1, "n_reversal": 8},
        {"i": 0, "j": 1, "n_reversal": 16},
    ]
    progressive = progressive_reversal_matrices(result, "clf")
    assert progressive[0]["matrix"] == [[None, None], [None, None]]
    assert progressive[1]["matrix"] == [[None, 4], [None, None]]
    assert progressive[2]["matrix"] == [[None, 8], [None, None]]
    assert progressive[3]["matrix"] == [[None, 16], [None, None]]


def test_pair_with_no_reversal_remains_undefined_in_r():
    result = make_result(
        {"clf": {2: [0.4, 0.5, 0.9], 4: [0.6, 0.5, 0.9]}},
        grid=(2, 4),
        subsets=((0,), (1,), (2,)),
    )
    events = winner_reversal_events(result, "clf")
    assert events == [{"i": 0, "j": 1, "n_reversal": 4}]
    for item in progressive_reversal_matrices(result, "clf"):
        assert item["matrix"][0][2] is None
        assert item["matrix"][1][2] is None


def test_reversal_events_use_only_i_lt_j_pairs():
    result = crossing_result(
        {2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.6, 0.5], 16: [0.4, 0.5]}
    )
    events = winner_reversal_events(result, "clf")
    assert events and all(event["i"] < event["j"] for event in events)


def test_progressive_reversal_matrix_has_values_only_in_upper_triangle():
    result = crossing_result(
        {2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.6, 0.5], 16: [0.4, 0.5]}
    )
    for item in progressive_reversal_matrices(result, "clf"):
        matrix = item["matrix"]
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i >= j:
                    assert matrix[i][j] is None


def test_effective_winner_matrix_remains_antisymmetric_after_carry_forward():
    result = crossing_result({2: [0.4, 0.5], 4: [0.5, 0.5], 6: [0.6, 0.5]})
    for item in effective_winner_matrices(result, "clf"):
        matrix = item["matrix"]
        assert matrix[0][1] == -matrix[1][0]


def test_winner_agreement_uses_carried_forward_effective_winner_through_tie():
    ref = crossing_result({2: [0.4, 0.5], 4: [0.5, 0.5]})
    arm = crossing_result({2: [0.3, 0.5], 4: [0.5, 0.5]})
    row = metric_row(winner_agreement_series({"ref": ref, "arm": arm}, "ref"), n=4)
    assert row["n_pairs_valid"] == 1
    assert row["n_pairs_matching"] == 1
    assert row["winner_agreement"] == 1.0
    assert row["status"] == "ok"


def test_reversal_fidelity_is_unavailable_at_first_prefix_and_json_safe():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    rows = progressive_reversal_fidelity({"ref": result, "arm": result}, "ref")
    first = metric_row(rows, n=2)
    assert first["reversal_existence_agreement"] is None
    assert first["mean_log2_reversal_distance"] is None
    assert first["reversal_sample_size_similarity"] is None
    assert first["status"] == "unavailable_first_prefix"
    assert "null" in json.dumps(first, allow_nan=False)


def test_reversal_fidelity_no_reversals_in_either():
    ref = crossing_result({2: [0.4, 0.5], 4: [0.3, 0.5]})
    arm = crossing_result({2: [0.2, 0.5], 4: [0.1, 0.5]})
    row = metric_row(progressive_reversal_fidelity({"ref": ref, "arm": arm}, "ref"), n=4)
    assert row["reversal_existence_agreement"] == 1.0
    assert row["mean_log2_reversal_distance"] is None
    assert row["reversal_sample_size_similarity"] is None
    assert row["status"] == "no_reversals_in_either"


def test_reversal_fidelity_no_shared_reversals_leaves_similarity_undefined():
    ref = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    arm = crossing_result({2: [0.4, 0.5], 4: [0.3, 0.5]})
    row = metric_row(progressive_reversal_fidelity({"ref": ref, "arm": arm}, "ref"), n=4)
    assert row["n_reference_reversal_pairs"] == 1
    assert row["n_arm_reversal_pairs"] == 0
    assert row["reversal_existence_agreement"] == 0.0
    assert row["mean_log2_reversal_distance"] is None
    assert row["reversal_sample_size_similarity"] is None
    assert row["status"] == "no_shared_reversals"


def test_reversal_fidelity_partial_overlap_across_independent_pairs():
    ref = make_result(
        {
            "clf": {
                2: [0.3, 0.2, 0.6, 0.5],
                4: [0.15, 0.35, 0.6, 0.5],
                8: [0.15, 0.35, 0.55, 0.9],
            }
        },
        grid=(2, 4, 8),
        subsets=((0,), (1,), (2,), (3,)),
    )
    arm = make_result(
        {
            "clf": {
                2: [0.3, 0.2, 0.6, 0.5],
                4: [0.15, 0.35, 0.6, 0.5],
                8: [0.15, 0.35, 0.6, 0.5],
            }
        },
        grid=(2, 4, 8),
        subsets=((0,), (1,), (2,), (3,)),
    )
    row = metric_row(
        progressive_reversal_fidelity({"ref": ref, "arm": arm}, "ref"), n=8
    )
    assert row["n_reference_reversal_pairs"] == 2
    assert row["n_arm_reversal_pairs"] == 1
    assert row["n_shared_reversal_pairs"] == 1
    assert row["n_union_reversal_pairs"] == 2
    assert row["reversal_existence_agreement"] == pytest.approx(0.5)
    assert row["mean_log2_reversal_distance"] == pytest.approx(0.0)
    assert row["reversal_sample_size_similarity"] == pytest.approx(1.0)
    assert row["status"] == "ok"


def test_reversal_fidelity_full_overlap_identical_reversal_sample_sizes():
    ref = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    arm = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    row = metric_row(progressive_reversal_fidelity({"ref": ref, "arm": arm}, "ref"), n=4)
    assert row["reversal_existence_agreement"] == 1.0
    assert row["mean_log2_reversal_distance"] == pytest.approx(0.0)
    assert row["reversal_sample_size_similarity"] == pytest.approx(1.0)
    assert row["status"] == "ok"


def test_reversal_fidelity_one_level_log2_displacement():
    ref = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5], 8: [0.4, 0.5]})
    arm = crossing_result({2: [0.6, 0.5], 4: [0.6, 0.5], 8: [0.4, 0.5]})
    row = metric_row(progressive_reversal_fidelity({"ref": ref, "arm": arm}, "ref"), n=8)
    assert row["mean_log2_reversal_distance"] == pytest.approx(1.0)
    assert row["reversal_sample_size_similarity"] == pytest.approx(0.5)
    assert row["status"] == "ok"


def test_reversal_fidelity_self_comparison_is_ok_with_zero_distance():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    row = metric_row(
        progressive_reversal_fidelity({"ref": result}, "ref"), arm="ref", n=4
    )
    assert row["reversal_existence_agreement"] == 1.0
    assert row["mean_log2_reversal_distance"] == pytest.approx(0.0)
    assert row["reversal_sample_size_similarity"] == pytest.approx(1.0)
    assert row["status"] == "ok"


def test_reversal_fidelity_has_no_composite_product_field():
    ref = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    rows = progressive_reversal_fidelity({"ref": ref, "arm": ref}, "ref")
    for row in rows:
        assert "nstar_similarity" not in row
        assert "crossing_jaccard" not in row
        assert "timing_similarity" not in row
        assert not any("product" in key for key in row)


def test_structural_schema_version_is_2_and_strict_json_safe():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    dynamics = simulation_structural_dynamics(result)
    assert dynamics["schema_version"] == 2
    fidelity = scenario_structural_fidelity(
        {"ref": result, "arm": result}, "ref", {"ref": "Ref", "arm": "Arm"}
    )
    assert fidelity["schema_version"] == 2
    assert "reversal_fidelity_series" in fidelity
    assert "nstar_similarity_series" not in fidelity
    encoded = json.dumps({"dynamics": dynamics, "fidelity": fidelity}, allow_nan=False)
    assert "NaN" not in encoded


def test_serialization_uses_sparse_effective_winners_and_reversal_events():
    result = crossing_result({2: [0.6, 0.5], 4: [0.4, 0.5]})
    data = simulation_structural_dynamics(result)
    assert data["subset_catalog"] == [[0], [1]]
    assert data["sample_sizes"] == [2, 4]
    classifier = data["classifiers"]["clf"]
    assert classifier["effective_winner_pairs_by_n"] == {
        "2": [{"i": 0, "j": 1, "outcome": -1}],
        "4": [{"i": 0, "j": 1, "outcome": 1}],
    }
    assert classifier["winner_reversal_events"] == [
        {"i": 0, "j": 1, "n_reversal": 4}
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
