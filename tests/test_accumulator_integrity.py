"""Integrity tests for complete Monte Carlo accumulator batches."""

from copy import deepcopy

import numpy as np
import pytest

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.simulation.replication import ReplicationResult
from coinfosim.simulation.stopping import StandardErrorStoppingRule

CELLS = [((0,), "linear_svm"), ((1,), "gaussian_nb")]


def _result(replication_id, losses=(0.1, 0.2), n_per_class=4):
    return ReplicationResult(
        n_per_class=n_per_class,
        replication_id=replication_id,
        losses=tuple(losses),
    )


def _assert_rejected_without_mutation(accumulator, action, match):
    before = deepcopy(accumulator._losses)
    with pytest.raises(ValueError, match=match):
        action()
    assert accumulator._losses == before


def test_add_replication_records_one_complete_vector():
    accumulator = LossAccumulator()

    accumulator.add_replication(4, 0, CELLS, (0.1, 0.2))

    assert np.array_equal(
        accumulator.losses(4, (0,), "linear_svm"), np.array([0.1])
    )
    assert np.array_equal(
        accumulator.losses(4, (1,), "gaussian_nb"), np.array([0.2])
    )


def test_add_batch_accepts_out_of_order_results_and_commits_in_id_order():
    accumulator = LossAccumulator()

    accumulator.add_batch(
        n_per_class=4,
        expected_replication_ids=[0, 1],
        cells=CELLS,
        results=[_result(1, (0.3, 0.4)), _result(0, (0.1, 0.2))],
    )

    assert np.array_equal(
        accumulator.losses(4, (0,), "linear_svm"), np.array([0.1, 0.3])
    )
    assert np.array_equal(
        accumulator.losses(4, (1,), "gaussian_nb"), np.array([0.2, 0.4])
    )


def test_add_batch_rejects_missing_or_unexpected_ids_atomically():
    accumulator = LossAccumulator()
    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [0, 1], CELLS, [_result(0)]),
        "do not match expected IDs",
    )


def test_add_batch_rejects_duplicate_received_ids_atomically():
    accumulator = LossAccumulator()
    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(
            4, [0, 1], CELLS, [_result(0), _result(0)]
        ),
        "received replication IDs must not contain duplicates",
    )


def test_add_batch_rejects_duplicate_expected_ids_atomically():
    accumulator = LossAccumulator()
    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(
            4, [0, 0], CELLS, [_result(0), _result(1)]
        ),
        "expected replication IDs must not contain duplicates",
    )


def test_add_batch_rejects_wrong_sample_size_atomically():
    accumulator = LossAccumulator()
    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [0], CELLS, [_result(0, n_per_class=8)]),
        "wrong n_per_class",
    )


def test_add_batch_rejects_wrong_loss_vector_length_atomically():
    accumulator = LossAccumulator()
    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [0], CELLS, [_result(0, (0.1,))]),
        "loss vector length",
    )


@pytest.mark.parametrize(
    ("loss", "message"),
    [
        (np.nan, "finite"),
        (np.inf, "finite"),
        (-0.1, "within"),
        (1.1, "within"),
    ],
)
def test_add_batch_rejects_invalid_losses_atomically(loss, message):
    accumulator = LossAccumulator()
    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [0], CELLS, [_result(0, (loss, 0.2))]),
        message,
    )


def test_add_batch_rejects_existing_cell_replication_atomically():
    accumulator = LossAccumulator()
    for subset, classifier_name in CELLS:
        accumulator.add(4, subset, classifier_name, 0, 0.1)

    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [0], CELLS, [_result(0)]),
        "replication already exists",
    )


def test_add_batch_rejects_noncontiguous_new_ids_atomically():
    accumulator = LossAccumulator()
    accumulator.add_batch(4, [0], CELLS, [_result(0)])

    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [2], CELLS, [_result(2)]),
        "continue contiguously",
    )


def test_add_batch_rejects_noncontiguous_existing_ids_atomically():
    accumulator = LossAccumulator()
    for subset, classifier_name in CELLS:
        accumulator.add(4, subset, classifier_name, 2, 0.1)

    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [1], CELLS, [_result(1)]),
        "existing replication IDs must be contiguous from zero",
    )


def test_add_batch_rejects_unequal_existing_cell_counts_atomically():
    accumulator = LossAccumulator()
    accumulator.add(4, CELLS[0][0], CELLS[0][1], 0, 0.1)

    _assert_rejected_without_mutation(
        accumulator,
        lambda: accumulator.add_batch(4, [1], CELLS, [_result(1)]),
        "unequal existing replication counts",
    )


def test_stopping_rejects_unequal_requested_cell_counts():
    accumulator = LossAccumulator()
    accumulator.add(4, CELLS[0][0], CELLS[0][1], 0, 0.1)
    rule = StandardErrorStoppingRule(
        min_replications=2,
        max_replications=4,
        ci_half_width_target=0.05,
    )

    with pytest.raises(ValueError, match="unequal replication counts"):
        rule.evaluate(accumulator, 4, CELLS)
