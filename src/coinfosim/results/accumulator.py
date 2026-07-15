"""
Monte Carlo loss accumulator for CoInfoSim Sprint 1.

:class:`LossAccumulator` stores individual replication losses indexed by
``(n_per_class, subset, classifier_name, replication_id)`` and provides
summary statistics (mean, standard deviation, standard error) and the
number of completed replications for each ``n_per_class``.

Only empirical test loss is stored. There is no notion of train loss,
theoretical loss, or Bayes error.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from coinfosim.simulation.replication import Cell, ReplicationResult

# A key identifying one (n_per_class, subset, classifier) cell.
CellKey = Tuple[int, Tuple[int, ...], str]


class LossAccumulator:
    """Accumulate per-replication empirical test losses and summarize them."""

    def __init__(self) -> None:
        # Map cell -> dict(replication_id -> loss).
        self._losses: Dict[CellKey, Dict[int, float]] = {}

    @staticmethod
    def _cell(n_per_class: int, subset, classifier_name: str) -> CellKey:
        return (int(n_per_class), tuple(subset), str(classifier_name))

    def add(
        self,
        n_per_class: int,
        subset,
        classifier_name: str,
        replication_id: int,
        loss: float,
    ) -> None:
        """Record one replication loss for a (n_per_class, subset, classifier) cell."""
        cell = self._cell(n_per_class, subset, classifier_name)
        self._losses.setdefault(cell, {})[int(replication_id)] = float(loss)

    def add_replication(
        self,
        n_per_class: int,
        replication_id: int,
        cells: Sequence[Cell],
        losses: Sequence[float],
    ) -> None:
        """Validate and atomically record one complete replication."""

        result = ReplicationResult(
            n_per_class=int(n_per_class),
            replication_id=int(replication_id),
            losses=tuple(float(loss) for loss in losses),
        )
        self.add_batch(
            n_per_class=n_per_class,
            expected_replication_ids=[replication_id],
            cells=cells,
            results=[result],
        )

    def add_batch(
        self,
        n_per_class: int,
        expected_replication_ids: Sequence[int],
        cells: Sequence[Cell],
        results: Sequence[ReplicationResult],
    ) -> None:
        """Validate and atomically record a complete replication batch."""

        n_per_class = int(n_per_class)
        expected_ids = [
            int(replication_id) for replication_id in expected_replication_ids
        ]
        normalized_cells = [
            (tuple(subset), str(classifier_name))
            for subset, classifier_name in cells
        ]
        if len(set(normalized_cells)) != len(normalized_cells):
            raise ValueError("cells must not contain duplicates")

        cell_keys = [
            self._cell(n_per_class, subset, classifier_name)
            for subset, classifier_name in normalized_cells
        ]
        counts = [len(self._losses.get(cell, {})) for cell in cell_keys]
        if len(set(counts)) > 1:
            raise ValueError(
                "cannot add batch: requested cells have unequal existing "
                f"replication counts {counts}"
            )
        completed_count = counts[0] if counts else 0

        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected replication IDs must not contain duplicates")

        received_ids = [int(result.replication_id) for result in results]
        if len(set(received_ids)) != len(received_ids):
            raise ValueError("received replication IDs must not contain duplicates")
        if sorted(received_ids) != sorted(expected_ids):
            raise ValueError(
                "received replication IDs do not match expected IDs: "
                f"received={sorted(received_ids)}, expected={sorted(expected_ids)}"
            )

        pending = []
        for result in results:
            if int(result.n_per_class) != n_per_class:
                raise ValueError(
                    "replication result uses the wrong n_per_class: "
                    f"received={result.n_per_class}, expected={n_per_class}"
                )
            if len(result.losses) != len(normalized_cells):
                raise ValueError(
                    "replication loss vector length does not match cells: "
                    f"received={len(result.losses)}, expected={len(normalized_cells)}"
                )

            losses = np.asarray(result.losses, dtype=float)
            if not np.all(np.isfinite(losses)):
                raise ValueError("replication losses must all be finite")
            if np.any(losses < 0.0) or np.any(losses > 1.0):
                raise ValueError("replication losses must all be within [0, 1]")

            replication_id = int(result.replication_id)
            for cell, loss in zip(cell_keys, losses):
                if replication_id in self._losses.get(cell, {}):
                    raise ValueError(
                        "replication already exists for cell: "
                        f"cell={cell}, replication_id={replication_id}"
                    )
                pending.append((cell, replication_id, float(loss)))

        required_ids = list(
            range(completed_count, completed_count + len(expected_ids))
        )
        if expected_ids != required_ids:
            raise ValueError(
                "expected replication IDs must continue contiguously from "
                f"{completed_count}: received={expected_ids}, required={required_ids}"
            )

        existing_ids = list(range(completed_count))
        for cell in cell_keys:
            actual_ids = sorted(self._losses.get(cell, {}))
            if actual_ids != existing_ids:
                raise ValueError(
                    "existing replication IDs must be contiguous from zero: "
                    f"cell={cell}, received={actual_ids}, required={existing_ids}"
                )

        for cell, replication_id, loss in sorted(pending, key=lambda item: item[1]):
            self._losses.setdefault(cell, {})[replication_id] = loss

    def losses(self, n_per_class: int, subset, classifier_name: str) -> np.ndarray:
        """Return the recorded losses for a cell, ordered by replication id."""
        cell = self._cell(n_per_class, subset, classifier_name)
        rep_map = self._losses.get(cell, {})
        if not rep_map:
            return np.empty(0, dtype=float)
        ordered = [rep_map[r] for r in sorted(rep_map)]
        return np.asarray(ordered, dtype=float)

    def count(self, n_per_class: int, subset, classifier_name: str) -> int:
        """Number of replications recorded for a cell."""
        cell = self._cell(n_per_class, subset, classifier_name)
        return len(self._losses.get(cell, {}))

    def mean_loss(self, n_per_class: int, subset, classifier_name: str) -> float:
        """Mean test loss for a cell."""
        values = self.losses(n_per_class, subset, classifier_name)
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    def std_loss(self, n_per_class: int, subset, classifier_name: str) -> float:
        """Sample standard deviation (ddof=1) of test losses for a cell.

        Returns ``0.0`` when fewer than two replications are present.
        """
        values = self.losses(n_per_class, subset, classifier_name)
        if values.size < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    def standard_error(self, n_per_class: int, subset, classifier_name: str) -> float:
        """Standard error of the mean test loss for a cell.

        Returns ``0.0`` when fewer than two replications are present.
        """
        values = self.losses(n_per_class, subset, classifier_name)
        n = values.size
        if n < 2:
            return 0.0
        return float(np.std(values, ddof=1) / np.sqrt(n))

    def replications_completed(self, n_per_class: int) -> int:
        """Return the common replication count across all cells for ``n_per_class``.

        In Sprint 1 every (subset, classifier) pair shares the same number of
        replications for a given ``n_per_class``. This returns the maximum
        observed count (which equals the common count by construction).
        """
        counts = [
            len(rep_map)
            for cell, rep_map in self._losses.items()
            if cell[0] == int(n_per_class)
        ]
        return max(counts) if counts else 0

    def cells_for(self, n_per_class: int) -> List[CellKey]:
        """Return all recorded cells for a given ``n_per_class``."""
        return [cell for cell in self._losses if cell[0] == int(n_per_class)]

    def sample_sizes(self) -> List[int]:
        """Return the sorted list of recorded ``n_per_class`` values."""
        return sorted({cell[0] for cell in self._losses})
