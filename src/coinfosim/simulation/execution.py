"""Execution configuration and executors for Monte Carlo replications."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from coinfosim.samplers.dataset import Dataset
from coinfosim.simulation.replication import (
    Cell,
    ReplicationResult,
    ReplicationSampler,
    ReplicationTask,
    evaluate_replication,
)

SUPPORTED_BACKENDS = ("sequential", "process")


@dataclass(frozen=True)
class ExecutionConfig:
    """Computational execution settings, separate from scientific settings."""

    backend: str = "sequential"
    n_jobs: int = 1
    start_method: str = "forkserver"
    worker_inner_threads: int = 1

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"unknown execution backend {self.backend!r}; "
                f"supported backends: {list(SUPPORTED_BACKENDS)}"
            )
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive")
        if self.worker_inner_threads <= 0:
            raise ValueError("worker_inner_threads must be positive")
        if self.backend == "process":
            available = mp.get_all_start_methods()
            if self.start_method not in available:
                raise ValueError(
                    f"multiprocessing start method {self.start_method!r} is not "
                    f"available; available methods: {available}"
                )


class ReplicationExecutor(Protocol):
    """Interface for evaluating a complete batch of replications."""

    def run_batch(
        self,
        n_per_class: int,
        replication_ids: Sequence[int],
    ) -> list[ReplicationResult]:
        ...


class SequentialReplicationExecutor:
    """Evaluate complete replications serially in the main process."""

    def __init__(
        self,
        sampler: ReplicationSampler,
        cells: Sequence[Cell],
        test_by_subset: Mapping[tuple[int, ...], Dataset],
    ) -> None:
        self.sampler = sampler
        self.cells = tuple(cells)
        self.test_by_subset = test_by_subset

    def run_batch(
        self,
        n_per_class: int,
        replication_ids: Sequence[int],
    ) -> list[ReplicationResult]:
        """Evaluate the requested replication IDs in their supplied order."""

        return [
            evaluate_replication(
                sampler=self.sampler,
                cells=self.cells,
                test_by_subset=self.test_by_subset,
                task=ReplicationTask(
                    n_per_class=n_per_class,
                    replication_id=replication_id,
                ),
            )
            for replication_id in replication_ids
        ]
