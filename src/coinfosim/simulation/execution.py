"""Execution configuration and executors for Monte Carlo replications."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

from threadpoolctl import threadpool_limits

from coinfosim.samplers.dataset import Dataset
from coinfosim.simulation.replication import (
    Cell,
    ReplicationResult,
    ReplicationSampler,
    ReplicationTask,
    evaluate_replication,
)

SUPPORTED_BACKENDS = ("sequential", "process")


def effective_worker_count(
    execution_config: "ExecutionConfig",
    replication_batch_size: int,
) -> int:
    """Return the number of workers that can execute one replication batch."""

    if execution_config.backend == "sequential":
        return 1
    return min(execution_config.n_jobs, int(replication_batch_size))


def estimate_fixed_test_cache_bytes(
    test_dataset: Dataset,
    subsets: Sequence[Sequence[int]],
) -> int:
    """Estimate bytes copied by pre-materializing every fixed-test subset."""

    values_per_row = sum(len(tuple(subset)) for subset in subsets)
    return int(
        test_dataset.n_samples
        * values_per_row
        * test_dataset.X.dtype.itemsize
    )


def build_execution_metadata(
    execution_config: "ExecutionConfig",
    replication_batch_size: int,
    test_dataset: Dataset,
    subsets: Sequence[Sequence[int]],
) -> dict[str, object]:
    """Build auditable execution metadata for one simulation arm."""

    return {
        "backend": execution_config.backend,
        "requested_workers": execution_config.n_jobs,
        "effective_workers": effective_worker_count(
            execution_config, replication_batch_size
        ),
        "worker_inner_threads": execution_config.worker_inner_threads,
        "start_method": execution_config.start_method,
        "logical_cpus": os.cpu_count(),
        "fixed_test_cache_bytes_per_worker": estimate_fixed_test_cache_bytes(
            test_dataset, subsets
        ),
    }


@dataclass
class _WorkerContext:
    """Fixed process-local context installed once by the worker initializer."""

    sampler: ReplicationSampler
    cells: tuple[Cell, ...]
    test_dataset: Dataset
    test_by_subset: Mapping[tuple[int, ...], Dataset]
    thread_limiter: Any


_WORKER_CONTEXT: Optional[_WorkerContext] = None


def _initialize_worker(
    sampler: ReplicationSampler,
    cells: Sequence[Cell],
    test_dataset: Dataset,
    test_by_subset: Mapping[tuple[int, ...], Dataset],
    worker_inner_threads: int,
) -> None:
    """Install immutable arm context and persistent numeric thread limits."""

    global _WORKER_CONTEXT
    thread_limiter = threadpool_limits(limits=worker_inner_threads)
    _WORKER_CONTEXT = _WorkerContext(
        sampler=sampler,
        cells=tuple(cells),
        test_dataset=test_dataset,
        test_by_subset=test_by_subset,
        thread_limiter=thread_limiter,
    )


def _run_worker_task(task: ReplicationTask) -> ReplicationResult:
    """Evaluate one complete replication using process-local arm context."""

    if _WORKER_CONTEXT is None:
        raise RuntimeError("Monte Carlo worker context is not initialized")
    return evaluate_replication(
        sampler=_WORKER_CONTEXT.sampler,
        cells=_WORKER_CONTEXT.cells,
        test_by_subset=_WORKER_CONTEXT.test_by_subset,
        task=task,
    )


class ReplicationExecutionError(RuntimeError):
    """Contextual failure raised when one replication cannot be evaluated."""

    def __init__(self, n_per_class: int, replication_id: int) -> None:
        self.n_per_class = int(n_per_class)
        self.replication_id = int(replication_id)
        super().__init__(
            "Monte Carlo replication failed: "
            f"n_per_class={self.n_per_class}, "
            f"replication_id={self.replication_id}"
        )


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

    def close(self) -> None:
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

    def close(self) -> None:
        """Release executor resources (a no-op for sequential execution)."""


class ProcessReplicationExecutor:
    """Evaluate complete replications in one persistent process pool."""

    def __init__(
        self,
        sampler: ReplicationSampler,
        cells: Sequence[Cell],
        test_dataset: Dataset,
        test_by_subset: Mapping[tuple[int, ...], Dataset],
        execution_config: ExecutionConfig,
        replication_batch_size: int,
    ) -> None:
        if execution_config.backend != "process":
            raise ValueError("ProcessReplicationExecutor requires process backend")
        self.effective_workers = effective_worker_count(
            execution_config,
            replication_batch_size,
        )
        self._closed = False
        self._pool = ProcessPoolExecutor(
            max_workers=self.effective_workers,
            mp_context=mp.get_context(execution_config.start_method),
            initializer=_initialize_worker,
            initargs=(
                sampler,
                tuple(cells),
                test_dataset,
                test_by_subset,
                execution_config.worker_inner_threads,
            ),
        )

    @staticmethod
    def _cancel(futures: Mapping[Future, int]) -> None:
        for future in futures:
            future.cancel()

    def run_batch(
        self,
        n_per_class: int,
        replication_ids: Sequence[int],
    ) -> list[ReplicationResult]:
        """Submit and collect one complete batch without committing results."""

        if self._closed:
            raise RuntimeError("process replication executor is closed")

        futures: dict[Future, int] = {}
        for replication_id in replication_ids:
            task = ReplicationTask(
                n_per_class=n_per_class,
                replication_id=replication_id,
            )
            try:
                future = self._pool.submit(_run_worker_task, task)
            except Exception as exc:
                self._cancel(futures)
                raise ReplicationExecutionError(
                    n_per_class, replication_id
                ) from exc
            futures[future] = replication_id

        results = []
        for future in as_completed(futures):
            replication_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                self._cancel(futures)
                raise ReplicationExecutionError(
                    n_per_class, replication_id
                ) from exc
        return results

    def close(self) -> None:
        """Shut down the persistent process pool exactly once."""

        if not self._closed:
            self._pool.shutdown(wait=True)
            self._closed = True


def make_replication_executor(
    execution_config: ExecutionConfig,
    sampler: ReplicationSampler,
    cells: Sequence[Cell],
    test_dataset: Dataset,
    test_by_subset: Mapping[tuple[int, ...], Dataset],
    replication_batch_size: int,
) -> ReplicationExecutor:
    """Construct the configured executor for one complete simulator run."""

    if execution_config.backend == "sequential":
        return SequentialReplicationExecutor(
            sampler=sampler,
            cells=cells,
            test_by_subset=test_by_subset,
        )
    return ProcessReplicationExecutor(
        sampler=sampler,
        cells=cells,
        test_dataset=test_dataset,
        test_by_subset=test_by_subset,
        execution_config=execution_config,
        replication_batch_size=replication_batch_size,
    )
