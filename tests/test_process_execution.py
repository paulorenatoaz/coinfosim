"""Focused tests for persistent process-based replication execution."""

from concurrent.futures import Future
from copy import deepcopy

import numpy as np
import pytest

import coinfosim.simulation.execution as execution_module
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.results.accumulator import LossAccumulator
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import (
    ExecutionConfig,
    ProcessReplicationExecutor,
    ReplicationExecutionError,
)
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator
from coinfosim.simulation.replication import ReplicationResult


def _worker_context():
    model = GaussianSimulationModel(
        means={0: [-0.6, -0.3], 1: [0.6, 0.3]},
        covariances={
            0: [[1.0, 0.2], [0.2, 1.0]],
            1: [[1.0, 0.2], [0.2, 1.0]],
        },
    )
    sampler = GaussianClassConditionalSampler(
        model,
        base_seed=7,
        test_samples_per_class=20,
    )
    test_dataset = sampler.sample_test()
    cells = [((0,), "gaussian_nb"), ((1,), "gaussian_nb")]
    test_by_subset = {
        subset: test_dataset.select_channels(subset) for subset, _ in cells
    }
    return model, sampler, test_dataset, cells, test_by_subset


def _result(replication_id, n_per_class=2, n_losses=2):
    return ReplicationResult(
        n_per_class=n_per_class,
        replication_id=replication_id,
        losses=tuple(0.1 + 0.1 * index for index in range(n_losses)),
    )


class _ImmediatePool:
    """ProcessPoolExecutor test double returning configurable futures."""

    instances = []
    failing_id = None
    pending_ids = set()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.futures_by_id = {}
        self.tasks = []
        self.shutdown_called = False
        type(self).instances.append(self)

    def submit(self, function, task):
        self.tasks.append(task)
        future = Future()
        if task.replication_id == type(self).failing_id:
            future.set_exception(RuntimeError("worker boom"))
        elif task.replication_id not in type(self).pending_ids:
            cells = self.kwargs["initargs"][1]
            future.set_result(
                _result(task.replication_id, task.n_per_class, len(cells))
            )
        self.futures_by_id[task.replication_id] = future
        return future

    def shutdown(self, wait=True):
        self.shutdown_called = True


@pytest.fixture
def immediate_pool(monkeypatch):
    _ImmediatePool.instances = []
    _ImmediatePool.failing_id = None
    _ImmediatePool.pending_ids = set()
    monkeypatch.setattr(execution_module, "ProcessPoolExecutor", _ImmediatePool)
    monkeypatch.setattr(
        execution_module,
        "as_completed",
        lambda futures: list(futures),
    )
    return _ImmediatePool


def test_two_worker_forkserver_execution_serializes_worker_context():
    _, sampler, test_dataset, cells, test_by_subset = _worker_context()
    executor = ProcessReplicationExecutor(
        sampler=sampler,
        cells=cells,
        test_dataset=test_dataset,
        test_by_subset=test_by_subset,
        execution_config=ExecutionConfig(
            backend="process",
            n_jobs=2,
            start_method="forkserver",
            worker_inner_threads=1,
        ),
        replication_batch_size=2,
    )

    try:
        results = executor.run_batch(2, [0, 1])
        worker_processes = len(executor._pool._processes)
    finally:
        executor.close()

    assert executor.effective_workers == 2
    assert worker_processes == 2
    assert sorted(result.replication_id for result in results) == [0, 1]
    assert all(len(result.losses) == len(cells) for result in results)
    assert all(np.all(np.isfinite(result.losses)) for result in results)


def test_worker_initializer_keeps_numeric_thread_limit_context(monkeypatch):
    _, sampler, test_dataset, cells, test_by_subset = _worker_context()
    limiter = object()
    monkeypatch.setattr(execution_module, "_WORKER_CONTEXT", None)
    monkeypatch.setattr(
        execution_module,
        "threadpool_limits",
        lambda limits: limiter,
    )

    execution_module._initialize_worker(
        sampler,
        cells,
        test_dataset,
        test_by_subset,
        worker_inner_threads=1,
    )

    context = execution_module._WORKER_CONTEXT
    assert context.sampler is sampler
    assert context.test_dataset is test_dataset
    assert context.thread_limiter is limiter


def test_process_executor_limits_workers_to_batch_size(immediate_pool):
    _, sampler, test_dataset, cells, test_by_subset = _worker_context()
    executor = ProcessReplicationExecutor(
        sampler=sampler,
        cells=cells,
        test_dataset=test_dataset,
        test_by_subset=test_by_subset,
        execution_config=ExecutionConfig(backend="process", n_jobs=8),
        replication_batch_size=2,
    )

    executor.close()

    pool = immediate_pool.instances[0]
    assert executor.effective_workers == 2
    assert pool.kwargs["max_workers"] == 2
    assert pool.shutdown_called is True


def test_out_of_order_completion_is_committed_in_replication_order(
    immediate_pool, monkeypatch
):
    _, sampler, test_dataset, cells, test_by_subset = _worker_context()
    executor = ProcessReplicationExecutor(
        sampler=sampler,
        cells=cells,
        test_dataset=test_dataset,
        test_by_subset=test_by_subset,
        execution_config=ExecutionConfig(backend="process", n_jobs=2),
        replication_batch_size=2,
    )
    pool = immediate_pool.instances[0]
    monkeypatch.setattr(
        execution_module,
        "as_completed",
        lambda futures: list(reversed(list(futures))),
    )

    results = executor.run_batch(2, [0, 1])
    executor.close()

    assert [result.replication_id for result in results] == [1, 0]
    accumulator = LossAccumulator()
    results.sort(key=lambda result: result.replication_id)
    accumulator.add_batch(2, [0, 1], cells, results)
    for subset, classifier_name in cells:
        key = (2, subset, classifier_name)
        assert list(accumulator._losses[key]) == [0, 1]
    assert pool.shutdown_called is True


def test_worker_failure_cancels_pending_tasks_and_cannot_partially_commit(
    immediate_pool,
):
    _, sampler, test_dataset, cells, test_by_subset = _worker_context()
    immediate_pool.failing_id = 1
    immediate_pool.pending_ids = {2}
    executor = ProcessReplicationExecutor(
        sampler=sampler,
        cells=cells,
        test_dataset=test_dataset,
        test_by_subset=test_by_subset,
        execution_config=ExecutionConfig(backend="process", n_jobs=2),
        replication_batch_size=3,
    )
    accumulator = LossAccumulator()
    accumulator.add_batch(2, [0], cells, [_result(0)])
    before = deepcopy(accumulator._losses)

    with pytest.raises(
        ReplicationExecutionError,
        match="n_per_class=2, replication_id=1",
    ):
        executor.run_batch(2, [0, 1, 2])
    executor.close()

    pool = immediate_pool.instances[0]
    assert pool.futures_by_id[2].cancelled()
    assert accumulator._losses == before


def test_simulator_uses_one_process_pool_for_all_sample_sizes(immediate_pool):
    model, sampler, _, _, _ = _worker_context()
    config = MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=7,
    )

    result = CooperativeMonteCarloSimulator(
        model,
        config,
        subsets=[(0,)],
        classifier_names=["gaussian_nb"],
        sampler=sampler,
        execution_config=ExecutionConfig(backend="process", n_jobs=2),
    ).run()

    assert result.sample_sizes == [2, 4]
    assert len(immediate_pool.instances) == 1
    pool = immediate_pool.instances[0]
    assert [(task.n_per_class, task.replication_id) for task in pool.tasks] == [
        (2, 0),
        (2, 1),
        (4, 0),
        (4, 1),
    ]
    assert pool.shutdown_called is True
