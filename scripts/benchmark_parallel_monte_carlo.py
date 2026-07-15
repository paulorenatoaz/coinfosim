#!/usr/bin/env python3
"""Run the bounded Block 8 parallel Monte Carlo benchmark.

The benchmark uses the Occupancy single-Gaussian-to-real arm without run
registries, reports, or persisted artifacts. Its workload is fixed by the
approved implementation plan: all 31 subsets, all three classifiers, sample
sizes ``(2, 32, 512)``, and exactly 40 replications in batches of 20.

Run from the repository root with::

    .venv/bin/python scripts/benchmark_parallel_monte_carlo.py

Machine-specific results are printed as a Markdown table and are intentionally
not written into the repository. Five workers remain the initial recommendation
unless six workers are at least five percent faster in this measurement.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from coinfosim.classifiers.registry import available_classifiers
from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios.occupancy import build_gaussian_anchored_occupancy_model
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import ExecutionConfig
from coinfosim.simulation.monte_carlo import (
    CooperativeMonteCarloSimulator,
    SimulationResult,
)


SAMPLE_SIZES = (2, 32, 512)
REPLICATIONS = 40
BATCH_SIZE = 20
BASE_SEED = 0
PROCESS_WORKERS = (5, 6)
REPEAT_THRESHOLD = 0.05


@dataclass
class BenchmarkRecord:
    backend: str
    requested_workers: int
    effective_workers: int
    runtimes: list[float] = field(default_factory=list)
    exact_results: list[bool] = field(default_factory=list)

    @property
    def runtime(self) -> float:
        return statistics.mean(self.runtimes)

    @property
    def exact(self) -> bool:
        return all(self.exact_results)


def _benchmark_config() -> MonteCarloConfig:
    return MonteCarloConfig(
        mode="benchmark",
        sample_sizes=SAMPLE_SIZES,
        min_replications=REPLICATIONS,
        max_replications=REPLICATIONS,
        replication_batch_size=BATCH_SIZE,
        test_samples_per_class=5000,
        ci_half_width_target=0.01,
        base_seed=BASE_SEED,
    )


def _sampler(model, test_dataset):
    train_sampler = GaussianClassConditionalSampler(
        model,
        base_seed=BASE_SEED,
        test_samples_per_class=5000,
    )
    return SyntheticTrainRealTestSampler(
        train_sampler,
        test_dataset,
        name="occupancy_single_gaussian_to_real_benchmark",
    )


def _run_once(model, test_dataset, execution: ExecutionConfig) -> SimulationResult:
    print(
        "Running "
        f"backend={execution.backend} n_jobs={execution.n_jobs} "
        f"threads_per_worker={execution.worker_inner_threads} ...",
        flush=True,
    )
    with threadpool_limits(limits=1):
        result = CooperativeMonteCarloSimulator(
            model,
            _benchmark_config(),
            sampler=_sampler(model, test_dataset),
            execution_config=execution,
            metadata={"experiment_arm": "single_gaussian_to_real_benchmark"},
        ).run()
    if len(result.subsets) != 31:
        raise AssertionError(f"expected 31 subsets, got {len(result.subsets)}")
    if len(result.classifier_names) != 3:
        raise AssertionError(
            f"expected 3 classifiers, got {len(result.classifier_names)}"
        )
    if result.classifier_names != available_classifiers():
        raise AssertionError("benchmark classifier order is not canonical")
    if any(
        info.replications != REPLICATIONS
        for info in result.stopping_info.values()
    ):
        raise AssertionError("benchmark did not execute exactly 40 replications")
    print(f"Completed in {result.runtime_seconds:.3f} s", flush=True)
    return result


def _assert_exact(reference: SimulationResult, candidate: SimulationResult) -> None:
    if reference.sample_sizes != candidate.sample_sizes:
        raise AssertionError("sample-size grids differ")
    if reference.subsets != candidate.subsets:
        raise AssertionError("subset catalogs differ")
    if reference.classifier_names != candidate.classifier_names:
        raise AssertionError("classifier catalogs differ")
    if reference.stopping_info != candidate.stopping_info:
        raise AssertionError("stopping information differs")
    for n_per_class in reference.sample_sizes:
        for subset in reference.subsets:
            for classifier_name in reference.classifier_names:
                expected = reference.accumulator.losses(
                    n_per_class, subset, classifier_name
                )
                actual = candidate.accumulator.losses(
                    n_per_class, subset, classifier_name
                )
                if not np.array_equal(expected, actual):
                    raise AssertionError(
                        "loss arrays differ at "
                        f"n_per_class={n_per_class}, subset={subset}, "
                        f"classifier={classifier_name}"
                    )


def _process_config(n_jobs: int) -> ExecutionConfig:
    return ExecutionConfig(
        backend="process",
        n_jobs=n_jobs,
        worker_inner_threads=1,
        start_method="forkserver",
    )


def _record_result(
    records: dict[int, BenchmarkRecord],
    result: SimulationResult,
    exact: bool,
) -> None:
    execution = result.metadata["execution"]
    requested = int(execution["requested_workers"])
    record = records.setdefault(
        requested,
        BenchmarkRecord(
            backend=str(execution["backend"]),
            requested_workers=requested,
            effective_workers=int(execution["effective_workers"]),
        ),
    )
    record.runtimes.append(float(result.runtime_seconds))
    record.exact_results.append(bool(exact))


def _print_table(records: Sequence[BenchmarkRecord]) -> None:
    sequential_runtime = records[0].runtime
    print()
    print(
        "| Backend | Requested workers | Effective workers | Passes | "
        "Runtime (s, mean) | Speedup | Efficiency | Exact equality |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for record in records:
        speedup = sequential_runtime / record.runtime
        efficiency = speedup / record.effective_workers
        print(
            f"| {record.backend} | {record.requested_workers} | "
            f"{record.effective_workers} | {len(record.runtimes)} | "
            f"{record.runtime:.3f} | {speedup:.3f}x | "
            f"{efficiency:.3f} | {'yes' if record.exact else 'no'} |"
        )


def run_benchmark(raw_dir: str) -> list[BenchmarkRecord]:
    data = load_occupancy_data(raw_dir)
    anchored = build_gaussian_anchored_occupancy_model(data)
    model = anchored.model
    test_dataset = data.test_dataset

    sequential = _run_once(model, test_dataset, ExecutionConfig())
    by_workers: dict[int, BenchmarkRecord] = {}
    _record_result(by_workers, sequential, exact=True)

    process_results = {}
    for n_jobs in PROCESS_WORKERS:
        result = _run_once(model, test_dataset, _process_config(n_jobs))
        _assert_exact(sequential, result)
        process_results[n_jobs] = result
        _record_result(by_workers, result, exact=True)

    runtime_five = process_results[5].runtime_seconds
    runtime_six = process_results[6].runtime_seconds
    if abs(runtime_five - runtime_six) / runtime_five < REPEAT_THRESHOLD:
        print(
            "Five and six workers differ by less than 5%; "
            "running one additional pass for each.",
            flush=True,
        )
        for n_jobs in PROCESS_WORKERS:
            result = _run_once(model, test_dataset, _process_config(n_jobs))
            _assert_exact(sequential, result)
            _record_result(by_workers, result, exact=True)

    records = [by_workers[n_jobs] for n_jobs in (1, *PROCESS_WORKERS)]
    if not all(record.exact for record in records):
        raise AssertionError("exact equality must pass before reporting speedup")
    _print_table(records)

    five = by_workers[5]
    six = by_workers[6]
    if six.runtime <= five.runtime * (1.0 - REPEAT_THRESHOLD):
        recommendation = "6 workers (at least 5% faster than 5 workers)"
    else:
        recommendation = "5 workers (the initial six-core desktop default)"
    print(f"\nRecommended worker count: {recommendation}")
    print("Peak memory: not measured; no additional profiling tooling was used.")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/raw/occupancy",
        help="Directory containing the Occupancy raw data files.",
    )
    args = parser.parse_args()
    run_benchmark(args.raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
