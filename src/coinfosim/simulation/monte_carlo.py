"""
Cooperative Monte Carlo simulator for CoInfoSim.

:class:`CooperativeMonteCarloSimulator` runs the experiment loop::

    n_per_class -> replication -> subset -> classifier

For each ``n_per_class`` it accumulates empirical test-loss replications for
every (subset, classifier) pair, reusing one fixed test set, and applies the
standard-error stopping rule at replication batch boundaries.

The simulator computes empirical test loss only. It does *not* compute
empirical train loss, theoretical loss, or Bayes error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from coinfosim.classifiers.registry import available_classifiers
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.results.accumulator import LossAccumulator
from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.execution import (
    ExecutionConfig,
    SequentialReplicationExecutor,
)
from coinfosim.simulation.progress import CooperativeProgressReporter
from coinfosim.simulation.stopping import StandardErrorStoppingRule
from coinfosim.simulation.subsets import all_nonempty_subsets


@dataclass
class StoppingInfo:
    """Final stopping status recorded for one ``n_per_class``."""

    n_per_class: int
    replications: int
    reason: str  # "converged" or "max_budget"
    max_ci_half_width: float


@dataclass
class SimulationResult:
    """Structured result returned by the cooperative Monte Carlo simulator."""

    model: Any
    config: MonteCarloConfig
    subsets: List[Tuple[int, ...]]
    classifier_names: List[str]
    accumulator: LossAccumulator
    stopping_info: Dict[int, StoppingInfo]
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def sample_sizes(self) -> List[int]:
        return list(self.config.sample_sizes)


class MonteCarloSampler(Protocol):
    """Sampler interface required by :class:`CooperativeMonteCarloSimulator`."""

    @property
    def model(self) -> Any:
        ...

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        ...

    def sample_test(self) -> Dataset:
        ...


class CooperativeMonteCarloSimulator:
    """Run a cooperative Monte Carlo experiment over subsets and classifiers."""

    def __init__(
        self,
        model: Any,
        config: MonteCarloConfig,
        subsets: Optional[Sequence[Sequence[int]]] = None,
        classifier_names: Optional[Sequence[str]] = None,
        stopping_rule: Optional[StandardErrorStoppingRule] = None,
        sampler: Optional[MonteCarloSampler] = None,
        metadata: Optional[Mapping[str, object]] = None,
        progress: Optional[CooperativeProgressReporter] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.execution_config = execution_config or ExecutionConfig()
        self.extra_metadata = dict(metadata or {})
        # Optional console progress reporter. When ``None`` the simulator is
        # completely silent, which keeps tests and programmatic use quiet.
        self.progress = progress

        if subsets is None:
            subsets = all_nonempty_subsets(model.d)
        self.subsets: List[Tuple[int, ...]] = [tuple(s) for s in subsets]

        if classifier_names is None:
            classifier_names = available_classifiers()
        self.classifier_names: List[str] = list(classifier_names)

        if stopping_rule is None:
            stopping_rule = StandardErrorStoppingRule(
                min_replications=config.min_replications,
                max_replications=config.max_replications,
                ci_half_width_target=config.ci_half_width_target,
            )
        self.stopping_rule = stopping_rule

        if sampler is None:
            sampler = GaussianClassConditionalSampler(
                model,
                base_seed=config.base_seed,
                test_samples_per_class=config.test_samples_per_class,
            )
        self.sampler = sampler

    def _cells(self) -> List[Tuple[Tuple[int, ...], str]]:
        return [
            (subset, clf)
            for subset in self.subsets
            for clf in self.classifier_names
        ]

    def run(self) -> SimulationResult:
        """Execute the full experiment and return a :class:`SimulationResult`."""
        if self.execution_config.backend != "sequential":
            raise NotImplementedError(
                "process execution is not implemented until Block 4"
            )

        start = time.time()
        accumulator = LossAccumulator()
        stopping_info: Dict[int, StoppingInfo] = {}
        cells = self._cells()

        # Fixed test set, generated once and reused everywhere.
        test_dataset = self.sampler.sample_test()
        # Pre-restrict the fixed test set per subset (test set is constant).
        test_by_subset = {
            subset: test_dataset.select_channels(subset) for subset in self.subsets
        }
        executor = SequentialReplicationExecutor(
            sampler=self.sampler,
            cells=cells,
            test_by_subset=test_by_subset,
        )

        if self.progress is not None:
            self.progress.simulation_start(
                arm=str(self.extra_metadata.get("experiment_arm", "unknown")),
                n_sample_sizes=len(self.config.sample_sizes),
                n_subsets=len(self.subsets),
                n_classifiers=len(self.classifier_names),
                n_cells=len(cells),
                fixed_test_size=test_dataset.n_samples,
                sample_sizes=list(self.config.sample_sizes),
            )

        total_sample_sizes = len(self.config.sample_sizes)
        for size_index, n_per_class in enumerate(self.config.sample_sizes, start=1):
            size_start = time.time()
            if self.progress is not None:
                self.progress.sample_size_start(
                    n_per_class, size_index, total_sample_sizes
                )
            replication_id = 0
            last_decision = None

            while True:
                # Run one batch of replications.
                batch_start = replication_id
                batch_end = replication_id + self.config.replication_batch_size
                replication_ids = range(batch_start, batch_end)
                batch_results = executor.run_batch(
                    n_per_class=n_per_class,
                    replication_ids=replication_ids,
                )
                replication_id = batch_end

                accumulator.add_batch(
                    n_per_class=n_per_class,
                    expected_replication_ids=replication_ids,
                    cells=cells,
                    results=batch_results,
                )

                # Evaluate stopping rule at the batch boundary.
                last_decision = self.stopping_rule.evaluate(
                    accumulator, n_per_class, cells
                )
                if self.progress is not None:
                    self.progress.batch_finish(
                        n_per_class,
                        last_decision.replications,
                        last_decision.max_ci_half_width,
                    )
                if last_decision.should_stop:
                    break

            stopping_info[n_per_class] = StoppingInfo(
                n_per_class=n_per_class,
                replications=last_decision.replications,
                reason=last_decision.reason,
                max_ci_half_width=last_decision.max_ci_half_width,
            )
            if self.progress is not None:
                self.progress.sample_size_finish(
                    n_per_class,
                    last_decision.replications,
                    last_decision.reason,
                    last_decision.max_ci_half_width,
                    elapsed=time.time() - size_start,
                )

        runtime = time.time() - start
        if self.progress is not None:
            self.progress.simulation_finish(runtime)

        channel_names = getattr(self.model, "channel_names", None)
        metadata = {
            "mode": self.config.mode,
            "base_seed": self.config.base_seed,
            "test_samples_per_class": self.config.test_samples_per_class,
            "fixed_test_size": test_dataset.n_samples,
            "ci_half_width_target": self.config.ci_half_width_target,
            "min_replications": self.config.min_replications,
            "max_replications": self.config.max_replications,
            "replication_batch_size": self.config.replication_batch_size,
            "metric": "empirical_test_loss",
            "d": self.model.d,
            "class_labels": list(self.model.class_labels),
        }
        if channel_names is not None:
            metadata["channel_names"] = list(channel_names)
        model_name = getattr(self.model, "name", None)
        if model_name is not None:
            metadata["model_name"] = model_name
        metadata.update(self.extra_metadata)

        return SimulationResult(
            model=self.model,
            config=self.config,
            subsets=self.subsets,
            classifier_names=self.classifier_names,
            accumulator=accumulator,
            stopping_info=stopping_info,
            runtime_seconds=runtime,
            metadata=metadata,
        )
