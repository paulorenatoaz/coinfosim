"""
Monte Carlo budget and execution-mode configuration for CoInfoSim.

Provides :class:`MonteCarloConfig`, :func:`get_mode_config`, and
:func:`resolve_sample_sizes_for_training_capacity` for the ``smoke``,
``fast``, ``full``, ``full-scale``, and ``strict`` execution modes.

Mode CI half-width targets
--------------------------
=======  =======================  ====================================
Mode     ci_half_width_target     Intended use
=======  =======================  ====================================
smoke    0.05                     Quick pipeline check / validation
fast     0.03                     Exploratory run with moderate precision
full     0.01                     Serious analysis
full-scale 0.01                   Dataset-capacity serious analysis
strict   0.005                    High-precision / paper-grade run
=======  =======================  ====================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for a Monte Carlo run.

    Attributes
    ----------
    mode:
        Execution-mode name (``smoke``, ``fast``, ``full``, ``full-scale``,
        or ``strict``).
    sample_sizes:
        Tuple of ``n_per_class`` values to evaluate.
    min_replications:
        Minimum replications before the stopping rule may trigger.
    max_replications:
        Maximum replications per ``n_per_class`` (budget cap).
    replication_batch_size:
        Number of replications between stopping-rule evaluations.
    test_samples_per_class:
        Size of the fixed test set per class.
    ci_half_width_target:
        Target 95% CI half-width for convergence.
    base_seed:
        Base random seed for reproducibility.
    """

    mode: str
    sample_sizes: Tuple[int, ...]
    min_replications: int
    max_replications: int
    replication_batch_size: int
    test_samples_per_class: int
    ci_half_width_target: float
    base_seed: int = 0

    def __post_init__(self) -> None:
        if len(self.sample_sizes) == 0:
            raise ValueError("sample_sizes must be non-empty")
        if any(n <= 0 for n in self.sample_sizes):
            raise ValueError("sample_sizes must be positive")
        if self.min_replications < 2:
            raise ValueError("min_replications must be at least 2")
        if self.max_replications < self.min_replications:
            raise ValueError("max_replications must be >= min_replications")
        if self.replication_batch_size <= 0:
            raise ValueError("replication_batch_size must be positive")
        if self.test_samples_per_class <= 0:
            raise ValueError("test_samples_per_class must be positive")
        if self.ci_half_width_target <= 0:
            raise ValueError("ci_half_width_target must be positive")


# Shared statistical budget for the fixed and dataset-aware full modes.
_FULL_PRESET = dict(
    sample_sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512),
    min_replications=100,
    max_replications=2000,
    replication_batch_size=20,
    test_samples_per_class=5000,
    ci_half_width_target=0.01,
    base_seed=0,
)


# Preset definitions for each execution mode.
_MODE_PRESETS: Dict[str, dict] = {
    "smoke": dict(
        sample_sizes=(2, 4, 8, 16, 32),
        min_replications=10,
        max_replications=40,
        replication_batch_size=5,
        test_samples_per_class=200,
        ci_half_width_target=0.05,
        base_seed=0,
    ),
    "fast": dict(
        sample_sizes=(2, 4, 8, 16, 32, 64, 128),
        min_replications=30,
        max_replications=300,
        replication_batch_size=10,
        test_samples_per_class=1000,
        ci_half_width_target=0.03,
        base_seed=0,
    ),
    "full": _FULL_PRESET,
    "full-scale": _FULL_PRESET,
    "strict": dict(
        sample_sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512),
        min_replications=100,
        max_replications=2000,
        replication_batch_size=20,
        test_samples_per_class=5000,
        ci_half_width_target=0.005,
        base_seed=0,
    ),
}

VALID_MODES = tuple(_MODE_PRESETS.keys())


def get_mode_config(mode: str) -> MonteCarloConfig:
    """Return the :class:`MonteCarloConfig` preset for ``mode``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``smoke``, ``fast``, ``full``,
        ``full-scale``, ``strict``.
    """
    if mode not in _MODE_PRESETS:
        raise ValueError(
            f"unknown mode {mode!r}; valid modes: {list(VALID_MODES)}"
        )
    return MonteCarloConfig(mode=mode, **_MODE_PRESETS[mode])


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and value & (value - 1) == 0


def resolve_sample_sizes_for_training_capacity(
    config: MonteCarloConfig,
    minority_class_count: int,
) -> MonteCarloConfig:
    """Resolve a full-scale power-of-two grid for training capacity.

    Configurations for all other modes are returned unchanged.  The
    full-scale base grid must start at two and double at each step.
    """
    if config.mode != "full-scale":
        return config
    if minority_class_count < 2:
        raise ValueError("minority_class_count must be at least 2")

    base_sizes = config.sample_sizes
    valid_base = (
        base_sizes[0] == 2
        and all(_is_power_of_two(value) for value in base_sizes)
        and all(
            current > previous and current == 2 * previous
            for previous, current in zip(base_sizes, base_sizes[1:])
        )
    )
    if not valid_base:
        raise ValueError(
            "full-scale sample_sizes must start at 2 and double using "
            "strictly increasing powers of two"
        )

    resolved_sizes = [
        value for value in base_sizes if value <= minority_class_count
    ]
    while 2 * resolved_sizes[-1] <= minority_class_count:
        resolved_sizes.append(2 * resolved_sizes[-1])

    return replace(config, sample_sizes=tuple(resolved_sizes))
