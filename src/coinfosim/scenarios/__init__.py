"""CoInfoSim scenario definitions and builders."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "GaussianAnchoredOccupancyScenario",
    "OCCUPANCY_SCENARIO_NAME",
    "OCCUPANCY_SCENARIO_QUESTION",
    "SCENARIO_1_NAME",
    "SCENARIO_1_QUESTION",
    "SyntheticScenario",
    "build_gaussian_anchored_occupancy_model",
    "make_synthetic_scenario_1",
]

# Resolved lazily (PEP 562): coinfosim.scenarios.occupancy transitively
# imports scikit-learn, and this package is imported by every scenario
# submodule, including the lightweight catalog used by `coinfosim scenario
# list`.
_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "GaussianAnchoredOccupancyScenario": ("coinfosim.scenarios.occupancy", "GaussianAnchoredOccupancyScenario"),
    "OCCUPANCY_SCENARIO_NAME": ("coinfosim.scenarios.occupancy", "OCCUPANCY_SCENARIO_NAME"),
    "OCCUPANCY_SCENARIO_QUESTION": ("coinfosim.scenarios.occupancy", "OCCUPANCY_SCENARIO_QUESTION"),
    "build_gaussian_anchored_occupancy_model": ("coinfosim.scenarios.occupancy", "build_gaussian_anchored_occupancy_model"),
    "SCENARIO_1_NAME": ("coinfosim.scenarios.synthetic", "SCENARIO_1_NAME"),
    "SCENARIO_1_QUESTION": ("coinfosim.scenarios.synthetic", "SCENARIO_1_QUESTION"),
    "SyntheticScenario": ("coinfosim.scenarios.synthetic", "SyntheticScenario"),
    "make_synthetic_scenario_1": ("coinfosim.scenarios.synthetic", "make_synthetic_scenario_1"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRIBUTES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRIBUTES))
