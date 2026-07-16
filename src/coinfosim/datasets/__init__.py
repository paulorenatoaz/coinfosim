"""Dataset loaders for CoInfoSim scenarios.

Resolved lazily (PEP 562): ``coinfosim.datasets.support2`` transitively
imports scikit-learn and every loader imports pandas, and this package is
imported by the lightweight dataset catalog (``coinfosim.datasets.catalog``)
used by ``coinfosim dataset list`` and ``coinfosim scenario list``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AIR_QUALITY_CHANNELS",
    "AIR_QUALITY_RAW_FILENAME",
    "AIR_QUALITY_REFERENCE",
    "AIR_QUALITY_TARGET",
    "AirQualityData",
    "DatasetAnchoredData",
    "OCCUPANCY_CHANNELS",
    "OCCUPANCY_RAW_FILENAMES",
    "OccupancyData",
    "StandardizationParameters",
    "SUPPORT2_CHANNELS",
    "SUPPORT2_RAW_FILENAME",
    "SUPPORT2_TARGET",
    "Support2Data",
    "load_air_quality_data",
    "load_occupancy_data",
    "load_support2_data",
]

_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "AIR_QUALITY_CHANNELS": ("coinfosim.datasets.air_quality", "AIR_QUALITY_CHANNELS"),
    "AIR_QUALITY_RAW_FILENAME": ("coinfosim.datasets.air_quality", "AIR_QUALITY_RAW_FILENAME"),
    "AIR_QUALITY_REFERENCE": ("coinfosim.datasets.air_quality", "AIR_QUALITY_REFERENCE"),
    "AIR_QUALITY_TARGET": ("coinfosim.datasets.air_quality", "AIR_QUALITY_TARGET"),
    "AirQualityData": ("coinfosim.datasets.air_quality", "AirQualityData"),
    "load_air_quality_data": ("coinfosim.datasets.air_quality", "load_air_quality_data"),
    "DatasetAnchoredData": ("coinfosim.datasets.common", "DatasetAnchoredData"),
    "StandardizationParameters": ("coinfosim.datasets.common", "StandardizationParameters"),
    "OCCUPANCY_CHANNELS": ("coinfosim.datasets.occupancy", "OCCUPANCY_CHANNELS"),
    "OCCUPANCY_RAW_FILENAMES": ("coinfosim.datasets.occupancy", "OCCUPANCY_RAW_FILENAMES"),
    "OccupancyData": ("coinfosim.datasets.occupancy", "OccupancyData"),
    "load_occupancy_data": ("coinfosim.datasets.occupancy", "load_occupancy_data"),
    "SUPPORT2_CHANNELS": ("coinfosim.datasets.support2", "SUPPORT2_CHANNELS"),
    "SUPPORT2_RAW_FILENAME": ("coinfosim.datasets.support2", "SUPPORT2_RAW_FILENAME"),
    "SUPPORT2_TARGET": ("coinfosim.datasets.support2", "SUPPORT2_TARGET"),
    "Support2Data": ("coinfosim.datasets.support2", "Support2Data"),
    "load_support2_data": ("coinfosim.datasets.support2", "load_support2_data"),
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
