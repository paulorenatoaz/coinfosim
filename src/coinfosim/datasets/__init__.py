"""Dataset loaders for CoInfoSim scenarios."""

from coinfosim.datasets.air_quality import (
    AIR_QUALITY_CHANNELS,
    AIR_QUALITY_RAW_FILENAME,
    AIR_QUALITY_REFERENCE,
    AIR_QUALITY_TARGET,
    AirQualityData,
    load_air_quality_data,
)
from coinfosim.datasets.common import DatasetAnchoredData, StandardizationParameters
from coinfosim.datasets.occupancy import (
    OCCUPANCY_CHANNELS,
    OCCUPANCY_RAW_FILENAMES,
    OccupancyData,
    load_occupancy_data,
)
from coinfosim.datasets.support2 import (
    SUPPORT2_CHANNELS,
    SUPPORT2_RAW_FILENAME,
    SUPPORT2_TARGET,
    Support2Data,
    load_support2_data,
)

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
