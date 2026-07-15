"""Loader-only tests for full-scale dataset-capacity resolution."""

from pathlib import Path

import pytest

from coinfosim.datasets.air_quality import load_air_quality_data
from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.scenarios.dataset_anchored_runner import (
    _config_dict,
    _training_class_counts,
)
from coinfosim.simulation.config import (
    get_mode_config,
    resolve_sample_sizes_for_training_capacity,
)


EXPECTED_GRID = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@pytest.mark.parametrize(
    ("loader", "raw_dir", "expected_counts"),
    [
        (
            load_occupancy_data,
            Path("data/raw/occupancy"),
            {"0": 6414, "1": 1729},
        ),
        (
            load_air_quality_data,
            Path("data/raw/air_quality"),
            {"0": 5374, "1": 1818},
        ),
    ],
)
def test_current_dataset_full_scale_resolution(
    loader, raw_dir, expected_counts
):
    data = loader(raw_dir)
    counts = _training_class_counts(data)
    minority_count = min(counts.values())
    requested = get_mode_config("full-scale")

    resolved = resolve_sample_sizes_for_training_capacity(
        requested, minority_count
    )
    payload = _config_dict(
        resolved,
        requested_sample_sizes=requested.sample_sizes,
        training_class_counts=counts,
    )

    assert counts == expected_counts
    assert resolved.sample_sizes == EXPECTED_GRID
    assert len(resolved.sample_sizes) == 10
    assert payload["mode"] == "full-scale"
    assert payload["sample_sizes"] == list(EXPECTED_GRID)
    assert payload["requested_sample_sizes"] == [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    assert payload["sample_size_strategy"] == (
        "powers_of_two_up_to_training_minority"
    )
    assert payload["training_class_counts"] == expected_counts
    assert payload["training_minority_class_count"] == minority_count
    assert payload["resolved_max_n_per_class"] == 1024


def test_fixed_mode_config_payload_schema_is_unchanged():
    config = get_mode_config("full")

    assert _config_dict(config) == {
        "mode": "full",
        "sample_sizes": [2, 4, 8, 16, 32, 64, 128, 256, 512],
        "min_replications": 100,
        "max_replications": 2000,
        "replication_batch_size": 20,
        "test_samples_per_class": 5000,
        "ci_half_width_target": 0.01,
        "base_seed": 0,
    }
