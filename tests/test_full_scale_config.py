"""Tests for dataset-capacity resolution of the full-scale mode."""

from dataclasses import replace

import pytest

from coinfosim.simulation.config import (
    get_mode_config,
    resolve_sample_sizes_for_training_capacity,
)


@pytest.mark.parametrize(
    ("capacity", "expected"),
    [
        (2, (2,)),
        (3, (2,)),
        (4, (2, 4)),
        (400, (2, 4, 8, 16, 32, 64, 128, 256)),
        (512, (2, 4, 8, 16, 32, 64, 128, 256, 512)),
        (700, (2, 4, 8, 16, 32, 64, 128, 256, 512)),
        (1024, (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)),
        (1729, (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)),
        (1818, (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)),
        (2048, (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)),
        (2055, (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)),
    ],
)
def test_resolve_full_scale_sample_sizes(capacity, expected):
    config = get_mode_config("full-scale")

    resolved = resolve_sample_sizes_for_training_capacity(config, capacity)

    assert resolved.sample_sizes == expected
    assert resolved is not config
    max_n = max(resolved.sample_sizes)
    assert max_n <= capacity < 2 * max_n


def test_non_full_scale_config_is_returned_unchanged():
    config = get_mode_config("full")

    resolved = resolve_sample_sizes_for_training_capacity(config, 2048)

    assert resolved is config


@pytest.mark.parametrize("capacity", [0, 1])
def test_full_scale_rejects_insufficient_capacity(capacity):
    with pytest.raises(ValueError, match="at least 2"):
        resolve_sample_sizes_for_training_capacity(
            get_mode_config("full-scale"), capacity
        )


@pytest.mark.parametrize(
    "sample_sizes",
    [
        (4, 8),
        (2, 4, 3),
        (2, 8),
        (2, 4, 4),
        (2, 4, 16, 8),
    ],
)
def test_full_scale_rejects_invalid_base_grid(sample_sizes):
    config = replace(
        get_mode_config("full-scale"), sample_sizes=sample_sizes
    )

    with pytest.raises(ValueError, match="start at 2 and double"):
        resolve_sample_sizes_for_training_capacity(config, 64)


def test_resolver_does_not_mutate_input_config():
    config = get_mode_config("full-scale")
    original_sizes = config.sample_sizes

    resolved = resolve_sample_sizes_for_training_capacity(config, 2048)

    assert config.sample_sizes == original_sizes
    assert resolved.sample_sizes[-1] == 2048
