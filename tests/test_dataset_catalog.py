"""Tests for the packaged, pinned dataset catalog."""

from __future__ import annotations

import os

import pytest

from coinfosim.datasets.catalog import (
    DatasetCatalogError,
    UnknownDatasetError,
    get_dataset,
    list_datasets,
    load_dataset_catalog,
    pages_base_url,
)

EXPECTED_FILENAMES = {
    "occupancy": ("datatraining.txt", "datatest.txt", "datatest2.txt"),
    "air-quality": ("AirQualityUCI.csv",),
    "support2": ("support2.csv",),
}


def test_load_dataset_catalog_returns_three_builtin_datasets():
    catalog = load_dataset_catalog()
    assert set(catalog) == {"occupancy", "air-quality", "support2"}


def test_list_datasets_returns_all_three():
    slugs = {dataset.slug for dataset in list_datasets()}
    assert slugs == {"occupancy", "air-quality", "support2"}


@pytest.mark.parametrize("slug", ["occupancy", "air-quality", "support2"])
def test_dataset_definitions_have_expected_filenames(slug):
    dataset = get_dataset(slug)
    assert dataset.filenames == EXPECTED_FILENAMES[slug]


def test_dataset_files_have_unique_filenames():
    for dataset in list_datasets():
        assert len(set(dataset.filenames)) == len(dataset.filenames)


def test_dataset_file_hashes_are_64_lowercase_hex_chars():
    for dataset in list_datasets():
        for file in dataset.files:
            assert len(file.sha256) == 64
            assert file.sha256 == file.sha256.lower()
            int(file.sha256, 16)  # raises ValueError if not hex


def test_dataset_file_urls_use_https_pages_prefix():
    base = pages_base_url()
    assert base.startswith("https://")
    for dataset in list_datasets():
        for file in dataset.files:
            assert file.url.startswith(base + "/")


def test_dataset_file_sizes_are_positive():
    for dataset in list_datasets():
        for file in dataset.files:
            assert file.size_bytes > 0


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("occupancy", "occupancy"),
        ("occupancy-detection", "occupancy"),
        ("occupancy_detection", "occupancy"),
        ("air-quality", "air-quality"),
        ("air_quality", "air-quality"),
        ("airquality", "air-quality"),
        ("support2", "support2"),
        ("support-2", "support2"),
        ("SUPPORT2", "support2"),
    ],
)
def test_get_dataset_resolves_aliases(alias, expected):
    assert get_dataset(alias).slug == expected


def test_get_dataset_unknown_slug_raises_typed_error():
    with pytest.raises(UnknownDatasetError):
        get_dataset("does-not-exist")


def test_air_quality_and_air_quality_alias_are_identical_object_after_install():
    assert get_dataset("air-quality") == get_dataset("air_quality")


def test_support2_license_does_not_claim_cc_by():
    support2 = get_dataset("support2")
    assert "CC BY" not in support2.license.name
    assert "acknowledgment" in support2.license.name.lower()


def test_occupancy_and_air_quality_license_is_cc_by_4_0():
    assert get_dataset("occupancy").license.name == "CC BY 4.0"
    assert get_dataset("air-quality").license.name == "CC BY 4.0"


def test_catalog_loading_does_not_depend_on_source_checkout_cwd(tmp_path, monkeypatch):
    load_dataset_catalog.cache_clear()
    monkeypatch.chdir(tmp_path)
    assert os.getcwd() == str(tmp_path)
    catalog = load_dataset_catalog()
    assert set(catalog) == {"occupancy", "air-quality", "support2"}
    load_dataset_catalog.cache_clear()


def test_catalog_error_types_are_exceptions():
    assert issubclass(DatasetCatalogError, Exception)
    assert issubclass(UnknownDatasetError, KeyError)
