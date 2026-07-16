"""Tests for coinfosim.scenarios.catalog."""

from __future__ import annotations

import subprocess
import sys

import pytest

from coinfosim.scenarios.catalog import (
    UnknownScenarioError,
    get_scenario,
    list_scenarios,
)


def test_list_scenarios_returns_exactly_three_canonical_slugs():
    slugs = {scenario.slug for scenario in list_scenarios()}
    assert slugs == {"occupancy", "air-quality", "support2"}


def test_no_numeric_scenario_identifiers_exposed():
    for scenario in list_scenarios():
        assert not scenario.slug.isdigit()
        assert all(not alias.isdigit() for alias in scenario.aliases)


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
    ],
)
def test_get_scenario_resolves_aliases(alias, expected):
    assert get_scenario(alias).slug == expected


def test_get_scenario_unknown_raises_typed_error():
    with pytest.raises(UnknownScenarioError):
        get_scenario("does-not-exist")


def test_occupancy_supports_dataset_report_only():
    assert get_scenario("occupancy").supports_dataset_report_only is True


def test_air_quality_and_support2_do_not_support_dataset_report_only():
    assert get_scenario("air-quality").supports_dataset_report_only is False
    assert get_scenario("support2").supports_dataset_report_only is False


def test_dataset_slug_matches_dataset_catalog_slug():
    from coinfosim.datasets.catalog import get_dataset

    for scenario in list_scenarios():
        # Must resolve in the dataset catalog too (raises if not).
        dataset = get_dataset(scenario.dataset_slug)
        assert dataset.slug == scenario.dataset_slug


def test_occupancy_spec_resolves_to_real_execution_spec():
    scenario = get_scenario("occupancy")
    spec = scenario.spec
    assert spec.scenario_slug == "occupancy_baseline"
    assert spec.dataset_slug == "occupancy"


def test_air_quality_spec_resolves_to_real_execution_spec():
    scenario = get_scenario("air-quality")
    spec = scenario.spec
    assert spec.scenario_slug == "air_quality_baseline"


def test_support2_spec_resolves_with_default_classifier_configuration():
    scenario = get_scenario("support2")
    spec = scenario.spec
    assert spec.scenario_slug == "support2_baseline"
    assert spec.classifier_names == ("linear_svm", "random_forest")
    config = scenario.default_classifier_configuration.resolve()
    assert config["classifier_names"] == ("linear_svm", "random_forest")


def test_occupancy_and_air_quality_have_no_default_classifier_configuration():
    assert get_scenario("occupancy").default_classifier_configuration is None
    assert get_scenario("air-quality").default_classifier_configuration is None


def test_listing_scenarios_does_not_import_matplotlib_or_sklearn():
    """Regression guard for the lazy-spec design: listing must stay cheap."""

    probe = (
        "import sys\n"
        "from coinfosim.scenarios.catalog import list_scenarios, get_scenario\n"
        "scenarios = list_scenarios()\n"
        "assert len(scenarios) == 3\n"
        "get_scenario('occupancy')\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib should not be imported by catalog listing'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
