"""Integration tests for Occupancy scenario/simulation run tracking.

These use a tiny Monte Carlo config so the full orchestration (registries,
folders, persistence, report generation, regeneration) runs quickly. They
require the Occupancy raw data files under ``data/raw/occupancy``.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from coinfosim.simulation.config import MonteCarloConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "occupancy"


def _load_script_module():
    script = REPO_ROOT / "scripts" / "run_occupancy_scenario.py"
    spec = importlib.util.spec_from_file_location("run_occupancy_scenario", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_occupancy_scenario"] = module
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "datatraining.txt").exists(),
    reason="Occupancy raw data files are not available",
)


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4,),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=30,
        ci_half_width_target=0.05,
        base_seed=3,
    )


def test_occupancy_run_creates_one_scenario_two_simulations(tmp_path):
    mod = _load_script_module()
    out = mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )

    assert out["scenario_run_id"] == 0
    assert out["real_simulation_run_id"] == 0
    assert out["single_gaussian_to_real_simulation_run_id"] == 1

    # Global registries exist.
    assert (tmp_path / "scenario_runs.json").exists()
    assert (tmp_path / "simulation_runs.json").exists()

    # Dedicated, id-stamped run directories.
    scenario_dir = tmp_path / "scenarios" / "000000_occupancy_baseline_smoke"
    real_dir = tmp_path / "simulations" / "000000_occupancy_real_data_smoke"
    gaussian_dir = (
        tmp_path
        / "simulations"
        / "000001_occupancy_single_gaussian_to_real_smoke"
    )
    assert scenario_dir.is_dir()
    assert real_dir.is_dir()
    assert gaussian_dir.is_dir()

    # Filenames include mode and zero-padded id.
    assert (
        scenario_dir / "occupancy_baseline_scenario_report_smoke_000000.html"
    ).exists()
    assert (
        real_dir
        / "occupancy_real_data_monte_carlo_report_smoke_000000.html"
    ).exists()
    assert (real_dir / "result_data_smoke_000000.json.gz").exists()
    assert (real_dir / "summary_smoke_000000.json").exists()
    assert (real_dir / "simulation.json").exists()
    assert (
        gaussian_dir
        / "occupancy_single_gaussian_to_real_monte_carlo_report_smoke_000001.html"
    ).exists()
    assert (gaussian_dir / "result_data_smoke_000001.json.gz").exists()
    assert (gaussian_dir / "simulation.json").exists()


def test_scenario_json_has_question_and_simulation_refs(tmp_path):
    mod = _load_script_module()
    mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )
    scenario_json = json.loads(
        (
            tmp_path
            / "scenarios"
            / "000000_occupancy_baseline_smoke"
            / "scenario.json"
        ).read_text(encoding="utf-8")
    )

    assert "cooperative structure" in scenario_json["question"]
    assert scenario_json["scenario_family"] == "dataset"
    assert scenario_json["status"] == "completed"
    assert scenario_json["simulation_run_ids"] == [0, 1]
    refs = scenario_json["simulation_refs"]
    assert "occupancy_real_data" in refs
    assert "occupancy_single_gaussian_to_real" in refs
    assert refs["occupancy_single_gaussian_to_real"]["simulation_family"] == (
        "single_gaussian_to_real"
    )
    # Scenario-level report data snapshots both main arms with clear semantics.
    assert set(scenario_json["report_data"]["arms"]) == {
        "real_to_real",
        "single_gaussian_to_real",
    }
    sgr_arm = scenario_json["report_data"]["arms"]["single_gaussian_to_real"]
    assert sgr_arm["train_source"] == "single_gaussian_synthetic"
    assert sgr_arm["test_source"] == "real_occupancy_evaluation_split"
    real_arm = scenario_json["report_data"]["arms"]["real_to_real"]
    assert real_arm["train_source"] == "real_occupancy_training_pool"
    assert real_arm["test_source"] == "real_occupancy_evaluation_split"


def test_simulation_json_has_report_ready_data(tmp_path):
    mod = _load_script_module()
    mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )
    sim_json = json.loads(
        (
            tmp_path
            / "simulations"
            / "000000_occupancy_real_data_smoke"
            / "simulation.json"
        ).read_text(encoding="utf-8")
    )

    assert sim_json["simulation_family"] == "real_dataset"
    assert sim_json["status"] == "completed"
    assert sim_json["summary_data"]["number_of_subsets"] == 31
    assert sim_json["summary_data"]["number_of_classifiers"] == 3
    # Report-ready tables are embedded.
    assert sim_json["result_data"]["summary_table"]
    assert sim_json["result_data"]["best_subset_rankings"]
    assert sim_json["result_data"]["threshold_comparisons"]
    # Pointer to full persisted result payload.
    assert sim_json["artifacts"]["result_data"].endswith(".json.gz")


def test_second_run_does_not_overwrite_first(tmp_path):
    mod = _load_script_module()
    first = mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )
    second = mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )

    assert first["scenario_run_id"] == 0
    assert second["scenario_run_id"] == 1
    assert first["scenario_run_dir"] != second["scenario_run_dir"]
    # First run's artifacts are still present.
    assert (
        tmp_path
        / "scenarios"
        / "000000_occupancy_baseline_smoke"
        / "scenario.json"
    ).exists()
    assert (
        tmp_path
        / "scenarios"
        / "000001_occupancy_baseline_smoke"
        / "scenario.json"
    ).exists()


def test_regeneration_does_not_rerun_monte_carlo(tmp_path, monkeypatch):
    mod = _load_script_module()
    mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=False,
    )

    # Any attempt to build a simulator during regeneration must fail loudly.
    def _boom(*args, **kwargs):
        raise AssertionError("Monte Carlo must not run during regeneration")

    monkeypatch.setattr(mod, "CooperativeMonteCarloSimulator", _boom)

    regenerated = mod.regenerate_from_scenario_run(0, output_dir=str(tmp_path))

    assert Path(regenerated["scenario_report"]).exists()
    assert Path(regenerated["real_report"]).exists()
    assert Path(regenerated["single_gaussian_to_real_report"]).exists()

    # Registries were not extended by regeneration.
    scenario_runs = json.loads(
        (tmp_path / "scenario_runs.json").read_text(encoding="utf-8")
    )
    assert scenario_runs["next_scenario_run_id"] == 1


def test_visualization_panels_written_and_registered(tmp_path):
    mod = _load_script_module()
    mod.run_scenario(
        raw_dir=str(RAW_DIR),
        output_dir=str(tmp_path),
        config=_tiny_config(),
        visualize=True,
    )
    scenario_dir = tmp_path / "scenarios" / "000000_occupancy_baseline_smoke"

    # Six visualization PNGs are written in the scenario folder.
    expected = [
        "viz_1d_real_smoke_000000.png",
        "viz_1d_gaussian_smoke_000000.png",
        "viz_2d_real_smoke_000000.png",
        "viz_2d_gaussian_smoke_000000.png",
        "viz_3d_real_smoke_000000.png",
        "viz_3d_gaussian_smoke_000000.png",
    ]
    for name in expected:
        assert (scenario_dir / name).exists(), name

    # scenario.json registers the visualization metadata and image paths.
    scenario_json = json.loads(
        (scenario_dir / "scenario.json").read_text(encoding="utf-8")
    )
    viz = scenario_json["report_data"]["visualization"]
    assert viz["metadata"]["visualization_seed"] == mod.VIZ_SEED
    assert set(viz["images"]) == set(scenario_json["report_data"]["visualization"]["images"])
    assert "visualization_images" in scenario_json["artifacts"]

    # Loss-vs-N graphs are generated and registered.
    graphs = scenario_json["report_data"]["graphs"]
    assert "graph_best_comparison_real" in graphs
    assert "graph_best_comparison_sgr" in graphs
    assert any(k.startswith("graph_topranked_") for k in graphs)
    assert any(k.startswith("graph_nstar_") for k in graphs)
    assert "graph_images" in scenario_json["artifacts"]
    for fname in graphs.values():
        assert (scenario_dir / fname).exists(), fname

    # The scenario report links the PNGs.
    report_text = (
        scenario_dir / "occupancy_baseline_scenario_report_smoke_000000.html"
    ).read_text(encoding="utf-8")
    assert "viz_1d_real_smoke_000000.png" in report_text
    assert "viz_3d_gaussian_smoke_000000.png" in report_text
    assert "class='carousel'" in report_text
    assert "graph_best_comparison_real_smoke_000000.png" in report_text


