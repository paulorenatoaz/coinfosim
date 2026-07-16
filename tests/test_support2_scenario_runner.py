import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from coinfosim.results.persistence import load_simulation_result
from coinfosim.datasets.support2 import load_support2_data
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios import dataset_anchored_runner as runner
from coinfosim.scenarios.support2 import build_gmm_anchored_support2_model
from coinfosim.simulation.config import MonteCarloConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "support2"


def _load_script_module():
    script = REPO_ROOT / "scripts" / "run_support2_scenario.py"
    spec = importlib.util.spec_from_file_location("run_support2_scenario", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_support2_scenario"] = module
    spec.loader.exec_module(module)
    return module


def _tiny_config(sample_sizes=(2, 4)):
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=sample_sizes,
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=3,
    )


def _fast_spec(module):
    original_resolver = module.SUPPORT2_SPEC.classifier_configuration_resolver

    def reduced_forest(data, configuration):
        plan = original_resolver(data, configuration)
        plan.parameters["random_forest"]["n_estimators"] = 2
        plan.provenance["classifier_configurations"]["random_forest"][
            "parameters"
        ]["n_estimators"] = 2
        return plan

    return replace(
        module.SUPPORT2_SPEC,
        gmm_builder=lambda data: build_gmm_anchored_support2_model(
            data, max_components=1, min_points_per_component=50, n_init=1, random_state=0
        ),
        classifier_configuration_resolver=reduced_forest,
    )


def _strict_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)))


def test_support2_runner_persists_three_arms_shared_test_and_protocol(tmp_path, monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(module, "SUPPORT2_SPEC", _fast_spec(module))
    captured = []
    captured_plans = []
    original = runner.CooperativeMonteCarloSimulator

    class CapturingSimulator(original):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs["sampler"])
            captured_plans.append(kwargs["classifier_plan"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(runner, "CooperativeMonteCarloSimulator", CapturingSimulator)
    output = module.run_scenario(
        raw_dir=str(RAW_DIR), output_dir=str(tmp_path), config=_tiny_config(), visualize=False
    )
    assert [output[key] for key in (
        "scenario_run_id", "real_simulation_run_id",
        "single_gaussian_to_real_simulation_run_id", "gmm_to_real_simulation_run_id"
    )] == [0, 0, 1, 2]
    assert isinstance(captured[0], RealDatasetSampler)
    assert all(isinstance(value, SyntheticTrainRealTestSampler) for value in captured[1:])
    fixed = [sampler.sample_test() for sampler in captured]
    assert fixed[0] is fixed[1] is fixed[2]
    assert all(np.array_equal(value.X, fixed[0].X) for value in fixed)
    assert all(np.array_equal(value.y, fixed[0].y) for value in fixed)
    assert captured_plans[0] is captured_plans[1] is captured_plans[2]
    assert captured_plans[0].names == ("linear_svm", "random_forest")

    scenario = _strict_json(output["scenario_json"])
    assert scenario["scenario_slug"] == "support2_baseline"
    assert scenario["status"] == "completed"
    report_data = scenario["report_data"]
    assert report_data["structural_snapshot_policy"] == "regenerate_from_result_data"
    assert "structural_fidelity" not in report_data
    assert all(
        "structural_dynamics" not in arm["report_data"]
        for arm in report_data["arms"].values()
    )
    assert Path(output["scenario_json"]).stat().st_size < 5_000_000
    assert report_data["target"]["name"] == "death_180d"
    assert report_data["target"]["raw_class_counts"] == {"0": 4840, "1": 4265}
    assert report_data["target"]["cohort_class_counts"] == {"0": 4711, "1": 4162}
    assert report_data["preprocessing"]["ddof"] == 0
    assert report_data["preprocessing"]["fit_scope"] == "training_reservoir_only"
    assert report_data["split"]["id_fingerprints"] == {
        "cohort": "5c42d0c15c34abaad9e81dce0c1749e1001e66ca8b663d680cd37c6fecd7c59e",
        "train": "154809eb0f6759485342138c97f7ef7efc7d45bfc13d5bfdace19e366bea8979",
        "test": "74731ff933b9a19cb77dc4c859e797020c7168497befccf4669680c86037f7a7",
    }
    assert report_data["exclusions"]["target_related"] == ["death", "d.time", "hospdead"]
    for artifact in ("split_manifest", "target_metadata", "preprocessing_metadata"):
        assert Path(scenario["artifacts"][artifact]).exists()
        _strict_json(scenario["artifacts"][artifact])
    manifest = _strict_json(scenario["artifacts"]["split_manifest"])
    assert len(manifest["cohort_ids"]) == 8873
    assert len(manifest["train_ids"]) == 7098
    assert len(manifest["test_ids"]) == 1775

    for slug, ref in scenario["simulation_refs"].items():
        assert Path(ref["simulation_json_path"]).stat().st_size < 2_000_000
        result = load_simulation_result(ref["result_data_path"])
        assert len(result.subsets) == 127
        assert result.classifier_names == ["linear_svm", "random_forest"]
        assert result.metadata["classifier_selection"]["ordered_keys"] == [
            "linear_svm",
            "random_forest",
        ]
        rf = result.metadata["classifier_configurations"]["random_forest"]
        assert rf["parameters"]["n_jobs"] == 1
        assert rf["calibration"]["artifact_sha256"]
        assert result.metadata["fixed_test_size"] == 1775
        assert Path(ref["report_path"]).exists(), slug
    for artifact in ("dataset_report", "scenario_report"):
        assert Path(scenario["artifacts"][artifact]).exists()
    for path in tmp_path.rglob("*.json"):
        _strict_json(path)


def test_support2_regeneration_uses_persisted_results_without_monte_carlo_or_raw_data(
    tmp_path, monkeypatch
):
    module = _load_script_module()
    monkeypatch.setattr(module, "SUPPORT2_SPEC", _fast_spec(module))
    output = module.run_scenario(
        raw_dir=str(RAW_DIR), output_dir=str(tmp_path), config=_tiny_config(), visualize=False
    )
    scenario_before = _strict_json(output["scenario_json"])

    def fail(*args, **kwargs):
        raise AssertionError("Monte Carlo or raw loader ran during regeneration")

    monkeypatch.setattr(runner, "CooperativeMonteCarloSimulator", fail)
    monkeypatch.setattr(module, "SUPPORT2_SPEC", replace(module.SUPPORT2_SPEC, loader=fail))
    regenerated = module.regenerate_from_scenario_run(0, output_dir=str(tmp_path))
    for key in ("scenario_report", "real_report", "single_gaussian_to_real_report", "gmm_to_real_report"):
        assert Path(regenerated[key]).exists()
    scenario_after = _strict_json(output["scenario_json"])
    assert scenario_after["report_data"]["target"] == scenario_before["report_data"]["target"]
    assert scenario_after["report_data"]["preprocessing"] == scenario_before["report_data"]["preprocessing"]
    assert _strict_json(tmp_path / "scenario_runs.json")["next_scenario_run_id"] == 1
    assert _strict_json(tmp_path / "simulation_runs.json")["next_simulation_run_id"] == 3


def test_support2_n_per_class_3331_fails_before_expensive_work(tmp_path, monkeypatch):
    module = _load_script_module()

    def expensive(*args, **kwargs):
        raise AssertionError("expensive work started")

    monkeypatch.setattr(
        module,
        "SUPPORT2_SPEC",
        replace(
            _fast_spec(module),
            dataset_report_callback=expensive,
            gaussian_builder=expensive,
            gmm_builder=expensive,
        ),
    )
    with pytest.raises(ValueError, match="minority-class count 3330.*3331"):
        module.run_scenario(
            raw_dir=str(RAW_DIR),
            output_dir=str(tmp_path),
            config=_tiny_config(sample_sizes=(3331,)),
            visualize=False,
        )
    assert _strict_json(tmp_path / "simulation_runs.json")["next_simulation_run_id"] == 0


def test_support2_random_forest_only_plan_uses_calibration_artifact():
    module = _load_script_module()
    data = load_support2_data(RAW_DIR)
    plan = module._resolve_support2_classifier_configuration(
        data,
        {
            "classifier_names": ("random_forest",),
            "classifier_selection_source": "cli_or_api",
            "rf_calibration_file": str(
                REPO_ROOT / "config/calibration/support2_random_forest.json"
            ),
        },
    )

    assert plan.names == ("random_forest",)
    assert tuple(plan.parameters) == ("random_forest",)
    assert plan.parameters["random_forest"]["n_jobs"] == 1
    assert plan.provenance["classifier_selection"] == {
        "source": "cli_or_api",
        "ordered_keys": ["random_forest"],
    }
    assert plan.provenance["classifier_configurations"]["random_forest"][
        "calibration"
    ]["artifact_sha256"]


def test_support2_non_forest_selection_does_not_require_calibration_artifact():
    module = _load_script_module()
    data = load_support2_data(RAW_DIR)
    plan = module._resolve_support2_classifier_configuration(
        data,
        {
            "classifier_names": ("gaussian_nb", "linear_svm"),
            "rf_calibration_file": "does-not-exist.json",
        },
    )

    assert plan.names == ("gaussian_nb", "linear_svm")
    assert plan.parameters == {"gaussian_nb": {}, "linear_svm": {}}
    assert list(plan.provenance["classifier_configurations"]) == [
        "gaussian_nb",
        "linear_svm",
    ]


def test_support2_run_scenario_applies_classifier_override(monkeypatch):
    module = _load_script_module()
    captured = {}

    def fake_runner(spec, **kwargs):
        captured["spec"] = spec
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "run_dataset_anchored_scenario", fake_runner)
    module.run_scenario(classifier_names=("random_forest",))

    assert captured["spec"].classifier_names == ("random_forest",)
    assert captured["classifier_configuration"]["classifier_names"] == (
        "random_forest",
    )
    assert captured["classifier_configuration"][
        "classifier_selection_source"
    ] == "cli_or_api"


def test_support2_cli_accepts_random_forest_only(monkeypatch):
    module = _load_script_module()
    captured = {}

    def fake_run_scenario(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "run_scenario", fake_run_scenario)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_support2_scenario.py",
            "--quiet",
            "--classifiers",
            "random_forest",
            "--execution-backend",
            "process",
            "--n-jobs",
            "12",
        ],
    )

    assert module.main() == 0
    assert captured["classifier_names"] == ["random_forest"]
    assert captured["execution_config"].backend == "process"
    assert captured["execution_config"].n_jobs == 12
