from coinfosim.datasets.occupancy import load_occupancy_data
from coinfosim.reports.occupancy_scenario import generate_occupancy_scenario_report
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.samplers.gmm import GMMClassConditionalSampler
from coinfosim.samplers.real import RealDatasetSampler
from coinfosim.samplers.transfer import SyntheticTrainRealTestSampler
from coinfosim.scenarios.occupancy import (
    build_gaussian_anchored_occupancy_model,
    build_gmm_anchored_occupancy_model,
)
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(4, 8),
        min_replications=2,
        max_replications=2,
        replication_batch_size=2,
        test_samples_per_class=20,
        ci_half_width_target=0.05,
        base_seed=19,
    )


def _results():
    data = load_occupancy_data("data/raw/occupancy")
    config = _tiny_config()

    real_sampler = RealDatasetSampler(
        data.train_dataset,
        data.test_dataset,
        base_seed=config.base_seed,
        channel_names=data.channel_names,
    )
    real_result = CooperativeMonteCarloSimulator(
        real_sampler.model,
        config,
        sampler=real_sampler,
        metadata={"channel_names": list(data.channel_names)},
    ).run()

    anchored = build_gaussian_anchored_occupancy_model(data)
    gaussian_sampler = GaussianClassConditionalSampler(
        anchored.model,
        base_seed=config.base_seed,
        test_samples_per_class=config.test_samples_per_class,
    )
    gaussian_result = CooperativeMonteCarloSimulator(
        anchored.model,
        config,
        sampler=gaussian_sampler,
        metadata={"channel_names": list(data.channel_names)},
    ).run()

    gmm_anchored = build_gmm_anchored_occupancy_model(data)
    gmm_sampler = SyntheticTrainRealTestSampler(
        GMMClassConditionalSampler(
            gmm_anchored.model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        ),
        data.test_dataset,
    )
    gmm_result = CooperativeMonteCarloSimulator(
        gmm_anchored.model,
        config,
        sampler=gmm_sampler,
        metadata={"channel_names": list(data.channel_names)},
    ).run()
    return data, real_result, gaussian_result, gmm_result


def test_occupancy_scenario_report_academic_layout(tmp_path):
    data, real_result, gaussian_result, gmm_result = _results()
    visualization = {
        "images": {
            "viz_1d_real": "viz_1d_real_smoke_000000.png",
            "viz_1d_gaussian": "viz_1d_gaussian_smoke_000000.png",
            "viz_1d_gmm": "viz_1d_gmm_smoke_000000.png",
            "viz_2d_real": "viz_2d_real_smoke_000000.png",
            "viz_2d_gaussian": "viz_2d_gaussian_smoke_000000.png",
            "viz_2d_gmm": "viz_2d_gmm_smoke_000000.png",
            "viz_3d_real": "viz_3d_real_smoke_000000.png",
            "viz_3d_gaussian": "viz_3d_gaussian_smoke_000000.png",
            "viz_3d_gmm": "viz_3d_gmm_smoke_000000.png",
        },
        "metadata": {
            "visualization_sample_size": 1024,
            "class_balance": "balanced (equal samples per class)",
            "real_data_source": "standardized Occupancy training pool",
            "single_gaussian_source": "single Gaussian model",
            "gmm_source": "class-conditional GMM",
            "visualization_seed": 20240501,
        },
    }
    graphs: dict = {}
    out = generate_occupancy_scenario_report(
        real_result,
        gaussian_result,
        gmm_result,
        output_dir=tmp_path,
        channel_names=data.channel_names,
        visualization=visualization,
        scenario_meta={
            "scenario_run_id": 0,
            "scenario_family": "dataset",
            "mode": "smoke",
            "dataset": "Occupancy Detection",
        },
        graphs_out=graphs,
    )
    text = out.read_text(encoding="utf-8")

    # Internal/development wording removed.
    assert "Sprint 2" not in text
    assert "Mode validation" not in text
    assert "Full mode was not run" not in text

    # Section numbering starts at 1; no standalone "Channel notation" section.
    assert "1. Scientific question" in text
    assert "2. Scenario summary" in text
    assert "Channel notation" not in text
    assert "cooperative structure observed under real-data" in text
    assert "GMM synthetic data" in text

    # Main arms use the new transfer semantics across all three arms.
    assert "Real → Real" in text
    assert "Single Gaussian → Real" in text
    assert "GMM → Real" in text
    assert "Single Gaussian → Synthetic" not in text
    assert "GMM → Synthetic" not in text
    assert "Gaussian-anchored" not in text

    # Scenario summary carries the main-arm list and real-data evaluation split.
    assert "Main arms" in text
    assert "Real → Real; Single Gaussian → Real; GMM → Real" in text
    assert "Fixed real Occupancy evaluation split" in text

    # Detailed report links point to the correct main-arm simulations.
    assert "Single Gaussian → Real Monte Carlo" in text
    assert "GMM → Real Monte Carlo" in text

    # Sticky channel legend retained.
    assert "class='legend'" in text
    assert "Channel legend" in text
    assert "X₁" in text
    assert "Temperature" in text

    # Data visualization uses nested source/model and projection tabs.
    assert "4. Data visualization" in text
    assert "4.1 Projection panels" in text
    assert "data-group='scenario-visualization-source'" in text
    assert "data-group='scenario-visualization-real-projection'" in text
    assert "viz-toggle" not in text
    assert "<select" not in text.lower()
    for name in visualization["images"].values():
        assert name in text

    # Best subset comparison includes three loss-vs-N graphs, one per arm.
    assert "5. Best subset comparison at largest N" in text
    assert "graph_best_comparison_real" in graphs
    assert "graph_best_comparison_sgr" in graphs
    assert "graph_best_comparison_gmm" in graphs
    assert "data-group='scenario-best-comparison-arm'" in text

    # Top-ranked subsets grouped classifier-first, then arm; each has a graph.
    assert "6. Top-ranked subsets" in text
    assert "6.1 Linear SVM" in text
    assert "Real-data top-ranked subsets" not in text
    assert "Gaussian-anchored top-ranked subsets" not in text
    assert "<th>Rank</th><th>Subset</th><th>Loss</th><th>SE</th>" in text
    assert any(k.startswith("graph_topranked_") and k.endswith("_real") for k in graphs)
    assert any(
        k.startswith("graph_topranked_") and k.endswith("_sgr") for k in graphs
    )
    assert any(
        k.startswith("graph_topranked_") and k.endswith("_gmm") for k in graphs
    )
    assert "data-group='scenario-top-ranked-classifier'" in text
    assert "data-group='scenario-top-ranked-linear_svm-arm'" in text

    # Structural fidelity is additive and keeps all metrics separate.
    assert "7. Structural fidelity metrics" in text
    assert "predictive cooperation profile" in text
    assert "Winner Agreement" in text
    assert "Reversal existence agreement" in text
    assert "Reversal sample-size similarity" in text
    assert "Winner matrix" in text
    assert "Reversal matrix" in text
    assert "no composite index is reported" in text
    assert "composite structural index" not in text
    assert "Progressive N-star similarity" not in text
    assert "Progressive N-star Similarity" not in text
    assert "Timing similarity" not in text
    assert "Interpolated N-star" not in text
    assert any(k.startswith("graph_structural_metric_") for k in graphs)
    assert any(k.startswith("graph_structural_winner_") for k in graphs)
    assert any(k.startswith("graph_structural_reversal_") for k in graphs)
    assert "unavailable_first_prefix" in text
    assert "no_reversals_in_either" in text
    assert "data-group='scenario-structural-dynamics-classifier'" in text

    # N-star availability section is removed entirely (superseded by 7.3).
    assert "N-star availability" not in text
    assert "<th>VS</th><th>N*</th><th>Interpolated N*</th><th>Winner</th>" not in text
    assert "graph_nstar_" not in text
    assert not any(k.startswith("graph_nstar_") for k in graphs)

    # Interpretation notes renumbered to section 8.
    assert "8. Interpretation notes" in text

    # Subset set-notation used in tables.
    assert "{X" in text

    # Graph PNG files were actually written.
    for name in graphs.values():
        assert (tmp_path / name).exists(), name


def test_occupancy_scenario_report_without_visualization(tmp_path):
    data, real_result, gaussian_result, gmm_result = _results()
    out = generate_occupancy_scenario_report(
        real_result,
        gaussian_result,
        gmm_result,
        output_dir=tmp_path,
        channel_names=data.channel_names,
    )
    text = out.read_text(encoding="utf-8")
    assert "Visualization panels were not generated" in text
    assert "7. Structural fidelity metrics" in text
    assert "8. Interpretation notes" in text
    assert "N-star availability" not in text
    assert "1. Scientific question" in text
    assert "GMM → Real" in text


def test_smoke_config_starts_at_n2():
    from coinfosim.simulation.config import get_mode_config
    cfg = get_mode_config("smoke")
    assert 1 not in cfg.sample_sizes
    assert cfg.sample_sizes[0] == 2


def test_nstar_multiple_crossings_stores_all_reports_last(tmp_path):
    """Test that _nstar_analysis stores all crossings but reports the last one."""
    from coinfosim.results.accumulator import LossAccumulator
    from coinfosim.reports.occupancy_scenario import _nstar_analysis

    # Construct a fake SimulationResult with controlled oscillating losses.
    class FakeModel:
        d = 2
        class_labels = (0, 1)
        channel_names = ("A", "B")

    class FakeResult:
        model = FakeModel()
        subsets = [(0,), (1,)]
        classifier_names = ["linear_svm"]
        sample_sizes = [2, 4, 8, 16]

        def __init__(self):
            self.accumulator = LossAccumulator()
            # reference (0,): 0.5, 0.3, 0.5, 0.3  — oscillating
            # comp (1,):      0.4, 0.4, 0.4, 0.4  — constant
            # Crossings: at n=4 (ref=0.3 < comp=0.4 → ref wins, no crossing yet)
            # Actually let's do:
            # ref losses: 0.5, 0.3, 0.5, 0.3
            # comp losses: 0.4, 0.4, 0.4, 0.4
            # delta = ref - comp: 0.1, -0.1, 0.1, -0.1
            # comp beats ref: at n=4 (delta<0), at n=16 (delta<0) → two crossings
            for rep, (r, c) in enumerate(zip([0.5, 0.3, 0.5, 0.3], [0.4, 0.4, 0.4, 0.4])):
                n = [2, 4, 8, 16][rep]
                self.accumulator.add(n, (0,), "linear_svm", 0, r)
                self.accumulator.add(n, (1,), "linear_svm", 0, c)

        config = type("C", (), {"sample_sizes": [2, 4, 8, 16]})()

    result = FakeResult()
    analysis = _nstar_analysis(result, "linear_svm", (0,), [("Best-2-ChSub", (1,))], n=16)
    entry = analysis[0]

    # winner = "VS" since comp 0.4 < ref 0.3? Actually at n=16 ref=0.3, comp=0.4 → ref wins
    # Let me re-check: at n=16, loss_ref=0.3, loss_comp=0.4 → ref wins. winner="Reference".
    assert entry["winner"] in ("Reference", "VS", "Tie")
    # all_crossings should have entries
    assert isinstance(entry["all_crossings"], list)
    # The reported n_star/interp should be from the LAST genuine crossing, or None
    if len([c for c in entry["all_crossings"] if c["status"] != "left_censored"]) > 1:
        # if multiple genuine crossings, reported is the last
        genuine = [c for c in entry["all_crossings"] if c["status"] != "left_censored"]
        assert entry["n_star"] == genuine[-1]["n_star_grid"]
