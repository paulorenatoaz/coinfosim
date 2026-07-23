from itertools import combinations
from pathlib import Path

from coinfosim.reports.air_quality_scenario import (
    generate_air_quality_scenario_report,
)
from coinfosim.results.accumulator import LossAccumulator


CHANNELS = (
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
)


class _TinyConfig:
    mode = "smoke"
    sample_sizes = (2, 4)
    min_replications = 2
    max_replications = 2
    replication_batch_size = 2
    ci_half_width_target = 0.05


class _TinyResult:
    classifier_names = ["linear_svm"]
    sample_sizes = [2, 4]
    subsets = [
        subset
        for cardinality in range(1, 6)
        for subset in combinations(range(5), cardinality)
    ]
    config = _TinyConfig()
    metadata = {"channel_names": list(CHANNELS)}

    def __init__(self, arm_offset: float, distortion: float):
        self.accumulator = LossAccumulator()
        for n_per_class in self.sample_sizes:
            for subset in self.subsets:
                subset_score = 0.002 * sum(i + 1 for i in subset) + 0.003 / len(subset)
                loss = (
                    0.28
                    + arm_offset
                    + distortion * subset_score
                    - 0.01 * (n_per_class / 2)
                )
                for replication in range(2):
                    self.accumulator.add(
                        n_per_class,
                        subset,
                        "linear_svm",
                        replication,
                        loss + replication * 0.0005,
                    )


def _linked_inputs(tmp_path: Path):
    names = {
        "dataset_report": "air_quality_dataset_report.html",
        "real_report": "air_quality_real_monte_carlo_report.html",
        "sg_real_report": "air_quality_single_gaussian_to_real_monte_carlo_report.html",
        "gmm_real_report": "air_quality_gmm_to_real_monte_carlo_report.html",
    }
    for name in names.values():
        (tmp_path / name).write_text("<html></html>", encoding="utf-8")
    return names


def _visualization(tmp_path: Path):
    images = {}
    for arm in ("real", "gaussian", "gmm"):
        for dimension in ("1d", "2d", "3d"):
            key = f"viz_{dimension}_{arm}"
            images[key] = f"{key}.png"
            (tmp_path / images[key]).write_bytes(b"preview")
    return {
        "images": images,
        "metadata": {
            "visualization_sample_size": 20,
            "class_balance": "balanced (equal samples per class)",
            "real_data_source": "chronological training pool",
            "single_gaussian_source": "training-only Gaussian fit",
            "gmm_source": "training-only GMM fit",
            "visualization_seed": 20240501,
        },
    }


def test_air_quality_scenario_report_uses_generic_academic_structure(tmp_path):
    real = _TinyResult(0.0, 1.0)
    gaussian = _TinyResult(0.01, 1.08)
    gmm = _TinyResult(0.005, 0.97)
    links = _linked_inputs(tmp_path)
    visualization = _visualization(tmp_path)
    graphs = {}

    report = generate_air_quality_scenario_report(
        real,
        gaussian,
        gmm,
        output_dir=tmp_path,
        channel_names=CHANNELS,
        visualization=visualization,
        graphs_out=graphs,
        **links,
    )
    text = report.read_text(encoding="utf-8")

    for section in (
        "1. Scientific question",
        "2. Scenario summary",
        "3. Experimental protocol",
        "4. Data visualization",
        "5. Best subset comparison at largest N",
        "6. Top-ranked subsets",
        "7. Structural fidelity metrics",
        "8. Interpretation notes",
    ):
        assert section in text

    assert "Real → Real" in text
    assert "Single Gaussian → Real" in text
    assert "GMM → Real" in text
    assert "fixed future real Air Quality test set" in text
    assert "C6H6(GT)" in text
    assert "75th-percentile threshold (14.5)" in text
    assert "class='legend'" in text
    for channel in CHANNELS:
        assert channel in text
    assert "<th>Rank</th><th>Subset</th><th>Loss</th><th>SE</th>" in text
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
    assert "unavailable_first_prefix" in text
    assert "data-group='scenario-visualization-source'" in text
    assert "data-group='scenario-top-ranked-classifier'" in text
    assert "data-group='scenario-structural-dynamics-classifier'" in text
    assert "Occupancy" not in text

    assert any(key.startswith("graph_topranked_") for key in graphs)
    assert any(key.startswith("graph_structural_winner_") for key in graphs)
    assert any(key.startswith("graph_structural_reversal_") for key in graphs)
    for filename in (*links.values(), *visualization["images"].values(), *graphs.values()):
        assert (tmp_path / filename).exists(), filename
