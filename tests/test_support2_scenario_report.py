from itertools import combinations
from pathlib import Path

from coinfosim.reports.support2_scenario import generate_support2_scenario_report
from coinfosim.results.accumulator import LossAccumulator

CHANNELS = ("meanbp", "hrt", "resp", "temp", "wblc", "crea", "sod")


class _TinyConfig:
    mode = "smoke"
    sample_sizes = (2, 4)
    min_replications = 2
    max_replications = 2
    replication_batch_size = 2
    ci_half_width_target = 0.05


class _TinyResult:
    classifier_names = ["linear_svm", "random_forest"]
    sample_sizes = [2, 4]
    subsets = [
        subset
        for cardinality in range(1, 8)
        for subset in combinations(range(7), cardinality)
    ]
    config = _TinyConfig()
    metadata = {
        "channel_names": list(CHANNELS),
        "classifier_configurations": {
            "linear_svm": {
                "estimator": "sklearn.svm.SVC",
                "parameters": {"kernel": "linear", "random_state": 0},
                "seed_policy": {"kind": "fixed", "value": 0},
            },
            "random_forest": {
                "estimator": "sklearn.ensemble.RandomForestClassifier",
                "parameters": {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": 1},
                "seed_policy": {"kind": "per_replication", "version": "classifier_seed_v1"},
            },
        },
    }

    def __init__(self, offset):
        self.accumulator = LossAccumulator()
        for n_per_class in self.sample_sizes:
            for subset in self.subsets:
                loss = 0.3 + offset - 0.001 * len(subset) - 0.002 * n_per_class
                for classifier in self.classifier_names:
                    for replication in range(2):
                        self.accumulator.add(
                            n_per_class,
                            subset,
                            classifier,
                            replication,
                            loss + replication * 0.0001,
                        )


def test_support2_scenario_report_links_hierarchy_and_identifies_protocol(tmp_path):
    names = {
        "dataset_report": "support2_dataset_report.html",
        "real_report": "support2_real.html",
        "sg_real_report": "support2_gaussian.html",
        "gmm_real_report": "support2_gmm.html",
    }
    for filename in names.values():
        (tmp_path / filename).write_text("<html></html>", encoding="utf-8")
    graphs = {}
    report = generate_support2_scenario_report(
        _TinyResult(0),
        _TinyResult(0.01),
        _TinyResult(0.005),
        output_dir=tmp_path,
        channel_names=CHANNELS,
        graphs_out=graphs,
        generate_graphs=False,
        **names,
    )
    text = report.read_text(encoding="utf-8")
    assert "death within 180 days after SUPPORT2 study entry" in text
    assert "same fixed 1,775-row real SUPPORT2 test set" in text
    assert "Real → Real" in text
    assert "Single Gaussian → Real" in text
    assert "GMM → Real" in text
    assert "Ranking Structural Fidelity" in text
    assert "Winner Agreement" in text
    assert "Progressive N-star Similarity" in text
    assert "exact-tie-aware" in text.lower()
    assert "127" in text
    assert "Linear SVM, Random Forest" in text
    assert "Logistic Regression" not in text
    assert "Gaussian Naive Bayes" not in text
    for filename in names.values():
        assert f"href='{filename}'" in text
        assert (tmp_path / filename).exists()


def test_support2_historical_report_uses_stored_classifier_names(tmp_path):
    class HistoricalResult(_TinyResult):
        classifier_names = ["linear_svm", "logistic_regression", "gaussian_nb"]
        metadata = {"channel_names": list(CHANNELS)}

    names = {
        "dataset_report": "dataset.html",
        "real_report": "real.html",
        "sg_real_report": "gaussian.html",
        "gmm_real_report": "gmm.html",
    }
    report = generate_support2_scenario_report(
        HistoricalResult(0),
        HistoricalResult(0.01),
        HistoricalResult(0.005),
        output_dir=tmp_path,
        channel_names=CHANNELS,
        generate_graphs=False,
        **names,
    )
    text = report.read_text(encoding="utf-8")
    assert "Linear SVM, Logistic Regression, Gaussian Naive Bayes" in text
    assert "Detailed classifier configuration was not recorded" in text
