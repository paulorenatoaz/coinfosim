"""Air Quality wrapper for the generic dataset-anchored scenario report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from coinfosim.reports.dataset_anchored_scenario import (
    DatasetAnchoredScenarioReportSpec,
    generate_dataset_anchored_scenario_report,
)
from coinfosim.simulation.monte_carlo import SimulationResult


AIR_QUALITY_SCENARIO_REPORT_SPEC = DatasetAnchoredScenarioReportSpec(
    title="CoInfoSim — UCI Air Quality Dataset-Anchored Scenario",
    scientific_question=(
        "Which training distribution best preserves the cooperative channel-subset "
        "structure observed on the fixed future real UCI Air Quality test set: real "
        "training data, single-Gaussian synthetic data, or GMM synthetic data?"
    ),
    dataset_identity="UCI Air Quality (DOI 10.24432/C59K5F)",
    target_definition=(
        "Positive when C6H6(GT) >= the training-only 75th-percentile threshold "
        "(14.5); C6H6(GT) is excluded from classifier input"
    ),
    channel_display_mapping={
        "PT08.S1(CO)": "PT08.S1(CO)",
        "PT08.S2(NMHC)": "PT08.S2(NMHC)",
        "PT08.S3(NOx)": "PT08.S3(NOx)",
        "PT08.S4(NO2)": "PT08.S4(NO2)",
        "PT08.S5(O3)": "PT08.S5(O3)",
    },
    real_training_description=(
        "chronologically first 80% complete-case Air Quality training pool, "
        "standardized using training-only means and population standard deviations"
    ),
    fixed_test_description=(
        "fixed future real Air Quality test set (chronologically last 20%)"
    ),
    arm_labels={
        "real_to_real": "Real → Real",
        "single_gaussian_to_real": "Single Gaussian → Real",
        "gmm_to_real": "GMM → Real",
    },
    arm_summaries={
        "real_to_real": (
            "Balanced training samples come from the chronologically first 80% real "
            "training pool and are evaluated on the unchanged fixed future real test set."
        ),
        "single_gaussian_to_real": (
            "One Gaussian per threshold-defined class is fitted only on the real training "
            "pool; synthetic balanced training samples are evaluated on the unchanged "
            "fixed future real test set."
        ),
        "gmm_to_real": (
            "A BIC-selected Gaussian mixture per threshold-defined class is fitted only "
            "on the real training pool; synthetic balanced training samples are evaluated "
            "on the unchanged fixed future real test set."
        ),
    },
    dataset_report_link_label="UCI Air Quality dataset report",
    limitations=(
        "The threshold defines a dataset-anchored classification target and is not a health-limit interpretation.",
        "Chronological evaluation measures transfer to this fixed future period only.",
        "Smoke-mode results are preliminary pipeline evidence, not final inferential conclusions.",
    ),
    interpretation_notes=(
        (
            "Structural fidelity metrics",
            "Ranking fidelity, winner agreement, and progressive N-star similarity measure separate properties and are interpreted separately alongside their availability statuses.",
        ),
        (
            "Agreement among the three arms",
            "Agreement in best subsets and cooperative thresholds indicates that a fitted synthetic training distribution preserves structure expressed on the same future real test set.",
        ),
        (
            "Single Gaussian vs GMM",
            "Closer GMM alignment with Real → Real suggests that multimodal class-conditional structure matters for synthetic-to-real transfer.",
        ),
        (
            "Divergences from Real → Real",
            "Differences identify channel-subset relationships in the real training distribution that a fitted synthetic distribution does not reproduce on future real observations.",
        ),
        (
            "Classifier-specific behavior",
            "Linear SVM, Logistic Regression, and Gaussian Naive Bayes can rank the same 31 subsets differently, so comparisons are classifier-specific.",
        ),
    ),
    scenario_metadata={
        "dataset": "UCI Air Quality",
        "doi": "10.24432/C59K5F",
        "split_strategy": "chronological_80_20",
        "threshold_quantile": "0.75",
    },
)


def generate_air_quality_scenario_report(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    output_dir: Path | str = "output/reports",
    dataset_report: str = "air_quality_dataset_report.html",
    real_report: str = "air_quality_real_monte_carlo_report.html",
    sg_real_report: str = "air_quality_single_gaussian_to_real_monte_carlo_report.html",
    gmm_real_report: str = "air_quality_gmm_to_real_monte_carlo_report.html",
    filename: str = "air_quality_scenario_report.html",
    channel_names: Sequence[str] | None = None,
    visualization: Optional[Dict] = None,
    scenario_meta: Optional[Dict] = None,
    graph_suffix: str = "",
    graphs_out: Optional[Dict] = None,
    generate_graphs: bool = True,
) -> Path:
    """Generate the Air Quality report through the shared scenario core."""

    return generate_dataset_anchored_scenario_report(
        real_result,
        gaussian_result,
        gmm_result,
        report_spec=AIR_QUALITY_SCENARIO_REPORT_SPEC,
        output_dir=output_dir,
        dataset_report=dataset_report,
        real_report=real_report,
        sg_real_report=sg_real_report,
        gmm_real_report=gmm_real_report,
        filename=filename,
        channel_names=channel_names,
        visualization=visualization,
        scenario_meta=scenario_meta,
        graph_suffix=graph_suffix,
        graphs_out=graphs_out,
        generate_graphs=generate_graphs,
    )
