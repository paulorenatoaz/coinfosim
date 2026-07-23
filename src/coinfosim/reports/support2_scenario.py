"""SUPPORT2 wrapper for the generic dataset-anchored scenario report."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from coinfosim.reports.dataset_anchored_scenario import (
    DatasetAnchoredScenarioReportSpec,
    generate_dataset_anchored_scenario_report,
)
from coinfosim.scenarios.support2 import SUPPORT2_SCENARIO_QUESTION
from coinfosim.simulation.monte_carlo import SimulationResult

SUPPORT2_SCENARIO_REPORT_SPEC = DatasetAnchoredScenarioReportSpec(
    title="CoInfoSim — SUPPORT2 180-Day Mortality Dataset-Anchored Scenario",
    scientific_question=SUPPORT2_SCENARIO_QUESTION,
    dataset_identity="SUPPORT2 (DOI 10.3886/ICPSR02957.v2)",
    target_definition="death within 180 days after SUPPORT2 study entry (day 180 inclusive)",
    channel_display_mapping={
        "meanbp": "meanbp", "hrt": "hrt", "resp": "resp", "temp": "temp",
        "wblc": "wblc", "crea": "crea", "sod": "sod",
    },
    real_training_description=(
        "fixed 7,098-row real training reservoir from the 8,873-row complete-case "
        "cohort, standardized with training-only population moments"
    ),
    fixed_test_description="same fixed 1,775-row real SUPPORT2 test set",
    arm_labels={
        "real_to_real": "Real → Real",
        "single_gaussian_to_real": "Single Gaussian → Real",
        "gmm_to_real": "GMM → Real",
    },
    arm_summaries={
        "real_to_real": "Balanced real training-reservoir draws evaluated on the unchanged fixed real test set.",
        "single_gaussian_to_real": "Training-only class-conditional Gaussian draws evaluated on the unchanged fixed real test set.",
        "gmm_to_real": "Training-only class-conditional GMM draws evaluated on the unchanged fixed real test set.",
    },
    dataset_report_link_label="SUPPORT2 dataset report",
    limitations=(
        "Complete-case selection can alter the represented patient population.",
        "Fixed-horizon binarization discards detailed time-to-event information.",
        "Smoke-mode results are preliminary pipeline evidence, not final inferential conclusions.",
    ),
    interpretation_notes=(
        ("Ranking fidelity", "Compare subset rankings against Real → Real separately for each classifier and sample size."),
        ("Exact-tie-aware winner agreement", "Pairwise agreement excludes comparisons tied in either arm and reports availability explicitly."),
        ("Reversal fidelity", "Compare reversal existence agreement and reversal sample-size similarity only where the required prefixes are available."),
        ("Near-balanced endpoint", "The cohort contains 4,711 negatives and 4,162 positives; balanced training draws isolate distributional fidelity."),
    ),
    scenario_metadata={
        "dataset": "SUPPORT2",
        "target": "death_180d",
        "horizon_days": "180",
        "cohort_rows": "8873",
        "train_rows": "7098",
        "test_rows": "1775",
        "subsets": "127",
    },
)


def generate_support2_scenario_report(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    output_dir: Path | str = "output/reports",
    dataset_report: str = "support2_dataset_report.html",
    real_report: str = "support2_real_data_monte_carlo_report.html",
    sg_real_report: str = "support2_single_gaussian_to_real_monte_carlo_report.html",
    gmm_real_report: str = "support2_gmm_to_real_monte_carlo_report.html",
    filename: str = "support2_scenario_report.html",
    channel_names: Sequence[str] | None = None,
    visualization: Optional[Dict] = None,
    scenario_meta: Optional[Dict] = None,
    graph_suffix: str = "",
    graphs_out: Optional[Dict] = None,
    generate_graphs: bool = True,
) -> Path:
    return generate_dataset_anchored_scenario_report(
        real_result,
        gaussian_result,
        gmm_result,
        report_spec=SUPPORT2_SCENARIO_REPORT_SPEC,
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
