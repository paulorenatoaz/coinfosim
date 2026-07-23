"""Generic academic report for a three-arm dataset-anchored scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from coinfosim.classifiers.registry import classifier_label
from coinfosim.reports.html_tabs import TAB_CSS, TAB_JS
from coinfosim.reports.occupancy_scenario import (
    _STYLE,
    _best_comparison_html,
    _carousel_html,
    _interpretation_html,
    _legend_html,
    _predictive_cooperation_profile_section,
    _protocol_html,
    _summary_table,
    _top_ranked_html,
)
from coinfosim.simulation.monte_carlo import SimulationResult


@dataclass(frozen=True)
class DatasetAnchoredScenarioReportSpec:
    """Dataset-specific language and metadata consumed by the generic renderer."""

    title: str
    scientific_question: str
    dataset_identity: str
    target_definition: str
    channel_display_mapping: Mapping[str, str]
    real_training_description: str
    fixed_test_description: str
    arm_labels: Mapping[str, str]
    arm_summaries: Mapping[str, str]
    dataset_report_link_label: str
    limitations: Tuple[str, ...]
    interpretation_notes: Tuple[Tuple[str, str], ...]
    scenario_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_arms = {
            "real_to_real",
            "single_gaussian_to_real",
            "gmm_to_real",
        }
        if set(self.arm_labels) != required_arms:
            raise ValueError("arm_labels must define the standard three arms")
        if set(self.arm_summaries) != required_arms:
            raise ValueError("arm_summaries must define the standard three arms")


def generate_dataset_anchored_scenario_report(
    real_result: SimulationResult,
    gaussian_result: SimulationResult,
    gmm_result: SimulationResult,
    *,
    report_spec: DatasetAnchoredScenarioReportSpec,
    output_dir: Path | str = "output/reports",
    dataset_report: str = "dataset_report.html",
    real_report: str = "real_monte_carlo_report.html",
    sg_real_report: str = "single_gaussian_to_real_monte_carlo_report.html",
    gmm_real_report: str = "gmm_to_real_monte_carlo_report.html",
    filename: str = "dataset_anchored_scenario_report.html",
    channel_names: Sequence[str] | None = None,
    visualization: Optional[Dict] = None,
    scenario_meta: Optional[Dict] = None,
    graph_suffix: str = "",
    graphs_out: Optional[Dict] = None,
    generate_graphs: bool = True,
) -> Path:
    """Render the shared eight-section academic scenario report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    if graphs_out is None:
        graphs_out = {}

    arm_results = {
        "real_to_real": real_result,
        "single_gaussian_to_real": gaussian_result,
        "gmm_to_real": gmm_result,
    }
    classifier_names = tuple(real_result.classifier_names)
    for arm_name, result in arm_results.items():
        if tuple(result.classifier_names) != classifier_names:
            raise ValueError(
                f"all three arms must use the same classifier plan; {arm_name} differs"
            )
    recorded_configurations = [
        result.metadata.get("classifier_configurations") for result in arm_results.values()
    ]
    present_configurations = [value for value in recorded_configurations if value]
    if present_configurations and (
        len(present_configurations) != len(recorded_configurations)
        or any(value != present_configurations[0] for value in present_configurations[1:])
    ):
        raise ValueError("all three arms must use the same classifier configuration")
    channel_names = tuple(
        channel_names
        or real_result.metadata.get("channel_names", [])
        or gaussian_result.metadata.get("channel_names", [])
    )
    n_max = max(real_result.sample_sizes)
    metadata = dict(report_spec.scenario_metadata)
    metadata.update(scenario_meta or {})
    metadata.setdefault("dataset", report_spec.dataset_identity)
    metadata["classifiers"] = ", ".join(classifier_names)
    classifier_display = ", ".join(classifier_label(name) for name in classifier_names)
    if present_configurations:
        classifier_configuration_note = (
            "<p><strong>Classifiers:</strong> "
            + html.escape(classifier_display)
            + ". Resolved classifier parameters, seed policy, and calibration "
            "provenance are recorded in each linked Monte Carlo report.</p>"
        )
    else:
        classifier_configuration_note = (
            "<p><strong>Classifiers:</strong> "
            + html.escape(classifier_display)
            + ". Detailed classifier configuration was not recorded for this "
            "historical result.</p>"
        )

    related = (
        "<p class='related'>Detailed reports: "
        f"<a href='{html.escape(dataset_report, quote=True)}'>"
        f"{html.escape(report_spec.dataset_report_link_label)}</a> &middot; "
        f"<a href='{html.escape(real_report, quote=True)}'>"
        f"{html.escape(report_spec.arm_labels['real_to_real'])} Monte Carlo</a> &middot; "
        f"<a href='{html.escape(sg_real_report, quote=True)}'>"
        f"{html.escape(report_spec.arm_labels['single_gaussian_to_real'])} Monte Carlo</a> &middot; "
        f"<a href='{html.escape(gmm_real_report, quote=True)}'>"
        f"{html.escape(report_spec.arm_labels['gmm_to_real'])} Monte Carlo</a>"
        "</p>"
    )
    arm_label_sequence = tuple(report_spec.arm_labels[arm] for arm in arm_results)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(report_spec.title)}</title>
<style>{_STYLE}{TAB_CSS}</style>
</head>
<body>

<h1>{html.escape(report_spec.title)}</h1>
{related}

{_legend_html(channel_names, report_spec.channel_display_mapping)}

<h2>1. Scientific question</h2>
<p class="question">{html.escape(report_spec.scientific_question)}</p>

<h2>2. Scenario summary</h2>
{_summary_table(real_result, metadata, channel_names, n_max, dataset_identity=report_spec.dataset_identity, arm_labels=arm_label_sequence, real_training_description=report_spec.real_training_description, fixed_test_description=report_spec.fixed_test_description, target_definition=report_spec.target_definition)}
{classifier_configuration_note}

{_protocol_html(real_result, report_spec.arm_labels, report_spec.arm_summaries, report_spec.fixed_test_description)}

{_carousel_html(visualization)}

{_best_comparison_html(real_result, gaussian_result, gmm_result, n_max, output_dir, graph_suffix, graphs_out, generate_graphs, report_spec.arm_labels)}

{_top_ranked_html(real_result, gaussian_result, gmm_result, n_max, output_dir, graph_suffix, graphs_out, generate_graphs=generate_graphs, arm_labels=report_spec.arm_labels)}

{_predictive_cooperation_profile_section(arm_results, "real_to_real", report_spec.arm_labels, output_dir, graph_suffix, graphs_out, generate_graphs)}

{_interpretation_html(report_spec.interpretation_notes, report_spec.limitations)}

{TAB_JS}
</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
