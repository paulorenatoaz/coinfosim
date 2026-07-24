"""SUPPORT2 wrappers for the shared structured Monte Carlo report."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coinfosim.reports.monte_carlo import (
    gaussian_parameters_section,
    generate_structured_monte_carlo_report,
    gmm_parameters_section,
)
from coinfosim.simulation.monte_carlo import SimulationResult

_TARGET = "death within 180 days after SUPPORT2 study entry"
_PROVENANCE = """
<p>Dataset: <strong>SUPPORT2</strong>, 9,105 seriously ill hospitalized adults
(DOI <code>10.3886/ICPSR02957.v2</code>). The target is derived before splitting:
<code>death_180d = (death == 1 and d.time &lt;= 180)</code>. The seven baseline
physiologic attributes use the complete-case cohort; death, d.time, hospdead,
survival estimates, ID, and disease group are excluded from classifier input.</p>
"""
_ROBUSTNESS = """
<p>Interpret loss curves with Monte Carlo precision diagnostics, subset rankings,
exact-tie-aware winner matrices, and progressive N-star matrices. Smoke-mode
results validate the pipeline and may be statistically unstable.</p>
"""
_VALIDITY = """
<p>External validity is limited to this historical seriously ill inpatient cohort.
Complete-case selection and fixed-horizon binarization can affect interpretation;
the endpoint does not represent eventual or in-hospital mortality.</p>
"""


def _common_kwargs(channel_names: Sequence[str]):
    return {
        "channel_names": channel_names,
        "fixed_test_description": "same fixed real SUPPORT2 test set",
        "dataset_provenance_html": _PROVENANCE,
        "dataset_name": "SUPPORT2",
        "target_description": _TARGET,
        "full_reference_label": f"Full-{len(tuple(channel_names))} reference",
        "robustness_html": _ROBUSTNESS,
        "validity_html": _VALIDITY,
    }


def generate_support2_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "support2_real_data_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — SUPPORT2 Real → Real Monte Carlo Report",
        arm_id="real_to_real",
        arm_label="Real → Real",
        arm_summary=(
            "Balanced training samples are drawn without replacement from the real "
            "training reservoir according to death_180d. Evaluation uses the same "
            "fixed real SUPPORT2 test set."
        ),
        scientific_role="Empirical reference for cooperative attribute-subset structure.",
        training_protocol_html=(
            "<dl class='meta'><dt>Training source</dt><dd>Balanced real SUPPORT2 "
            "training-reservoir draws labeled by death_180d</dd><dt>Evaluation "
            "source</dt><dd>Same fixed real SUPPORT2 test set</dd><dt>Target</dt><dd>"
            + _TARGET
            + "</dd></dl>"
        ),
        reproducibility_html=(
            "<dl class='meta'><dt>Sampler</dt><dd>RealDatasetSampler</dd>"
            "<dt>Split</dt><dd>Fixed joint-stratified 80/20 split, seed 0</dd>"
            "<dt>Standardization</dt><dd>Training-only z-score, ddof=0</dd></dl>"
        ),
        nstar_selection_result=nstar_selection_result or result,
        **_common_kwargs(channel_names),
    )


def generate_support2_single_gaussian_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "support2_single_gaussian_to_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — SUPPORT2 Single Gaussian → Real Monte Carlo Report",
        arm_id="single_gaussian_to_real",
        arm_label="Single Gaussian → Real",
        arm_summary=(
            "Balanced synthetic training samples come from one seven-dimensional "
            "class-conditional Gaussian fitted only on the real training partition. "
            "Evaluation uses the same fixed real SUPPORT2 test set."
        ),
        scientific_role="Tests unimodal class-conditional synthetic-to-real fidelity.",
        training_protocol_html=(
            "<dl class='meta'><dt>Fitting source</dt><dd>Standardized real training "
            "partition only</dd><dt>Target</dt><dd>" + _TARGET + "</dd><dt>Evaluation "
            "source</dt><dd>Same fixed real SUPPORT2 test set</dd></dl>"
        ),
        model_section_html=(
            "<h3>Estimated Gaussian training parameters</h3>"
            + gaussian_parameters_section(result.model, channel_names)
        ),
        reproducibility_html=(
            "<dl class='meta'><dt>Model</dt><dd>One full seven-dimensional Gaussian "
            "per death_180d class</dd><dt>Test semantics</dt><dd>Fixed real rows, "
            "unchanged across arms</dd></dl>"
        ),
        nstar_selection_result=nstar_selection_result,
        **_common_kwargs(channel_names),
    )


def generate_support2_gmm_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "support2_gmm_to_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — SUPPORT2 GMM → Real Monte Carlo Report",
        arm_id="gmm_to_real",
        arm_label="GMM → Real",
        arm_summary=(
            "Balanced synthetic training samples come from class-conditional Gaussian "
            "mixtures fitted only on the real training partition. Evaluation uses the "
            "same fixed real SUPPORT2 test set."
        ),
        scientific_role="Tests multimodal class-conditional synthetic-to-real fidelity.",
        training_protocol_html=(
            "<dl class='meta'><dt>Fitting source</dt><dd>Standardized real training "
            "partition only</dd><dt>Target</dt><dd>" + _TARGET + "</dd><dt>Model "
            "selection</dt><dd>BIC over configured component candidates</dd><dt>Evaluation "
            "source</dt><dd>Same fixed real SUPPORT2 test set</dd></dl>"
        ),
        model_section_html=(
            "<h3>Estimated GMM training model</h3>"
            + gmm_parameters_section(result.model, channel_names)
        ),
        reproducibility_html=(
            "<dl class='meta'><dt>Model</dt><dd>Full-space class-conditional GMM</dd>"
            "<dt>Test semantics</dt><dd>Fixed real rows, unchanged across arms</dd>"
            "<dt>Standardization</dt><dd>Training-only z-score, ddof=0</dd></dl>"
        ),
        nstar_selection_result=nstar_selection_result,
        **_common_kwargs(channel_names),
    )
