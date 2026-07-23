"""Air Quality wrappers for the shared structured Monte Carlo report."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coinfosim.reports.monte_carlo import (
    gaussian_parameters_section,
    generate_structured_monte_carlo_report,
    gmm_parameters_section,
)
from coinfosim.simulation.monte_carlo import SimulationResult

_PROVENANCE = """
<p>Dataset: <strong>UCI Air Quality</strong> (ID 360, DOI
<code>10.24432/C59K5F</code>). Source file: <code>AirQualityUCI.csv</code>.
The five channels are PT08 metal-oxide sensor responses. <code>C6H6(GT)</code>
is used only to define the training-period target and is excluded from classifier
inputs. Complete cases are split chronologically: first 80% training and last
20% fixed future real evaluation.</p>
"""

_TARGET = (
    "benzene concentration elevated relative to the training-period distribution "
    "using the training-only 75th percentile of C6H6(GT)"
)

_REAL_SUMMARY = (
    "Balanced real training samples are drawn without replacement from the "
    "standardized chronological training reservoir. Evaluation uses the fixed "
    "future real Air Quality test period."
)
_REAL_ROLE = (
    "This empirical reference arm measures cooperative sensor-response structure "
    "under real training and the fixed future real evaluation period."
)
_REAL_PROTOCOL = """
<dl class="meta"><dt>Training source</dt><dd>Balanced draws without replacement
from the standardized chronological real training reservoir</dd>
<dt>Evaluation source</dt><dd>Fixed future real Air Quality evaluation set</dd>
<dt>Target</dt><dd>Training-only C6H6(GT) 75th-percentile binary target</dd></dl>
"""
_REAL_REPRODUCIBILITY = """
<dl class="meta"><dt>Sampler</dt><dd>RealDatasetSampler</dd>
<dt>Split</dt><dd>Chronological 80/20 after complete-case filtering</dd>
<dt>Standardization</dt><dd>Training-only z-score, ddof=0</dd>
<dt>Test semantics</dt><dd>Fixed future real rows, unchanged across arms</dd></dl>
"""

_GAUSSIAN_SUMMARY = (
    "Balanced training samples are generated from one class-conditional "
    "multivariate Gaussian fitted to standardized historical training rows. "
    "Evaluation uses the fixed future real Air Quality test period."
)
_GAUSSIAN_ROLE = (
    "This arm tests whether a unimodal class-conditional approximation preserves "
    "the cooperative sensor-response behavior observed under future real evaluation."
)
_GAUSSIAN_PROTOCOL = """
<dl class="meta"><dt>Training source</dt><dd>Single-Gaussian synthetic draws from
class-conditional models fitted on standardized training rows</dd>
<dt>Evaluation source</dt><dd>Fixed future real Air Quality evaluation set</dd>
<dt>Reference exclusion</dt><dd>C6H6(GT) is not a model input</dd></dl>
"""
_GAUSSIAN_REPRODUCIBILITY = """
<dl class="meta"><dt>Model</dt><dd>One full five-dimensional Gaussian per class</dd>
<dt>Fitting source</dt><dd>Standardized chronological training reservoir only</dd>
<dt>Sampler</dt><dd>SyntheticTrainRealTestSampler with GaussianClassConditionalSampler</dd>
<dt>Test semantics</dt><dd>Fixed future real rows, unchanged across arms</dd></dl>
"""

_GMM_SUMMARY = (
    "Balanced training samples are generated from class-conditional Gaussian "
    "mixtures fitted to standardized historical training rows with BIC component "
    "selection. Evaluation uses the fixed future real Air Quality test period."
)
_GMM_ROLE = (
    "This arm tests whether a multimodal class-conditional approximation better "
    "preserves the real-evaluation predictive cooperation profile than a single "
    "Gaussian."
)
_GMM_PROTOCOL = """
<dl class="meta"><dt>Training source</dt><dd>Class-conditional GMM synthetic draws</dd>
<dt>Fitting source</dt><dd>Standardized chronological training reservoir only</dd>
<dt>Model selection</dt><dd>BIC over the configured candidate component counts</dd>
<dt>Evaluation source</dt><dd>Fixed future real Air Quality evaluation set</dd></dl>
"""
_GMM_REPRODUCIBILITY = """
<dl class="meta"><dt>Model</dt><dd>One full-space class-conditional GMM per class</dd>
<dt>Sampler</dt><dd>SyntheticTrainRealTestSampler with GMMClassConditionalSampler</dd>
<dt>Test semantics</dt><dd>Fixed future real rows, unchanged across arms</dd>
<dt>Standardization</dt><dd>Training-only z-score, ddof=0</dd></dl>
"""

_ROBUSTNESS = """
<p>Read loss curves together with Monte Carlo precision diagnostics, subset rankings,
winner matrices and progressive N-star matrices. Empirical test loss is affected by
class prevalence in the fixed future period.</p>
"""
_VALIDITY = """
<p>PT08 variables are cross-sensitive sensor responses. Sensor drift, concept drift,
chronological distribution shift and artificial binarization can limit transfer.
The experimental threshold is not a health or regulatory limit.</p>
"""


def _common_kwargs(channel_names: Sequence[str]):
    return {
        "channel_names": channel_names,
        "fixed_test_description": "fixed future real Air Quality evaluation set",
        "dataset_provenance_html": _PROVENANCE,
        "dataset_name": "UCI Air Quality",
        "target_description": _TARGET,
        "full_reference_label": f"Full-{len(tuple(channel_names))} reference",
        "robustness_html": _ROBUSTNESS,
        "validity_html": _VALIDITY,
    }


def generate_air_quality_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "air_quality_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — Air Quality Real → Real Monte Carlo Report",
        arm_id="real_to_real",
        arm_label="Real → Real",
        arm_summary=_REAL_SUMMARY,
        scientific_role=_REAL_ROLE,
        training_protocol_html=_REAL_PROTOCOL,
        reproducibility_html=_REAL_REPRODUCIBILITY,
        nstar_selection_result=nstar_selection_result or result,
        **_common_kwargs(channel_names),
    )


def generate_air_quality_single_gaussian_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "air_quality_single_gaussian_to_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    model_html = (
        "<h3>Estimated Gaussian training parameters</h3>"
        + gaussian_parameters_section(result.model, channel_names)
    )
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — Air Quality Single Gaussian → Real Monte Carlo Report",
        arm_id="single_gaussian_to_real",
        arm_label="Single Gaussian → Real",
        arm_summary=_GAUSSIAN_SUMMARY,
        scientific_role=_GAUSSIAN_ROLE,
        training_protocol_html=_GAUSSIAN_PROTOCOL,
        model_section_html=model_html,
        reproducibility_html=_GAUSSIAN_REPRODUCIBILITY,
        nstar_selection_result=nstar_selection_result,
        **_common_kwargs(channel_names),
    )


def generate_air_quality_gmm_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "air_quality_gmm_to_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    model_html = (
        "<h3>Estimated GMM training model</h3>"
        + gmm_parameters_section(result.model, channel_names)
    )
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — Air Quality GMM → Real Monte Carlo Report",
        arm_id="gmm_to_real",
        arm_label="GMM → Real",
        arm_summary=_GMM_SUMMARY,
        scientific_role=_GMM_ROLE,
        training_protocol_html=_GMM_PROTOCOL,
        model_section_html=model_html,
        reproducibility_html=_GMM_REPRODUCIBILITY,
        nstar_selection_result=nstar_selection_result,
        **_common_kwargs(channel_names),
    )
