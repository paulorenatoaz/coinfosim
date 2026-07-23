"""Occupancy-specific Monte Carlo report wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coinfosim.reports.monte_carlo import (
    gaussian_parameters_section,
    generate_monte_carlo_report,
    generate_structured_monte_carlo_report,
    gmm_parameters_section,
)
from coinfosim.simulation.monte_carlo import SimulationResult


# --------------------------------------------------------------------------- #
# Arm-specific section content
# --------------------------------------------------------------------------- #

_REAL_ARM_SUMMARY = (
    "Training samples are drawn as balanced batches from the standardized "
    "datatraining.txt pool. Evaluation uses the fixed real Occupancy evaluation "
    "split (standardized datatest.txt + datatest2.txt)."
)

_REAL_SCIENTIFIC_ROLE = (
    "This arm is the empirical baseline. It directly measures cooperative "
    "channel structure under real Occupancy training data. Results from the "
    "other arms are compared against this reference."
)

_REAL_TRAINING_PROTOCOL = """
<dl class="meta">
  <dt>Training source</dt>
  <dd>Balanced draws from the standardized <code>datatraining.txt</code> pool</dd>
  <dt>Evaluation source</dt>
  <dd>Fixed real evaluation split: standardized <code>datatest.txt + datatest2.txt</code></dd>
  <dt>Sampling mechanism</dt>
  <dd>Deterministic balanced sampling by <code>base_seed</code>, class label and
      replication ID</dd>
</dl>
"""

_REAL_REPRODUCIBILITY = """
<dl class="meta">
  <dt>Training pool</dt><dd><code>datatraining.txt</code></dd>
  <dt>Test split</dt><dd><code>datatest.txt + datatest2.txt</code> (concatenated)</dd>
  <dt>Standardization</dt><dd>z-score, fitted on training pool only (ddof=0)</dd>
  <dt>Sampler type</dt><dd>RealDatasetSampler (balanced, deterministic by seed)</dd>
</dl>
"""

_SG2R_ARM_SUMMARY = (
    "Class-conditional Gaussian parameters are estimated from the standardized "
    "datatraining.txt pool. Balanced training samples are generated synthetically "
    "from this single Gaussian model. Evaluation uses the fixed real Occupancy "
    "evaluation split (standardized datatest.txt + datatest2.txt)."
)

_SG2R_SCIENTIFIC_ROLE = (
    "This arm tests whether a simple unimodal class-conditional Gaussian "
    "approximation of the training distribution is sufficient to preserve the "
    "cooperative channel structure observed under real evaluation."
)

_SG2R_TRAINING_PROTOCOL = """
<dl class="meta">
  <dt>Training source</dt>
  <dd>Synthetic — class-conditional Gaussian model estimated from
      <code>datatraining.txt</code></dd>
  <dt>Evaluation source</dt>
  <dd>Fixed real evaluation split: standardized <code>datatest.txt + datatest2.txt</code></dd>
  <dt>Standardization</dt><dd>z-score, fitted on <code>datatraining.txt</code> only</dd>
</dl>
"""

_SG2R_REPRODUCIBILITY = """
<dl class="meta">
  <dt>Model type</dt><dd>Single class-conditional Gaussian per class</dd>
  <dt>Fitting source</dt><dd><code>datatraining.txt</code> (standardized)</dd>
  <dt>Test split</dt><dd><code>datatest.txt + datatest2.txt</code> (real, fixed)</dd>
  <dt>Sampler type</dt><dd>SyntheticTrainRealTestSampler (GaussianClassConditionalSampler)</dd>
</dl>
"""

_GMM2R_ARM_SUMMARY = (
    "Class-conditional Gaussian mixture models are fitted to the standardized "
    "datatraining.txt pool. Balanced training samples are generated synthetically "
    "from these GMMs. Evaluation uses the fixed real Occupancy evaluation split "
    "(standardized datatest.txt + datatest2.txt)."
)

_GMM2R_SCIENTIFIC_ROLE = (
    "This arm tests whether a more flexible multimodal class-conditional "
    "approximation better preserves the real-data predictive cooperation profile "
    "than the single-Gaussian approximation. If GMM results closely match "
    "Real → Real, the multimodal synthetic distribution captures the same "
    "predictive cooperation profile."
)

_GMM2R_TRAINING_PROTOCOL = """
<dl class="meta">
  <dt>Training source</dt>
  <dd>Synthetic — class-conditional GMMs fitted to
      <code>datatraining.txt</code></dd>
  <dt>Evaluation source</dt>
  <dd>Fixed real evaluation split: standardized <code>datatest.txt + datatest2.txt</code></dd>
  <dt>Model selection</dt><dd>BIC criterion over candidate component counts</dd>
  <dt>Standardization</dt><dd>z-score, fitted on <code>datatraining.txt</code> only</dd>
</dl>
"""

_GMM2R_REPRODUCIBILITY = """
<dl class="meta">
  <dt>Model type</dt><dd>Class-conditional Gaussian Mixture Model per class</dd>
  <dt>Fitting source</dt><dd><code>datatraining.txt</code> (standardized)</dd>
  <dt>Model selection criterion</dt><dd>BIC</dd>
  <dt>Test split</dt><dd><code>datatest.txt + datatest2.txt</code> (real, fixed)</dd>
  <dt>Sampler type</dt><dd>SyntheticTrainRealTestSampler (GMMClassConditionalSampler)</dd>
</dl>
"""


# --------------------------------------------------------------------------- #
# Report generators
# --------------------------------------------------------------------------- #

def generate_occupancy_real_monte_carlo_report(
    result: SimulationResult,
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    """Generate the structured Real → Real Monte Carlo arm report."""
    channel_names = result.metadata.get("channel_names")
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — Occupancy Real → Real Monte Carlo Report",
        arm_id="real_to_real",
        arm_label="Real → Real",
        arm_summary=_REAL_ARM_SUMMARY,
        scientific_role=_REAL_SCIENTIFIC_ROLE,
        channel_names=channel_names,
        fixed_test_description=(
            "fixed real Occupancy evaluation split "
            "(standardized datatest.txt + datatest2.txt)"
        ),
        training_protocol_html=_REAL_TRAINING_PROTOCOL,
        reproducibility_html=_REAL_REPRODUCIBILITY,
        nstar_selection_result=nstar_selection_result or result,
    )


def generate_occupancy_gaussian_anchored_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_gaussian_anchored_monte_carlo_report.html",
) -> Path:
    """Generate the detailed Gaussian-anchored Monte Carlo report (legacy)."""
    return generate_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim - Occupancy Gaussian-Anchored Monte Carlo Report",
        experiment_arm="Gaussian-anchored Monte Carlo",
        description=(
            "Class-conditional Gaussian parameters are estimated from the "
            "standardized datatraining.txt pool, then synthetic balanced "
            "training and fixed test samples are generated from that model."
        ),
        channel_names=channel_names,
        fixed_test_description="fixed synthetic Gaussian test set",
        extra_sections={
            "Estimated Gaussian parameters": gaussian_parameters_section(
                result.model, channel_names
            )
        },
    )


def generate_occupancy_single_gaussian_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_single_gaussian_to_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    """Generate the structured Single Gaussian → Real Monte Carlo arm report."""
    model_html = (
        "<h3>Estimated Gaussian training parameters</h3>"
        + gaussian_parameters_section(result.model, list(channel_names))
    )
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — Occupancy Single Gaussian → Real Monte Carlo Report",
        arm_id="single_gaussian_to_real",
        arm_label="Single Gaussian → Real",
        arm_summary=_SG2R_ARM_SUMMARY,
        scientific_role=_SG2R_SCIENTIFIC_ROLE,
        channel_names=channel_names,
        fixed_test_description=(
            "fixed real Occupancy evaluation split "
            "(standardized datatest.txt + datatest2.txt)"
        ),
        training_protocol_html=_SG2R_TRAINING_PROTOCOL,
        model_section_html=model_html,
        reproducibility_html=_SG2R_REPRODUCIBILITY,
        nstar_selection_result=nstar_selection_result,
    )


def generate_occupancy_gmm_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_gmm_to_real_monte_carlo_report.html",
    nstar_selection_result: SimulationResult | None = None,
) -> Path:
    """Generate the structured GMM → Real Monte Carlo arm report."""
    model_html = (
        "<h3>Estimated GMM training model</h3>"
        + gmm_parameters_section(result.model, list(channel_names))
    )
    return generate_structured_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim — Occupancy GMM → Real Monte Carlo Report",
        arm_id="gmm_to_real",
        arm_label="GMM → Real",
        arm_summary=_GMM2R_ARM_SUMMARY,
        scientific_role=_GMM2R_SCIENTIFIC_ROLE,
        channel_names=channel_names,
        fixed_test_description=(
            "fixed real Occupancy evaluation split "
            "(standardized datatest.txt + datatest2.txt)"
        ),
        training_protocol_html=_GMM2R_TRAINING_PROTOCOL,
        model_section_html=model_html,
        reproducibility_html=_GMM2R_REPRODUCIBILITY,
        nstar_selection_result=nstar_selection_result,
    )
