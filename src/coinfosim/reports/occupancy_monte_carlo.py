"""Occupancy-specific Monte Carlo report wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from coinfosim.reports.monte_carlo import (
    gaussian_parameters_section,
    generate_monte_carlo_report,
    gmm_parameters_section,
)
from coinfosim.simulation.monte_carlo import SimulationResult


def generate_occupancy_real_monte_carlo_report(
    result: SimulationResult,
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_real_monte_carlo_report.html",
) -> Path:
    """Generate the detailed real-data Monte Carlo report."""

    channel_names = result.metadata.get("channel_names")
    return generate_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim - Occupancy Real-Data Monte Carlo Report",
        experiment_arm="Real-data Monte Carlo",
        description=(
            "Balanced training samples are drawn from the standardized "
            "datatraining.txt pool and evaluated on the fixed standardized "
            "datatest.txt + datatest2.txt test set."
        ),
        channel_names=channel_names,
        fixed_test_description="standardized datatest.txt + datatest2.txt",
    )


def generate_occupancy_gaussian_anchored_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_gaussian_anchored_monte_carlo_report.html",
) -> Path:
    """Generate the detailed Gaussian-anchored Monte Carlo report."""

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
) -> Path:
    """Generate the detailed single-Gaussian-to-real Monte Carlo report.

    Training samples are drawn from a single class-conditional Gaussian model
    estimated from the standardized Occupancy training pool, while evaluation is
    performed on the fixed real Occupancy evaluation split.
    """

    return generate_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim - Occupancy Single Gaussian to Real Monte Carlo Report",
        experiment_arm="Single Gaussian → Real Monte Carlo",
        description=(
            "Class-conditional Gaussian parameters are estimated from the "
            "standardized datatraining.txt pool. Balanced training samples are "
            "generated synthetically from that single Gaussian model, while "
            "evaluation uses the fixed real Occupancy evaluation split "
            "(standardized datatest.txt + datatest2.txt)."
        ),
        channel_names=channel_names,
        fixed_test_description=(
            "fixed real Occupancy evaluation split "
            "(standardized datatest.txt + datatest2.txt)"
        ),
        extra_sections={
            "Estimated Gaussian training parameters": gaussian_parameters_section(
                result.model, channel_names
            )
        },
    )


def generate_occupancy_gmm_to_real_monte_carlo_report(
    result: SimulationResult,
    channel_names: Sequence[str],
    output_dir: Path | str = "output/reports",
    filename: str = "occupancy_gmm_to_real_monte_carlo_report.html",
) -> Path:
    """Generate the detailed GMM-to-real Monte Carlo report.

    Training samples are drawn from class-conditional Gaussian mixture models
    fitted to the standardized Occupancy training pool, while evaluation is
    performed on the fixed real Occupancy evaluation split.
    """

    return generate_monte_carlo_report(
        result=result,
        output_dir=output_dir,
        filename=filename,
        title="CoInfoSim - Occupancy GMM to Real Monte Carlo Report",
        experiment_arm="GMM → Real Monte Carlo",
        description=(
            "Class-conditional Gaussian mixture models are fitted to the "
            "standardized Occupancy training pool. Balanced training samples are "
            "generated synthetically from these GMMs, while evaluation uses the "
            "fixed real Occupancy evaluation split."
        ),
        channel_names=channel_names,
        fixed_test_description=(
            "fixed real Occupancy evaluation split "
            "(standardized datatest.txt + datatest2.txt)"
        ),
        extra_sections={
            "Estimated GMM training model": gmm_parameters_section(
                result.model, channel_names
            )
        },
    )
