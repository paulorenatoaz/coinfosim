"""Compatibility shim for the canonical predictive-cooperation-profile analysis.

The active implementation lives in :mod:`coinfosim.results.predictive_profile`.
This module re-exports the canonical functions under their previous names for
one compatibility cycle, and retains the historical directed-crossing / N*
helpers that are not part of the canonical framework (see
``docs/migration-predictive-profile-schema.md``). Active code must import
from :mod:`coinfosim.results.predictive_profile` directly; do not add new
callers of this module.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from coinfosim.results.predictive_profile import (
    PredictiveProfileCompatibility as StructuralCompatibility,
    Subset,
    effective_winner_matrices,
    loss_vector,
    observed_pairwise_outcome,
    progressive_reversal_agreement as progressive_reversal_fidelity,
    progressive_reversal_matrices,
    rank_vector,
    ranking_fidelity_series,
    scenario_predictive_profile_agreement as scenario_structural_fidelity,
    select_display_subsets_by_cardinality,
    simulation_pairwise_profile_dynamics as simulation_structural_dynamics,
    validate_predictive_profile_compatibility as validate_structural_compatibility,
    winner_agreement_series,
    winner_matrix,
    winner_reversal_events,
)
from coinfosim.simulation.monte_carlo import SimulationResult

__all__ = [
    "StructuralCompatibility",
    "validate_structural_compatibility",
    "loss_vector",
    "rank_vector",
    "observed_pairwise_outcome",
    "ranking_fidelity_series",
    "winner_matrix",
    "winner_agreement_series",
    "effective_winner_matrices",
    "winner_reversal_events",
    "progressive_reversal_matrices",
    "progressive_reversal_fidelity",
    "select_display_subsets_by_cardinality",
    "simulation_structural_dynamics",
    "scenario_structural_fidelity",
    "directed_crossing_events",
    "progressive_directed_nstar",
]


# --------------------------------------------------------------------------- #
# Historical directed-crossing / N* helpers (deprecated, not canonical).
#
# These implement the retired first-threshold / geometric-crossing framework.
# They are kept only for the legacy tests/figures that still exercise them
# directly; they are not part of the active predictive-cooperation-profile
# report path (see coinfosim.results.predictive_profile) and must not be used
# by new code.
# --------------------------------------------------------------------------- #
def directed_crossing_events(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, int]]:
    """Extract observed-grid row-becomes-better events for ordered pairs. Deprecated."""

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    sample_sizes = tuple(int(n) for n in result.sample_sizes)
    losses = {
        n: loss_vector(result, classifier, n, catalog) for n in sample_sizes
    }
    events: List[Dict[str, int]] = []
    for row in range(len(catalog)):
        for col in range(len(catalog)):
            if row == col:
                continue
            for previous_n, current_n in zip(sample_sizes, sample_sizes[1:]):
                previous_delta = losses[previous_n][row] - losses[previous_n][col]
                current_delta = losses[current_n][row] - losses[current_n][col]
                if previous_delta >= 0 and current_delta < 0:
                    events.append(
                        {"row": row, "col": col, "n_crossing": current_n}
                    )
    return events


def progressive_directed_nstar(
    result: SimulationResult,
    classifier: str,
    subsets: Optional[Sequence[Subset]] = None,
) -> List[Dict[str, object]]:
    """Reconstruct latest same-direction crossing matrices from sparse events. Deprecated."""

    catalog = tuple(tuple(s) for s in (subsets or result.subsets))
    sample_sizes = tuple(int(n) for n in result.sample_sizes)
    events = directed_crossing_events(result, classifier, catalog)
    latest: Dict[Tuple[int, int], int] = {}
    by_n: Dict[int, List[Dict[str, int]]] = {}
    for event in events:
        by_n.setdefault(event["n_crossing"], []).append(event)
    progressive: List[Dict[str, object]] = []
    for n_prefix in sample_sizes[1:]:
        for event in by_n.get(n_prefix, []):
            latest[(event["row"], event["col"])] = event["n_crossing"]
        matrix: List[List[Optional[int]]] = []
        for row in range(len(catalog)):
            matrix.append(
                [
                    None if row == col else latest.get((row, col))
                    for col in range(len(catalog))
                ]
            )
        progressive.append({"n_prefix": n_prefix, "matrix": matrix})
    return progressive
