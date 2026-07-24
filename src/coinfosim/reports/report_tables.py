"""Reusable report-table helpers for CoInfoSim Occupancy reports.

Block 1 infrastructure: shared helpers for compact subset notation,
full loss tables, CI95 columns, precision diagnostics, and stable
classifier ordering.  All functions are pure (no side effects) and
return either strings or pandas DataFrames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from coinfosim.classifiers.registry import CLASSIFIER_KEYS, classifier_label
from coinfosim.simulation.monte_carlo import SimulationResult


# --------------------------------------------------------------------------- #
# Subset notation helpers
# --------------------------------------------------------------------------- #

def compact_subset_label(
    subset: Sequence[int],
    channel_names: Optional[Sequence[str]] = None,
) -> str:
    """Return a compact curly-brace label for a zero-based attribute subset.

    With attribute names: ``{Temperature, Light}``
    Without attribute names: ``{X1, X3}``

    Parameters
    ----------
    subset:
        Zero-based attribute indices, e.g. ``(0, 2)``.
    channel_names:
        Optional sequence of attribute display names. When provided, attribute
        names are used instead of ``Xi`` placeholders.
    """
    indices = [int(i) for i in subset]
    if not indices:
        raise ValueError("subset must be non-empty")
    if channel_names is not None:
        parts = [str(channel_names[i]) for i in indices]
    else:
        parts = [f"X{i + 1}" for i in indices]
    return "{" + ", ".join(parts) + "}"


def compact_subset_label_xi(subset: Sequence[int]) -> str:
    """Return ``{X1, X3}`` notation regardless of attribute names."""
    return compact_subset_label(subset, channel_names=None)


# --------------------------------------------------------------------------- #
# Subset metadata table
# --------------------------------------------------------------------------- #

_CHANNEL_FLAGS: List[Tuple[str, str]] = [
    ("Temperature",    "has_temperature"),
    ("Humidity",       "has_humidity"),
    ("Light",          "has_light"),
    ("CO2",            "has_co2"),
    ("HumidityRatio",  "has_humidity_ratio"),
]


def subset_metadata_table(
    subsets: Sequence[Sequence[int]],
    channel_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return a metadata table for a list of attribute subsets.

    Columns: ``subset_id``, ``subset_label``, ``subset_size``, plus one
    boolean column per attribute (``has_temperature``, ``has_humidity``,
    ``has_light``, ``has_co2``, ``has_humidity_ratio``) when *channel_names*
    matches the standard five Occupancy attributes.

    Parameters
    ----------
    subsets:
        Ordered list of zero-based attribute-index tuples.
    channel_names:
        Optional attribute names used to populate boolean membership columns.
    """
    ch = list(channel_names) if channel_names is not None else []
    rows: List[Dict[str, Any]] = []
    for idx, subset in enumerate(subsets):
        s = tuple(int(i) for i in subset)
        row: Dict[str, Any] = {
            "subset_id": idx,
            "x_label": compact_subset_label_xi(s),
            "subset_label": (
                compact_subset_label(s, ch) if ch else compact_subset_label_xi(s)
            ),
            "subset_size": len(s),
        }
        # Boolean membership columns (only populated when attribute names match).
        for ch_name, flag in _CHANNEL_FLAGS:
            if ch:
                try:
                    pos = ch.index(ch_name)
                    row[flag] = pos in s
                except ValueError:
                    row[flag] = False
            else:
                row[flag] = False
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CI95 helpers
# --------------------------------------------------------------------------- #

_Z95 = 1.96  # z-score for 95 % confidence interval


def add_ci95_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ci95_half_width``, ``ci95_low``, ``ci95_high`` to a summary DataFrame.

    Expects ``mean_loss`` and ``standard_error`` columns.  CI95 half-width is
    computed as ``1.96 * SE``.  Returns a copy with the three new columns
    appended.
    """
    out = df.copy()
    out["ci95_half_width"] = out["standard_error"].apply(
        lambda se: _Z95 * se if (se is not None and not _is_nan(se)) else float("nan")
    )
    out["ci95_low"] = out.apply(
        lambda r: (
            r["mean_loss"] - r["ci95_half_width"]
            if not _is_nan(r["ci95_half_width"])
            else float("nan")
        ),
        axis=1,
    )
    out["ci95_high"] = out.apply(
        lambda r: (
            r["mean_loss"] + r["ci95_half_width"]
            if not _is_nan(r["ci95_half_width"])
            else float("nan")
        ),
        axis=1,
    )
    return out


def _is_nan(v: Any) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Full loss table
# --------------------------------------------------------------------------- #

def full_loss_table(
    result: SimulationResult,
    channel_names: Optional[Sequence[str]] = None,
    arm: Optional[str] = None,
    scenario_id: Optional[Any] = None,
) -> pd.DataFrame:
    """Return the full long-format loss table for one simulation result.

    Columns:

    ``scenario_id``, ``arm``, ``n_per_class``, ``classifier``,
    ``classifier_label``, ``subset_id``, ``subset_size``, ``x_label``,
    ``subset_label``, ``mean_loss``, ``std_loss``, ``se_loss``,
    ``ci95_low``, ``ci95_high``, ``ci95_half_width``, ``replications``,
    ``stopping_reason``.

    The ``test_loss`` scale is preserved as-is (empirical misclassification
    rate); no normalization is applied.

    Parameters
    ----------
    result:
        A completed :class:`~coinfosim.simulation.monte_carlo.SimulationResult`.
    channel_names:
        Optional attribute names for human-readable subset labels.
    arm:
        Optional arm identifier string (e.g. ``"real_to_real"``).
    scenario_id:
        Optional scenario run identifier.
    """
    ch = list(channel_names) if channel_names is not None else []
    subsets = list(result.subsets)
    clf_names = stable_classifier_order(result.classifier_names)

    rows: List[Dict[str, Any]] = []
    for n in result.sample_sizes:
        info = result.stopping_info.get(n)
        stopping_reason = info.reason if info is not None else None
        for s_idx, subset in enumerate(subsets):
            s = tuple(int(i) for i in subset)
            x_label = compact_subset_label_xi(s)
            sub_label = compact_subset_label(s, ch) if ch else x_label
            for clf in clf_names:
                mean = result.accumulator.mean_loss(n, s, clf)
                std = result.accumulator.std_loss(n, s, clf)
                se = result.accumulator.standard_error(n, s, clf)
                reps = result.accumulator.count(n, s, clf)
                hw = _Z95 * se if (se is not None and not _is_nan(se)) else float("nan")
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "arm": arm,
                        "n_per_class": int(n),
                        "classifier": clf,
                        "classifier_label": classifier_label(clf),
                        "subset_id": s_idx,
                        "subset_size": len(s),
                        "x_label": x_label,
                        "subset_label": sub_label,
                        "mean_loss": mean,
                        "std_loss": std,
                        "se_loss": se,
                        "ci95_low": mean - hw if not _is_nan(hw) else float("nan"),
                        "ci95_high": mean + hw if not _is_nan(hw) else float("nan"),
                        "ci95_half_width": hw,
                        "replications": reps,
                        "stopping_reason": stopping_reason,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Compact precision diagnostics
# --------------------------------------------------------------------------- #

def compact_precision_diagnostics(result: SimulationResult) -> pd.DataFrame:
    """Return a compact per-n_per_class Monte Carlo precision diagnostics table.

    Columns:

    ``n_per_class``, ``replications``, ``max_se``, ``max_ci95_half_width``,
    ``target_ci95_half_width``, ``stopping_reason``.

    The ``max_ci95_half_width`` is the stopping rule's recorded maximum CI95
    half-width (``1.96 * SE``) across all (subset, classifier) cells for that
    sample size; ``max_se`` is the corresponding standard error
    (``max_ci95_half_width / 1.96``). ``target_ci95_half_width`` is the
    configured convergence target, which the stopping rule compares directly
    against ``max_ci95_half_width`` (both are on the CI95 half-width scale).

    Parameters
    ----------
    result:
        A completed :class:`~coinfosim.simulation.monte_carlo.SimulationResult`.
    """
    target_hw = result.config.ci_half_width_target
    rows: List[Dict[str, Any]] = []
    for n in result.sample_sizes:
        info = result.stopping_info.get(n)
        replications = info.replications if info is not None else None
        max_ci_hw = info.max_ci_half_width if info is not None else float("nan")
        max_se = max_ci_hw / _Z95 if (max_ci_hw is not None and not _is_nan(max_ci_hw)) else float("nan")
        stopping_reason = info.reason if info is not None else None
        rows.append(
            {
                "n_per_class": int(n),
                "replications": replications,
                "max_se": max_se,
                "max_ci95_half_width": max_ci_hw,
                "target_ci95_half_width": target_hw,
                "stopping_reason": stopping_reason,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Stable classifier ordering
# --------------------------------------------------------------------------- #

def stable_classifier_order(classifier_names: Sequence[str]) -> List[str]:
    """Return *classifier_names* sorted in the canonical report order.

    The canonical order is ``linear_svm`` first, ``logistic_regression``
    second, ``gaussian_nb`` third.  Any unknown classifiers are appended at
    the end in their original relative order.

    Parameters
    ----------
    classifier_names:
        Classifier keys to sort.
    """
    priority = {k: i for i, k in enumerate(CLASSIFIER_KEYS)}
    known = [c for c in CLASSIFIER_KEYS if c in set(classifier_names)]
    unknown = [c for c in classifier_names if c not in set(CLASSIFIER_KEYS)]
    return known + unknown
