"""Schema versioning and backward-compatible readers for persisted profile payloads.

Three schema generations exist in persisted ``scenario.json`` /
``simulation.json`` report data:

- **schema 1** (historical, pre-W/R): top-level keys ``structural_fidelity`` /
  ``structural_dynamics``, containing the retired composite metric
  (``nstar_similarity``, ``crossing_jaccard``, ``timing_similarity``,
  directed-crossing counts). Readable for historical display only; its
  composite values must never be reinterpreted as new metrics.
- **schema 2** (W/R, no semantic layer): same top-level keys as schema 1, but
  with the separated reversal metrics (``reversal_existence_agreement``,
  ``mean_log2_reversal_distance``, ``reversal_sample_size_similarity``) and no
  composite. Upgradeable losslessly to schema 3 by renaming fields.
- **schema 3** (canonical): top-level keys ``predictive_cooperation_profile`` /
  ``pairwise_profile_dynamics``, with ``semantic_vocabulary_version`` and
  ``semantic_type`` fields. Produced by
  :mod:`coinfosim.results.predictive_profile`.

This module never recomputes scientific values; it only reads, classifies,
and (schema 2 only) losslessly renames already-persisted fields.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from coinfosim.semantics.vocabulary import canonical_key_to_id, vocabulary_version

SCHEMA_VERSION = 3

CANONICAL_KEYS = {
    "scenario": "predictive_cooperation_profile",
    "simulation": "pairwise_profile_dynamics",
}
LEGACY_KEYS = {
    "scenario": "structural_fidelity",
    "simulation": "structural_dynamics",
}

# Fields present only in schema 1 (the retired composite/N* era). These
# describe a non-decomposable product metric and must never be reinterpreted
# as coinfosim:ReversalExistenceAgreement or coinfosim:ReversalSampleSizeSimilarity.
SCHEMA1_NON_UPGRADABLE_FIELDS = frozenset(
    {
        "nstar_similarity_series",
        "nstar_similarity",
        "n_reference_crossings",
        "n_arm_crossings",
        "n_shared_crossings",
        "n_union_crossings",
        "crossing_jaccard",
        "timing_similarity",
        "nstar_status",
    }
)

# Lossless schema-2 -> schema-3 field renames for reversal-agreement rows.
SCHEMA2_TO_SCHEMA3_FIELD_RENAMES = {
    "n_reference_reversal_pairs": "reference_reversal_pair_count",
    "n_arm_reversal_pairs": "arm_reversal_pair_count",
    "n_shared_reversal_pairs": "shared_reversal_pair_count",
    "n_union_reversal_pairs": "union_reversal_pair_count",
}

# Lossless schema-2 -> schema-3 top-level container key renames.
SCHEMA2_TO_SCHEMA3_CONTAINER_RENAMES = {
    "reversal_fidelity_series": "reversal_agreement_series",
}

# Lossless schema-2 -> schema-3 per-classifier simulation-level field renames.
SCHEMA2_TO_SCHEMA3_SIMULATION_RENAMES = {
    "effective_winner_pairs_by_n": "effective_winner_relations_by_n",
}


class ProfileSchemaError(ValueError):
    """Raised for malformed payloads or disallowed schema upgrades."""


@dataclass(frozen=True)
class LoadedProfilePayload:
    """A predictive-profile payload as found in persisted report data."""

    level: str
    schema_version: int
    is_canonical: bool
    payload: Mapping[str, Any]

    @property
    def is_historical(self) -> bool:
        return self.schema_version < SCHEMA_VERSION


@dataclass(frozen=True)
class UpgradeResult:
    """Result of attempting to upgrade a loaded payload to canonical schema 3."""

    payload: Dict[str, Any]
    upgraded: bool
    non_upgradable_fields: List[str] = field(default_factory=list)


def load_predictive_profile_payload(
    report_data: Mapping[str, Any], *, level: str
) -> LoadedProfilePayload:
    """Locate and classify a scenario- or simulation-level profile payload.

    Looks for the canonical key first, then the legacy key. Raises
    :class:`ProfileSchemaError` if neither is present.
    """

    if level not in CANONICAL_KEYS:
        raise ProfileSchemaError(f"unknown payload level {level!r}")

    canonical_key = CANONICAL_KEYS[level]
    legacy_key = LEGACY_KEYS[level]

    if canonical_key in report_data:
        payload = report_data[canonical_key]
        version = int(payload.get("schema_version", SCHEMA_VERSION))
        return LoadedProfilePayload(
            level=level,
            schema_version=version,
            is_canonical=(version == SCHEMA_VERSION and canonical_key in report_data),
            payload=payload,
        )
    if legacy_key in report_data:
        payload = report_data[legacy_key]
        version = int(payload.get("schema_version", 1))
        return LoadedProfilePayload(
            level=level, schema_version=version, is_canonical=False, payload=payload
        )
    raise ProfileSchemaError(
        f"no {level} predictive-profile payload found "
        f"(looked for {canonical_key!r} and {legacy_key!r})"
    )


def _rename_keys(row: Mapping[str, Any], renames: Mapping[str, str]) -> Dict[str, Any]:
    return {renames.get(key, key): value for key, value in row.items()}


def _upgrade_schema2_scenario_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    upgraded = copy.deepcopy(dict(payload))
    for old_container_key, new_container_key in SCHEMA2_TO_SCHEMA3_CONTAINER_RENAMES.items():
        if old_container_key in upgraded:
            rows = upgraded.pop(old_container_key)
            upgraded[new_container_key] = [
                _rename_keys(row, SCHEMA2_TO_SCHEMA3_FIELD_RENAMES) for row in rows
            ]
    if "final_summary" in upgraded:
        upgraded["final_summary"] = [
            _rename_keys(row, SCHEMA2_TO_SCHEMA3_FIELD_RENAMES)
            for row in upgraded["final_summary"]
        ]
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded["semantic_vocabulary_version"] = vocabulary_version()
    upgraded["semantic_type"] = canonical_key_to_id(CANONICAL_KEYS["scenario"])
    return upgraded


def _upgrade_schema2_simulation_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    upgraded = copy.deepcopy(dict(payload))
    classifiers = upgraded.get("classifiers", {})
    for classifier_data in classifiers.values():
        for old_key, new_key in SCHEMA2_TO_SCHEMA3_SIMULATION_RENAMES.items():
            if old_key in classifier_data:
                pairs_by_n = classifier_data.pop(old_key)
                classifier_data[new_key] = [
                    {
                        "n_per_class": int(n_str),
                        "relations": relations,
                    }
                    for n_str, relations in sorted(pairs_by_n.items(), key=lambda kv: int(kv[0]))
                ]
        classifier_data.setdefault("reversal_matrices_by_prefix", [])
    upgraded["schema_version"] = SCHEMA_VERSION
    upgraded["semantic_vocabulary_version"] = vocabulary_version()
    upgraded["semantic_type"] = canonical_key_to_id(CANONICAL_KEYS["simulation"])
    return upgraded


def upgrade_predictive_profile_payload(loaded: LoadedProfilePayload) -> UpgradeResult:
    """Attempt a lossless upgrade of a loaded payload to canonical schema 3.

    Schema 3 payloads are returned unchanged. Schema 2 payloads are upgraded
    losslessly (field/container renames only; no recomputation). Schema 1
    payloads are refused: their composite metric cannot be decomposed into
    the new reversal-existence-agreement / reversal-sample-size-similarity
    pair, so raises :class:`ProfileSchemaError` rather than fabricating new
    metrics from old composite values.
    """

    if loaded.schema_version >= SCHEMA_VERSION:
        return UpgradeResult(payload=dict(loaded.payload), upgraded=False, non_upgradable_fields=[])

    if loaded.schema_version == 1:
        present_non_upgradable = sorted(
            _find_fields(loaded.payload, SCHEMA1_NON_UPGRADABLE_FIELDS)
        )
        raise ProfileSchemaError(
            "schema 1 payload contains a non-decomposable composite reversal "
            f"metric and cannot be upgraded to schema {SCHEMA_VERSION}; fields: "
            f"{present_non_upgradable}. Load it for historical display only."
        )

    if loaded.schema_version == 2:
        if loaded.level == "scenario":
            upgraded = _upgrade_schema2_scenario_payload(loaded.payload)
        else:
            upgraded = _upgrade_schema2_simulation_payload(loaded.payload)
        return UpgradeResult(payload=upgraded, upgraded=True, non_upgradable_fields=[])

    raise ProfileSchemaError(f"unknown schema_version {loaded.schema_version!r}")


def _find_fields(payload: Any, needles: frozenset) -> List[str]:
    found: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in needles:
                    found.append(key)
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return sorted(set(found))
