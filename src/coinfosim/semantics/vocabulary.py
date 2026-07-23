"""Loader for the packaged CoInfoSim scientific vocabulary.

The vocabulary is the stable, machine-readable ledger of canonical semantic
identifiers for the predictive-cooperation-profile framework
(``coinfosim/resources/scientific_vocabulary.json``). It underlies the JSON-LD
context and later provenance/ontology work; it defines no scientific
calculation itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional

_RESOURCE_PACKAGE = "coinfosim.resources"
_VOCABULARY_FILENAME = "scientific_vocabulary.json"
_CONTEXT_FILENAME = "coinfosim-context.jsonld"


class VocabularyError(ValueError):
    """Raised when the packaged vocabulary is malformed or a lookup fails."""


@dataclass(frozen=True)
class Concept:
    """One canonical semantic concept from the vocabulary ledger."""

    name: str
    id: str
    labels: Mapping[str, str]
    definitions: Mapping[str, str]
    symbol: Optional[str]
    broader_concept: Optional[str]
    canonical_python_key: Optional[str]
    deprecated_aliases: List[str]

    @property
    def label_en(self) -> str:
        return self.labels["en"]

    @property
    def label_pt(self) -> str:
        return self.labels["pt"]


def _read_packaged_text(filename: str) -> str:
    return resources.files(_RESOURCE_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def _validate_and_index(raw: Dict[str, Any]) -> Dict[str, Any]:
    for required in ("vocabulary_version", "namespace", "preferred_language", "concepts"):
        if required not in raw:
            raise VocabularyError(f"vocabulary is missing required field {required!r}")

    concepts = raw["concepts"]
    seen_ids: Dict[str, str] = {}
    seen_keys: Dict[str, str] = {}
    seen_en_labels: Dict[str, str] = {}
    indexed: Dict[str, Concept] = {}

    for name, entry in concepts.items():
        concept_id = entry["id"]
        if concept_id in seen_ids:
            raise VocabularyError(
                f"duplicate semantic ID {concept_id!r} on concepts "
                f"{seen_ids[concept_id]!r} and {name!r}"
            )
        seen_ids[concept_id] = name

        label_en = entry["labels"]["en"]
        if label_en in seen_en_labels:
            raise VocabularyError(
                f"duplicate English preferred label {label_en!r} on concepts "
                f"{seen_en_labels[label_en]!r} and {name!r}"
            )
        seen_en_labels[label_en] = name

        canonical_key = entry.get("canonical_python_key")
        if canonical_key is not None:
            if canonical_key in seen_keys:
                raise VocabularyError(
                    f"duplicate canonical_python_key {canonical_key!r} on concepts "
                    f"{seen_keys[canonical_key]!r} and {name!r}"
                )
            seen_keys[canonical_key] = name

        deprecated_aliases = list(entry.get("deprecated_aliases", []))
        for alias in deprecated_aliases:
            if alias == label_en or alias == entry["labels"].get("pt"):
                raise VocabularyError(
                    f"deprecated alias {alias!r} on concept {name!r} must not equal "
                    "a preferred label"
                )

        indexed[name] = Concept(
            name=name,
            id=concept_id,
            labels=dict(entry["labels"]),
            definitions=dict(entry["definitions"]),
            symbol=entry.get("symbol"),
            broader_concept=entry.get("broader_concept"),
            canonical_python_key=canonical_key,
            deprecated_aliases=deprecated_aliases,
        )

    return {
        "vocabulary_version": raw["vocabulary_version"],
        "namespace": raw["namespace"],
        "preferred_language": raw["preferred_language"],
        "concepts": indexed,
        "concepts_by_key": {c.canonical_python_key: c for c in indexed.values() if c.canonical_python_key},
        "deprecated_terms": dict(raw.get("deprecated_terms", {})),
    }


@lru_cache(maxsize=1)
def load_vocabulary() -> Dict[str, Any]:
    """Load, validate, and index the packaged scientific vocabulary.

    Works both from a source checkout and from an installed wheel via
    :mod:`importlib.resources`.
    """
    try:
        raw_text = _read_packaged_text(_VOCABULARY_FILENAME)
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise VocabularyError(
            f"packaged vocabulary resource {_VOCABULARY_FILENAME!r} not found"
        ) from exc
    raw = json.loads(raw_text)
    return _validate_and_index(raw)


@lru_cache(maxsize=1)
def load_context() -> Dict[str, Any]:
    """Load and parse the packaged JSON-LD context as plain JSON."""
    raw_text = _read_packaged_text(_CONTEXT_FILENAME)
    return json.loads(raw_text)


def vocabulary_version() -> str:
    return str(load_vocabulary()["vocabulary_version"])


def get_concept(name: str) -> Concept:
    """Return the :class:`Concept` for a concept name (e.g. ``"WinnerMatrix"``).

    Raises :class:`VocabularyError` for unknown concept names.
    """
    concepts = load_vocabulary()["concepts"]
    try:
        return concepts[name]
    except KeyError as exc:
        raise VocabularyError(f"unknown canonical concept id {name!r}") from exc


def get_concept_by_python_key(canonical_python_key: str) -> Concept:
    """Return the :class:`Concept` whose ``canonical_python_key`` matches."""
    concepts_by_key = load_vocabulary()["concepts_by_key"]
    try:
        return concepts_by_key[canonical_python_key]
    except KeyError as exc:
        raise VocabularyError(
            f"no concept with canonical_python_key {canonical_python_key!r}"
        ) from exc


def canonical_key_to_id(canonical_python_key: str) -> str:
    """Return the stable semantic ID for a canonical persisted JSON key."""
    return get_concept_by_python_key(canonical_python_key).id


def semantic_ids() -> List[str]:
    """Return every stable semantic ID in the vocabulary, sorted."""
    return sorted(concept.id for concept in load_vocabulary()["concepts"].values())
