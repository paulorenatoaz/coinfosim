"""Tests for the canonical predictive-cooperation semantic vocabulary."""

from __future__ import annotations

import json

import pytest

from coinfosim.semantics import (
    VocabularyError,
    canonical_key_to_id,
    get_concept,
    load_vocabulary,
    semantic_ids,
    vocabulary_version,
)
from coinfosim.semantics.vocabulary import get_concept_by_python_key, load_context

REQUIRED_CONCEPT_NAMES = [
    "PredictiveCooperation",
    "PredictiveComplementarity",
    "PredictiveRedundancy",
    "PredictiveCooperationProfile",
    "PredictiveCooperationPattern",
    "Attribute",
    "InformationChannel",
    "EstimatedMeanTestLoss",
    "PairwiseWinnerRelation",
    "WinnerMatrix",
    "WinnerReversal",
    "ReversalMatrix",
    "RankingFidelity",
    "WinnerAgreement",
    "ReversalExistenceAgreement",
    "MeanLog2ReversalDistance",
    "ReversalSampleSizeSimilarity",
]

CANONICAL_PERSISTED_KEYS = [
    "predictive_cooperation_profile",
    "pairwise_profile_dynamics",
    "ranking_fidelity_series",
    "winner_agreement_series",
    "reversal_agreement_series",
    "reversal_existence_agreement",
    "mean_log2_reversal_distance",
    "reversal_sample_size_similarity",
]

DEPRECATED_TERMS = [
    "structural_fidelity",
    "structural_dynamics",
    "nstar_similarity",
    "timing_similarity",
    "crossing_jaccard",
    "directed_crossing_events",
    "progressive_directed_nstar",
    "cooperative_threshold",
    "last_crossing",
]


def test_vocabulary_version_is_1_0_0():
    assert vocabulary_version() == "1.0.0"


def test_namespace_matches_expected_ns():
    vocab = load_vocabulary()
    assert vocab["namespace"] == "https://paulorenatoaz.github.io/coinfosim/ns#"


@pytest.mark.parametrize("name", REQUIRED_CONCEPT_NAMES)
def test_required_concept_is_present(name):
    concept = get_concept(name)
    assert concept.id == f"coinfosim:{name}"
    assert concept.label_en
    assert concept.label_pt
    assert concept.definitions["en"]
    assert concept.definitions["pt"]


def test_unknown_concept_raises_vocabulary_error():
    with pytest.raises(VocabularyError):
        get_concept("NotARealConcept")


@pytest.mark.parametrize("key", CANONICAL_PERSISTED_KEYS)
def test_every_canonical_persisted_key_maps_to_one_semantic_id(key):
    concept = get_concept_by_python_key(key)
    assert concept.id.startswith("coinfosim:")
    assert canonical_key_to_id(key) == concept.id


def test_semantic_ids_are_unique_and_sorted():
    ids = semantic_ids()
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert len(ids) >= len(REQUIRED_CONCEPT_NAMES)


@pytest.mark.parametrize("term", DEPRECATED_TERMS)
def test_deprecated_terms_are_never_preferred_labels(term):
    vocab = load_vocabulary()
    preferred_en_labels = {c.label_en for c in vocab["concepts"].values()}
    preferred_pt_labels = {c.label_pt for c in vocab["concepts"].values()}
    assert term not in preferred_en_labels
    assert term not in preferred_pt_labels


def test_deprecated_aliases_never_equal_a_preferred_label():
    vocab = load_vocabulary()
    preferred_en_labels = {c.label_en for c in vocab["concepts"].values()}
    preferred_pt_labels = {c.label_pt for c in vocab["concepts"].values()}
    for concept in vocab["concepts"].values():
        for alias in concept.deprecated_aliases:
            assert alias not in preferred_en_labels
            assert alias not in preferred_pt_labels


def test_cooperative_threshold_documented_without_false_successor():
    vocab = load_vocabulary()
    assert "cooperative_threshold" in vocab["deprecated_terms"]
    assert vocab["deprecated_terms"]["cooperative_threshold"]["replaced_by"] is None


def test_jsonld_context_parses_as_strict_json_and_declares_required_prefixes():
    context = load_context()
    assert "@context" in context
    for prefix in ("coinfosim", "prov", "rdf", "rdfs", "xsd"):
        assert prefix in context["@context"]


def test_jsonld_context_does_not_map_deprecated_keys_as_preferred_terms():
    context = load_context()["@context"]
    for term in ("structural_fidelity", "structural_dynamics", "nstar_similarity",
                 "timing_similarity", "crossing_jaccard"):
        assert term not in context


def test_vocabulary_resource_file_is_strict_json():
    from importlib import resources

    text = resources.files("coinfosim.resources").joinpath(
        "scientific_vocabulary.json"
    ).read_text(encoding="utf-8")
    json.loads(text)  # must not raise


def test_context_resource_file_is_strict_json():
    from importlib import resources

    text = resources.files("coinfosim.resources").joinpath(
        "coinfosim-context.jsonld"
    ).read_text(encoding="utf-8")
    json.loads(text)  # must not raise


def test_package_data_declares_jsonld_glob():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    package_data = data["tool"]["setuptools"]["package-data"]["coinfosim.resources"]
    assert "*.jsonld" in package_data
