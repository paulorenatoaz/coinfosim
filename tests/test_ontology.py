"""Tests for the CoInfoSim OWL ontology (Blocks O1/O2).

Parses ``ontology/coinfosim.owl.ttl`` with ``rdflib`` and checks structural
requirements from the task plan Sections 14-19. No reasoner is used or
required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rdflib
from rdflib.namespace import OWL, RDF, RDFS, XSD

REPO_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = REPO_ROOT / "ontology" / "coinfosim.owl.ttl"
VOCAB_PATH = REPO_ROOT / "src" / "coinfosim" / "resources" / "scientific_vocabulary.json"

COINFOSIM = rdflib.Namespace("https://paulorenatoaz.github.io/coinfosim/ns#")
PROV = rdflib.Namespace("http://www.w3.org/ns/prov#")
ONTOLOGY_IRI = rdflib.URIRef("https://paulorenatoaz.github.io/coinfosim/ontology/coinfosim")
VERSION_IRI = rdflib.URIRef(
    "https://paulorenatoaz.github.io/coinfosim/ontology/coinfosim/1.0.0"
)
PROV_O_IRI = rdflib.URIRef("http://www.w3.org/ns/prov-o#")

REQUIRED_SCIENTIFIC_CLASSES = (
    # 15.1
    "PredictiveCooperation",
    "PredictiveComplementarity",
    "PredictiveRedundancy",
    "PredictiveCooperationProfile",
    "PredictiveCooperationPattern",
    # 15.2
    "Attribute",
    "InformationChannel",
    "ChannelSubset",
    "Classifier",
    "TrainingCondition",
    # 15.3
    "PairwiseWinnerRelation",
    "WinnerMatrix",
    "WinnerReversal",
    "ReversalMatrix",
    # 15.4
    "PredictiveCooperationMetric",
    "EstimatedMeanTestLoss",
    "RankingFidelity",
    "WinnerAgreement",
    "ReversalExistenceAgreement",
    "MeanLog2ReversalDistance",
    "ReversalSampleSizeSimilarity",
)

REQUIRED_PROVENANCE_ENTITY_CLASSES = (
    "SourceDataset",
    "PreparedDataset",
    "TrainingReservoir",
    "FixedRealTestSet",
    "TargetSpecification",
    "PreprocessingSpecification",
    "FittedGenerator",
    "FittedSingleGaussianGenerator",
    "FittedGMMGenerator",
    "RandomForestCalibrationArtifact",
    "ResultData",
    "PredictiveCooperationProfile",
    "ReportArtifact",
    "CodeRevision",
    "ExecutionEnvironment",
    "RecoveredArtifactSet",
)

REQUIRED_ACTIVITY_CLASSES = (
    "DatasetPreparation",
    "GeneratorFitting",
    "FitSingleGaussian",
    "FitGMM",
    "SimulationRun",
    "RealSimulationRun",
    "SingleGaussianSimulationRun",
    "GMMSimulationRun",
    "ProfileComputation",
    "ReportGeneration",
    "ArtifactRecovery",
)


@pytest.fixture(scope="module")
def graph() -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(str(ONTOLOGY_PATH), format="turtle")
    return g


@pytest.fixture(scope="module")
def vocabulary() -> dict:
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def test_ontology_file_exists():
    assert ONTOLOGY_PATH.exists()


def test_turtle_parses_with_rdflib(graph):
    assert len(graph) > 0


def test_ontology_node_is_owl_ontology(graph):
    assert (ONTOLOGY_IRI, RDF.type, OWL.Ontology) in graph


def test_version_iri_exists(graph):
    assert (ONTOLOGY_IRI, OWL.versionIRI, VERSION_IRI) in graph


def test_prov_o_import_exists(graph):
    assert (ONTOLOGY_IRI, OWL.imports, PROV_O_IRI) in graph


@pytest.mark.parametrize("name", REQUIRED_SCIENTIFIC_CLASSES)
def test_required_scientific_class_exists(graph, name):
    assert (COINFOSIM[name], RDF.type, OWL.Class) in graph


@pytest.mark.parametrize("name", REQUIRED_PROVENANCE_ENTITY_CLASSES)
def test_required_provenance_entity_class_exists(graph, name):
    assert (COINFOSIM[name], RDF.type, OWL.Class) in graph
    assert (COINFOSIM[name], RDFS.subClassOf, PROV.Entity) in graph


@pytest.mark.parametrize("name", REQUIRED_ACTIVITY_CLASSES)
def test_required_activity_class_exists(graph, name):
    assert (COINFOSIM[name], RDF.type, OWL.Class) in graph
    assert (COINFOSIM[name], RDFS.subClassOf, PROV.Activity) in graph


def test_experiment_configuration_is_a_plan(graph):
    assert (COINFOSIM["ExperimentConfiguration"], RDFS.subClassOf, PROV.Plan) in graph


def test_coinfosim_software_agent_is_a_software_agent(graph):
    assert (
        COINFOSIM["CoInfoSimSoftwareAgent"],
        RDFS.subClassOf,
        PROV.SoftwareAgent,
    ) in graph


def test_predictive_cooperation_hierarchy(graph):
    assert (
        COINFOSIM["PredictiveComplementarity"],
        RDFS.subClassOf,
        COINFOSIM["PredictiveCooperation"],
    ) in graph
    assert (
        COINFOSIM["PredictiveRedundancy"],
        RDFS.subClassOf,
        COINFOSIM["PredictiveCooperation"],
    ) in graph


@pytest.mark.parametrize(
    "name",
    (
        "EstimatedMeanTestLoss",
        "RankingFidelity",
        "WinnerAgreement",
        "ReversalExistenceAgreement",
        "MeanLog2ReversalDistance",
        "ReversalSampleSizeSimilarity",
    ),
)
def test_metric_hierarchy(graph, name):
    assert (
        COINFOSIM[name],
        RDFS.subClassOf,
        COINFOSIM["PredictiveCooperationMetric"],
    ) in graph


def test_fitted_generator_secondary_hierarchy(graph):
    assert (
        COINFOSIM["FittedSingleGaussianGenerator"],
        RDFS.subClassOf,
        COINFOSIM["FittedGenerator"],
    ) in graph
    assert (
        COINFOSIM["FittedGMMGenerator"],
        RDFS.subClassOf,
        COINFOSIM["FittedGenerator"],
    ) in graph


def test_generator_fitting_secondary_hierarchy(graph):
    assert (
        COINFOSIM["FitSingleGaussian"],
        RDFS.subClassOf,
        COINFOSIM["GeneratorFitting"],
    ) in graph
    assert (
        COINFOSIM["FitGMM"],
        RDFS.subClassOf,
        COINFOSIM["GeneratorFitting"],
    ) in graph


def test_simulation_run_secondary_hierarchy(graph):
    for name in (
        "RealSimulationRun",
        "SingleGaussianSimulationRun",
        "GMMSimulationRun",
    ):
        assert (
            COINFOSIM[name],
            RDFS.subClassOf,
            COINFOSIM["SimulationRun"],
        ) in graph


def test_predictive_cooperation_pattern_is_not_subclass_of_profile(graph):
    assert (
        COINFOSIM["PredictiveCooperationPattern"],
        RDFS.subClassOf,
        COINFOSIM["PredictiveCooperationProfile"],
    ) not in graph


def test_attribute_is_not_declared_equivalent_to_information_channel(graph):
    assert (
        COINFOSIM["Attribute"],
        OWL.equivalentClass,
        COINFOSIM["InformationChannel"],
    ) not in graph
    assert (
        COINFOSIM["InformationChannel"],
        OWL.equivalentClass,
        COINFOSIM["Attribute"],
    ) not in graph


def test_code_revision_is_entity_subclass_and_not_agent_subclass(graph):
    assert (COINFOSIM["CodeRevision"], RDFS.subClassOf, PROV.Entity) in graph
    assert (COINFOSIM["CodeRevision"], RDFS.subClassOf, PROV.Agent) not in graph
    assert (COINFOSIM["CodeRevision"], RDFS.subClassOf, PROV.SoftwareAgent) not in graph


@pytest.mark.parametrize(
    "name",
    (
        "PredictiveCooperation",
        "PredictiveComplementarity",
        "PredictiveRedundancy",
        "PredictiveCooperationProfile",
        "PredictiveCooperationPattern",
        "Attribute",
        "InformationChannel",
        "PairwiseWinnerRelation",
        "WinnerMatrix",
        "WinnerReversal",
        "ReversalMatrix",
        "EstimatedMeanTestLoss",
        "RankingFidelity",
        "WinnerAgreement",
        "ReversalExistenceAgreement",
        "MeanLog2ReversalDistance",
        "ReversalSampleSizeSimilarity",
    ),
)
def test_vocabulary_matched_labels_and_definitions_are_copied_verbatim(
    graph, vocabulary, name
):
    concept = vocabulary["concepts"][name]
    subject = COINFOSIM[name]
    labels_en = {str(o) for o in graph.objects(subject, RDFS.label) if o.language == "en"}
    labels_pt = {str(o) for o in graph.objects(subject, RDFS.label) if o.language == "pt"}
    comments_en = {
        str(o) for o in graph.objects(subject, RDFS.comment) if o.language == "en"
    }
    comments_pt = {
        str(o) for o in graph.objects(subject, RDFS.comment) if o.language == "pt"
    }
    assert concept["labels"]["en"] in labels_en
    assert concept["labels"]["pt"] in labels_pt
    assert concept["definitions"]["en"] in comments_en
    assert concept["definitions"]["pt"] in comments_pt


def test_no_retired_term_is_a_canonical_preferred_label(graph):
    retired_terms = (
        "N*",
        "N-star",
        "cooperative threshold",
        "structural fidelity",
        "structural dynamics",
        "nstar_similarity",
    )
    all_labels = {
        str(o).lower()
        for _s, _p, o in graph.triples((None, RDFS.label, None))
    }
    for retired in retired_terms:
        assert retired.lower() not in all_labels
