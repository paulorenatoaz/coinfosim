# Provenance Mapping — PROV-O Roles

This document defines the canonical mapping between CoInfoSim objects and
[W3C PROV-O](https://www.w3.org/TR/prov-o/) roles, and how that mapping is
implemented and serialized.

There is exactly **one** canonical provenance model per scenario run: a
`prov.model.ProvDocument`, built by
[`build_scenario_prov_document`](../../src/coinfosim/provenance/model.py)
from normalized `ProvenanceEvidence`
([`src/coinfosim/provenance/evidence.py`](../../src/coinfosim/provenance/evidence.py)).
Every serialized/rendered artifact — PROV-JSON, PROV-N, PROV-O/Turtle,
and (when Graphviz is available) PNG/PDF — is derived from that same
document by
[`export_provenance_artifacts`](../../src/coinfosim/provenance/export.py).
The CoInfoSim domain-specific entity/activity types (`coinfosim:*`) used
below are declared by the OWL ontology at
[`ontology/coinfosim.owl.ttl`](../../ontology/coinfosim.owl.ttl), which
specializes PROV-O.

## Role table

| CoInfoSim object | PROV-O role | CoInfoSim type |
|---|---|---|
| Source dataset | `prov:Entity` | `coinfosim:SourceDataset` |
| Prepared dataset | `prov:Entity` | `coinfosim:PreparedDataset` |
| Training reservoir | `prov:Entity` | `coinfosim:TrainingReservoir` |
| Fixed real test set | `prov:Entity` | `coinfosim:FixedRealTestSet` |
| Target specification | `prov:Entity` | `coinfosim:TargetSpecification` |
| Preprocessing specification | `prov:Entity` | `coinfosim:PreprocessingSpecification` |
| Experiment configuration | `prov:Entity` (`prov:Plan`) | `coinfosim:ExperimentConfiguration` |
| Fitted single Gaussian generator | `prov:Entity` | `coinfosim:FittedSingleGaussianGenerator` |
| Fitted GMM generator | `prov:Entity` | `coinfosim:FittedGMMGenerator` |
| Random Forest calibration artifact (only when used) | `prov:Entity` | `coinfosim:RandomForestCalibrationArtifact` |
| Per-arm Monte Carlo result data | `prov:Entity` | `coinfosim:ResultData` |
| Predictive cooperation profile | `prov:Entity` | `coinfosim:PredictiveCooperationProfile` |
| Report artifact (dataset/arm/consolidated) | `prov:Entity` | `coinfosim:ReportArtifact` |
| Git commit | `prov:Entity` | `coinfosim:CodeRevision` |
| Execution environment | `prov:Entity` | `coinfosim:ExecutionEnvironment` |
| Recovered `gh-pages` artifact set (only when recovery evidence exists) | `prov:Entity` | `coinfosim:RecoveredArtifactSet` |
| Dataset preparation | `prov:Activity` | `coinfosim:DatasetPreparation` |
| Single Gaussian fitting | `prov:Activity` | `coinfosim:FitSingleGaussian` |
| GMM fitting | `prov:Activity` | `coinfosim:FitGMM` |
| Real-arm Monte Carlo simulation run | `prov:Activity` | `coinfosim:RealSimulationRun` |
| Single-Gaussian-arm Monte Carlo simulation run | `prov:Activity` | `coinfosim:SingleGaussianSimulationRun` |
| GMM-arm Monte Carlo simulation run | `prov:Activity` | `coinfosim:GMMSimulationRun` |
| Predictive cooperation profile computation | `prov:Activity` | `coinfosim:ProfileComputation` |
| Report generation | `prov:Activity` | `coinfosim:ReportGeneration` |
| Historical artifact recovery (only when recovery evidence exists) | `prov:Activity` | `coinfosim:ArtifactRecovery` |
| CoInfoSim (the software itself) | `prov:SoftwareAgent` | `coinfosim:CoInfoSimSoftwareAgent` |

A **simulation run is always a `prov:Activity`**, never a `prov:Entity` — it
is a computation, not a data artifact. A **Git commit is always a
`prov:Entity`** (`coinfosim:CodeRevision`), never a `prov:Agent`: the only
agent in the graph is CoInfoSim itself
(`coinfosim:CoInfoSimSoftwareAgent`, a `prov:SoftwareAgent`).

## Granularity

Provenance is scenario/arm-level, not per-replication: one
`DatasetPreparation` activity, up to two generator-fitting activities, three
simulation-run activities (one per training condition), one
`ProfileComputation` activity, and one `ReportGeneration` activity per
scenario run. There is no provenance node per Monte Carlo replication,
sample size, attribute subset, classifier fit, matrix cell, or loss estimate.

## Causal structure

```text
SourceDataset --> DatasetPreparation --> PreparedDataset, TrainingReservoir, FixedRealTestSet
TargetSpecification, PreprocessingSpecification --> DatasetPreparation

TrainingReservoir --> FitSingleGaussian --> FittedSingleGaussianGenerator
TrainingReservoir --> FitGMM --> FittedGMMGenerator

TrainingReservoir, FixedRealTestSet, ExperimentConfiguration --> RealSimulationRun --> ResultData (real)
FittedSingleGaussianGenerator, FixedRealTestSet, ExperimentConfiguration --> SingleGaussianSimulationRun --> ResultData (Gaussian)
FittedGMMGenerator, FixedRealTestSet, ExperimentConfiguration --> GMMSimulationRun --> ResultData (GMM)

ResultData (real, Gaussian, GMM) --> ProfileComputation --> PredictiveCooperationProfile
PredictiveCooperationProfile --> ReportGeneration --> ReportArtifact(s)
```

All three simulation-run activities `prov:used` the *same*
`FixedRealTestSet` entity.

## No-test-leakage invariant

The graph makes the scientific train/test separation testable:

- **Required:** `TrainingReservoir` is `prov:used` by `FitSingleGaussian` and
  `FitGMM`.
- **Forbidden:** `FixedRealTestSet` is never `prov:used` by
  `FitSingleGaussian` or `FitGMM`.
- **Required:** `FixedRealTestSet` is `prov:used` by all three simulation-run
  activities.

This invariant is covered by automated tests in
[`tests/test_provenance_model.py`](../../tests/test_provenance_model.py).

## Historical regeneration

When a scenario is regenerated from persisted results (no Monte Carlo
rerun), the historical `DatasetPreparation`/`FitSingleGaussian`/`FitGMM`/
`*SimulationRun` activities are **never** associated with the current commit
or execution environment — only the activities that actually run now
(`ProfileComputation`, `ReportGeneration`, and, when recovery evidence is
present, `ArtifactRecovery`) are. An original simulation commit is never
guessed; it is only recorded when genuinely persisted. When
`docs/provenance/gh_pages_recovery_manifest.json` gives an explicit recovery
source commit, a `coinfosim:RecoveredArtifactSet` entity and a
`coinfosim:ArtifactRecovery` activity are added to the graph.

## Serialized artifact formats

For each processed scenario directory:

| File | Format | Status |
|---|---|---|
| `semantic_manifest.json` | JSON | always written |
| `provenance.provjson` | PROV-JSON | always written |
| `provenance.provn` | PROV-N | always written |
| `provenance.ttl` | PROV-O/Turtle | always written |
| `provenance.png` | Graphviz PNG | written only when `dot` is available |
| `provenance.pdf` | Graphviz PDF | written only when `dot` is available |

Graphviz being unavailable never fails a completed scenario run — the three
machine-readable formats are always produced, and a warning is logged.

## Path and determinism rules

- Persist repository-relative paths only. Never persist `/home/...`,
  `/tmp/...`, usernames, or other machine-specific/temporary worktree paths.
- Use SHA-256 for all artifact hashes.
- PROV-JSON is rewritten to disk with `indent=2, sort_keys=True,
  allow_nan=False` for deterministic output.
- A generated timestamp may appear as metadata but is never the sole
  identity of an artifact — identity is derived from deterministic
  `urn:coinfosim:...` identifiers built from run IDs, artifact hashes, and
  relative paths (see `src/coinfosim/provenance/model.py`).

## Legacy JSON-LD compatibility

[`src/coinfosim/provenance/jsonld.py`](../../src/coinfosim/provenance/jsonld.py)
is retained **only** as a legacy-compatibility module. It built a
hand-rolled JSON-LD graph that (incorrectly) modeled a simulation run as a
`prov:Entity`; the canonical model above corrects this. No new scenario
execution or regeneration uses it as the canonical builder. Historical
`provenance.jsonld` publications are not deleted and remain a publisher
fallback (see
[`src/coinfosim/publish/site.py`](../../src/coinfosim/publish/site.py)):
the Pages home page links the canonical PROV-JSON/PROV-N/PROV-O/Turtle/PNG/
PDF artifacts when present, and falls back to a sibling `provenance.jsonld`
link only for older scenario runs that have no canonical artifacts.

## Minimum JSON-LD context (legacy)

```json
{
  "@context": {
    "coinfosim": "https://paulorenatoaz.github.io/coinfosim/ns#",
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#"
  }
}
```

The full context, including canonical persisted-key mappings, lives in
[`src/coinfosim/resources/coinfosim-context.jsonld`](../../src/coinfosim/resources/coinfosim-context.jsonld).
