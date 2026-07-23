# Predictive Cooperation Profile — Schema and Terminology Migration

This document records the migration from the retired cooperation-structure /
`N*` framework, through the intermediate winner/reversal (`W`/`R`) schema, to
the canonical predictive-cooperation-profile framework with a machine-readable
semantic vocabulary and provenance layer.

## Schema generations

| Schema | Top-level scenario key | Top-level simulation key | Central metric object |
|---|---|---|---|
| 1 (historical) | `structural_fidelity` | `structural_dynamics` | composite `nstar_similarity` (product of a Jaccard term and a timing term) |
| 2 (`W`/`R`, no semantic layer) | `structural_fidelity` | `structural_dynamics` | separate `reversal_existence_agreement` / `reversal_sample_size_similarity`, no composite |
| 3 (canonical, this refactor) | `predictive_cooperation_profile` | `pairwise_profile_dynamics` | same separate metrics, plus `semantic_vocabulary_version` and `semantic_type` |

Schema 3 is produced by [`src/coinfosim/results/predictive_profile.py`](../src/coinfosim/results/predictive_profile.py).
Backward-compatible readers live in
[`src/coinfosim/results/profile_schema.py`](../src/coinfosim/results/profile_schema.py):

- `load_predictive_profile_payload(report_data, level=...)` locates the
  canonical or legacy key and classifies its schema version.
- `upgrade_predictive_profile_payload(loaded)` losslessly renames schema-2
  fields to schema-3 names. It **refuses** to upgrade schema-1 payloads,
  raising `ProfileSchemaError`, because schema 1's composite metric cannot be
  decomposed into the new separated indicators.

## Deprecated-to-canonical mapping

| Deprecated | Canonical |
|---|---|
| cooperation structure | predictive cooperation profile |
| structural fidelity (as the name of the central scientific object) | predictive cooperation profile / profile agreement |
| directed crossing | winner reversal |
| `N*` / N-star matrix | reversal matrix (`R`) |
| crossing Jaccard | reversal existence agreement |
| timing similarity | reversal sample-size similarity |
| `nstar_similarity` (composite/product metric) | *(retired, no successor — see below)* |
| `structural_fidelity` JSON key | `predictive_cooperation_profile` |
| `structural_dynamics` JSON key | `pairwise_profile_dynamics` |
| `n_reference_reversal_pairs` / `n_arm_reversal_pairs` / `n_shared_reversal_pairs` / `n_union_reversal_pairs` | `reference_reversal_pair_count` / `arm_reversal_pair_count` / `shared_reversal_pair_count` / `union_reversal_pair_count` |
| `effective_winner_pairs_by_n` (dict keyed by stringified `n`) | `effective_winner_relations_by_n` (list of `{n_per_class, relations}`) |
| `_structural_fidelity_section` | `_predictive_cooperation_profile_section` |
| `_structural_dynamics_panel` | `_pairwise_profile_dynamics_panel` |
| `metric_series_figure` | `profile_metric_series_figure` |
| `progressive_nstar_matrix_figure` | `reversal_matrix_figure` (+ new `paired_winner_reversal_figure`) |
| `structural.py` / `structural_visualization.py` | `predictive_profile.py` / `predictive_profile_visualization.py` |

The complete vocabulary ledger, including stable semantic IDs, is in
[`docs/semantics/predictive_cooperation_vocabulary.md`](semantics/predictive_cooperation_vocabulary.md).

## Binding statements

- **Historical artifacts are not rewritten.** Everything recovered from
  `origin/gh-pages` (scenarios `000000`–`000008`, their HTML, and their
  schema-1 `structural_fidelity`/`structural_dynamics` payloads) remains
  byte-for-byte as published. It is read for inventory, provenance-source
  hashing, and historical display only.
- **Old composite values cannot be decomposed.** A schema-1
  `nstar_similarity` value (or its supporting `crossing_jaccard` /
  `timing_similarity` / crossing-count fields) cannot be mapped onto
  `reversal_existence_agreement` and `reversal_sample_size_similarity`. Doing
  so would fabricate a scientific claim the original computation never made.
  `upgrade_predictive_profile_payload` raises rather than guessing.
- **Recovered `gh-pages` results are source entities.** In the provenance
  graph (`provenance.jsonld`), every recovered `result_data_*.json.gz` file is
  a `prov:Entity` attributed to the `gh-pages` recovery commit
  (`prov:wasAttributedTo`).
- **Regenerated reports are derived entities.** Every regenerated scenario
  report and its `predictive_cooperation_profile` entity carry
  `prov:wasDerivedFrom` relations back to the exact source `result_data`
  hashes, and `prov:wasGeneratedBy` relations to the recomputation/
  report-regeneration activities associated with the current code commit.
- **Semantic IDs are stable independently of labels.** `coinfosim:ReversalMatrix`,
  `coinfosim:WinnerAgreement`, etc. do not change if their English or
  Portuguese preferred label text is later revised; code and provenance
  graphs reference the ID, never the label string.

## Recovery and validation tooling

- [`scripts/inventory_recovered_reports.py`](../scripts/inventory_recovered_reports.py) —
  directory-scanning inventory of the `gh-pages` recovery snapshot; writes
  [`docs/provenance/gh_pages_recovery_manifest.json`](provenance/gh_pages_recovery_manifest.json).
- [`scripts/validate_predictive_profile_refactor.py`](../scripts/validate_predictive_profile_refactor.py) —
  the single permitted multi-minute end-to-end validation: regenerates the
  five academic scenarios (`000002`, `000005`, `000006`, `000007`, `000008`)
  from persisted results only, validates canonical schema/semantic
  manifests/provenance JSON-LD, and runs the targeted regression tests.

## Non-goals of this migration

This migration did not rerun any Monte Carlo simulation, did not change
datasets/splits/classifiers/generators, did not build a complete OWL
ontology or deploy an RDF triple store, and did not publish anything to
`gh-pages`. See Section 12 of
[`docs/tasks/predictive_profile_semantic_refactor_plan.md`](tasks/predictive_profile_semantic_refactor_plan.md)
for the full non-goals list.
