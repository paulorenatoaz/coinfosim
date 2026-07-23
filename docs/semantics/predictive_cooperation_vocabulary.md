# Predictive Cooperation Profile — Semantic Vocabulary

This document describes the canonical, machine-readable scientific vocabulary
for CoInfoSim's predictive-cooperation-profile framework. The vocabulary is
the terminology ledger for the migration away from the retired
cooperation-structure / `N*` framework, and the semantic foundation later
ontology (OWL 2) and provenance (PROV-O) phases build on.

- Source of truth: [`src/coinfosim/resources/scientific_vocabulary.json`](../../src/coinfosim/resources/scientific_vocabulary.json)
- JSON-LD context: [`src/coinfosim/resources/coinfosim-context.jsonld`](../../src/coinfosim/resources/coinfosim-context.jsonld)
- Loader: [`src/coinfosim/semantics/vocabulary.py`](../../src/coinfosim/semantics/vocabulary.py)
- Namespace: `https://paulorenatoaz.github.io/coinfosim/ns#`
- Full deprecated-to-canonical mapping and schema history: [`docs/migration-predictive-profile-schema.md`](../migration-predictive-profile-schema.md)
- Vocabulary version: `1.0.0`

## How to use it

```python
from coinfosim.semantics import get_concept, canonical_key_to_id

concept = get_concept("ReversalMatrix")
concept.id            # "coinfosim:ReversalMatrix"
concept.label_en       # "reversal matrix"
concept.label_pt       # "matriz de inversões"
concept.symbol         # "\\mathbf{R}(n_t)"

canonical_key_to_id("reversal_existence_agreement")
# "coinfosim:ReversalExistenceAgreement"
```

Unknown concept names or canonical keys raise `coinfosim.semantics.VocabularyError`
rather than returning a guessed value.

## Canonical concepts

| Stable ID | English label | Portuguese label | Canonical Python key |
|---|---|---|---|
| `coinfosim:PredictiveCooperation` | predictive cooperation | cooperação preditiva | — |
| `coinfosim:PredictiveComplementarity` | predictive complementarity | complementaridade preditiva | — |
| `coinfosim:PredictiveRedundancy` | predictive redundancy | redundância preditiva | — |
| `coinfosim:PredictiveCooperationProfile` | predictive cooperation profile | perfil de cooperação preditiva | `predictive_cooperation_profile` |
| `coinfosim:PredictiveCooperationPattern` | predictive cooperation pattern | padrão de cooperação preditiva | — |
| `coinfosim:Attribute` | attribute | atributo | `attribute` |
| `coinfosim:InformationChannel` | information channel | canal de informação | `information_channel` |
| `coinfosim:EstimatedMeanTestLoss` | estimated mean test loss | perda média estimada no conjunto de teste | `estimated_mean_test_loss` |
| `coinfosim:PairwiseWinnerRelation` | pairwise winner relation | relação de vencedor par a par | `pairwise_winner_relation` |
| `coinfosim:WinnerMatrix` | winner matrix | matriz de vencedores | `winner_matrix` |
| `coinfosim:WinnerReversal` | winner reversal | inversão de vencedor | `winner_reversal` |
| `coinfosim:ReversalMatrix` | reversal matrix | matriz de inversões | `reversal_matrix` |
| `coinfosim:RankingFidelity` | ranking fidelity | fidelidade de ranking | `ranking_fidelity_series` |
| `coinfosim:WinnerAgreement` | Winner Agreement | concordância de vencedores | `winner_agreement_series` |
| `coinfosim:ReversalExistenceAgreement` | reversal existence agreement | concordância de existência de inversões | `reversal_existence_agreement` |
| `coinfosim:MeanLog2ReversalDistance` | mean log2 reversal distance | distância média das inversões em escala log2 | `mean_log2_reversal_distance` |
| `coinfosim:ReversalSampleSizeSimilarity` | reversal sample-size similarity | similaridade dos tamanhos amostrais das inversões | `reversal_sample_size_similarity` |
| `coinfosim:PairwiseProfileDynamics` | pairwise profile dynamics | dinâmica de perfil par a par | `pairwise_profile_dynamics` |
| `coinfosim:ReversalAgreementSeries` | reversal agreement series | série de concordância de inversões | `reversal_agreement_series` |

The first 17 rows are the required terminology ledger. `PairwiseProfileDynamics`
and `ReversalAgreementSeries` are supplementary container concepts added to
give the two new top-level persisted schema keys (`pairwise_profile_dynamics`,
and the `reversal_agreement_series` array inside `predictive_cooperation_profile`)
a stable semantic type.

## Required terminology distinctions

- **Attribute vs. information channel**: an attribute is the formal
  model-input variable a classifier consumes; an information channel is the
  conceptual/operational source an attribute (or attribute group)
  represents. They are not automatically the same ontology class.
- **Profile vs. pattern**: a predictive cooperation profile is a computed
  quantitative artifact; a predictive cooperation pattern is an
  *interpretation* derived from one or more profiles. A report JSON may
  contain a profile; it must not claim to contain a pattern unless an
  explicit interpretation activity produced one.
- **Reversal vs. crossing**: a winner reversal is defined by a change in the
  *effective pairwise winner relation*, not by geometric interpolation of
  continuous loss curves.

## Deprecated terms

Deprecated aliases are recorded as metadata on the concept they most closely
relate to and must never be emitted as a preferred label. Two terms have no
clean 1:1 successor and are recorded separately under `deprecated_terms` in
the vocabulary JSON:

| Deprecated term | Status |
|---|---|
| `cooperative_threshold` (`N*`) | Retired without a single-scalar successor; superseded by the winner-reversal trajectory. |
| composite/product reversal metric (`nstar_similarity`) | Cannot be decomposed into `ReversalExistenceAgreement` and `ReversalSampleSizeSimilarity`; new output never emits a composite metric. |

See [`docs/migration-predictive-profile-schema.md`](../migration-predictive-profile-schema.md)
for the full deprecated-to-canonical mapping table.
