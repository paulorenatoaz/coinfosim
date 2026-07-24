# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- First-release terminology and identity consolidation: adopt **attribute**
  / **attribute subset** as the formal terms for a classifier's model-input
  variables and their combinations, with **predictor variable** /
  **predictive attribute** as natural prose variants. **Information
  channel** is preserved as a distinct, narrower operational/motivational
  concept (the idea that a measurement source, sensor, procedure, or other
  acquisition mechanism provides information to a classifier), and no
  longer serves as the default synonym for every classifier input. Adopts
  the public software title **CoInfoSim: A Simulator for Predictive
  Cooperation Across Attribute Subsets** and the Portuguese academic title
  **Preservação do perfil de cooperação preditiva entre atributos** across
  README, CITATION.cff, package/CLI metadata, the published site, the
  scientific report, the presentation, and HTML reports. Updates the
  `PredictiveCooperation`/`PredictiveComplementarity`/`PredictiveRedundancy`/
  `PredictiveCooperationProfile` definitions in
  `scientific_vocabulary.json` (bumped to `1.1.0`) and
  `ontology/coinfosim.owl.ttl` to be defined primarily over attributes and
  attribute subsets. No stable IRIs, persisted JSON keys/fields (including
  `channel_names`), or scientific results/protocol change.
- Add the canonical predictive-cooperation-profile semantic vocabulary
  (`src/coinfosim/resources/scientific_vocabulary.json`, 17 concepts with
  stable `coinfosim:` IDs, EN/PT labels and definitions) and a JSON-LD 1.1
  context (`coinfosim-context.jsonld`) mapping canonical persisted keys to
  those IDs. Canonicalize the analysis and visualization modules into
  `coinfosim.results.predictive_profile` and
  `coinfosim.reports.predictive_profile_visualization` (the previous
  `structural.py`/`structural_visualization.py` modules remain as thin
  compatibility shims). Version the persisted report schema to 3, replacing
  the `structural_fidelity`/`structural_dynamics` top-level keys with
  `predictive_cooperation_profile`/`pairwise_profile_dynamics` (each carrying
  a `semantic_vocabulary_version` and `semantic_type`); provide
  backward-compatible readers in `coinfosim.results.profile_schema` that
  losslessly upgrade schema 2 but refuse to reinterpret schema 1's retired
  composite metric as a new one. Add a PROV-O-compatible provenance layer
  (`coinfosim.provenance`): every regenerated scenario now emits
  `semantic_manifest.json` and `provenance.jsonld` recording source
  `result_data` hashes, the recovered `gh-pages` commit, and the current
  code commit as distinct agents. The publication page now links these
  artifacts on each scenario's card when present. See
  `docs/migration-predictive-profile-schema.md` for the full
  deprecated-to-canonical mapping.
- Replace the `N*` / directed-crossing / composite-similarity structural
  framework with an explicit pairwise winner-reversal framework: an effective
  winner matrix `W` with exact-tie carry-forward propagation, an unordered
  upper-triangular reversal matrix `R` that exists only after a pair's first
  valid winner reversal, and two separate metrics (reversal existence
  agreement, reversal sample-size similarity) with no composite/product metric.
  Updated the persisted `structural_dynamics`/`structural_fidelity` JSON
  schema to version 2 and the scenario/arm HTML reports (summary tables,
  metric curves, and paired `W`/`R` matrix panels) accordingly.
- Reformulate the project as **CoInfoSim: A Simulator for Cooperative
  Classification from Multiple Information Channels**. Documentation, metadata,
  and identity updated from the sensor-network framing (CoSenSim) to a general
  multi-channel classification framing based on information channels.
- Rename the functional package `cosensim` -> `coinfosim` and the console
  command `cosensim` -> `coinfosim` (history preserved via `git mv`). Updated
  imports, entry points, environment variables (`COINFOSIM_*`), tests, and docs.
- Rewrite README around the CoInfoSim modeling framework: input vector
  `X=(X_1,...,X_d)` of information channels, class-conditional Gaussian models
  `{(mu_c, Sigma_c)}` with `Sigma_0 != Sigma_1` allowed, balanced per-class
  sampling, standardized channels, the core object `L_bar_{A,f}(n)`, and the
  cooperative advantage threshold `N*`.
- Document the three-phase research plan (idealized synthetic scenarios,
  dataset-anchored simulation, cost-aware channel selection) and the planned
  layered reporting structure. No new simulator logic implemented yet.

## [0.1.0] - 2026-06-15
- Initial repository scaffold (identity, docs, package skeleton).
- Refactor: renamed the functional package from `slacgs` to `cosensim`,
  preserving the SLACGS implementation (CLI, simulation, reporting,
  configuration, logging, and publishing). Derived from SLACGS 0.2.0 and
  licensed under GPL-3.0.
