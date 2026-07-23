# Changelog

All notable changes to this project will be documented in this file.

# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
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
