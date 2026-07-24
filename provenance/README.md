# CoInfoSim provenance — readable diagrams

This directory holds the **final, readable provenance diagrams** for CoInfoSim:
one dataset-independent generic model and one instantiation per scientific
scenario (Occupancy Detection, UCI Air Quality, SUPPORT2). Each subdirectory
preserves the diagram's `.tex` source (self-contained: it includes its own
copies of `prov_diagram_style.tex` and `preamble_diagrams.tex`, so it compiles
with `xelatex <file>.tex` on its own, without depending on
`coinfosim-report-latex/` or `coinfosim-presentation/`) and its compiled
`.pdf`.

```text
provenance/
├── generic/         # Level 1 — dataset-independent architecture
│   ├── coinfosim_provenance_model.tex / .pdf        (portrait, used in the report)
│   └── coinfosim_provenance_model_wide.tex / .pdf   (16:9, used in the presentation)
├── occupancy/        # Level 2 — Occupancy Detection instantiation
├── air_quality/       # Level 2 — UCI Air Quality instantiation
└── support2/          # Level 2 — SUPPORT2 instantiation
```

## What these diagrams are

These are **editorial readability projections** of CoInfoSim's canonical
provenance model, built by hand in TikZ from the same type/relation mapping
the software actually emits (see
[`docs/semantics/provenance_mapping.md`](../docs/semantics/provenance_mapping.md)),
using one shared visual grammar across the whole family:

- `prov:Entity` → yellow, rounded box
- `prov:Activity` → blue, sharp-cornered box
- `prov:SoftwareAgent` → orange box
- `used` / `wasGeneratedBy` / `wasAssociatedWith` → one arrow style per
  relation, labeled along the edge

The three scenario diagrams also carry small dashed annotation notes (file
roles, split rules, target thresholds) drawn from each dataset's own
documentation (`data/raw/<dataset>/README.md`) and from the scientific
report's dataset chapter — these notes are captions, not additional PROV
nodes; every node and edge actually drawn is backed by that scenario's
persisted `provenance.provn`.

**These diagrams are not `prov_to_dot` output.** They are a human-authored,
readability-oriented adaptation of the canonical graph, meant to be legible
in a report page or a presentation slide — something the genuine Graphviz
rendering, which is wide and dense, is not.

## Where the canonical (audit-layer) artifacts are

The formal, machine-generated provenance — the actual evidentiary layer — is
produced by the software itself, per scenario run, and is **not** duplicated
into this directory:

- Persisted locally next to each scenario's output: `semantic_manifest.json`,
  `provenance.provjson` (PROV-JSON), `provenance.provn` (PROV-N),
  `provenance.ttl` (PROV-O/Turtle), and, when Graphviz is available,
  `provenance.png`/`provenance.pdf` (the genuine `prov_to_dot` rendering).
- A copy of the genuine Graphviz rendering for the three scenarios discussed
  in the report is also checked in at
  [`coinfosim-report-latex/figures/provenance/`](../coinfosim-report-latex/figures/provenance/)
  (e.g. `provenance_occupancy_000002.pdf`) — that is the nearest thing to a
  direct "original graph" artifact in this repository.
- `semantic_manifest.json` for the three scenarios discussed in the report is
  also published on GitHub Pages next to each scenario's HTML report, e.g.
  <https://paulorenatoaz.github.io/coinfosim/reports/scenarios/000002_occupancy_baseline_full/semantic_manifest.json>.
- Any scenario can be regenerated from persisted results, without rerunning
  Monte Carlo, via `coinfosim scenario regenerate <scenario> --run-id <id>`
  (see the root [`README.md`](../README.md#report-regeneration)), which
  rewrites all of the above deterministically from one canonical
  `prov.model.ProvDocument`.

See [`coinfosim-report-latex/appendices/apendice_e_proveniencia.tex`](../coinfosim-report-latex/appendices/apendice_e_proveniencia.tex)
for the complete formal mapping, serialization-format table, and
traceability table.

## Distinction from `src/coinfosim/provenance/`

[`src/coinfosim/provenance/`](../src/coinfosim/provenance/) is **implementation
code** — the Python package that builds the canonical `ProvDocument` and
exports its serializations (`model.py`, `export.py`, `evidence.py`). This
`provenance/` directory at the repository root is unrelated to that code
path: it holds only the static, hand-authored readable diagrams described
above, for academic/editorial use.
