# Academic Documents — Predictive Cooperation Profile Revision Log

**Block:** Resumed academic-document block only (final report + presentation LaTeX).
**Scope boundary:** does not touch `src/coinfosim/**` scientific logic, does not merge into `main`, does not begin the final integration block.

---

## 1. Branch and repository state

- Repository: `paulorenatoaz/coinfosim`, working directory `/home/pr/coinfosim`.
- `main` at session start: `ce1bafd83a34fc3aeef504ab450fa8d75dbcaff3` (`Merge feature/support2-random-forest`), 11 commits ahead of `origin/main` (`4ceba2f8`).
- `main` already contains, fully merged:
  - the complete `predictive_cooperation_profile_refactor_tasks.md` block history (`feaa8bf5` … `5e8ff924`, merged at `8f1e8da6`);
  - the `feature/support2-random-forest` branch (`7cdb1567`, merged at `ce1bafd8`), which is the commit that first added `coinfosim-report-latex/` and `coinfosim-presentation/` to the tracked tree.
- New work branch for this block: **`feature/academic-report-revision`**, created from `ce1bafd8`.
- Uncommitted state found at session start (both intentional inputs to this exact block, not unrelated WIP):
  - `M .gitignore` — two lines re-commented (`#coinfosim-report-latex/`, `#coinfosim-presentation/`); harmless since both directories are already tracked (git tracks already-added paths regardless of `.gitignore`) — left as found, will be committed with the revision.
  - `?? coinfosim_scientific_editorial_revision_map.md` (repo root) — the editorial specification supplied with this prompt; copied verbatim to `docs/tasks/coinfosim_scientific_editorial_revision_map.md` per instructions.
- No other tracked file was modified before this block began. `git fetch origin` returned no new refs.

## 2. Report and presentation source roots (bounded discovery)

```text
find . -maxdepth 6 -type f \( -name "*.tex" -o -name "*.bib" -o -name "latexmkrc" -o -name "Makefile" -o -name "*.sty" -o -name "*.cls" \)
```

- **Report root:** `coinfosim-report-latex/main.tex` (memoir/abnTeX2-based, `\input`-driven, `latexmkrc` present, `$out_dir = 'build'`, `$xelatex`, `$biber`). 17 direct chapter/frontmatter/appendix includes (listed below), plus `references/references.bib`.
- **Presentation root:** `coinfosim-presentation/presentation.tex` (`beamer`, `aspectratio=169`, `# !TeX program = xelatex`, `biblatex` + `biber`), includes `config/theme.tex`, `config/commands.tex`, `slides/main.tex`, `slides/backup.tex`, `references.bib`.
- `coinfosim_research_proposal_v4.tex` (repo root) was inspected only to confirm it is **not** the 121-page report: it is a single-file short proposal, already updated in Block 6 of the implementation task, and out of scope here.
- No other `.tex`/`.bib` roots exist within the bounded search depth; no ambiguity to report.

### Direct dependency graph

Report (`main.tex` → `\input`):
`config/{metadata,preamble,commands}` · `frontmatter/{cover,titlepage,resumo,abstract,lists}` · `chapters/{01_introducao … 11_conclusao}` · `appendices/{apendice_a_resultados,apendice_b_reprodutibilidade}` · `references/references.bib`.

Presentation (`presentation.tex` → `\input`):
`config/{theme,commands}` · `slides/{main,backup}` · `references.bib`.

Figure-generation sources: `coinfosim-report-latex/figures/conceptual/src/*.tex` (6 standalone TikZ diagrams built by a local `Makefile`), `coinfosim-report-latex/scripts/{extract_embedded_report_figures.py,fetch_published_figures.py}`, `coinfosim-presentation/scripts/{generate_air_quality_gnb_curves.py,generate_air_quality_gnb_progressive_metrics.py}`.

## 3. Build commands (existing, unmodified)

- Report: `cd coinfosim-report-latex && latexmk` (uses `latexmkrc`: `xelatex -interaction=nonstopmode -file-line-error -synctex=1`, `biber`, `-outdir=build`).
- Presentation: `cd coinfosim-presentation && make presentation` (`latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build presentation.tex`, plus a `figures` prerequisite that reruns two Python plotting scripts against `../.venv`).

### Toolchain audit — blocker

```text
which pdflatex latexmk biber bibtex pdftotext
  /usr/bin/pdflatex
  /usr/bin/latexmk
  /usr/bin/bibtex     (biber NOT present)
  /usr/bin/pdftotext
```

`xelatex` and `biber` are **not installed** in this environment (`texlive-xetex` and `biber` packages are absent; only `texlive-latex-base`/`-extra`/`-recommended` are present). Both `main.tex` and `presentation.tex` require `fontspec` (XeLaTeX/LuaLaTeX-only) and `biblatex`+`biber`; `pdflatex`+`bibtex` is not a compatible fallback without editing the preamble (out of scope — "use the existing build system", "do not install packages"). **This is reported as a blocker in Section 11 below; no package was installed.**

## 4. Compiled-PDF locations already tracked

- `coinfosim-report-latex/main.pdf` (tracked, old N★ framework, ~ full report).
- `coinfosim-presentation/presentation.pdf` (tracked, old framework, 2.35 MB).
- `coinfosim-report-latex/build/.gitkeep` only — no build artifacts tracked beyond the top-level PDFs and their `.aux`/`.bbl`/etc. companions (also tracked, see `.gitignore` inside `coinfosim-report-latex/` which only ignores `build/*` and `*.synctex.gz`).

## 5. Persisted schema-v2 data availability

- `output/reports/scenario_runs.json` — **absent**; `coinfosim runs scenarios` returns an empty registry.
- Only local persisted Monte Carlo payload found anywhere under a bounded `find . -maxdepth 6` search: `output/validation/support2-random-forest-smoke/` — a **smoke-mode** SUPPORT2 run (`scenario_run_id 0`, `support2_baseline`), sample grid `n ∈ {2,4,8,16,32}`, classifiers `linear_svm` and `random_forest`, three arms (`support2_real_data`, `support2_single_gaussian_to_real`, `support2_gmm_to_real`). Raw per-arm `SimulationResult` payloads exist as `result_data_smoke_0000{0,1,2}.json.gz`.
- **No locally persisted raw results exist for Occupancy or Air Quality**, and **no full-scale (`n` up to 512) SUPPORT2 payload exists locally** for any classifier. `coinfosim-report-latex/figures/FIGURE_SOURCES.md` independently confirms this: it documents that the original report's full-scale figures for scenario `000008`/simulations `000024–000026` (SUPPORT2 full) were sourced from `origin/gh-pages`, not from local `output/reports/`, because "o cenário 000008 e as simulações 000024–000026 não existem em `output/reports/` desta máquina."
- `coinfosim scenario regenerate support2 --run-id 0 --output-dir output/academic_revision_validation` was run (copy of the smoke registry; Monte Carlo **not** rerun) to obtain a current rendered report. **Its HTML still shows `Progressive N-star similarity` and the old "exact ties are excluded" wording** — i.e., the *renderer* (`dataset_anchored_scenario.py`/`support2_scenario.py`/`structural_visualization.py`) has not actually been migrated to the `W`/`R` schema in this checkout, despite `structural.py` (the analysis layer) already containing complete, correct `effective_winner_matrices`, `winner_reversal_events`, `progressive_reversal_matrices`, `progressive_reversal_fidelity`, and `scenario_structural_fidelity` (schema_version 2, no composite metric) implementations that match the task specification exactly. **Exact blocker symbol:** `src/coinfosim/reports/structural_visualization.py::progressive_nstar_matrix_figure` and the "Progressive N-star similarity" / "Exact-tie-aware winner agreement" strings baked into `src/coinfosim/reports/dataset_anchored_scenario.py` (or its `support2_scenario.py` wrapper) are still active on the report-rendering path, even though `src/coinfosim/results/structural.py` and `src/coinfosim/runs/report_data.py::generic_scenario_structural_report_data` already call the new, correct schema-v2 functions. This is a Python-source inconsistency, out of scope for this academic-document block (per policy, not modified).
- **Workaround used (data layer only, no code changes, no Monte Carlo rerun):** loaded the three persisted `SimulationResult` payloads directly with `coinfosim.results.persistence.load_simulation_result` and called the already-approved, already-implemented `coinfosim.results.structural.scenario_structural_fidelity(...)` function directly in a throwaway Python one-liner. This produced genuine, current, schema-v2 `W`/`R` numbers (ρ_rank, `A_W`, `A_R`, `D_R`, `S_R`, all pair counts) for SUPPORT2 at smoke scale (`n` up to 32), classifiers `linear_svm` and `random_forest`, all three arms. Saved to `output/academic_revision_validation/computed_structural_fidelity/support2_smoke_run0_scenario_structural_fidelity.json` (untracked/gitignored, reproducible by rerunning the same one-liner against the same `result_data_smoke_*.json.gz` files — command recorded in Appendix B of the report).
- **Consequence for Chapter 7 / results slides:** Occupancy and Air Quality numeric tables/figures, and SUPPORT2 numeric tables/figures **above `n=32`**, cannot be regenerated locally under the current schema. Per the master task's numerical-integrity policy, these are **not** filled with old composite values, invented numbers, or a new Monte Carlo run. They are marked explicitly unavailable in the rewritten Chapter 7, with the exact missing path/run ID recorded, and conceptual/structural rewriting is completed regardless. The one dataset with genuinely regenerated current numbers (SUPPORT2, smoke scale, both classifiers) is presented as a clearly labeled worked/reproducibility example, not as a substitute for the original full-scale published results.

## 6. Stale-terminology census (bounded grep, case-insensitive, report + presentation source trees only)

Search terms: `estrutura de cooperação · fidelidade estrutural · cooperation structure · structural fidelity · N* / N-star / N⋆ / nstar · Last Crossing / último cruzamento · crossing matrix / cruzamento dirigido / directed crossing · timing similarity / similaridade temporal · position similarity / similaridade de posição · crossing Jaccard / Jaccard de cruzamentos · produto final / nstar_similarity · reservatório real / reservatório de treinamento · "empates exatos são excluídos"`.

| File | Hits | Type of change required |
|---|---:|---|
| `chapters/01_introducao.tex` | 6 | conceptual rewrite (motivation → predictive cooperation profile) |
| `chapters/02_fundamentacao.tex` | 11 | conceptual rewrite (profile vs. pattern, attribute vs. channel) |
| `chapters/03_perguntas.tex` | 6 | conceptual rewrite (RQ1–RQ6 restructuring) |
| `chapters/04_datasets.tex` | 6 | editorial polish only ("reservatório real" → "dados reais de treinamento") |
| `chapters/05_metodologia.tex` | 7 | conceptual + protocol-figure rewrite |
| `chapters/06_metricas.tex` | 37 | full mathematical rewrite (largest chapter change) |
| `chapters/07_resultados.tex` | 74 | numerical regeneration + table/figure replacement (largest change; partially blocked, see §5) |
| `chapters/08_discussao.tex` | 17 | conceptual rewrite |
| `chapters/09_implicacoes.tex` | 3 | editorial polish |
| `chapters/10_limitacoes.tex` | 12 | conceptual rewrite (new limitation list from editorial map §6.13) |
| `chapters/11_conclusao.tex` | 8 | conceptual rewrite (new conclusion core) |
| `frontmatter/abstract.tex` | 2 | full rewrite (not patchable sentence-by-sentence per mandate) |
| `frontmatter/resumo.tex` | 2 | full rewrite |
| `frontmatter/lists.tex` | 1 | symbol-list rewrite |
| `frontmatter/cover.tex` | 0 | title string still needs replacing (title check independent of grep) |
| `frontmatter/titlepage.tex` | 0 | title string still needs replacing |
| `appendices/apendice_a_resultados.tex` | 2 | editorial + run-ID/provenance update |
| `appendices/apendice_b_reprodutibilidade.tex` | 3 | editorial + regeneration-command update |
| `coinfosim-presentation/slides/main.tex` | 28 | full rewrite of slides 1–12 per editorial map §7 |
| `coinfosim-presentation/slides/backup.tex` | 11 | terminology + numeric-table updates |
| `coinfosim-presentation/presentation.tex` | 1 (title) | title/subtitle replacement |

Figures whose **embedded image content** (not just captions) encodes the retired framework, identified by filename/content and cross-checked against `FIGURE_SOURCES.md` and `figures/conceptual/src/*.tex`:

- `figures/conceptual/src/nstar_progressivo.tex` → compiled `nstar_progressivo.pdf` (Ch. 6) — **TikZ source, will be replaced conceptually** (progressive `N*` construction → effective winner trajectory + tie carry-forward).
- `figures/conceptual/src/metricas_tres_paineis.tex` → `metricas_tres_paineis.pdf` (Ch. 6) — three-panel metric figure; panel 3 currently depicts the old third metric and must become paired `W`/`R`.
- `figures/{occupancy,air_quality,support2}_nstar.pdf/png`, `figures/selected_nstar_cross_dataset.pdf/png`, `figures/support2_nstar_all_classifiers.pdf/png`, `figures/support2_rf_nstar_components.pdf/png` — synthesis vector charts (Apêndice A note: "reconstruídos a partir das tabelas finais publicadas") built from the **old published composite table**, not from a script in this repo → cannot be mechanically regenerated from local data (no script, no full-scale local numbers); flagged unavailable, not silently kept.
- `figures/nstar_matrices/support2_random_forest/*.png`, `figures/nstar_curves/{occupancy,air_quality,support2}/*.png`, `figures/structural_metrics/{occupancy,air_quality,support2}/*_nstar_similarity_*.png` — original-report screenshots pulled from historical HTML (`FIGURE_SOURCES.md`), full-scale, not regenerable locally (§5) → flagged unavailable for direct replacement; captions/discussion around any that are *kept* for historical geometry/loss-curve content (e.g. `geometry/*`, `winners/*`, `structural_metrics/*_rho_rank_*`, `*_winner_agreement_*`) are retained since those are not old-metric-dependent.
- `coinfosim-presentation/figures/air_quality_gnb_*.pdf` — generated by tracked scripts (`generate_air_quality_gnb_curves.py`, `generate_air_quality_gnb_progressive_metrics.py`); need inspection for embedded `N*` labelling in Block "regenerate figures" step.

No product/composite-metric figure other than the ones above was found within the bounded search.

## 7. Change-type classification (per master-task requirement)

- **Conceptual rewriting:** front matter framing, Ch. 1–3, Ch. 6 §6.1–6.8 narrative, Ch. 8 discussion structure, Ch. 10 limitation categories, Ch. 11 conclusion core, presentation slides 1, 2, 3, 4, 7, 8, 12.
- **Mathematical rewriting:** Ch. 6 full equation set (`U`, `W`, tie rule, `R`, `A_R`, `D_R`, `S_R`), list of symbols, presentation slide 8 equations/boxes.
- **Numerical regeneration:** Ch. 7 tables 3–6 and all per-dataset result subsections, presentation slides 9–11 and backup slide 21 — **partially blocked**, see §5.
- **Figure regeneration:** Ch. 6 conceptual TikZ figures (regenerable locally), Ch. 7 SUPPORT2-smoke-scale W/R figures (regenerable locally from the one-liner data in §5), Ch. 7 full-scale/other-dataset figures (**not** regenerable locally, flagged).
- **Purely editorial polishing:** Ch. 4 dataset prose, Ch. 5 protocol prose retention, Ch. 9, Apêndices A/B provenance text, presentation slides 5, 6, backup slides 13–20, 22.

---

## 8. Build attempts and results

**Update:** the owner explicitly approved installing the missing toolchain mid-session (`sudo apt-get install -y texlive-xetex biber`, later `texlive-publishers` for `abntex2.cls`), run by the owner in their own shell. With the toolchain present, both documents were compiled for real:

- Report: `xelatex` → `biber --input-directory=build --output-directory=build main` → `xelatex` ×2, from `coinfosim-report-latex/`. Final state: **120 pages**, 0 undefined references/citations, 0 multiply-defined labels, 0 overfull `\hbox` warnings exceeding 30pt (down from a worst case of 254pt before fixes; see §8.1). `main.pdf` and `main.bbl` installed at the tracked paths.
- Presentation: same `xelatex`/`biber`/`xelatex`×2 sequence from `coinfosim-presentation/`. Final state: **30 pages** (one overloaded backup slide, "Definições formais dos quatro indicadores", was split into two frames to fix a genuine content overflow — see §8.1), 0 undefined references. `presentation.pdf` and `presentation.bbl` installed at the tracked paths.
- The original blocker analysis (missing `xelatex`/`biber`, `abntex2.cls` requiring `texlive-publishers`) was correct; it is recorded here as historical context for why compilation was initially deferred, not as the final state.

### 8.1 Defects found only by compiling and rendering (not visible from source review)

Compiling and visually inspecting the actual PDFs surfaced real defects that a source-only pass could not have caught:

- **Landscape figure overflow:** the SUPPORT2 paired `W`/`R` matrix figure (Ch. 6) used `width=\linewidth` on four stacked PNG panels, overflowing the landscape page height and clipping the second row. Fixed by constraining to `height=6.2cm,keepaspectratio`.
- **TikZ node overlap:** presentation slide 8's redesigned winner-reversal trajectory diagram had the `R_{ij}` callout box overlapping the state nodes below it. Rebuilt with a taller canvas and a curved leader line.
- **Footer/content collisions:** slides 3, 5, 9, 10, 11, 12 and backup slides "Apoio 1" (formal definitions), "Apoio 2" (datasets table), "Apoio 4" (Occupancy/Air Quality results), "Apoio 6" (SUPPORT2 results), "Apoio 7" (limitations) all had added text push slide content into the footer bar. Fixed by shortening captions, reducing font sizes/`\vspace`, and — for "Apoio 1", which had six formula blocks — splitting into two frames ("Apoio 1a"/"Apoio 1b").
- **Stale commit hash in BibTeX:** `coinfosim-presentation/references.bib`'s `azevedo2026coinfosim` entry carried a `note` field with a pre-refactor commit hash (`ec5b70b...`, not `git log`-resolvable in this repository) that surfaced in the compiled bibliography slide. Updated to this revision's actual commit and re-ran biber.
- **Stale terms embedded in TikZ image text, invisible to a chapter-prose grep:** the `quadrante_desempenho` conceptual figure's axis label and all four quadrant labels still said "estrutura [de cooperação/preservada/pouco preservada]" (only one caption line had been fixed earlier); presentation slide 7 had a leftover "colapso da estrutura" callout. Both are inside TikZ node text, not `.tex` chapter prose, so the original `grep`-based stale-term sweep over chapter files could not see them — only the post-compile `pdftotext` audit on the actual PDF caught them (see §9).
- **Macro-driven double periods:** the shared `\notailustracao{#1}` macro appends its own trailing period; five figure notes in the report supplied text that already ended in a period, rendering as "..". Fixed by stripping the redundant periods.

All fixes were re-verified by recompiling and re-rendering the affected pages/slides to PNG via `pdftoppm` and visually inspecting them again.

## 9. Stale-term PDF audit and visual QA — status

Both completed against the **actual final compiled PDFs**, not just source:

```bash
pdftotext main.pdf - | grep -inE "estrutura de coopera|fidelidade estrutural|N-star|último cruzamento|cruzamento dirigido|timing similarity|similaridade temporal|crossing jaccard|produto final|reservat[oó]rio real|reservat[oó]rio de treinamento"
```

- **Report** (`main.pdf`, 120 pages): exactly the same two explicitly historical/contrastive sentences identified in the source-level sweep (§9 below), nothing else.
- **Presentation** (`presentation.pdf`, 30 pages): **zero** matches of any kind, including historical ones.
- Visual QA was performed on: title page, Resumo, Abstract, list of symbols, Ch. 6 opening + all 4 new/edited conceptual figures (rendered from actual compiled PDF, not just the standalone figure), the paired `W`/`R` SUPPORT2 figure, Ch. 7's Occupancy results table, Ch. 8's performance/preservation quadrant figure, Ch. 10's limitations table, Ch. 11's conclusion page, and — for the presentation — all 12 main slides plus 6 backup slides with tables/formulas ("Apoio 1a", "Apoio 1b", "Apoio 2", "Apoio 4", "Apoio 6", "Apoio 7"). Every defect found (§8.1) was fixed and re-verified visually.

### Final source-level stale-term sweep (this session)

```bash
grep -rinE "estrutura de coopera(ção|tive)|fidelidade estrutural|cooperation structure|structural fidelity|N⋆|último cruzamento|cruzamento dirigido|directed crossing|similaridade temporal|timing similarity|similaridade de posi|position similarity|jaccard de cruzamentos|crossing jaccard|produto final|nstar_similarity|reservat[oó]rio real|reservat[oó]rio de treinamento" \
  coinfosim-report-latex/{chapters,frontmatter,appendices,config} \
  coinfosim-presentation/{slides,config,presentation.tex}
```

Result: **zero active matches.** The pattern matches exactly two sentences, both explicitly historical/contrastive (Ch. 7, SUPPORT2 §7.4, and Appendix B's renderer-blocker note), e.g. "as antigas figuras de matrizes dirigidas de $N^{\star}$ e de curvas cooperativas com marcação de último cruzamento, publicadas para este cenário, dependiam do objeto retirado do arcabouço ativo e não são reproduzidas" — typeset with the dedicated `\NstarHistorical` macro added specifically so such sentences remain typographically valid without reintroducing `\Nstar` as an active symbol. No match names the retired framework as the current one; every match is grammatically past-tense/contrastive ("antigas figuras... não são reproduzidas", "arcabouço retirado").

## 10. Cross-document consistency matrix

| Check | Report | Presentation | Consistent? |
|---|---|---|---|
| Title | "Preservação do perfil de cooperação preditiva entre canais de informação" (`config/metadata.tex`) | identical (`presentation.tex`, slide 1) | ✅ |
| Subtitle / central oral question | "Um estudo com classificadores treinados em dados sintéticos ancorados em dados reais" | subtitle: "O treinamento com amostras sintéticas preserva o perfil observado com treinamento real?" (compatible, presentation-appropriate phrasing per editorial map §4.2) | ✅ |
| Central research question wording | full form, Ch. 1 | concise form, slide 4 | ✅ compatible (map §4.3 sanctions two lengths) |
| Profile definition | Ch. 2 §"Perfil de cooperação preditiva" | slide 3 | ✅ same definition, same profile-vs-pattern distinction |
| Attribute vs. channel distinction | Ch. 1 | slide 2 speaker notes | ✅ |
| `W` tie rule | Ch. 6 §6.3, Eq. (winner-relation) | slide 7 (corrected from old "ties excluded" wording) | ✅ identical rule stated in both |
| `R` existence rule | Ch. 6 §6.5, definitionbox | slide 8 | ✅ identical ("iff at least one valid reversal") |
| Meaning of `A_R`, `D_R`, `S_R` | Ch. 6 §6.6–6.7 | slide 8 + backup Apoio 1 | ✅ same formulas, same conventions (empty-union `A_R=1`, `S_R` never `1` by default) |
| No composite metric | Ch. 6 §2.7 statement + throughout Ch. 7/8 | slide 8 takeaway ("não há produto entre as métricas") + backup Apoio 1 | ✅ |
| No active `N*` | verified Section 9 above | verified Section 9 above | ✅ |
| Shared numeric values (SUPPORT2 smoke) | Table `tab:support2-smoke`, Ch. 7 | main slide 11 + backup Apoio 6 | ✅ same run ID (`scenario_run_id=0`), same $\rankrho$/$A_W$/$A_R$/$S_R$ values to 3 decimals |
| Shared numeric values (herdados, full-scale) | Tables Ch. 7 (`tab:occupancy-final`, `tab:air-final`, `tab:support2-original`) | main slide 9 + backup Apoio 4/6 | ✅ identical $\rankrho$/$A_W$ values, identical "indisponível"/"—" treatment for $A_R$/$S_R$ |
| Run IDs / prefixes | Ch. 7 guide section + Apêndice A/B | backup Apoio 4/6/8 | ✅ scenario 000002/000005/000007/000008 (herdados) and `scenario_run_id=0` smoke (recalculated) named identically in both documents |
| Data-availability caveat | Ch. 7 §"Guia de leitura" limitationbox | main slides 9–11 captions/notes + backup Apoio 4/6 | ✅ both explicitly mark $A_R$/$D_R$/$S_R$ as unavailable at full scale, never as zero |

No inconsistency was found between the two documents at the source level. This matrix should be re-verified visually once both documents can be compiled (Section 8).

## 11. Known follow-up items not corrected in this session

- `coinfosim-presentation/roteiro_apresentacao.md` (speaker-script companion, not `\input` by any `.tex` file, not part of the compiled deck) still contains ~13 occurrences of retired terminology. Out of the bounded LaTeX-source scope of this block; flagged for a future editorial pass.
- The Python renderer inconsistency documented in Appendix B (§"Estado do renderizador HTML no commit auditado") — `structural_visualization.py`/`dataset_anchored_scenario.py` still emit old-framework labels despite `structural.py` being fully migrated — is a `src/coinfosim` code issue, out of scope for this document-only block, and was not modified.
- `coinfosim-presentation/scripts/generate_air_quality_gnb_progressive_metrics.py` reads a non-existent local path (`output/reports/scenarios/000006_air_quality_baseline_full-scale/scenario.json`) and computes the retired N★ panel; it was not invoked and its output figure was removed from slide 10. Regenerating a current-schema replacement requires the same missing raw Air Quality data flagged throughout this log.
