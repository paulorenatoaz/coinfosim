# CoInfoSim — Scientific Editorial Revision Map
## Final Academic Report and Presentation

**Repository:** `paulorenatoaz/coinfosim`  
**Scope:** final academic report (approximately 121 pages) and academic presentation (12 main slides plus backup material; approximately 29 PDF pages)  
**Purpose:** guide a scientifically coherent migration from the retired `N*`/last-crossing/composite-similarity formulation to the approved predictive-cooperation-profile framework.

---

# 1. Editorial mandate

This is not a terminology-only revision.

The report and presentation must be rewritten so that the following elements form one consistent scientific argument:

1. title and research question;
2. definition of the scientific object;
3. experimental protocol;
4. mathematical notation;
5. metric definitions;
6. figures and tables;
7. numerical results;
8. interpretation and discussion;
9. limitations;
10. conclusion.

The revised documents should present CoInfoSim as a study of:

> **the preservation of predictive cooperation profiles among information channels when the same classifiers are trained under real and synthetic training conditions and evaluated on the same fixed real test set.**

The academic report must read as a rigorous scientific study rather than as an accumulation of software-report outputs.

The presentation must communicate one clear oral narrative:

1. why relationships among attribute subsets matter;
2. what a predictive cooperation profile is;
3. how real and synthetic training conditions are compared;
4. what each preservation indicator measures;
5. what the experiments support;
6. what the study does not establish.

---

# 2. Current-document audit that motivates the rewrite

The currently rendered report and presentation are still organized around the retired framework.

## 2.1 Final academic report

The current report:

- uses **“estrutura de cooperação”** in the title and throughout the scientific framing;
- defines preservation with ranking correlation, Winners Agreement, and progressive `N*` similarity;
- defines `N*` through the last observed crossing of loss curves;
- includes a list of symbols containing `N*`;
- structures Chapter 6 around cooperative curves, directed crossings, last `N*`, and progressive `N*` similarity;
- contains figures that explicitly display:
  - `N*`;
  - directed crossings;
  - crossing Jaccard;
  - temporal/timing similarity;
  - the final multiplicative product;
- contains result tables and conclusions based on old composite values;
- states conclusions such as a classifier or generator being superior under old `N*` similarity.

The abstract, methods, results, discussion, and conclusion are therefore semantically coupled to the retired formulation. They cannot be repaired by replacing labels alone.

## 2.2 Academic presentation

The current presentation:

- uses **“estrutura de cooperação”** in the title, object definition, question, and conclusion;
- uses **“reservatório real de treinamento”** in the experimental diagram;
- presents the old metric set in the protocol;
- states that exact ties are excluded from Winners Agreement;
- dedicates one slide to `N*` and the product “existence × position”;
- contains result tables and charts with old composite `N*` values;
- states comparative conclusions based on those values;
- repeats old terminology and values in backup slides.

The new presentation must therefore be rebuilt conceptually and numerically in the affected slides.

---

# 3. Canonical terminology ledger

## 3.1 Portuguese terminology

| Concept | Canonical term |
|---|---|
| Umbrella phenomenon | **cooperação preditiva** |
| Distinct predictive contributions that improve joint performance | **complementaridade preditiva** |
| Little or no additional predictive contribution | **redundância preditiva** |
| Quantitative object evolving with sample size | **perfil de cooperação preditiva** |
| Interpreted regularity extracted from one or more profiles | **padrão de cooperação preditiva** |
| Formal model-input variable | **atributo** |
| Conceptual or operational information source | **canal de informação** |
| Monte Carlo comparison quantity | **perda média estimada no conjunto de teste** |
| Pairwise current relation | **relação de vencedor par a par** |
| Full formal event name | **inversão de superioridade preditiva par a par** |
| Short event name after first definition | **inversão de vencedor** |
| Current effective-winner matrix | **matriz de vencedores** \(\mathbf W(n)\) |
| Matrix storing the latest valid reversal sample size | **matriz de inversões** \(\mathbf R(n_t)\) |
| Set-based preservation indicator | **concordância de existência de inversões** |
| Unnormalized sample-size displacement | **distância média das inversões em escala \(\log_2\)** |
| Conditional sample-size agreement | **similaridade dos tamanhos amostrais das inversões** |
| Current-winner agreement metric | **concordância de vencedores (Winners Agreement)** |

## 3.2 English terminology

| Concept | Canonical term |
|---|---|
| Umbrella phenomenon | **predictive cooperation** |
| Quantitative object | **predictive cooperation profile** |
| Interpreted regularity | **predictive cooperation pattern** |
| Formal event | **pairwise predictive-superiority reversal** |
| Short event name | **winner reversal** |
| Current effective-winner matrix | **winner matrix** |
| Latest-reversal matrix | **reversal matrix** |
| Set-based agreement | **reversal existence agreement** |
| Unnormalized displacement | **mean log2 reversal distance** |
| Conditional sample-size agreement | **reversal sample-size similarity** |

## 3.3 Terms removed from the active scientific framework

Do not use as current principal terminology:

- estrutura de cooperação;
- cooperation structure;
- fidelidade estrutural;
- structural fidelity;
- `N*`, `N-star`, `N⋆`;
- Last Crossing;
- último cruzamento as the formal event;
- crossing matrix;
- directed crossing;
- cruzamento dirigido;
- timing similarity;
- similaridade temporal;
- position similarity;
- similaridade de posição;
- progressive N-star similarity;
- multiplicative/composite reversal metric.

Historical occurrences may remain only when they are explicitly identified as historical and immediately contrasted with the current formulation.

## 3.4 Training/evaluation wording

Prefer:

- **dados reais de treinamento**;
- **conjunto de dados reais de treinamento**;
- **amostras reais de treinamento**;
- **modelos geradores ajustados aos dados reais de treinamento**;
- **amostras sintéticas de treinamento**;
- **mesmo conjunto de teste real fixo**.

Avoid in the active scientific narrative:

- reservatório real;
- reservatório de treinamento;
- pool real;
- real reservoir;
- reference set, when it ambiguously denotes either training or test data.

---

# 4. Canonical scientific formulation

## 4.1 Report title

> **Preservação do perfil de cooperação preditiva entre canais de informação**

Subtitle:

> **Um estudo com classificadores treinados em dados sintéticos ancorados em dados reais**

## 4.2 Presentation title

Use the same title.

A concise subtitle or oral question may be:

> **O treinamento sintético preserva o perfil observado com treinamento real?**

## 4.3 Central research question

Full report version:

> **Em que medida classificadores treinados com amostras sintéticas, produzidas por modelos geradores ajustados aos dados reais de treinamento, preservam o perfil de cooperação preditiva observado quando os mesmos classificadores são treinados com amostras reais?**

Presentation version:

> **Em que medida o treinamento com amostras sintéticas preserva o perfil de cooperação preditiva observado com treinamento real?**

## 4.4 Predictive cooperation profile

Recommended definition:

> **O perfil de cooperação preditiva é a representação quantitativa de como o desempenho relativo dos subconjuntos de atributos ou canais evolui ao longo do tamanho amostral, para um dataset, classificador e condição de treinamento determinados.**

Immediately distinguish profile from pattern:

> **Um padrão de cooperação preditiva é uma regularidade interpretada a partir de um ou mais perfis; não é o objeto quantitativo em si.**

## 4.5 Methodological statement

> **O CoInfoSim compara perfis de cooperação preditiva produzidos pelos mesmos classificadores em duas condições: treinamento com amostras reais e treinamento com amostras sintéticas geradas por modelos ajustados aos dados reais de treinamento. Em ambas as condições, a avaliação é realizada sobre o mesmo conjunto de teste real fixo.**

---

# 5. Mathematical content that must be identical across documents

## 5.1 Estimated mean test loss

For subset \(S_i\), classifier \(f\), training condition \(g\), and sample size \(n\):

\[
\widehat L_{S_i,f}^{(g)}(n)
=
\frac{1}{R_n}
\sum_{r=1}^{R_n}
\widehat L_{S_i,f,r}^{(g)}(n).
\]

Use:

- **perda média estimada no conjunto de teste**;
- **estimated mean test loss**.

Do not use:

- “estimated mean of the loss metric”;
- “média estimada da métrica loss.”

## 5.2 Observed pairwise outcome

For unordered pair \(i<j\):

\[
U_{ij}(n_k)=
\begin{cases}
+1, & \widehat L_i(n_k)<\widehat L_j(n_k),\\
-1, & \widehat L_i(n_k)>\widehat L_j(n_k),\\
0, & \widehat L_i(n_k)=\widehat L_j(n_k).
\end{cases}
\]

Interpretation:

- \(+1\): \(S_i\) has lower estimated mean test loss;
- \(-1\): \(S_j\) has lower estimated mean test loss;
- \(0\): exact observed tie at the current grid point.

## 5.3 Effective winner relation and tie rule

\[
W_{ij}(n_k)=
\begin{cases}
U_{ij}(n_k), & U_{ij}(n_k)\neq 0,\\
W_{ij}(n_{k-1}), &
U_{ij}(n_k)=0\text{ and a previous winner exists},\\
0, &
U_{ij}(n_k)=0\text{ and no previous winner exists}.
\end{cases}
\]

Required prose interpretation:

- initial ties leave the pair unresolved;
- the first strict winner initializes the relation;
- a later exact tie carries the previous winner forward;
- a tie neither creates nor erases a reversal;
- the first winner after initial ties is not a reversal.

This corrects the current presentation statement that “exact ties are excluded.” Under the current formulation:

- unresolved initial pairs are unavailable;
- later exact ties preserve the effective winner;
- Winner Agreement compares effective winners.

## 5.4 Valid winner reversal

A valid reversal at \(n_k\) requires:

\[
W_{ij}(n_{k-1})\neq 0,\qquad
W_{ij}(n_k)\neq 0,\qquad
W_{ij}(n_k)\neq W_{ij}(n_{k-1}).
\]

This means:

> one previously defined pairwise winner is replaced by the other.

The following do not constitute reversals:

- an initial tie;
- repeated initial ties;
- the appearance of the first strict winner;
- a later tie that carries the previous winner;
- curves approaching without changing the effective winner.

## 5.5 Reversal matrix

For the prefix ending at \(n_t\):

\[
\mathcal E_{ij}(n_t)
=
\left\{
n_k\leq n_t:
W_{ij}(n_{k-1})\neq 0,\;
W_{ij}(n_k)\neq 0,\;
W_{ij}(n_k)\neq W_{ij}(n_{k-1})
\right\},
\]

\[
R_{ij}(n_t)=
\begin{cases}
\max\mathcal E_{ij}(n_t),
& \mathcal E_{ij}(n_t)\neq\varnothing,\\
\varnothing,
& \mathcal E_{ij}(n_t)=\varnothing.
\end{cases}
\]

Required existence statement:

> **\(R_{ij}(n_t)\) exists if and only if at least one valid winner reversal occurred for the unordered pair in the evaluated prefix.**

Representation rules:

- one value per unordered pair;
- upper triangle only;
- diagonal and lower triangle unavailable;
- a defined cell stores the latest observed reversal sample size;
- the current winner is recovered from \(\mathbf W(n_t)\);
- \(\mathbf R(n_t)\) does not duplicate directions.

## 5.6 Preservation indicators

### Ranking correlation

\[
\rho_{\mathrm{rank}}.
\]

It compares complete subset rankings at a given \(n\), with average-rank treatment for exact loss ties.

### Winner Agreement

\[
A_W.
\]

It compares effective pairwise winners between a synthetic condition and the Real \(\rightarrow\) Real reference condition.

### Reversal support

\[
\mathcal C_g(n_t)
=
\left\{
(i,j): i<j,\;R_{ij}^{(g)}(n_t)\neq\varnothing
\right\}.
\]

### Reversal existence agreement

\[
A_R(n_t)
=
\frac{
|\mathcal C_{\mathrm{real}}(n_t)\cap\mathcal C_g(n_t)|
}{
|\mathcal C_{\mathrm{real}}(n_t)\cup\mathcal C_g(n_t)|
}.
\]

Interpretation:

> Do real and synthetic training conditions identify valid winner reversals for the same unordered subset pairs?

Conventions:

- first prefix: unavailable;
- empty union after the first prefix: \(A_R=1\), with an explicit “no reversals in either condition” status;
- nonempty union and empty intersection: \(A_R=0\).

### Shared reversal support

\[
\mathcal H(n_t)
=
\mathcal C_{\mathrm{real}}(n_t)\cap\mathcal C_g(n_t).
\]

### Mean log2 reversal distance

\[
D_R(n_t)
=
\frac{1}{|\mathcal H(n_t)|}
\sum_{(i,j)\in\mathcal H(n_t)}
\left|
\log_2 R_{ij}^{(g)}(n_t)
-
\log_2 R_{ij}^{(\mathrm{real})}(n_t)
\right|.
\]

Interpretation:

- measured in levels of the power-of-two sample-size grid;
- defined only when there is at least one shared reversal pair.

### Reversal sample-size similarity

\[
S_R(n_t)
=
\operatorname{clip}
\left(
1-
\frac{D_R(n_t)}
{\log_2(n_t)-\log_2(n_1)},
0,1
\right).
\]

Interpretation:

> For pairs that reverse in both conditions, how similar are the sample-size regimes of their latest observed reversals?

Conventions:

- first prefix: unavailable;
- no shared reversal pairs: unavailable;
- no reversals in either condition: unavailable, not \(1\);
- identical shared reversal sample sizes: \(S_R=1\).

## 5.7 No composite indicator

Do not calculate, report, plot, or interpret:

\[
A_R(n_t)\times S_R(n_t).
\]

Existence agreement and sample-size similarity answer different questions and must remain separate.

---

# 6. Final report — section-by-section revision map

The current report is approximately 121 pages. The largest changes are required in the front matter, Chapters 1–3, Chapter 6, Chapter 7, discussion, limitations, conclusion, lists, figures, tables, and appendices.

## 6.1 Cover, title page, and metadata

Replace the old title based on “estrutura de cooperação.”

Check all occurrences in:

- primary title page;
- secondary title page;
- PDF metadata;
- running headers;
- internal citation metadata;
- appendix references to the document title.

## 6.2 Resumo and Abstract

Rewrite both as complete scientific abstracts. Do not patch the old paragraphs sentence by sentence.

Recommended structure:

1. scientific problem;
2. real and synthetic training conditions;
3. same fixed real test set;
4. definition of predictive cooperation profile;
5. datasets and classifiers;
6. four preservation indicators:
   - ranking correlation;
   - Winner Agreement;
   - reversal existence agreement;
   - reversal sample-size similarity;
7. principal empirical findings using current values only;
8. main conclusion;
9. scope limitation.

Recommended Portuguese opening:

> **Este trabalho investiga em que medida classificadores treinados com amostras sintéticas preservam o perfil de cooperação preditiva observado quando os mesmos classificadores são treinados com amostras reais.**

Recommended English opening:

> **This study investigates the extent to which classifiers trained on synthetic samples preserve the predictive cooperation profile observed when the same classifiers are trained on real samples.**

Do not retain these old claims without current recalculation:

- Random Forest had the highest `N*` similarity;
- thousands of directed crossings support a specific conclusion;
- one generator was better under the old composite;
- any exact old `N*`-similarity value.

Recommended keywords:

- canais de informação;
- cooperação preditiva;
- perfil de cooperação preditiva;
- dados sintéticos;
- seleção de subconjuntos;
- simulação de Monte Carlo.

## 6.3 Lists of figures, tables, and symbols

Replace every caption based on:

- `N*`;
- progressive `N*`;
- directed crossing;
- crossing Jaccard;
- temporal similarity;
- final product.

The symbol list should include at least:

- \(X_j\): attribute/information-channel index;
- \(S_i\): nonempty subset;
- \(n\): training samples per class;
- \(\widehat L_{S_i,f}^{(g)}(n)\);
- \(U_{ij}(n)\);
- \(W_{ij}(n)\);
- \(R_{ij}(n_t)\);
- \(\rho_{\mathrm{rank}}\);
- \(A_W\);
- \(\mathcal C_g\);
- \(A_R\);
- \(D_R\);
- \(S_R\).

Remove `N*`.

## 6.4 Chapter 1 — Introduction

### Motivation

Preserve useful operational examples but clarify:

- an **attribute** is the formal model input;
- an **information channel** is the conceptual or operational source represented by one attribute or an explicitly defined attribute group.

Avoid information-theoretic claims such as:

- “information content”;
- “reduces uncertainty”;
- “mutual information” interpretations;

unless formally defined and measured. This study is predictive and empirical.

### Scientific problem

Replace “preservation of cooperation structure” with preservation of the predictive cooperation profile.

### Central question and objectives

Use the approved research question.

Map objectives directly to the analytical indicators:

1. compare global ordering;
2. compare effective pairwise winners;
3. compare which unordered pairs undergo valid reversals;
4. compare latest reversal sample sizes on shared support;
5. study dependence on dataset, classifier, generator, and sample size.

### Contributions

Suggested contribution structure:

- formal definition of the predictive cooperation profile;
- explicit effective winner relation with tie carry-forward;
- winner matrix \(\mathbf W\);
- upper-triangular reversal matrix \(\mathbf R\);
- separation of reversal existence and reversal sample-size agreement;
- dataset-anchored real-versus-synthetic protocol;
- reproducible implementation and report regeneration.

## 6.5 Chapter 2 — Conceptual foundation

### Channels and subsets

Clarify attribute versus information channel.

### Cooperation, complementarity, and redundancy

Use:

- predictive cooperation;
- predictive complementarity;
- predictive redundancy.

Do not imply that these are information-theoretic quantities.

### Current “estrutura de cooperação” section

Rename:

> **Perfil de cooperação preditiva**

Explain:

- the profile is the quantitative object;
- a pattern is an interpreted regularity;
- the profile is not only the best subset or ranking winner;
- it includes relative subset performance over sample size and complementary summaries of that evolution.

## 6.6 Chapter 3 — Research questions and hypotheses

Recommended architecture:

- **RQ1 — Preservação da ordenação global**  
  How closely does synthetic training reproduce complete subset rankings?

- **RQ2 — Preservação das relações de vencedor**  
  How closely does synthetic training reproduce effective pairwise winners?

- **RQ3 — Concordância quanto à existência de inversões**  
  Do the same unordered subset pairs undergo at least one valid winner reversal?

- **RQ4 — Concordância quanto ao tamanho amostral das inversões**  
  For shared reversal pairs, how similar are the latest reversal sample sizes?

- **RQ5 — Dependência do classificador**

- **RQ6 — Dependência do modelo gerador**

Retain geometry as a separate exploratory question.

Rewrite hypotheses so they do not assume that a visually more realistic generator must preserve the predictive cooperation profile better.

## 6.7 Chapter 4 — Datasets

Most dataset facts may remain.

Required editorial changes:

- replace “reservatório real” with “dados reais de treinamento”;
- distinguish attributes from conceptual channels;
- preserve target construction, provenance, leakage controls, split logic, and ethical limitations;
- avoid redefining the global scientific object in dataset-specific terms.

## 6.8 Chapter 5 — Experimental methodology

Update the protocol figure and prose:

- same classifiers;
- real versus synthetic training samples;
- generators fitted only on real training data;
- same fixed real test set;
- estimated mean test loss;
- four separate preservation indicators;
- no progressive `N*` similarity;
- no composite metric.

Preserve valid details concerning:

- standardization;
- balanced sampling;
- nested samples;
- sample-size grid;
- Monte Carlo stopping;
- classifier calibration;
- parallelization;
- persistence;
- reproducibility.

## 6.9 Chapter 6 — Complete scientific rewrite

Recommended structure:

### 6.1 Da perda média estimada ao perfil de cooperação preditiva

Introduce the profile and three analytical levels:

1. global ordering;
2. current effective pairwise winners;
3. winner-reversal dynamics.

Explain that the third level produces two separate preservation indicators.

### 6.2 Ordenação dos subconjuntos e correlação de ranking

Define ranking and \(\rho_{\mathrm{rank}}\).

Preserve average-rank treatment for exact loss ties.

### 6.3 Relação efetiva de vencedor e matriz \(\mathbf W\)

Introduce \(U\), \(W\), and the tie rule.

### 6.4 Concordância de vencedores

Define \(A_W\) using effective winners.

Explain unresolved initial pairs.

### 6.5 Inversões de superioridade preditiva e matriz \(\mathbf R\)

Define:

- valid reversal;
- exact existence rule;
- upper-triangular storage;
- relation between \(\mathbf W\) and \(\mathbf R\).

### 6.6 Concordância de existência de inversões

Define \(\mathcal C_g\) and \(A_R\).

Explain empty-support semantics.

### 6.7 Distância e similaridade dos tamanhos amostrais das inversões

Define \(D_R\) and \(S_R\).

State:

- the object concerns sample size, not time;
- \(S_R\) is conditional on shared support;
- no product is used.

### 6.8 Leitura integrada do perfil

Explain:

- \(\rho_{\mathrm{rank}}\): global ordering;
- \(A_W\): current pairwise decisions;
- \(A_R\): which pairs have undergone valid reversals;
- \(S_R\): sample-size agreement for shared reversals;
- no indicator subsumes the others.

## 6.10 Conceptual figures in Chapter 6

Replace or regenerate all figures whose image content contains:

- `N*`;
- crossing arrows;
- directed crossing cells;
- progressive `N*`;
- Jaccard × timing;
- product metric.

Do not merely change captions while leaving old symbols inside images.

Recommended new conceptual figures:

1. **Predictive cooperation profile and analytical levels**
2. **Effective winner trajectory with initial tie, first winner, carried tie, and valid reversal**
3. **Paired \(\mathbf W\) and upper-triangular \(\mathbf R\) matrices**
4. **Separate construction of \(A_R\) and \(S_R\)**

## 6.11 Chapter 7 — Experimental results

This chapter requires numerical regeneration from current persisted schema-v2 data.

### Global numerical rules

- do not reuse old composite values;
- do not derive \(A_R\) or \(S_R\) algebraically from the old product;
- do not infer new metric values from old charts;
- use current persisted result data or regenerated report-ready JSON;
- confirm \(\rho_{\mathrm{rank}}\) and \(A_W\) under the current implementation;
- label unavailable values explicitly;
- record run ID and prefix for every result.

### Tables 3–6

Replace the old third metric with columns such as:

- \(\rho_{\mathrm{rank}}\);
- \(A_W\);
- \(A_R\);
- \(D_R\);
- \(S_R\);
- reference, arm, shared, and union reversal-pair counts when useful.

### Occupancy

Rename the old subsection based on `N*` to:

> **Existência e tamanho amostral das inversões**

Preserve the argument that geometric resemblance does not guarantee profile preservation only if current metrics support it.

### Air Quality

Replace vague statements that `N*` is “more demanding” with precise findings about:

- reversal support;
- shared reversal support;
- latest reversal sample sizes;
- metric availability.

### SUPPORT2

Replace the old heading:

> Random Forest: baixa qualidade preditiva e alta similaridade de \(N*\)

with a neutral heading:

> **Random Forest: desempenho absoluto e preservação do perfil**

Re-evaluate all Random Forest conclusions using \(A_R\) and \(S_R\) separately.

### Figures dependent on the retired metric

Replace old:

- `N*` bars;
- progressive `N*` curves;
- directed `N*` matrices;
- Jaccard/timing/product decompositions;
- cross-classifier `N*` comparisons.

Use:

- separate \(A_R\) and \(S_R\) curves/bars;
- optional \(D_R\) diagnostic;
- triangular \(\mathbf R\);
- paired \(\mathbf W/\mathbf R\);
- no product curve.

### Qualitative synthesis

Rewrite around:

- ranking preservation;
- effective-winner preservation;
- reversal-existence preservation;
- reversal-sample-size preservation;
- geometric resemblance;
- absolute predictive performance.

## 6.12 Discussion

Preserve the central insight that preservation is multidimensional, but express the dimensions correctly.

Required points:

- strong ranking agreement does not guarantee identical effective winners;
- strong Winner Agreement does not guarantee identical reversal support;
- identical reversal support does not guarantee similar reversal sample sizes;
- \(S_R\) is conditional on shared reversal support;
- \(A_R=1\) with empty supports means agreement in absence, not matched reversal locations;
- geometry, absolute loss, and profile preservation are distinct empirical properties;
- results depend on dataset, classifier, generator, sample-size grid, and prefix;
- no generator should be ranked universally.

## 6.13 Limitations

Explicitly address:

- all relations are based on estimated mean test loss;
- Monte Carlo uncertainty propagates to derived profile objects;
- exact ties are treated deterministically through carry-forward after initialization;
- \(\mathbf R\) depends on the evaluated grid;
- \(\mathbf R\) stores only the latest reversal;
- \(\mathbf R\) discards reversal count, order, duration, and loss magnitude;
- \(A_R\) is set-based and ignores reversal direction and magnitude;
- \(S_R\) is defined only on shared support;
- normalization of \(S_R\) depends on the evaluated prefix;
- only 0–1 loss is studied;
- stopping controls loss-estimation precision, not direct stability of every derived metric;
- no validated extrapolation beyond the grid;
- no observed channel costs;
- computational cost grows as \(2^d-1\);
- external validity remains limited by the evaluated datasets and classifiers.

## 6.14 Conclusion

Recommended conceptual core:

> **Os experimentos indicam que a preservação do perfil de cooperação preditiva é multidimensional. O treinamento sintético pode reproduzir a ordenação global e as relações de vencedor sem reproduzir necessariamente os mesmos pares de inversão ou os mesmos regimes amostrais em que essas inversões ocorrem. Por isso, correlação de ranking, concordância de vencedores, concordância de existência de inversões e similaridade dos tamanhos amostrais devem ser interpretadas separadamente.**

Then state:

- evidence is scenario-, classifier-, generator-, grid-, and prefix-dependent;
- the study does not claim that synthetic training improves accuracy;
- cost reduction was not demonstrated;
- cost-aware and analytical extensions remain future work.

## 6.15 Appendices and reproducibility

Update:

- notation;
- schema version;
- report-generation commands;
- figure-generation commands;
- repository commit anchor;
- run IDs;
- compilation instructions;
- reproduction checklist;
- links to newly regenerated HTML reports when republished.

---

# 7. Presentation — slide-by-slide revision map

The current presentation contains 12 main slides and backup material.

## Main slide 1 — Title

Replace the title with the canonical title.

Keep the key clarification that the study is not data augmentation for accuracy.

Suggested statement:

> **O objetivo não é melhorar a acurácia com dados sintéticos, mas verificar se o treinamento sintético preserva o perfil observado com treinamento real.**

## Main slide 2 — Why compare subsets?

Preserve the motivation.

Clarify that \(X_j\) is an attribute representing an information channel.

Avoid a visual label “information” if it suggests an information-theory measure. Prefer:

- desempenho preditivo;
- complementaridade;
- redundância;
- complexidade;
- custo operacional.

## Main slide 3 — Scientific object

Rename:

> **O que é o perfil de cooperação preditiva?**

Use the approved profile definition.

Keep the loss-curve sketch, but make clear that the profile is not only the winning subset.

## Main slide 4 — Scientific question

Use the concise approved question.

Bottom boxes:

- não é aumento de dados;
- não é busca de maior acurácia;
- é avaliação da preservação do perfil.

## Main slide 5 — Three arms

Replace:

- “reservatório real” with “dados reais de treinamento”;
- old metric footer with:
  - correlação de ranking;
  - concordância de vencedores;
  - concordância de existência de inversões;
  - similaridade dos tamanhos amostrais das inversões.

Use “perda média estimada no teste real.”

## Main slide 6 — Metric 1

Preserve ranking correlation.

Remove stale `N*` overview artwork from the shared conceptual graphic.

## Main slide 7 — Metric 2

Retain pairwise winner logic but correct tie semantics:

- initial exact ties are unresolved;
- after a winner is defined, an exact tie carries the previous winner;
- Winner Agreement compares effective winners;
- only unresolved pairs are unavailable.

Replace:

> empates exatos são excluídos

with a concise accurate explanation.

## Main slide 8 — Replace old Metric 3

Replace the old `N*` and product slide completely.

Recommended title:

> **Métricas 3 e 4 — quais pares invertem e em que regime amostral?**

Recommended layout:

### Left

A small effective-winner trajectory showing:

- initial tie;
- first winner;
- carried tie;
- valid winner reversal;
- \(R_{ij}\) receiving the latest reversal sample size.

### Right, independent boxes

1. **Concordância de existência \(A_R\)**  
   Os mesmos pares apresentam ao menos uma inversão válida?

2. **Similaridade amostral \(S_R\)**  
   Nos pares compartilhados, as últimas inversões ocorrem em tamanhos amostrais semelhantes?

### Bottom takeaway

> **Existência e tamanho amostral respondem a perguntas diferentes; não há produto entre as métricas.**

## Main slide 9 — Occupancy

The current table contains old composite values.

Regenerate with:

- \(\rho_{\mathrm{rank}}\);
- \(A_W\);
- \(A_R\);
- \(S_R\), or `—` if unavailable.

Preserve the geometry-versus-profile argument only if current data support it.

Replace “reservatório real” in the geometry panel.

## Main slide 10 — Air Quality progression

The current chart contains one old progressive `N*` curve.

Replace with either:

- four legible curves per arm; or
- two compact panels per arm:
  - ranking and winners;
  - \(A_R\) and \(S_R\).

Do not plot unavailable \(S_R\) as zero.

Rewrite the takeaway using current results.

## Main slide 11 — SUPPORT2

Recommended title:

> **SUPPORT2 — desempenho absoluto e perfil são dimensões distintas**

Preserve the majority-class baseline and absolute-loss comparison if unchanged.

Replace the right chart with separate indicators.

Do not preserve the old conclusion that one generator is better under `N*`.

## Main slide 12 — Synthesis

Replace:

- “ranking, vitórias e \(N*\)”;

with:

- ordenação;
- vencedores;
- existência de inversões;
- tamanhos amostrais das inversões.

Replace “estrutura” with “perfil” in the central conclusion.

Suggested synthesis:

> **A preservação do perfil depende da interação entre dataset, classificador, modelo gerador e tamanho amostral. Nenhum indicador isolado resume todas essas dimensões.**

## Backup slide 13 — Dataset selection

Mostly preserve.

Replace stale “structure” wording where it denotes the formal object.

## Backup slide 14 — Provenance

Replace “reservatório de treinamento” with “dados reais de treinamento.”

## Backup slides 15–20

Review for stale terminology in:

- reproducibility;
- software description;
- responsible SUPPORT2 use;
- stopping and validity explanations.

Preserve unrelated factual content.

## Backup slide 21 — Additional numerical results

Regenerate every row containing old composite values.

Use separate \(A_R\) and \(S_R\).

Include the sample-size prefix and run ID.

## Backup slide 22 — Cooperative curves

The loss curves may remain.

Update interpretation from “structure” to “profile.”

## Backup metric/matrix slides

Replace embedded:

- `N*`;
- directed crossing cells;
- product diagrams;
- temporal similarity;
- old composite legends.

## Backup repository-anchor slide

Update the audited scientific commit only after the final academic revision is committed and reviewed.

---

# 8. Professional scientific editing standards

## 8.1 Cohesion

Each report section should:

- open with its purpose;
- introduce notation before use;
- connect equations to the scientific question;
- explain what an object measures and what it does not;
- transition explicitly to the next analytical level;
- avoid isolated definitions without interpretation.

## 8.2 Terminological discipline

Do not alternate casually among:

- structure;
- profile;
- pattern;
- organization;
- dynamics.

Use:

- **profile** for the quantitative object;
- **pattern** for an interpreted regularity;
- **organization** only as informal prose when it cannot be mistaken for a formal object;
- **dynamics** for evolution over sample size.

## 8.3 Claims and evidence

Every comparative claim about:

- a generator;
- a classifier;
- a dataset;
- a metric;
- a sample-size regime;

must be traceable to a current table, figure, or persisted result.

Avoid:

- universal superiority claims;
- causal claims;
- interpreting undefined values as zero;
- interpreting empty reversal support as perfect sample-size similarity;
- presenting the latest reversal as an optimum or critical threshold.

## 8.4 Readability

### Report

- use paragraphs with one main argumentative function;
- reduce repeated descriptions of the same protocol;
- retain methodological precision;
- make the English abstract idiomatic scientific English;
- avoid software-centric prose in the main scientific argument unless it supports reproducibility.

### Presentation

- one principal message per slide;
- no unreadably dense table;
- use `—` for unavailable metrics;
- use short declarative takeaways;
- keep backup slides detailed but terminologically consistent.

---

# 9. Numerical integrity policy

The old composite `N*` similarity cannot be decomposed from its published value alone.

Therefore:

1. use current persisted schema-v2 report data;
2. regenerate reports from persisted simulation results;
3. extract current metric values;
4. update tables and plots;
5. record source run ID and prefix;
6. never rerun Monte Carlo;
7. never invent or estimate a missing value;
8. never reuse a composite value as \(A_R\) or \(S_R\).

When current data are unavailable:

- complete conceptual editing where possible;
- do not publish final PDFs with disguised placeholders;
- report the exact missing run, payload, path, or field;
- stop before declaring the academic revision complete.

---

# 10. Cross-document consistency checklist

Before completion, confirm that the report and presentation have:

- identical title;
- compatible central-question wording;
- identical profile definition;
- identical attribute/channel distinction;
- identical tie carry-forward rule;
- identical reversal existence rule;
- identical meanings for \(A_R\), \(D_R\), and \(S_R\);
- identical numerical values for shared results;
- identical run IDs and sample-size prefixes;
- no multiplicative metric;
- no active `N*` terminology;
- no stale old symbols embedded in figures.
