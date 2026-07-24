# Apresentação acadêmica do CoInfoSim

Deck 16:9 em LaTeX Beamer, com **14 slides principais** e material de apoio. A
narrativa principal foi planejada para **aproximadamente 21 minutos**. Cada
slide principal possui notas do apresentador no próprio fonte.

Após a conclusão principal, a seção **Complemento: boas práticas de Ciência de
Dados** reúne o material de apoio (backup), fora da duração da narrativa
principal: seleção dos datasets, reprodutibilidade computacional, a camada
semântica e o modelo de proveniência (incluindo as três instanciações
legíveis por cenário — Occupancy, Air Quality e SUPPORT2), software
científico aberto, responsabilidade metodológica no uso do SUPPORT2,
definições formais dos quatro indicadores de preservação do perfil,
configurações dos geradores/classificadores, resultados numéricos adicionais
dos três cenários, limitações e ameaças à validade, e a bibliografia
completa.

## Compilação

No diretório `coinfosim-presentation`:

```bash
make presentation
```

O comando regenera as figuras vetoriais derivadas dos artefatos persistidos e
executa XeLaTeX/Biber via `latexmk`. O PDF final é produzido em
`build/presentation.pdf`.

Para gerar a versão de apoio com slide e notas lado a lado:

```bash
make notes
```

O resultado é `build/presentation-notes.pdf`.

## Fontes científicas

O relatório final `../coinfosim-report-latex/main.pdf` é a referência
científica principal. As figuras vetoriais e os PNGs originais são importados
de `../coinfosim-report-latex/figures/`. As figuras específicas da apresentação
são regeneradas a partir dos artefatos persistidos:

- `scripts/generate_air_quality_gnb_curves.py`: curvas cooperativas do Air
  Quality (Gaussian NB), a partir dos `full_loss_table_*.csv` das simulações
  000015--000017 (usada no material de apoio);
- `scripts/generate_full_scale_metric_curves.py`: séries de
  \(\rho_{\mathrm{rank}}\), \(A_W\), \(A_R\) e \(S_R\) por \(n\) para
  Occupancy/SVM, Air Quality/Gaussian NB e SUPPORT2/Random Forest, a partir do
  payload canônico `predictive_cooperation_profile` ou, no cenário 000008, dos
  resultados brutos persistidos. O script valida os valores finais e usa
  exclusivamente os cenários `full` 000002, 000005 e 000008 (slides 9--11).
  Valores matematicamente indefinidos nos primeiros prefixos são mantidos
  como indisponíveis e marcados por \(\times\), sem serem convertidos em zero.
