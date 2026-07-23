# Manifesto das figuras originais do CoInfoSim

Cada figura abaixo foi copiada ou extraída sem modificação dos relatórios
gerados pelo CoInfoSim. Nenhuma figura foi regenerada manualmente e nenhuma
captura de tela foi utilizada.

Métodos de extração:

- **copiado**: PNG já referenciado por `<img src="...">` no HTML do relatório
  de cenário; o arquivo existe ao lado do HTML e foi copiado sem alteração.
- **decodificado**: figura embutida no HTML do relatório de simulação como
  data URI `data:image/png;base64,...`; o payload foi decodificado com
  `scripts/extract_embedded_report_figures.py`, identificado pelo atributo
  `alt` do `<img>`, e validado pela assinatura PNG.

Fontes locais: `output/reports/` no repositório CoInfoSim. O cenário 000008 e
as simulações 000024–000026 não existem em `output/reports/` desta máquina;
seus arquivos foram obtidos da publicação (branch `origin/gh-pages`,
`reports/...`), que corresponde a
`https://paulorenatoaz.github.io/coinfosim/reports/...`.

## geometry/ — projeções 2D (painéis geométricos, Capítulos 4 e 7)

| Destino | Relatório HTML de origem | Arquivo original | Dataset | Braço | Método |
|---|---|---|---|---|---|
| `geometry/occupancy/viz_2d_real_full_000002.png` | `scenarios/000002_occupancy_baseline_full/occupancy_baseline_scenario_report_full_000002.html` | idem ao destino | Occupancy | treino real | copiado |
| `geometry/occupancy/viz_2d_gaussian_full_000002.png` | idem | idem | Occupancy | amostra Gaussiana única | copiado |
| `geometry/occupancy/viz_2d_gmm_full_000002.png` | idem | idem | Occupancy | amostra GMM | copiado |
| `geometry/air_quality/viz_2d_real_full_000005.png` | `scenarios/000005_air_quality_baseline_full/air_quality_scenario_report_full_000005.html` | idem | Air Quality | treino real | copiado |
| `geometry/air_quality/viz_2d_gaussian_full_000005.png` | idem | idem | Air Quality | amostra Gaussiana única | copiado |
| `geometry/air_quality/viz_2d_gmm_full_000005.png` | idem | idem | Air Quality | amostra GMM | copiado |
| `geometry/support2/viz_2d_real_full_000008.png` | `scenarios/000008_support2_baseline_full/support2_scenario_report_full_000008.html` (gh-pages) | idem | SUPPORT2 | treino real | copiado |
| `geometry/support2/viz_2d_gaussian_full_000008.png` | idem | idem | SUPPORT2 | amostra Gaussiana única | copiado |
| `geometry/support2/viz_2d_gmm_full_000008.png` | idem | idem | SUPPORT2 | amostra GMM | copiado |

Sem classificador associado (visualização dos dados); todas em modo *full*.

## winners/ — matrizes de vencedores (Capítulos 6 e 7)

| Destino | Relatório HTML de origem | Dataset | Classificador | Braço | N | Método |
|---|---|---|---|---|---|---|
| `winners/occupancy/graph_structural_winner_linear_svm_real_to_real_n2_full_000002.png` | cenário 000002 | Occupancy | Linear SVM | Real → Real | 2 | copiado |
| `winners/occupancy/graph_structural_winner_linear_svm_real_to_real_n32_full_000002.png` | cenário 000002 | Occupancy | Linear SVM | Real → Real | 32 | copiado |
| `winners/occupancy/graph_structural_winner_linear_svm_real_to_real_n512_full_000002.png` | cenário 000002 | Occupancy | Linear SVM | Real → Real | 512 | copiado |

## nstar_matrices/ — matrizes progressivas de N★ (Capítulos 6 e 7)

| Destino | Relatório HTML de origem | Dataset | Classificador | Braço | Prefixo | Método |
|---|---|---|---|---|---|---|
| `nstar_matrices/support2_random_forest/graph_structural_nstar_random_forest_real_to_real_prefix512_full_000008.png` | cenário 000008 (gh-pages) | SUPPORT2 | Random Forest | Real → Real | 512 | copiado |
| `nstar_matrices/support2_random_forest/graph_structural_nstar_random_forest_single_gaussian_to_real_prefix512_full_000008.png` | cenário 000008 (gh-pages) | SUPPORT2 | Random Forest | Gaussiana única → Real | 512 | copiado |
| `nstar_matrices/support2_random_forest/graph_structural_nstar_random_forest_gmm_to_real_prefix512_full_000008.png` | cenário 000008 (gh-pages) | SUPPORT2 | Random Forest | GMM → Real | 512 | copiado |

## nstar_curves/ — curvas cooperativas com N★ (Capítulos 6 e 7)

Identificador da imagem embutida = atributo `alt` do `<img>` no HTML.

| Destino | Relatório HTML de origem | Identificador embutido (`alt`) | Dataset | Classificador | Braço | Método |
|---|---|---|---|---|---|---|
| `nstar_curves/occupancy/nstar_curve_linear_svm_best1_real_to_real_sim000006.png` | `simulations/000006_occupancy_real_data_full/occupancy_real_data_monte_carlo_report_full_000006.html` | `N-star graph Linear SVM best 1` | Occupancy | Linear SVM | Real → Real | decodificado |
| `nstar_curves/occupancy/nstar_curve_linear_svm_best1_single_gaussian_to_real_sim000007.png` | `simulations/000007_occupancy_single_gaussian_to_real_full/occupancy_single_gaussian_to_real_monte_carlo_report_full_000007.html` | `N-star graph Linear SVM best 1` | Occupancy | Linear SVM | Gaussiana única → Real | decodificado |
| `nstar_curves/air_quality/nstar_curve_gaussian_nb_best1_real_to_real_sim000015.png` | `simulations/000015_air_quality_real_data_full/air_quality_real_data_monte_carlo_report_full_000015.html` | `N-star graph Gaussian Naive Bayes best 1` | Air Quality | Gaussian Naive Bayes | Real → Real | decodificado |
| `nstar_curves/air_quality/nstar_curve_gaussian_nb_best1_single_gaussian_to_real_sim000016.png` | `simulations/000016_air_quality_single_gaussian_to_real_full/air_quality_single_gaussian_to_real_monte_carlo_report_full_000016.html` | `N-star graph Gaussian Naive Bayes best 1` | Air Quality | Gaussian Naive Bayes | Gaussiana única → Real | decodificado |
| `nstar_curves/air_quality/nstar_curve_gaussian_nb_best1_gmm_to_real_sim000017.png` | `simulations/000017_air_quality_gmm_to_real_full/air_quality_gmm_to_real_monte_carlo_report_full_000017.html` | `N-star graph Gaussian Naive Bayes best 1` | Air Quality | Gaussian Naive Bayes | GMM → Real | decodificado |
| `nstar_curves/support2/nstar_curve_random_forest_best1_real_to_real_sim000024.png` | `simulations/000024_support2_real_data_full/support2_real_data_monte_carlo_report_full_000024.html` (gh-pages) | `N-star graph Random Forest best 1` | SUPPORT2 | Random Forest | Real → Real | decodificado |
| `nstar_curves/support2/nstar_curve_random_forest_best1_single_gaussian_to_real_sim000025.png` | `simulations/000025_support2_single_gaussian_to_real_full/support2_single_gaussian_to_real_monte_carlo_report_full_000025.html` (gh-pages) | `N-star graph Random Forest best 1` | SUPPORT2 | Random Forest | Gaussiana única → Real | decodificado |
| `nstar_curves/support2/nstar_curve_random_forest_best1_gmm_to_real_sim000026.png` | `simulations/000026_support2_gmm_to_real_full/support2_gmm_to_real_monte_carlo_report_full_000026.html` (gh-pages) | `N-star graph Random Forest best 1` | SUPPORT2 | Random Forest | GMM → Real | decodificado |

Em todas as curvas, a referência do painel é o melhor subconjunto singular
(`best 1`), e as linhas tracejadas marcam o último cruzamento observado.

## structural_metrics/ — séries progressivas das métricas (Capítulo 7)

Cada imagem compara os braços Gaussiana única → Real e GMM → Real contra a
referência real ao longo da grade amostral.

| Destino | Relatório HTML de origem | Dataset | Classificador | Métrica | Método |
|---|---|---|---|---|---|
| `structural_metrics/occupancy/graph_structural_metric_linear_svm_rho_rank_full_000002.png` | cenário 000002 | Occupancy | Linear SVM | correlação de ranking | copiado |
| `structural_metrics/occupancy/graph_structural_metric_linear_svm_winner_agreement_full_000002.png` | cenário 000002 | Occupancy | Linear SVM | Winners Agreement | copiado |
| `structural_metrics/occupancy/graph_structural_metric_linear_svm_nstar_similarity_full_000002.png` | cenário 000002 | Occupancy | Linear SVM | similaridade progressiva de N★ | copiado |
| `structural_metrics/air_quality/graph_structural_metric_gaussian_nb_rho_rank_full_000005.png` | cenário 000005 | Air Quality | Gaussian Naive Bayes | correlação de ranking | copiado |
| `structural_metrics/air_quality/graph_structural_metric_gaussian_nb_winner_agreement_full_000005.png` | cenário 000005 | Air Quality | Gaussian Naive Bayes | Winners Agreement | copiado |
| `structural_metrics/air_quality/graph_structural_metric_gaussian_nb_nstar_similarity_full_000005.png` | cenário 000005 | Air Quality | Gaussian Naive Bayes | similaridade progressiva de N★ | copiado |
| `structural_metrics/support2/graph_structural_metric_random_forest_rho_rank_full_000008.png` | cenário 000008 (gh-pages) | SUPPORT2 | Random Forest | correlação de ranking | copiado |
| `structural_metrics/support2/graph_structural_metric_random_forest_winner_agreement_full_000008.png` | cenário 000008 (gh-pages) | SUPPORT2 | Random Forest | Winners Agreement | copiado |
| `structural_metrics/support2/graph_structural_metric_random_forest_nstar_similarity_full_000008.png` | cenário 000008 (gh-pages) | SUPPORT2 | Random Forest | similaridade progressiva de N★ | copiado |

## Demais arquivos em figures/

Os PDFs/PNGs na raiz de `figures/` (`occupancy_ranking.pdf`,
`air_quality_nstar.pdf`, `support2_rf_metrics.pdf` etc.) são gráficos
vetoriais de síntese reconstruídos a partir das tabelas finais publicadas,
conforme registrado no Apêndice A; não fazem parte deste manifesto de
figuras originais.
