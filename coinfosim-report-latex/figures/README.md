# Figuras

As figuras originais dos relatórios do CoInfoSim estão organizadas em:

- `geometry/{occupancy,air_quality,support2}/` — projeções 2D dos três braços;
- `winners/occupancy/` — matrizes de vencedores (SVM linear, Real → Real);
- `nstar_matrices/support2_random_forest/` — matrizes progressivas de N★ no prefixo 512;
- `nstar_curves/{occupancy,air_quality,support2}/` — curvas cooperativas extraídas
  dos relatórios de simulação (imagens embutidas em base64, decodificadas por
  `scripts/extract_embedded_report_figures.py`);
- `structural_metrics/{occupancy,air_quality,support2}/` — séries progressivas das métricas.

A origem exata de cada arquivo (relatório HTML, identificador da imagem, dataset,
classificador, braço, N/prefixo e método de extração) está registrada em
`FIGURE_SOURCES.md`. Os PDFs/PNGs na raiz deste diretório são gráficos vetoriais
de síntese reconstruídos a partir das tabelas finais publicadas.

Cada ilustração deve ter título acima e fonte abaixo, inclusive quando elaborada
pelo próprio autor.
