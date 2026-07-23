# Plano editorial consolidado - etapa 2

## Ajustes finais incorporados

1. A notação principal é `S_I`, com `I` como conjunto de índices.
2. A métrica chama-se **Winners Agreement**, pois compara matrizes de relações
   vencedoras par a par.
3. A estrutura formal de winners é triangular; a matriz quadrada é apenas uma
   visualização redundante.
4. A seção de Winners Agreement prevê matrizes do cenário Occupancy Full,
   Real -> Real, Linear SVM, no início (`n=2`) e no fim (`n=512`), além de um
   terceiro prefixo opcional a selecionar após inspeção.
5. A seção de Progressive N*-Similarity prevê matrizes do SUPPORT2, modelo id08,
   Random Forest, em `n=512`.
6. Nenhuma métrica é descrita como simples identidade da primeira posição.
   Ranking e Winners Agreement são especialmente relevantes para decisões de
   custo e eficiência.
7. Visualizações geométricas serão mantidas, inclusive em páginas paisagem.
8. Occupancy é o principal exemplo de distância visual em relação à Gaussiana.
   Air Quality e SUPPORT2 parecem ajustar-se melhor a regiões gaussianas.
9. No Occupancy, o GMM pode imitar visualmente melhor os dados reais, enquanto a
   Gaussiana única apresenta, no maior `n`, indicadores de ranking e winners
   ligeiramente superiores. Esse contraste será usado para separar fidelidade
   geométrica de fidelidade estrutural.
10. O Random Forest do SUPPORT2 será apresentado como estudo adicional, não como
    comparação simétrica entre datasets.
