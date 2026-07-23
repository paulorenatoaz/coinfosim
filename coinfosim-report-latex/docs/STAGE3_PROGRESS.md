# Etapa 3 - progresso de redação

## Concluído

- Capítulo 1: Introdução.
- Capítulo 2: Fundamentação conceitual e trabalhos relacionados.
- Capítulo 3: Perguntas de pesquisa e hipóteses.
- Capítulo 4: seleção, caracterização e preparação dos três datasets.
- Capítulo 5: metodologia experimental auditada a partir do código do repositório.
- Capítulo 6: formalização de ranking, Winners Agreement, cruzamentos dirigidos e Progressive N-star Similarity.
- Bibliografia ampliada com artigos de origem dos datasets, aplicações posteriores, leakage, validação temporal, Monte Carlo, critério de parada e desbalanceamento.

## Decisões preservadas

- Notação por subconjunto `S_I`.
- Métrica denominada **Winners Agreement**.
- A estrutura formal de Winners Agreement usa pares únicos/triângulo; a matriz quadrada é visualização redundante.
- Winners Agreement não é interpretada como identidade da primeira posição.
- A Progressive N-star Similarity usa cruzamentos dirigidos e o último cruzamento observado no prefixo.
- Separação entre fidelidade geométrica, preditiva e estrutural.
- Geometrias mantidas como figuras importantes nos três datasets.
- GMM não tratado como modelo automaticamente preferível.
- Projeção para valores maiores de `n` tratada como hipótese futura.
- Random Forest tratado como estudo adicional específico do SUPPORT2.

## Pontos que dependem da auditoria dos resultados

- Inserir os painéis geométricos reais, Gaussiana única e GMM.
- Inserir as matrizes de vencedores do Occupancy Full/Real-to-Real/Linear SVM em `n=2` e `n=512`; manter painel intermediário apenas se for didático.
- Inserir as matrizes de N-star do SUPPORT2 id08/Random Forest em `n=512`, com Jaccard, timing similarity, total similarity e contagens de cruzamentos.
- Confirmar no payload persistido os valores usados nas afirmações comparativas dos capítulos 7 e 8.

## Próxima rodada

- Capítulo 7: resultados experimentais, com seleção e extração das figuras.
- Capítulo 8: discussão transversal.
- Capítulos 9 a 11: implicações, limitações e conclusão.
