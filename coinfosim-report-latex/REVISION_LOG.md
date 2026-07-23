# Registro de revisão

Este registro consolida a revisão editorial e visual do relatório. A coluna
"significado científico" indica se a alteração modificou hipóteses, método ou
resultados, e não apenas sua formulação.

| Seção | Tipo de alteração | Razão | Significado científico alterado? | Figura ou tabela nova? |
|---|---|---|---|---|
| Elementos pré-textuais | Reescrita do resumo e do *abstract* | Explicitar que o objeto comparado são classificadores treinados em amostras reais ou sintéticas e preservar a distinção entre parametrização e avaliação | Não; houve maior precisão da formulação | Não |
| Capítulo 1 | Reordenação da motivação, revisão linguística e definição operacional de canal de informação | Introduzir custos, redundância e complementaridade antes da notação e justificar a terminologia central | Não | Não |
| Capítulo 2 | Revisão conceitual e integração de diagramas | Definir cooperação, espaço de subconjuntos e distinção entre fidelidade geométrica, preditiva e estrutural antes de seu uso | Não | Sim: cooperação entre canais e reticulado de subconjuntos |
| Capítulo 3 | Reformulação das perguntas, objetivos e hipóteses | Incorporar as formulações definitivas do autor e manter hipóteses separadas de conclusões | Não; a intenção científica foi preservada e explicitada | Não |
| Capítulo 4 | Revisão dos datasets e das tabelas de caracterização | Tornar canais, alvos, domínios e estratégias de particionamento comparáveis e legíveis | Não | Tabelas redesenhadas |
| Capítulo 5 | Auditoria metodológica, revisão terminológica e diagrama do protocolo | Evidenciar a parametrização dos geradores a partir do treinamento real e a avaliação comum no conjunto de teste real fixo | Não | Sim: pipeline das três condições |
| Capítulo 6 | Revisão das definições, exemplos e integração visual | Distinguir ranking, Winners Agreement, matriz triangular formal, visualização quadrada, cruzamentos dirigidos e similaridade progressiva de N* | Não | Sim: métricas em três painéis e N* progressivo |
| Capítulo 7 | Reescrita dos resultados, redesenho de tabelas e integração das figuras experimentais | Relacionar cada afirmação à evidência e encerrar subseções com resultado, interpretação e limite | Não; valores foram auditados nos relatórios persistidos | Tabelas redesenhadas e figuras experimentais integradas |
| Capítulo 8 | Revisão da discussão transversal e quadrante conceitual | Separar desempenho preditivo, semelhança geométrica e preservação estrutural | Não | Sim: quadrante desempenho versus preservação |
| Capítulo 9 | Revisão editorial e caixa de alcance | Distinguir evidência metodológica, possível uso operacional e agenda de custos ainda não testada | Não | Sim: caixa de interpretação |
| Capítulo 10 | Revisão editorial, tradução de termos e síntese das ameaças à validade | Agrupar limitações por tipo e explicitar consequência e mitigação | Não | Sim: tabela-síntese |
| Capítulo 11 | Revisão da resposta final e da linguagem causal | Responder às perguntas sem atribuir ao gerador propriedades demonstradas apenas na interação com o classificador | Não | Não |
| Apêndice A | Complementação dos IDs e URLs de Air Quality e atualização do texto | Completar a rastreabilidade e remover instruções editoriais já cumpridas | Não | Não |
| Apêndice B | Revisão terminológica da documentação computacional | Usar português no texto sem alterar nomes literais de comandos e arquivos | Não | Não |
| Projeto gráfico | Paleta, títulos, legendas, tabelas e caixas reutilizáveis | Modernizar a leitura mantendo A4, margens e hierarquia acadêmica da UFRJ | Não | Ambientes e comandos reutilizáveis |
| Figuras conceituais | Criação de seis diagramas vetoriais originais | Explicar protocolo, cooperação, subconjuntos e métricas antes dos resultados | Não | Sim: seis PDFs vetoriais em `figures/conceptual/` |
| Revisão terminológica final | Revisão contextual de “ajuste”, “estimação”, “parametrização”, “geração” e “treinamento” no texto completo, no resumo, no *abstract* e no diagrama do protocolo | Impedir que a estrutura de cooperação seja atribuída às amostras ou ao gerador e reservar “parametrização” à definição dos modelos probabilísticos a partir dos dados reais | Não; a cadeia conceitual foi explicitada sem alterar método ou resultados | Diagrama do protocolo atualizado |

## Revisão subsequente: fechamento da lacuna de dados brutos ($A_R$/$D_R$/$S_R$)

A revisão registrada na tabela acima reportava $A_R$, $D_R$ e $S_R$ como indisponíveis para Occupancy (000002), Air Quality (000005) e SUPPORT2 (000007, 000008) em escala completa, por ausência local dos resultados brutos por replicação dessas execuções. Essa lacuna foi fechada em uma revisão editorial subsequente, após a recuperação e regeneração desses resultados brutos e a republicação dos relatórios HTML sob o esquema $\Wmat/\Rmat$ vigente (refatoramento semântico do arcabouço de perfil de cooperação preditiva, `coinfosim.results.predictive_profile`, `schema_version=3`).

| Seção | Tipo de alteração | Razão | Significado científico alterado? | Figura ou tabela nova? |
|---|---|---|---|---|
| Cap. 6 (nota SUPPORT2 RF) | Substituição da ressalva de indisponibilidade de figuras full-scale por referência aos indicadores agora recalculados no Capítulo 7 | Os dados brutos da execução full do Random Forest no SUPPORT2 foram regenerados | Não; apenas remove uma limitação de disponibilidade já resolvida | Não |
| Cap. 7 | Reescrita da caixa de limitação inicial; adição das colunas $A_R$/$S_R$ (e nota de $D_R$) às três tabelas de escala completa; reescrita das subseções de existência/regime amostral de inversões do Occupancy e SUPPORT2 RF; reescrita da síntese de cenários | Os quatro indicadores foram recalculados diretamente pelo código atual para os quatro cenários de escala completa | Sim: revela dissociações entre indicadores (p.ex. GNB no Occupancy tem o maior $\rankrho$/$\AW$ e o menor $\AR$ do dataset) não observáveis na revisão anterior | Não; tabelas existentes foram completadas |
| Cap. 8 | Reescrita das respostas a RQ3/RQ4/RQ5/RQ6 e das seções de geometria/desempenho absoluto, incorporando os quatro indicadores em escala completa | As afirmações anteriores estavam condicionadas à ausência de $A_R$/$S_R$ em escala completa | Sim; RQ3/RQ4 passam a ter evidência de escala completa nos três datasets, não apenas no exemplo *smoke* | Não |
| Cap. 10 | Remoção da linha de limitação sobre dados brutos indisponíveis da tabela-síntese; reescrita da seção "Disponibilidade dos dados brutos sob o esquema atual" como nota de resolução | A limitação de disponibilidade foi resolvida | Não; documenta resolução de uma limitação, não introduz nem remove hipótese científica | Não |
| Cap. 11 | Reescrita da resposta a RQ3/RQ4 e do parágrafo de trabalhos imediatos, removendo a regeneração de dados brutos da lista de trabalhos pendentes | A regeneração foi concluída nesta revisão | Sim; a conclusão agora responde a RQ3/RQ4 com evidência de escala completa | Não |
| Apêndice A | Adição dos caminhos de simulação do cenário 000007; atualização da política de uso do material suplementar e da nota sobre proveniência dos valores $\rankrho$/$\AW$/$A_R$/$D_R$/$S_R$ | Refletir que os quatro indicadores são agora computados diretamente pelo código atual, não herdados nem extraídos de PDF | Não | Não |
| Apêndice B | Atualização do comando de reprodução para os módulos canônicos (`predictive_profile`, `predictive_profile_visualization`); documentação do caminho de regeneração dos quatro cenários de escala completa; correção de `schema_version` (2→3); resolução das caixas de limitação sobre inconsistência terminológica do renderizador e sobre compilação não validada | Refletir o estado atual do código e do ambiente de compilação (TeXLive completo já instalado) | Não | Não |
| Apresentação, slides 9–12 | Preenchimento das células de $A_R$/$S_R$ antes marcadas com travessão; reescrita das notas do apresentador e caixas de destaque | Os mesmos dados agora disponíveis no relatório | Sim, no mesmo sentido do Capítulo 7 | Não |
| Apresentação, slides de apoio 4 e 6 | Preenchimento das tabelas completas de $A_R$/$S_R$ para os quatro cenários de escala completa | Idem | Sim, no mesmo sentido do Capítulo 7 | Não |

## Revisão de 2026-07-23: resultados exclusivamente em escala completa

Esta revisão elimina o uso de execuções *smoke* como evidência científica nos
documentos acadêmicos. Todos os números, tabelas e figuras de resultados agora
provêm dos cenários `full` 000002, 000005, 000007 e 000008. O modo *smoke*
permanece citado apenas na documentação do fluxo de validação computacional,
sem resultados associados.

| Seção | Tipo de alteração | Razão | Significado científico alterado? | Figura ou tabela nova? |
|---|---|---|---|---|
| Cap. 6 | Substituição das matrizes $\Wmat$/$\Rmat$ do exemplo SUPPORT2 *smoke* ($n\leq32$) por matrizes recalculadas do cenário SUPPORT2 Random Forest `full` 000008 ($n\leq512$) | A execução completa e seus resultados brutos passaram a estar disponíveis | Não muda o método; fortalece a escala da evidência ilustrativa | Sim: quatro matrizes `support2_full` |
| Cap. 7 | Remoção integral da caixa, tabela e figuras SUPPORT2 *smoke*; inclusão de quatro figuras de evolução por $n$ para cada uma das seções Occupancy/GNB, Air Quality/GNB, SUPPORT2/SVM e SUPPORT2/Random Forest | Evitar que uma execução de validação reduzida seja apresentada junto dos resultados científicos e mostrar a trajetória, não apenas o ponto final | Sim: a interpretação passa a se apoiar exclusivamente nos quatro cenários completos e explicita a evolução amostral dos indicadores | Sim: 16 gráficos de métricas por $n$ |
| Caps. 8, 10 e 11 | Remoção das comparações e conclusões baseadas no exemplo *smoke*; atualização das respostas a RQ3--RQ6 e das limitações | Manter discussão e conclusão no mesmo escopo de evidência do Capítulo 7 | Não altera os valores; restringe corretamente o alcance das inferências à escala completa | Não |
| Apêndice A | Remoção da seção de artefatos SUPPORT2 *smoke* e correção das referências cruzadas; declaração explícita da política de resultados `full`/`full-scale` | Os relatórios HTML e resultados completos estão disponíveis na publicação GitHub Pages | Não | Não |
| Apêndice B | Reescrita do comando de regeneração de $\Wmat$/$\Rmat$ para as simulações `full` 000024--000026 e documentação das figuras atuais | Garantir que o procedimento documentado reproduza exatamente a evidência exibida | Não | Não |
| Apresentação, slides 9--11 | Substituição das tabelas estáticas/resultado *smoke* por gráficos compactos de $\rankrho$, $\AW$, $\AR$ e $\SR$ versus $n$, gerados dos cenários `full` 000002, 000005 e 000008; atualização das notas do apresentador | Mostrar a evolução dos quatro indicadores e remover a mistura de escalas | Não altera os dados; melhora a leitura temporal e a consistência entre relatório e apresentação | Sim: três PDFs vetoriais |
| Apresentação, apoio 6 e conclusão | Remoção da tabela SUPPORT2 *smoke* e de todas as citações de resultados em escala reduzida | Manter somente evidência completa em todo o deck | Não | Tabela de apoio simplificada |
| Proveniência de figuras | Atualização de `FIGURE_SOURCES.md`, criação de gerador reprodutível para o deck e exclusão dos oito PNGs *smoke* sem uso | Tornar explícita a origem de cada figura e impedir reutilização acidental de evidência reduzida | Não | Sim |
| Compilação e QA | `latexmk -xelatex` no relatório e `make presentation` no deck; inspeção visual das matrizes full, das quatro páginas de métricas do Capítulo 7, dos slides 9--12 e do apoio 6 | Validar referências, margens, legibilidade e ausência de colisões | Não | PDFs finais com 124 e 30 páginas, respectivamente |

### Ajuste visual subsequente dos slides de resultados

| Seção | Tipo de alteração | Razão | Significado científico alterado? | Figura ou tabela nova? |
|---|---|---|---|---|
| Apresentação, slides 9--11 | Reintegração das projeções bidimensionais dos treinamentos real, Gaussiana única e GMM; reorganização dos quatro indicadores em dois painéis por braço sintético | Permitir a comparação simultânea entre geometria das amostras de treinamento e preservação do perfil ao longo de \(n\) | Não; os mesmos resultados `full` foram recompostos | Não |
| Apresentação, slide 11 | Marcação explícita, por \(\times\) colorido, dos prefixos em que uma métrica do SUPPORT2 é matematicamente indefinida | Evitar que o início posterior de uma curva seja interpretado como dado ausente ou deslocamento da série | Não; nenhum valor foi imputado nem convertido em zero | Não |
