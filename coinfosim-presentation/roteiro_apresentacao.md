# Roteiro de apresentação — CoInfoSim

**Duração alvo:** 18 min 15 s de narrativa principal (12 slides), com folga até 20 min para respiração, pausas e pequenas variações de ritmo. Os tempos por slide abaixo são os mesmos gravados no rodapé de cada slide (`\slidetime`), então servem como checkpoints reais: se você chegar muito adiantado ou atrasado em relação à marca, ajuste o ritmo dos slides seguintes.

Leia isso em voz alta pelo menos duas vezes antes de apresentar — o texto foi escrito para ser falado, não para ser lido. Convém não decorar palavra por palavra: decore a sequência de ideias e deixe a frase sair naturalmente.

---

## Slide 1 — Título (0:00 – 0:45)

Boa tarde a todos. O trabalho que vou apresentar chama-se *Preservação do perfil de cooperação preditiva entre atributos*, e a pergunta que ele responde é esta: **o treinamento com amostras sintéticas reproduz a evolução da cooperação observada com amostras reais?**

Uma coisa precisa ficar clara desde o primeiro minuto, porque ela organiza tudo o que vem depois: amostras sintéticas, aqui, **não** são usadas para aumentar a acurácia do classificador. Elas são usadas para investigar se o treinamento sintético reproduz o comportamento cooperativo dos subconjuntos de canais que observamos quando o treinamento é feito com dados reais. Se você guardar só uma frase desta apresentação, guarde essa.

*(Transição: Para entender por que essa pergunta importa, começo pela decisão operacional que existe antes de qualquer gerador sintético.)*

---

## Slide 2 — Por que comparar subconjuntos de atributos? (0:45 – 2:05)

Em operações reais, cada variável — cada canal — pode representar um sensor ambiental, um exame laboratorial, uma medição fisiológica, a saída de um equipamento, ou até uma informação fornecida por um especialista. Quando decidimos quais canais usar num classificador, adicionar mais um canal pode trazer sinal complementar — mas também pode trazer redundância, mais dimensionalidade, mais complexidade de integração e mais custo operacional.

E há uma armadilha estatística aqui: com poucas amostras, acrescentar dimensões pode até **piorar** a estimação — é o efeito de pequenas amostras associado ao fenômeno de Hughes. Então "mais canais" não é uma resposta óbvia.

Este trabalho não resolve a decisão econômica de quais canais vale a pena manter. Ele constrói a etapa anterior a essa decisão: mapear como **todos** os subconjuntos possíveis de atributos se comportam na tarefa de classificação, treinando um classificador para cada subconjunto. E a aposta é a seguinte: se essa organização puder ser reproduzida por treinamento sintético, teremos uma ferramenta estrutural útil para estudos operacionais futuros — sem precisar coletar mais dados reais para cada nova pergunta.

*(Transição: Essa motivação operacional leva à definição precisa do que estou chamando de estrutura de cooperação.)*

---

## Slide 3 — O que é a estrutura de cooperação? (2:05 – 4:05)

Aqui está o coração conceitual da apresentação, então vou com calma.

A estrutura começa com curvas de perda no mesmo teste real. Para cada tamanho de treinamento por classe — vou chamar esse tamanho de **n** — cada subconjunto de canais origina um classificador próprio, e esse classificador tem uma perda média, estimada por simulação de Monte Carlo.

Repare no gráfico: com poucas amostras, um canal isolado pode ter a menor perda, porque exige menos estimação — é mais simples de aprender. Mas, à medida que o treinamento cresce, um par de canais, ou um conjunto maior, pode passar a aproveitar informação complementar e ultrapassar o canal isolado. É exatamente isso que as três curvas mostram: uma inversão de posições ao longo de n.

Isso me dá uma hierarquia de três degraus, que peço que vocês guardem, porque ela reaparece em todo o resto da apresentação:

Primeiro, o **desempenho** pertence a cada subconjunto — é a perda média daquele classificador específico.

Segundo, o **ranking** compara todos os subconjuntos entre si, mas só para um valor fixo de n — é uma fotografia.

Terceiro, e este é o objeto novo, a **estrutura de cooperação** é a evolução dessas posições relativas conforme n cresce — é o filme, não a fotografia.

Então, para ser preciso: cooperação, aqui, não é correlação bruta entre variáveis, e não é uma propriedade que já está pronta dentro da amostra. É uma organização dinâmica que só existe depois que treinamos os classificadores e comparamos as perdas deles ao longo da grade amostral.

*(Transição: Com esse objeto bem definido, posso agora formular a pergunta científica sem confundi-la com aumento de dados ou melhoria preditiva.)*

---

## Slide 4 — Pergunta científica e escopo (4:05 – 5:25)

A pergunta de pesquisa é: em que medida a estrutura de cooperação que emerge do treinamento de classificadores com amostras sintéticas reproduz aquela que emerge do treinamento dos mesmos classificadores com amostras reais?

Três coisas para deixar muito claras sobre o escopo, porque são os erros de interpretação mais fáceis de cometer aqui.

Isto **não é** aumento de dados — não estamos misturando sintético com real para treinar melhor.

Isto **não é** busca de maior acurácia — o objetivo não é o classificador acertar mais.

O que **é**: avaliar a preservação da estrutura de cooperação. Comparamos duas estruturas emergentes — uma vem do treinamento com amostras reais, a outra vem do treinamento com amostras produzidas por um modelo gerador parametrizado a partir desses mesmos dados reais. Em ambos os casos, a avaliação acontece no mesmo teste real fixo.

Esse desenho lembra o protocolo *Train on Synthetic, Test on Real*, mas o desfecho que nos interessa não é a perda isolada de um classificador — é o padrão relativo entre **todos** os subconjuntos: a ordenação, as vitórias par a par, e a dinâmica das transições entre eles.

*(Transição: Essa pergunta exige um protocolo experimental em que somente a origem das amostras de treinamento mude — é isso que o próximo slide mostra.)*

---

## Slide 5 — Três braços, um único teste real (5:25 – 7:25)

O protocolo parte de um único reservatório real de treinamento. Toda a padronização e toda a parametrização dos modelos geradores usam **somente** esse reservatório — o teste nunca entra nessa conta.

A partir daí, temos três braços experimentais. No primeiro, Real para Real, cada replicação simplesmente sorteia amostras reais balanceadas por classe — é a nossa referência. No segundo braço, uma Gaussiana única por classe é parametrizada a partir dos dados reais de treinamento, no espaço completo de canais, e produz as amostras sintéticas usadas para treinar os classificadores. No terceiro braço, um GMM — uma mistura de gaussianas — por classe é parametrizado a partir dos mesmos dados reais; o número de componentes é escolhido pelo critério BIC.

Para cada tamanho da grade — n igual a 2, 4, 8, e assim por diante até 512 amostras por classe — repetimos o treinamento de cada classificador em **todos** os subconjuntos não vazios de atributos: são 31 subconjuntos quando há cinco atributos, e 127 quando há sete. As replicações de Monte Carlo estimam as perdas médias com a precisão que definimos como alvo.

O ponto crucial: a **única** coisa que muda entre os três braços é a origem das amostras de treinamento. O conjunto de teste real é fixo e compartilhado pelos três — é ele que nos permite comparar as estruturas obtidas em uma base absolutamente comum.

*(Transição: Dessas perdas médias derivam três métricas estruturais. Vou apresentar as três, começando pela mais global.)*

---

## Slide 6 — Métrica 1: a ordenação global foi preservada? (7:25 – 8:45)

Em cada tamanho amostral, ordenamos **todos** os subconjuntos pela perda média no teste real — não é só "quem ganhou", é a lista inteira, do melhor ao pior.

A primeira métrica, a correlação de ranking, usa a correlação de Spearman para comparar o vetor de postos do braço sintético com o vetor de postos da referência Real para Real.

É uma medida global, e isso é proposital: se dois subconjuntos quase empatados trocam de posição entre si, a correlação continua muito alta — porque essa troca é irrelevante para a organização geral. E isso importa porque a estrutura não deve ser reduzida a "quem ficou em primeiro lugar". Tivemos um caso, no Occupancy, em que o braço treinado com GMM acertou exatamente o vencedor do SVM linear, mas foi o braço da Gaussiana única que produziu o ranking **completo** mais próximo da referência. Ou seja: acertar o topo e preservar a lista inteira são duas afirmações diferentes, e a correlação de ranking mede a segunda.

*(Transição: Mas essa visão global ainda pode esconder quais decisões específicas entre pares foram preservadas — é isso que a segunda métrica abre.)*

---

## Slide 7 — Métrica 2: decisões "A vence B" (8:45 – 10:05)

Agora comparamos cada par de subconjuntos diretamente. Se a perda média de A é menor que a de B, dizemos que A vence B — simples assim, e empates exatos são excluídos porque não têm vencedor estrito comparável.

A Winners Agreement é a proporção dessas decisões par a par que coincide entre o braço sintético e a referência real. Para dar uma noção de escala: com 31 subconjuntos, existem 465 pares distintos; com 127 subconjuntos, são 8.001 pares.

Por que essa métrica importa na prática: mesmo que haja uma troca residual entre os dois primeiros colocados, centenas — ou milhares — de outras relações entre alternativas podem continuar exatamente iguais. Uma troca no topo não é um colapso da estrutura. A Winners Agreement nos diz quanto do mapa decisório inteiro foi conservado, não apenas quem está na primeira linha.

*(Transição: Ranking e vencedores observam um ponto fixo da grade. A terceira métrica adiciona explicitamente a dinâmica do tamanho amostral.)*

---

## Slide 8 — Métrica 3: quando a superioridade muda? (10:05 – 11:55)

Este é o gráfico canônico do que observamos nos experimentos: duas curvas de perda, suaves, decrescentes, com exatamente um cruzamento. Antes de N-estrela, o subconjunto S-A tem menor perda; depois de N-estrela, é o subconjunto S-B que passa a ser superior.

Formalmente, um evento de cruzamento acontece quando, no ponto anterior da grade, o primeiro subconjunto não era estritamente melhor, e no ponto seguinte passa a ser — sem interpolação entre pontos, só o que foi observado. Na prática, a maioria dos pares de subconjuntos exibe zero ou um cruzamento ao longo de toda a grade; dois cruzamentos já são incomuns, e três ou mais são raríssimos nos cenários que avaliamos. Para os casos excepcionais em que há mais de um cruzamento, a implementação registra o **último** cruzamento observado até aquele prefixo da grade — isso mantém a definição bem-posta sem exigir que a métrica central da apresentação seja sobre oscilação.

A similaridade progressiva de N-estrela combina duas coisas: primeiro, existência — os mesmos pares de subconjuntos cruzam nos dois braços? Segundo, posição — esse último cruzamento acontece num regime amostral parecido, numa escala logarítmica de n? O produto dessas duas componentes explica por que ranking e Winners Agreement podem estar muito próximos de um, enquanto a similaridade de N-estrela permanece moderada: preservar a ordenação final é mais fácil do que reproduzir exatamente a trajetória das inversões.

*(Transição: Com as três métricas definidas, passo aos resultados. O primeiro caso separa aparência geométrica de preservação real da estrutura.)*

---

## Slide 9 — Occupancy: parecer realista não basta (11:55 – 13:35)

No Occupancy, a projeção real de temperatura e umidade tem trajetórias estreitas, concentrações irregulares, formas pouco elípticas. A Gaussiana única regulariza fortemente essa geometria — fica um contorno bem mais suave e simples do que a realidade. O GMM recupera melhor essa multimodalidade visível. Visualmente, o GMM parece claramente mais fiel.

Mas o resultado estrutural do SVM linear não segue essa aparência. Em n igual a 512, os classificadores treinados com amostras da Gaussiana única alcançaram correlação de ranking de 0,9528 e Winners Agreement de 0,9204 — acima dos 0,8774 e 0,8624 obtidos pelos classificadores treinados com amostras do GMM. Só na similaridade de N-estrela o GMM levou uma pequena vantagem: 0,4949 contra 0,4479.

A distribuição sintética que parece mais realista aos nossos olhos não é necessariamente aquela cujo treinamento melhor reproduz a estrutura de cooperação. Semelhança geométrica, ordenação global e dinâmica dos cruzamentos são três critérios distintos.

*(Transição: O segundo caso mostra o extremo positivo — uma organização quase inteiramente reproduzida ao longo de toda a grade amostral.)*

---

## Slide 10 — Air Quality + Gaussian Naive Bayes: preservação ao longo de n (13:35 – 15:15)

Aqui acompanhamos a preservação ao longo de **toda** a grade, não só no ponto final — e essa visão progressiva é importante, porque um único número em n igual a 512 esconde como a estrutura se comporta ao longo do caminho.

Cada painel é um braço sintético: à esquerda, Gaussiana única; à direita, GMM, os dois parametrizados a partir dos dados reais de treinamento. Reparem que a correlação de ranking e a Winners Agreement — as linhas azul e amarela — já ficam muito próximas de 1 desde os prefixos pequenos, e permanecem lá até o fim da grade, que aqui vai até n igual a 1024. Isso significa que a ordenação global e as decisões par a par são preservadas quase integralmente, e cedo.

Já a similaridade progressiva de N-estrela, em roxo, é claramente menor e evolui de forma diferente: cresce ao longo da grade e estabiliza em torno de 0,73 no braço da Gaussiana única e 0,81 no braço do GMM.

A leitura correta não é "preservação perfeita" — é que a preservação é **multidimensional**. Os dois treinamentos sintéticos reproduzem fortemente a ordenação global e as relações de vencedores; reproduzir a posição exata dos últimos cruzamentos é uma exigência mais estrita, e ela permanece abaixo de 1.

*(Transição: O terceiro caso separa outra dupla de conceitos que é fácil confundir: qualidade preditiva absoluta e preservação da estrutura.)*

---

## Slide 11 — SUPPORT2: predição ≠ estrutura (15:15 – 16:55)

No SUPPORT2, o Random Forest teve perda mínima próxima de 0,455 nos três braços. Para efeito de comparação, um classificador que sempre prevê a classe majoritária erra 0,4687 — então a melhoria absoluta é pequena, e o desempenho preditivo desse classificador, nesse problema, é fraco.

Ainda assim — e este é o ponto — a comparação relativa entre os 127 subconjuntos mostra que uma parte substancial da estrutura de cooperação foi preservada. Com amostras da Gaussiana única, correlação de ranking, Winners Agreement e similaridade de N-estrela foram 0,8387, 0,8282 e 0,6799. Com amostras do GMM, foram 0,8665, 0,8429 e 0,6340. O GMM levou vantagem nas relações estáticas; a Gaussiana única, na dinâmica dos últimos cruzamentos.

Isso não transforma o Random Forest no melhor classificador para esse problema clínico — longe disso. O que isso mostra é que perda absoluta e preservação da estrutura de cooperação respondem a perguntas diferentes: um classificador pode ter desempenho preditivo fraco e, mesmo assim, responder de forma estruturalmente muito parecida ao treinamento real e ao sintético.

*(Transição: Os três estudos de caso nos permitem agora uma síntese cuidadosa sobre o que foi, e o que não foi, demonstrado.)*

---

## Slide 12 — Síntese e conclusão (16:55 – 18:15)

Fecho com três ideias, e depois a conclusão central.

Primeiro: a preservação estrutural é um perfil **multidimensional**. Ranking, Winners Agreement e similaridade de N-estrela contam histórias complementares, e não devem ser fundidos numa única afirmação.

Segundo: geometria, perda preditiva e estrutura de cooperação são propriedades **distintas**. Nenhuma delas permite inferir automaticamente as outras — vimos isso no Occupancy, e vimos isso no SUPPORT2.

Terceiro: nenhum gerador foi universalmente superior nos cenários que avaliamos. A preservação da estrutura de cooperação depende da interação entre dataset, classificador, modelo gerador **e** tamanho amostral.

A conclusão central, então, não é que amostras sintéticas melhoram o classificador — isso nunca foi o objetivo. É que, em determinados cenários, o treinamento com amostras sintéticas reproduz a organização cooperativa que observamos com o treinamento real.

Uma ressalva final, importante: extrapolação para tamanhos amostrais maiores e decisões de custo operacional **não** foram validadas neste trabalho. Elas permanecem como hipóteses para pesquisa futura, não como resultado demonstrado aqui.

Muito obrigado. Fico à disposição para perguntas — e tenho slides de apoio com as definições formais das métricas, as configurações completas dos geradores e classificadores, e resultados adicionais, caso seja útil aprofundar algum ponto.

---

## Notas de ritmo para o apresentador

- **Checkpoints de tempo:** slide 3 deve terminar por volta de 4:05; slide 6 por volta de 8:45; slide 9 por volta de 13:35; slide 12 por volta de 18:15. Se estiver 1 minuto ou mais atrasado num desses pontos, corte adjetivos e exemplos secundários nos slides seguintes — nunca corte as frases de definição (elas evitam mal-entendidos na arguição).
- **Se sobrar tempo até os 20 minutos:** use a folga para pausas depois das transições marcadas, não para adicionar conteúdo novo — o roteiro já cobre tudo o que os slides mostram.
- **Perguntas prováveis de banca:** por que o GMM não é sistematicamente melhor (slide 12, terceira ideia); por que N-estrela é mais baixo que as outras métricas (slide 8, definição); se isso é data augmentation (slide 4, primeira negação); se o resultado se generaliza para outros datasets (não — é condicional a dataset, classificador, gerador e n, como o slide 12 deixa explícito).
