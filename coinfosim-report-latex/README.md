# Relatório CoInfoSim em LaTeX

Projeto modular do relatório. O pacote contém a redação integral dos capítulos 1 a 11, resumo, abstract, referências e apêndices de resultados suplementares e reprodutibilidade.

## Compilação

Requisitos: XeLaTeX, BibTeX e `latexmk`.

```bash
latexmk -xelatex main.tex
```

Para limpar arquivos auxiliares:

```bash
latexmk -C
```

O PDF é gerado como `build/main.pdf`.

## Estrutura

- `config/`: metadados, preâmbulo e comandos;
- `frontmatter/`: capa, folha de rosto, resumos, listas e sumário;
- `chapters/`: capítulos modulares;
- `appendices/`: apêndices;
- `references/`: BibTeX e lista de referências pendentes;
- `figures/`: destino das figuras auditadas;
- `docs/`: decisões normativas e plano editorial.

## Modo rascunho

Em `config/metadata.tex`, `\draftdocumenttrue` exibe caixas editoriais e inclui
todas as referências iniciais. Para a versão final, trocar para
`\draftdocumentfalse` e remover os marcadores pendentes.

## Normas

A UFRJ governa margens, paginação, espaçamento, estrutura pré-textual,
ilustrações e referências. O template SBC é usado como referência tipográfica e
de apresentação de tabelas/legendas. Consulte `docs/STYLE_DECISIONS.md`.

### Windows / MiKTeX

No TeXworks, TeXstudio ou VS Code/LaTeX Workshop, selecione XeLaTeX e Biber. O comando `latexmk -xelatex main.tex` continua sendo a forma preferida quando `latexmk` estiver disponível.


## Progresso editorial

Consulte `docs/STAGE5_PROGRESS.md` para o estado atual da redação.

## Etapa 4 - figuras publicadas

Os gráficos de síntese numérica já estão em `figures/`. As geometrias e matrizes
originais permanecem como marcadores na prévia porque o ambiente de compilação
não possui acesso direto aos binários da branch `gh-pages`. Em uma máquina com
internet, execute:

```bash
python scripts/fetch_published_figures.py
```

O script baixa as figuras selecionadas para `figures/published/`. A inserção
definitiva deve manter as escalas, a ordem dos subconjuntos e as legendas dos
relatórios publicados.
