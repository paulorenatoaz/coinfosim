# Compila os diagramas conceituais (determinístico; requer XeLaTeX + TikZ).
# Uso: make -C coinfosim-report-latex/figures/conceptual/src
DIAGRAMAS = pipeline_condicoes cooperacao_canais reticulado_subconjuntos \
            metricas_tres_paineis construcao_ar_sr quadrante_desempenho \
            trajetoria_vencedor

all: $(DIAGRAMAS:%=../%.pdf)

../%.pdf: %.tex preamble_diagrams.tex
	xelatex -interaction=nonstopmode -halt-on-error $<
	mv $*.pdf ../$*.pdf

clean:
	rm -f *.aux *.log *.xdv

.PHONY: all clean
