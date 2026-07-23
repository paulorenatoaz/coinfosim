LATEXMK = latexmk
LATEXFLAGS = -xelatex -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all presentation notes clean figures

all: presentation

figures:
	../.venv/bin/python scripts/generate_air_quality_gnb_curves.py
	../.venv/bin/python scripts/generate_full_scale_metric_curves.py

presentation: figures
	$(LATEXMK) $(LATEXFLAGS) -outdir=build presentation.tex

notes: figures
	$(LATEXMK) $(LATEXFLAGS) -outdir=build presentation-notes.tex

clean:
	$(LATEXMK) -C -outdir=build presentation.tex
	$(LATEXMK) -C -outdir=build presentation-notes.tex
