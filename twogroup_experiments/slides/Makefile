# Build the GSEA / two-group committee deck.
#   make figures   - build all figures (with provenance sidecars)
#   make check     - verify every figure's geometry lines up on the slide
#   make deck      - check, then compile deck.tex -> deck.pdf
#   make           - figures, check, deck
#   make clean     - remove latex aux files (keeps deck.pdf and figures)

.PHONY: all figures check deck clean

all: figures check deck

figures:
	uv run snakemake -s figures.smk all -c4

# Fail the build if any figure would not line up in the slide box.
check:
	uv run --with pymupdf python scripts/check_figures.py --strict

deck: check deck.tex refs.bib
	latexmk -pdf -interaction=nonstopmode deck.tex

clean:
	latexmk -c deck.tex
	rm -f deck.nav deck.snm deck.vrb
