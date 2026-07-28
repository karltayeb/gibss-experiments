# GSEA / two-group committee slides

Section 4 ("Revisiting GSEA") of the 2026-07-30 committee deck.

## Ground rules

- **Beamer**, simple construction (content over polish).
- **Bibliography** included (`refs.bib`); real entries added later.
- **Provenance for every generated figure**: figures are built by `figures.smk`
  + `scripts/`, and each `figures/<name>.pdf` gets a `figures/<name>.prov.json`
  sidecar (script, git commit, timestamp, params). No hand-run one-offs.

## Build

```sh
make            # figures + check + deck
make figures    # just the figures (uv run snakemake -s figures.smk all)
make check      # verify every figure lines up in the slide box (strict)
make deck       # check, then latexmk -pdf deck.tex
```

Requires `uv` (Python + matplotlib + snakemake) and a LaTeX toolchain
(`latexmk`, `pdflatex`, `bibtex`).

## Keeping figures lined up

Figures line up because of two things working together:

1. **Fixed-footprint placement.** `\slidefig` in `deck.tex` centres every figure in
   the *same* box (`0.70\textwidth x 0.72\textheight`, `keepaspectratio`) inside a
   fixed-height `\parbox`. So every figure occupies the same rectangle and the caption
   always starts at the same height, regardless of the figure's aspect ratio.
2. **A geometry check.** `scripts/check_figures.py` reads each `figures/*.pdf`, computes
   its aspect ratio, and flags any figure that would render noticeably narrower than the
   box (i.e. wouldn't line up). It runs as part of `make deck` (strict, fails the build)
   and standalone via `make check`. Intentional exceptions (e.g. B2's 4:3) are listed in
   its `ALLOWLIST`.

To keep new figures in line: target an aspect ratio in **[1.7, 2.75]** (width:height) -
`_common.py` placeholders already do. If a figure must be a different shape, add it to the
checker's `ALLOWLIST` with a reason.

## Layout

```
deck.tex        beamer source (one frame per slide; \includegraphics from figures/)
refs.bib        bibliography (stub)
figures.smk     snakemake: one rule per figure, writes provenance sidecars
scripts/        one script per figure
  _common.py        provenance save() + placeholder() + house style
  _pipeline_stub.py placeholder for results-pipeline-backed figures
  figNN_*.py        standalone figure scripts (local, no cluster)
figures/        build outputs (*.pdf) + provenance (*.prov.json)
outline.md      slide-by-slide plan and task board
```

## Status

All figures are **placeholders** that compile the deck end-to-end. Fill them in:

- **Local (no migration)**: fig06, fig08, fig09, fig10, fig14, fig15, B1, B2.
- **Pipeline-backed (after gibss-mono migration + rerun)**: fig17, fig18, fig19,
  fig20, B3, B4 -- currently `_pipeline_stub.py` placeholders.

Naming: **cox / cox-reverse** (= Plackett-Luce partial likelihood).
The review-surface artifact holds the slide-by-slide with cartoon mockups.
