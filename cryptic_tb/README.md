# cryptic_tb

Logistic SuSiE (gibss) applied to *M. tuberculosis* drug resistance in the CRyPTIC data,
with the WHO 2023 mutation catalogue as the truth set. See `HANDOFF.md` for the original
brief, `docs/data_notes.md` for what was downloaded and how the two sources were joined,
and `results/summary.md` for the fits.

Self-contained uv project, pinned to its own gibss-mono rev. Run from this directory.

```
uv sync
uv run python scripts/01_filter_mutations.py   # 88M rows -> candidate-gene parquet (~3 min)
uv run python scripts/02_build_design.py       # per-drug X, y, covariates, WHO grades
uv run python scripts/03_fit.py                # logistic SuSiE + marginal comparators
uv run python scripts/04_summarize.py          # results/summary.md
uv run python scripts/05_stability.py          # L sensitivity; results/stability_L.md
```

`scripts/03_fit.py` takes drug codes as arguments (`uv run python scripts/03_fit.py RIF INH`)
and reads `L` from the `L` environment variable, defaulting to the last entry of
`fit.L` in `config.yaml`.

Raw data and the processed matrices are gitignored.

## What to know before reading the results

- **Report only unsaturated fits.** A fit whose active components fill every available slot
  has run out of components, and the extra credible sets are an artifact of that cap. At
  L=20 isoniazid produced six ahpC promoter credible sets that vanish at L=40 and L=60.
  `scripts/05_stability.py` checks this; L=40 is the reported setting.
- **The WHO join is by genomic coordinate, not name.** The two sources disagree about which
  gene some variants belong to. CRyPTIC files the main isoniazid promoter variant under
  `fabG1` at `c-15`; WHO files the same base under `inhA` at `c.-777` and lists no `fabG1`
  tier for isoniazid. A name join drops the second strongest isoniazid signal in the cohort.
- **Pyrazinamide is not in this cohort.** The UKMYC plates do not include it.
- **Covariates enter as a fixed offset**, since the gibss front door has no fixed-covariate
  block. Lineage indicators stand in for genome-wide principal components for now, and three
  credible sets land on common WHO grade-5 variants as a result.
