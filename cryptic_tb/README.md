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

- **The credible-set count depends on L, so do not read it as a property of the data.**
  Two separate effects, both shown by `scripts/05_stability.py`:
  1. A *saturated* fit, one whose active components fill every available slot, has run out
     of components and invents sets. At L=20 isoniazid produced six ahpC promoter credible
     sets that are absent at L=40 and L=60. Those fits are not reportable.
  2. Even unsaturated, the count still drifts: ethambutol gives 16 sets at L=40 and 12 at
     L=60. So "N credible sets" is not a stable summary for the strong-signal drugs, and
     the per-variant question ("is rpoB Ser450Leu in a set at every L?") is the one that
     has a stable answer. `stability_L.md` reports that set.

  L=40 is the reported setting, and it is unsaturated for every drug.
- **The WHO join is by genomic coordinate, not name.** The two sources disagree about which
  gene some variants belong to. CRyPTIC files the main isoniazid promoter variant under
  `fabG1` at `c-15`; WHO files the same base under `inhA` at `c.-777` and lists no `fabG1`
  tier for isoniazid. A name join drops the second strongest isoniazid signal in the cohort.
- **Pyrazinamide is not in this cohort.** The UKMYC plates do not include it.
- **Covariates are fit jointly** (`fit_glm_susie(covariates=Z)`, gibss 7c7e462), not frozen
  as an offset. This matters: the frozen-offset route *caused* lineage-marker false
  positives that joint fitting does not have, while finding fewer true positives. See
  `results/covariate_mode.md` and the covariates section of `docs/data_notes.md`.
  Clofazimine's mmpL5 Asp767Asn survives joint adjustment and is still unexplained;
  genome-wide principal components are the next thing to try.
