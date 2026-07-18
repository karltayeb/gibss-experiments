# pbmc_purified logistic-SuSiE GSEA

Gene-set enrichment on the purified-PBMC topic-model differential expression
(Zheng et al. 2017), using the same logistic / Cox / ORA SuSiE suite as the sibling
[`lifan`](../lifan) example. Built to be **parallel to lifan**: `scripts/prep.py`,
`scripts/fit_cell.py`, and `scripts/scoring.py` are shared verbatim; only
`scripts/convert.R` is bespoke, because the input is an `.RData`
`topic_model_de_analysis` object keyed by Ensembl ids.

## Analysis units

Six topics of the purified-PBMC model (`config.yaml: units`): `k1 .. k6`. Each is a
per-gene DE signature (`postmean` loading + Wald `z`) for one topic.

## Pipeline

```
convert (R)   RData topic-model DE -> per-topic tidy TSV (entrez, score, signed_effect) + Entrez map
   -> prep    align genes to MSigDB c2.all, cache the sparse design per topic
   -> fit     one job per topic x method (logistic / cox / ora)
   -> gather  concat a topic's SuSiE methods into results/{topic}/results.parquet
```

- **Score.** `config.yaml: score_kind` sets the per-gene ranking statistic:
  `abs_z` = `|Wald z|` (two-sided significance, default), `signed_postmean`
  (topic-up first), or `abs_postmean`. `lfsr` in this object is degenerate, so we
  rank on `z`; `signed_effect` (`postmean`) is carried only to annotate set direction.
- **Hit lists / Cox censoring** use the top `thresholds` fractions `[0.05, 0.10, 0.20]`.
- **Methods** (`config.yaml: methods`): logistic (L1, L20); Cox with the exact
  profiled (`coarse`) baseline - censored at L1, forward and reversed at L1 and L20;
  and ORA. 8 methods per topic. One entry = one fit =
  `results/fits/{topic}/{method}.parquet`. (The fast `shared_baseline` Cox is
  omitted: on lifan it gave ~identical rankings to the exact baseline.)

## Run

Self-contained uv project (pins `gibss-mono@1a9ba7a`). The `convert` rule shells out
to **R** (`Rscript` must be on PATH) to read the `.RData`. See [RUNNING.md](RUNNING.md).

```bash
cd gsea_examples/pbmc_purified
uv sync
uv run snakemake -c6                  # local  (61 jobs: convert + 6 prep + 48 fit + 6 gather)
uv run snakemake --profile profile    # Midway3 SLURM
```

Input data (`de-pbmc-purified-noshrink.RData`) is gitignored and transferred
out-of-band (rsync); `resources/` (NCBI `gene_info`) and `results/` regenerate.
