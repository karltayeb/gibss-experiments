# covid (GSE147507) logistic-SuSiE GSEA

Gene-set enrichment on the GSE147507 NHBE differential-expression contrasts, using
the same logistic / Cox / ORA SuSiE suite as the sibling [`lifan`](../lifan)
example. Built to be **parallel to lifan**: `scripts/prep.py`, `scripts/fit_cell.py`,
and `scripts/scoring.py` are shared verbatim; only `scripts/convert.py` is bespoke,
because the input here is a DESeq2 table (HGNC symbols; `log2FoldChange`, `stat`,
`padj`) rather than a factor-analysis RDS.

## Analysis units

Two NHBE contrasts (`config.yaml: units`), each a public DESeq2 comparison:

- `nhbe_sars_cov_2_vs_mock`
- `nhbe_iav_vs_mock`

## Pipeline

```
convert (py)  DESeq2 tables -> per-contrast tidy TSV (entrez, score, signed_effect) + Entrez map
   -> prep    align genes to MSigDB c2.all, cache the sparse design per contrast
   -> fit     one job per contrast x method (logistic / cox / ora)
   -> gather  concat a contrast's SuSiE methods into results/{contrast}/results.parquet
```

- **Score.** `config.yaml: score_kind` sets the per-gene ranking statistic:
  `abs_stat` = `|Wald z|` (two-sided significance, default), `signed_stat`
  (up-regulated first), or `neglog_padj`. `signed_effect` (log2FC) is carried only
  to annotate each set's direction.
- **Hit lists / Cox censoring** use the top `thresholds` fractions `[0.05, 0.10, 0.20]`.
- **Methods** (`config.yaml: methods`): logistic (L1, L20); Cox with the exact
  profiled (`coarse`) baseline - censored at L1, forward and reversed at L1 and L20;
  and ORA. 8 methods per contrast. One entry = one fit =
  `results/fits/{contrast}/{method}.parquet`. (The fast `shared_baseline` Cox is
  omitted: on lifan it gave ~identical rankings to the exact baseline.)

## Run

Self-contained uv project (pins `gibss-mono@1a9ba7a`, the front-door
`fit_glm_susie` / `fit_cox_susie` APIs). See [RUNNING.md](RUNNING.md).

```bash
cd gsea_examples/covid
uv sync
uv run snakemake -c6                  # local  (21 jobs: convert + 2 prep + 16 fit + 2 gather)
uv run snakemake --profile profile    # Midway3 SLURM
```

Input data (`de/`, `raw/`, `gene_lists/`) is gitignored and transferred
out-of-band (rsync); `resources/` (NCBI `gene_info`) and `results/` regenerate.
