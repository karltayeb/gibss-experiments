# covid (GSE147507) logistic-SuSiE GSEA

Gene-set enrichment on the GSE147507 NHBE differential-expression contrasts, using
the same logistic / Cox / ORA SuSiE suite as the sibling [`lifan`](../lifan)
example. Built to be **parallel to lifan**: `scripts/prep.py`, `scripts/fit_cell.py`,
and `scripts/scoring.py` are shared verbatim; only `scripts/convert.py` is bespoke,
because the input here is a DESeq2 table (HGNC symbols; `log2FoldChange`, `stat`,
`padj`) rather than a factor-analysis RDS.

## Analysis units

Two NHBE contrasts (`config.yaml: targets`), each a public DESeq2 comparison:

- `nhbe_sars_cov_2_vs_mock`
- `nhbe_iav_vs_mock`

## Pipeline

```
deseq2 -> convert -> prep -> {fit_susie, fit_ora, fit_gsea, fit_slpr, fit_mgsa} -> gather
```

Fanned out over `score x collection x target x method`. The only provided input is
`resources/raw/` (GSE147507 human counts + sample sheet); everything else - DE
included - is generated into `results/`.

| rule | inputs | outputs | role |
|---|---|---|---|
| `download_geneinfo` | *(NCBI URL)* | `resources/Homo_sapiens.gene_info.gz` | fetch symbol->Entrez reference |
| `deseq2` | `resources/raw/` counts + `sample_metadata.tsv` | `results/de/de_results.tsv` | DESeq2 (pydeseq2) per target -> DE table |
| `convert` | `de_results.tsv`, `gene_info.gz` | `results/targets/{score}/{target}.tsv`, `results/genes/{score}/gene_mapping.tsv` | tidy DE -> per-target `(entrez, score, signed_effect)` + Entrez map |
| `prep` | `results/targets/{score}/{target}.tsv` | `results/prep/{score}/{collection}/{target}_design.npz` + `_sets.parquet` | align genes to the collection, cache the sparse design |
| `fit_susie` | design + sets | `results/fits/{score}/{collection}/{target}/{method}.parquet` | logistic/cox/linear SuSiE fit (component rows) - heavy |
| `fit_ora` | design + sets | `.../{method}.parquet` (`ora`) | Fisher over-representation per threshold - light |
| `fit_gsea` | design + sets | `.../{method}.parquet` (`gsea`) | gseapy prerank (NES/FDR) - light |
| `fit_slpr` | design + sets | `.../{method}.parquet` (`slpr`, `slpr_abs`) | SLPR: LASSO+OLS multiset selection (R glmnet) - moderate |
| `fit_mgsa` | design + sets | `.../{method}.parquet` (`mgsa`) | MGSA: Bayesian multiset MCMC per threshold (R `mgsa`) - heavy (MCMC) |
| `gather` | `fit_susie` outputs for a target | `results/{score}/{collection}/{target}/results.parquet` | concat the SuSiE fits into one table |

- **Score.** `config.yaml: score_kind` sets the per-gene ranking statistic:
  `abs_stat` = `|Wald z|` (two-sided significance, default), `signed_stat`
  (up-regulated first), or `neglog_padj`. `signed_effect` (log2FC) is carried only
  to annotate each set's direction.
- **Hit lists / Cox censoring** use the top `thresholds` fractions `[0.05, 0.10, 0.20]`.
- **Methods** (`config.yaml: methods`): logistic (L1, L20); Cox with the exact
  profiled (`coarse`) baseline - censored at L1, forward and reversed at L1 and L20;
  ORA; GSEA; and the R-backed multiset baselines SLPR (`slpr`, `slpr_abs`) and MGSA
  (`mgsa`, see below). One entry = one fit =
  `results/fits/{score}/{collection}/{target}/{method}.parquet`. (The fast
  `shared_baseline` Cox is omitted: on lifan it gave ~identical rankings.)

## R-backed multiset methods (SLPR, MGSA)

Two *multiset* baselines (they jointly model the whole collection to select a
parsimonious, minimally-overlapping set of active pathways, unlike the per-set
`ora`/`gsea`). Both have no faithful Python implementation, so their numeric core
runs in R; `fit_slpr.py` / `fit_mgsa.py` load the cached design, hand R the sparse
membership through a per-job temp dir (MatrixMarket + TSV, see `scripts/_rbridge.py`),
and write the per-set parquet. Base R + `Matrix` + the method package only - no R
Arrow/data.table needed.

- **SLPR** - gene set Selection via LASSO Penalized Regression (Frost & Amos 2017,
  NAR 45(12):e114; `scripts/slpr.R` ports the paper's reference `slpr()`). Regresses
  the per-gene **effect size** (log2FC, cached by `prep` as `gene_signed_effect`) on
  the genes x sets membership under a LASSO (`cv.glmnet`, `standardize=FALSE`,
  `alpha=1`, `lambda.min`), then refits an unpenalized OLS on the selected sets
  (two-stage "Gauss-Lasso"). Effect sizes - **not** z/t-statistics - are the paper's
  recommended response. Two variants: `slpr` (signed log2FC; coef sign = direction),
  `slpr_abs` (`abs_response`, |log2FC|; scale alternative). Output per set:
  `coef_lasso` (shrunken), `coef_ols`/`ols_pvalue` (unshrunken, non-inferential),
  `selected`, `direction`, `rank` (by |coef_lasso|), `lambda`, `n_selected`.
- **MGSA** - model-based gene set analysis (Bauer et al. 2010; Bioconductor `mgsa`;
  `scripts/mgsa.R`). Bayesian model over the **binary** top-fraction hit list (the
  same hit lists `ora`/logistic use, built in Python), run once per `threshold`.
  Returns each set's posterior `estimate` = P(active), plus posterior-mean global
  `alpha`/`beta`/`p`. Tunables in `config.yaml`: `steps`/`restarts`/`thin`
  (`restarts>1` yields the `std_error`s; mgsa `threads` parallelize restarts).

Both need R (`Rscript` on `PATH`) with the packages installed once per machine
(local and each cluster node):

```r
install.packages("glmnet")
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install("mgsa")
```

**Measured runtime** (one target, all 3 thresholds; local, 4 cores; ~4 s of that is
Python import + design I/O). Both are cheap even on the largest collections - MGSA's
per-step MCMC cost grows only mildly with set count, so `steps` (not #sets) dominates:

| collection | genes x sets | SLPR | MGSA (`steps=1e6`, `restarts=1`) |
|---|---|---|---|
| hallmark | 3.9k x 50 | ~2 s | ~2.5 s |
| go_bp | 14.4k x 4815 | ~13 s | ~11 s |
| c2_all | 15.4k x 6098 | ~14 s | ~13 s |

`restarts=5` (for `std_error`s) adds only ~2 s/threshold. The full grid adds a few
minutes total, not hours.

## Run

Self-contained uv project (pins `gibss-mono@1a9ba7a`, the front-door
`fit_glm_susie` / `fit_cox_susie` APIs). See [RUNNING.md](RUNNING.md).

```bash
cd gsea_examples/covid
uv sync
uv run snakemake -c6                  # local  (135 jobs across score x collection x target x method)
uv run snakemake --profile profile    # Midway3 SLURM
```

The only provided input is `resources/raw/` (GSE147507 human counts + sample sheet),
gitignored and transferred out-of-band (rsync). Everything else - the DESeq2 table
(`results/de/`, via `deseq2`), NCBI `gene_info` (downloaded by `download_geneinfo`),
and all fits/figures - is generated into `results/`.
