# Running the purified-PBMC GSEA pipeline

The workflow (`convert -> prep -> fit -> gather`) is 61 jobs (6 topics x 8 methods
+ convert/prep/gather). `fit` cells are JAX-heavy; the rest is light. Per-rule
`mem_mb` / `runtime` (minutes) / `threads` are declared in the `Snakefile`.

## Isolated environment

This example is a **self-contained uv project** (its own `pyproject.toml` +
`.venv`), deliberately kept apart from the repo-root env: it pins the gibss-mono
front-door APIs (`fit_glm_susie` / `fit_cox_susie`) at rev `1a9ba7a`. The `convert`
rule additionally shells out to **R** to read the `.RData`, so `Rscript` must be on
PATH (Midway3 has `R-4.3.1`).

```bash
cd gsea_examples/pbmc_purified
uv sync                 # builds ./.venv (fetches gibss-mono@1a9ba7a from GitHub)
which Rscript           # the `convert` rule needs R
```

The input (`de-pbmc-purified-noshrink.RData`) is transferred out-of-band (rsync),
not committed. The `convert` rule downloads NCBI `Homo_sapiens.gene_info.gz` into `resources/` on first
run (`download_geneinfo`); `resources/` and `results/` regenerate.

## Launch

```bash
# local
uv run snakemake -c6

# dry run (preview the DAG without executing)
uv run snakemake -n

# cluster (Midway3 SLURM) - run under tmux/screen; the driver polls SLURM
uv run snakemake --profile profile
```

`profile/config.yaml` submits via the cluster-generic plugin
(`--account=pi-mstephens --partition=caslake --qos=caslake`); **edit the account /
partition if yours differs.** Job logs land in `./logs/%j_<rule>.out`.

## Notes

- **Thread capping.** `fit` requests `threads: 2` and `fit_cell.py` pins the
  BLAS / XLA threadpools to that count before importing jax.
- **JAX recompiles per job.** Each `fit` cell is its own process (XLA compile once).
- **Score choice.** Flip `score_kind` in `config.yaml` between `abs_z` (two-sided,
  default), `signed_postmean` (directional), or `abs_postmean`. `lfsr` is degenerate
  in this object, so ranking uses `z`.
