# Running the covid (GSE147507) GSEA pipeline

The workflow (`convert -> prep -> fit -> gather`) is 21 jobs (2 contrasts x 8
methods + convert/prep/gather). `fit` cells are JAX-heavy; the rest is light.
Per-rule `mem_mb` / `runtime` (minutes) / `threads` are declared in the `Snakefile`.

## Isolated environment

This example is a **self-contained uv project** (its own `pyproject.toml` +
`.venv`), deliberately kept apart from the repo-root env: it pins the gibss-mono
front-door APIs (`fit_glm_susie` / `fit_cox_susie`) at rev `1a9ba7a`. Everything
below runs from this directory.

```bash
cd gsea_examples/covid
uv sync                 # builds ./.venv (fetches gibss-mono@1a9ba7a from GitHub)
```

The only provided input is `resources/raw/` (GSE147507 human counts +
`sample_metadata.tsv`), transferred out-of-band (rsync), not committed. The DESeq2
table is generated in-pipeline by `deseq2` (pydeseq2) into `results/de/`; the
`download_geneinfo` rule fetches NCBI `Homo_sapiens.gene_info.gz` into `resources/`
on first run. `resources/` (beyond raw/) and `results/` regenerate.

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
  BLAS / XLA threadpools to that count before importing jax, so jobs don't
  oversubscribe a shared node.
- **JAX recompiles per job.** Each `fit` cell is its own process, so it pays the
  XLA compile once (seconds).
- **Score choice.** Flip `score_kind` in `config.yaml` between `abs_stat`
  (two-sided, default), `signed_stat` (directional), or `neglog_padj`.
