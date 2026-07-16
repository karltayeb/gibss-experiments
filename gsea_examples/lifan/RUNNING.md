# Running the Lifan GSEA pipeline on Midway3 (SLURM)

The workflow (`convert -> prep -> fit -> gather`) is ~2400 jobs for the full
matrix (140 factors x 15 methods + prep/gather). `fit` cells are JAX-heavy; the
rest is light. Per-rule `mem_mb` / `runtime` (minutes) / `threads` are declared
in the `Snakefile`; the SLURM submit line lives in `profile/config.yaml`.

## Isolated environment

This example is a **self-contained uv project** (its own `pyproject.toml` +
`.venv`), deliberately kept apart from the repo-root env: it pins the gibss-mono
front-door APIs (`fit_glm_susie` / `fit_cox_susie`) at rev `1a9ba7a`. Everything
below runs from this directory.

## One-time setup on the cluster

```bash
cd gsea_examples/lifan
uv sync                 # builds ./.venv (fetches gibss-mono@1a9ba7a from GitHub)
which Rscript           # the `convert` rule shells out to R; Midway3 has R-4.3.1 on PATH
```

`factor_sumstats.RDS` is the only external input and is **not** version
controlled (74 MB). Copy it into this directory before running (e.g.
`rsync factor_sumstats.RDS midway3:/project/mstephens/ktayeb/gibss-experiments/gsea_examples/lifan/`).
`results/` and `resources/` regenerate.

## Launch

Snakemake stays running on the login node and submits/polls SLURM jobs, so run
it under `tmux`/`screen`:

```bash
# smoke test the SLURM path on 3 factors first
uv run snakemake --profile profile --config factors="[1,2,3]"

# full run (all 140 factors)
tmux new -s lifan
uv run snakemake --profile profile
```

`profile/config.yaml` submits via the cluster-generic plugin
(`--account=pi-mstephens --partition=caslake --qos=caslake`), caps concurrency at
`jobs: 100`, retries a failed job once, and uses `keep-going`. Job logs land in
`./logs/%j_<rule>.out`.

> **Verify the allocation** before a big run and edit the `--account` /
> `--partition` in `profile/config.yaml` if yours differs.

## Notes

- **Thread capping.** `fit` requests `threads: 2` and `fit_cell.py` pins the
  BLAS / XLA Eigen threadpools to that count before importing jax, so jobs don't
  oversubscribe a shared node.
- **JAX recompiles per job.** Each `fit` cell is its own process, so it pays the
  XLA compile once (seconds); `runtime: 120` on `fit` absorbs the slow `coarse` /
  `L=auto` cells with margin.
- **Dry run** any time to preview the plan without submitting:
  `uv run snakemake --profile profile -n`.
