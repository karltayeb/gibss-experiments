# run/ — cluster batching helpers

Batch the pipeline on SLURM **one supercollection at a time**, sliced by
`batch_hash`, without snakemake job groups.

Why not `--groups` / `--batch`:
- The full DAG is ~1e5 jobs. Snakemake **groups** add a whole-DAG partition +
  connected-components pass that is slow to build at that scale.
- `--batch rule=i/N` batches only **one** rule — batching `fit` still pulls every
  `reduce_pip`/`reduce_cs` job into the DAG.

Instead: the `reduce_chunk` rule exposes a native target

    results/supercollections/<SC>/.done_<i>_of_<N>

whose input function returns only chunk `i`'s reductions (both pip + cs),
partitioned by whole `batch_hash`. Snakemake then builds a *small* DAG (that slice
+ its fits/sims) — fast build, no groups. Each array task runs one chunk with the
local executor on the node's cores.

## Usage

```bash
cd .../logistic_susie_experiments
mkdir -p logs                       # --output=logs/... must be writable at submit
# 1. reductions, as a SLURM array (use --array=1-N; N derived from it).
#    account=pi-mstephens / partition=caslake baked into the sbatch.
sbatch --array=1-100 --export=SC=000_markov_ser run/fit_array.sbatch

# 2. plots, once all reductions exist (single, non-array step)
uv run snakemake -j8 results/supercollections/000_markov_ser/.done
```

Manually build one chunk (no SLURM):
```bash
uv run snakemake --executor local -j8 \
  results/supercollections/000_markov_ser/.done_1_of_100
```

**Alternative — profile submission, chunk-by-chunk (sequential driver).** Instead
of a SLURM array + `--executor local`, drive the chunks through the snakemake
`profile/` (which submits group jobs to SLURM). `run_chunks.sh` loops
`.done_i_of_N` for i=1..N; each chunk builds a small DAG and submits its group
jobs, snakemake waits, then the next chunk. Run under tmux/screen (long-lived):
```bash
tmux new -s gibss 'bash run/run_chunks.sh 000_markov_ser 1000 | tee run/chunks.log'
# resume a range:  bash run/run_chunks.sh 000_markov_ser 1000 250 500
```
Needs the untracked `profile/` (SLURM submit-cmd with `--cpus-per-task={threads}`,
`mem_mb: 4000`). Bound: a group needs `threads ≤ 48` and `threads×mem_mb ≤ 184 GB`
(threads ≈ group-components × fits-per-batch) — keep `group-components` modest.

## Sizing N
Aim each task ~15–60 min. Cheap SCs (score/global_taylor ~1–2s/fit) need small N;
exact-quadrature / profile+GH5 cells and the n=50k tier want large N. If some
batches are far heavier (50k), weight the split rather than round-robin.

## Notes
- Partition is by whole batch, so a batch's pip, cs, and shared
  `simulations.parquet` co-locate in one chunk (no recompute / racing).
- `N` == chunk count == array size; the sbatch derives it from
  `SLURM_ARRAY_TASK_COUNT` (or the array `MAX-MIN+1`). Use a 1-based array.
- Chunks with no batches (N > #batches) just touch the sentinel — harmless.
- Scope every submission to ONE `SC`; never the global default target.
