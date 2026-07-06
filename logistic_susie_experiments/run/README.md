# run/ — cluster batching helpers

Batch the pipeline on SLURM **one supercollection at a time**, sliced by
`batch_hash`, without snakemake job groups.

Why not `--groups` / `--batch`:
- The full DAG is ~1e5 jobs. Snakemake **groups** add a whole-DAG partition +
  connected-components pass that is slow to build at that scale.
- `--batch rule=i/N` batches only **one** rule — batching `fit` still pulls every
  `reduce_pip`/`reduce_cs` job into the DAG.

Instead: partition by whole batch and target the reductions directly. Each array
task builds a *small* DAG (its batch slice) and runs simulate -> fit ->
reduce_pip + reduce_cs on the node's cores.

## Usage

```bash
# 1. reductions, as a SLURM array (N must equal the array size)
sbatch --array=1-100 --export=SC=000_markov_ser,N=100 run/fit_array.sbatch

# 2. plots, once all reductions exist (single, non-array step)
uv run snakemake -j8 results/supercollections/000_markov_ser/.done
```

`emit_targets.py --sc <SC> --chunk i --n N` prints the reduction parquet targets
for chunk `i` (batches where `index % N == i-1`; both reductions, method-filtered).

## Sizing N
Aim each task ~15–60 min. Cheap SCs (score/global_taylor ~1–2s/fit) need small N;
the exact-quadrature / profile+GH5 cells and the n=50k tier want large N. If some
batches are far heavier (50k), weight the split rather than round-robin.

## Notes
- Partition is by whole batch, so a batch's pip, cs, and shared
  `simulations.parquet` always co-locate in one chunk (no recompute / racing).
- Local executor per task (`--executor local -j $CPUS`) — no per-fit sbatch.
- Scope every submission to ONE `SC`; never the global default target.
