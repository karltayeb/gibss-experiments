#!/bin/bash
# SLURM array: batch a supercollection's simulate->fit->reduce over disjoint
# batch-hash slices, one array task per slice, snakemake local executor per task.
# No snakemake groups (slow DAG build at ~1e5 jobs), no per-job sbatch overhead;
# batches BOTH reduce_pip and reduce_cs (+ their fits/sims).
#
# Submit (N must match the --array size):
#     sbatch --array=1-100 --export=SC=000_markov_ser,N=100 run/fit_array.sbatch
# Then, once all reductions exist, build plots in a single non-array step:
#     uv run snakemake -j8 results/supercollections/$SC/.done
#
#SBATCH --job-name=gibss-reduce
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${SC:?set SC=<supercollection>, e.g. --export=SC=000_markov_ser,N=100}"
: "${N:?set N=<num chunks> (must equal the --array size)}"

cd "$(dirname "$0")/.."          # -> logistic_susie_experiments (Snakefile lives here)
mkdir -p logs run/targets

T="run/targets/${SC}_${SLURM_ARRAY_TASK_ID}.txt"
uv run python run/emit_targets.py --sc "$SC" --chunk "${SLURM_ARRAY_TASK_ID}" --n "$N" > "$T"

if [ -s "$T" ]; then
    # small per-task DAG (one batch slice) -> fast build, run on the node's cores
    uv run snakemake --executor local -j "${SLURM_CPUS_PER_TASK}" \
        --rerun-incomplete --nocolor $(cat "$T")
else
    echo "chunk ${SLURM_ARRAY_TASK_ID}/${N} for ${SC}: no batches in slice — skip"
fi
