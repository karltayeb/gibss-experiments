#!/bin/bash
# SLURM array: batch a supercollection's simulate->fit->reduce over disjoint
# batch-hash slices, one array task per slice, snakemake local executor per task.
# No snakemake groups (slow DAG build at ~1e5 jobs), no per-job sbatch overhead;
# batches BOTH reduce_pip and reduce_cs (+ their fits/sims).
#
# N (the chunk count) == the array size. It is derived from the array itself, so
# you only pass SC. Submit:
#     sbatch --array=1-100 --export=SC=000_markov_ser run/fit_array.sbatch
# (Override only if you know better: --export=SC=...,N=100 — must match --array.)
# Then, once all reductions exist, build plots in a single non-array step:
#     uv run snakemake -j8 results/supercollections/$SC/.done
#
#SBATCH --job-name=gibss-reduce
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail

: "${SC:?set SC=<supercollection>, e.g. --export=SC=000_markov_ser}"
# N = chunk count = array size. Prefer SLURM_ARRAY_TASK_COUNT; else derive from
# the array range MAX-MIN+1; else require an explicit N. (Assumes a contiguous
# array like --array=1-100.)
if [ -z "${N:-}" ]; then
    if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
        N="${SLURM_ARRAY_TASK_COUNT}"
    elif [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ -n "${SLURM_ARRAY_TASK_MIN:-}" ]; then
        N=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
    else
        echo "cannot derive N (array size); pass --export=SC=...,N=<size>" >&2; exit 1
    fi
fi

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
