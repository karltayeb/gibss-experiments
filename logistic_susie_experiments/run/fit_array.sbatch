#!/bin/bash
# SLURM array: batch a supercollection's simulate->fit->reduce over disjoint
# batch-hash slices, one array task per slice, snakemake local executor per task.
# No snakemake groups (slow DAG build at ~1e5 jobs), no per-job sbatch overhead;
# batches BOTH reduce_pip and reduce_cs (+ their fits/sims).
#
# N (the chunk count) == the array size. It is derived from the array itself, so
# you only pass SC. Submit (midway3 / caslake), from logistic_susie_experiments/
# (mkdir -p logs once so --output can be written):
#     mkdir -p logs
#     sbatch --array=1-100 --export=SC=000_markov_ser run/fit_array.sbatch
# (account=pi-mstephens / partition=caslake baked in below; override on the CLI if
#  needed. Override chunk count with --export=SC=...,N=100 — must match --array.)
# Then, once all reductions exist, build plots in a single non-array step:
#     uv run snakemake -j8 results/supercollections/$SC/.done
#
#SBATCH --job-name=gibss-reduce
#SBATCH --account=pi-mstephens
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
mkdir -p logs

# JAX/XLA persistent compile cache: keep it NODE-LOCAL. gibss auto-enables a cache
# at ~/.cache/gibss/jax (shared FS); loading an AOT-compiled kernel built on a node
# with different CPU features (e.g. avx512vnni) on another node -> SIGILL. A
# node-local dir avoids the cross-machine mismatch (still reused across tasks on the
# same node). Set GIBSS_NO_JAX_CACHE=1 instead to disable entirely.
export GIBSS_JAX_CACHE_DIR="/tmp/gibss_jax_${USER}"

# Native chunk target: reduce_chunk's input function returns only this slice's
# reductions -> small per-task DAG (fast build, no groups), run on the node's cores.
# Chunks with no batches just touch the sentinel (no work).
uv run snakemake --executor local -j "${SLURM_CPUS_PER_TASK}" --rerun-incomplete --nocolor \
    "results/supercollections/${SC}/.done_${SLURM_ARRAY_TASK_ID}_of_${N}"
