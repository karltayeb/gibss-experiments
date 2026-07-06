#!/bin/bash
# Drive a supercollection chunk-by-chunk via the snakemake profile (SLURM submit),
# SEQUENTIALLY. Each chunk builds a small DAG (`.done_i_of_N` -> only that batch
# slice, fast build) and submits its group jobs through profile/; snakemake blocks
# until the chunk's jobs finish, then the next chunk starts. Chunks are serial;
# jobs WITHIN a chunk run in parallel (profile jobs: 100 + cpus-per-task).
#
# Run on the login node under tmux/screen/nohup (it's long-lived):
#   tmux new -s gibss 'bash run/run_chunks.sh 000_markov_ser 1000 | tee run/chunks_000_markov_ser.log'
# Resume a range:
#   bash run/run_chunks.sh 000_markov_ser 1000 250 500
set -uo pipefail

SC=${1:?usage: run_chunks.sh <supercollection> <N> [start] [end]}
N=${2:?usage: run_chunks.sh <supercollection> <N> [start] [end]}
START=${3:-1}
END=${4:-$N}

cd "$(dirname "$0")/.."          # -> logistic_susie_experiments (Snakefile here)

# Disable the XLA compile cache for the SUBMITTED jobs too (sbatch --export=ALL
# propagates this env to the compute nodes) -- avoids the cross-machine AOT SIGILL.
export GIBSS_NO_JAX_CACHE=1
unset JAX_COMPILATION_CACHE_DIR 2>/dev/null || true
export JAX_ENABLE_COMPILATION_CACHE=false

fails=()
for i in $(seq "$START" "$END"); do
    tgt="results/supercollections/${SC}/.done_${i}_of_${N}"
    echo "==================================================================="
    echo "[$(date '+%F %T')] chunk ${i}/${N}  ->  ${tgt}"
    uv run snakemake --unlock >/dev/null 2>&1 || true   # clear a stale lock from a killed chunk
    if uv run snakemake "$tgt" --profile profile/; then
        echo "[$(date '+%F %T')] chunk ${i}/${N} OK"
    else
        rc=$?
        echo "[$(date '+%F %T')] chunk ${i}/${N} FAILED (exit ${rc})"
        fails+=("$i")
    fi
done

echo "==================================================================="
if [ "${#fails[@]}" -eq 0 ]; then
    echo "[$(date '+%F %T')] ALL chunks ${START}..${END} of ${SC} done, 0 failures."
else
    echo "[$(date '+%F %T')] done with ${#fails[@]} failed chunk(s): ${fails[*]}"
    echo "re-run e.g.:  bash run/run_chunks.sh ${SC} ${N} <i> <i>"
    exit 1
fi
