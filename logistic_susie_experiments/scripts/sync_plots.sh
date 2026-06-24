#!/usr/bin/env bash
# Sync plots from Midway to local results/plots/.
# Remote symlinks are dereferenced (-L) so local gets real PDF copies.
#
# Usage:
#   ./scripts/sync_plots.sh [user@midway3.rcc.uchicago.edu] [remote_results_dir]
#
# Defaults:
#   HOST  = midway3.rcc.uchicago.edu
#   RDIR  = ~/research/gibss-experiments/logistic_susie_experiments/results

set -euo pipefail

HOST="${1:-midway3.rcc.uchicago.edu}"
RDIR="${2:-~/research/gibss-experiments/logistic_susie_experiments/results}"
LDIR="results/plots"

mkdir -p "$LDIR"

rsync -avL --progress \
    "${HOST}:${RDIR}/plots/" \
    "${LDIR}/"
