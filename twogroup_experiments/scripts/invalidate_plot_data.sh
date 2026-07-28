#!/usr/bin/env bash
# Delete all plot data parquet files and sentinel out.txt files so Snakemake
# rebuilds the full collection + supercollection pipeline on next run.
set -euo pipefail

RESULTS="$(dirname "$0")/../results"

echo "Deleting collection plot data..."
find "$RESULTS/collections" -name "*.parquet" -delete
find "$RESULTS/collections" -name "out.txt" -delete

echo "Deleting supercollection sentinels..."
find "$RESULTS/supercollections" -name "out.txt" -delete

echo "Done."
