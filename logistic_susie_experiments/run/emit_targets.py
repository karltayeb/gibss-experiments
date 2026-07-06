"""Emit reduction-parquet targets for ONE supercollection, sliced into chunk
`--chunk` of `--n` by batch_hash.

Feeds a SLURM array (see fit_array.sbatch): each array task runs
`snakemake --executor local` on its slice, doing simulate -> fit ->
reduce_pip + reduce_cs for a disjoint set of batches. This avoids snakemake job
GROUPS (whose whole-DAG partition + connected-components pass is slow at ~1e5
jobs) and the per-job sbatch overhead, while still batching BOTH reductions
(and the fits/sims they need).

Partitioning is by whole batch_hash, so a batch's pip and cs — and its shared
simulations.parquet — always land in the SAME chunk (no recomputation / racing
across tasks).

Usage:
    uv run python run/emit_targets.py --sc 000_markov_ser --chunk $i --n $N
Emits the target parquet paths (one per line) for chunk i on stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# put logistic_susie_experiments on sys.path (this file lives in run/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import loader  # noqa: E402


def sc_reduction_targets(config: dict, sc_name: str) -> list[str]:
    """Every reduction parquet for one supercollection (both reductions,
    method-filtered) — the per-SC analogue of loader.all_reduction_targets."""
    library = config["library"]
    targets: list[str] = []
    seen: set[str] = set()
    for reduction in library["reductions"]:
        mfilter = loader.reduction_method_filter(library, reduction)
        pred = loader.resolve_predicate(mfilter) if mfilter else None
        for coll in loader.collection_method_pairs(config, sc_name).values():
            for bh, mh, _name, mcoord, _sim in coll["pairs"]:
                if pred is not None and not pred(mcoord):
                    continue
                path = loader.reduction_output(bh, mh, reduction)
                if path not in seen:
                    seen.add(path)
                    targets.append(path)
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True, help="supercollection name")
    ap.add_argument("--chunk", type=int, required=True, help="1-based chunk index (SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--n", type=int, required=True, help="number of chunks (array size)")
    a = ap.parse_args()

    cfg = loader.load_config()
    if a.sc not in cfg["supercollections"]:
        sys.exit(f"unknown supercollection {a.sc!r}; have {sorted(cfg['supercollections'])}")
    if not (1 <= a.chunk <= a.n):
        sys.exit(f"--chunk must be in 1..{a.n}; got {a.chunk}")

    # group targets by batch_hash, then assign whole batches round-robin to chunks
    by_batch: dict[str, list[str]] = {}
    for t in sc_reduction_targets(cfg, a.sc):
        bh = t.split("/by_batch/")[1].split("/")[0]
        by_batch.setdefault(bh, []).append(t)

    for k, bh in enumerate(sorted(by_batch)):
        if k % a.n == (a.chunk - 1):
            for t in by_batch[bh]:
                print(t)


if __name__ == "__main__":
    main()
