#!/usr/bin/env python3
"""
Diff manifest against results/by_batch/.

Symbols:
  x  all batches complete
  /  partial (some batches have fits)
  o  none complete
  -  not in manifest (orphaned)

Usage:
    uv run python manifest_status.py
    uv run python manifest_status.py --missing   # list missing (sim, method) pairs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_ROOT = Path("results")
BY_BATCH = RESULTS_ROOT / "by_batch"
MANIFEST_PATH = RESULTS_ROOT / "manifest.json"
HASH_KEY = "__spec_hash__"

manifest = json.loads(MANIFEST_PATH.read_text())
batches: dict = manifest["batches"]
method_specs: dict = manifest["method_specs"]

# ── Disk scan ──────────────────────────────────────────────────────────────────
disk_fits: set[tuple[str, str]] = set()  # (batch_hash, method_hash)
disk_batch_hashes: set[str] = set()

if BY_BATCH.exists():
    for batch_dir in sorted(BY_BATCH.iterdir()):
        if not batch_dir.is_dir():
            continue
        bh = batch_dir.name
        disk_batch_hashes.add(bh)
        fits_dir = batch_dir / "fits"
        if fits_dir.exists():
            for method_dir in fits_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                if (method_dir / "fits.parquet").exists():
                    disk_fits.add((bh, method_dir.name))

# ── Expected pairs ─────────────────────────────────────────────────────────────
manifest_batch_hashes = set(batches)
manifest_method_hashes = set(method_specs)
expected: set[tuple[str, str]] = {
    (bh, mh) for bh in manifest_batch_hashes for mh in manifest_method_hashes
}

complete = disk_fits & expected
orphaned = disk_fits - expected

print(f"Expected : {len(expected)}")
print(f"Complete : {len(complete)} ({100 * len(complete) / max(len(expected), 1):.1f}%)")
print(f"Missing  : {len(expected) - len(complete)}")
print(f"Orphaned : {len(orphaned)}  (on disk but not in manifest)")
print()

# ── Sim → batch list ───────────────────────────────────────────────────────────
sim_to_batches: dict[str, list[str]] = {}
sim_to_name: dict[str, str] = {}

for bh, bnode in batches.items():
    sim_node = bnode["simulation_spec"]
    sh = sim_node[HASH_KEY]
    sim_to_batches.setdefault(sh, []).append(bh)
    sim_to_name.setdefault(sh, sim_node["fields"]["name"])

# ── Method label ───────────────────────────────────────────────────────────────
def _method_label(mh: str) -> str:
    fields = method_specs[mh].get("fields", {})
    name = str(fields.get("name", mh[:8]))
    kwargs = dict(fields.get("kwargs", {}))
    t = kwargs.get("threshold")
    return f"{name}@{t:g}" if t is not None else name


# ── Cell symbol ───────────────────────────────────────────────────────────────
def _cell(sim_hash: str, method_hash: str) -> str:
    bhs = sim_to_batches[sim_hash]
    n = sum(1 for bh in bhs if (bh, method_hash) in disk_fits)
    if n == len(bhs):
        return "x"
    if n == 0:
        return "o"
    return "/"


# ── Sort ───────────────────────────────────────────────────────────────────────
sim_hashes = sorted(sim_to_name, key=lambda h: sim_to_name[h])
method_hashes = sorted(manifest_method_hashes, key=_method_label)
method_labels = [_method_label(mh) for mh in method_hashes]

SIM_W = max(len(sim_to_name[sh]) for sh in sim_hashes)
N = len(method_hashes)
BLOCK = 10  # space every 10 columns

# ── Column index header ────────────────────────────────────────────────────────
def _col_header_row(digit_fn) -> str:
    parts = []
    for i in range(N):
        if i and i % BLOCK == 0:
            parts.append(" ")
        parts.append(digit_fn(i))
    return " " * (SIM_W + 2) + "".join(parts)

print(_col_header_row(lambda i: str(i // 10) if i % 10 == 0 else " "))
print(_col_header_row(lambda i: str(i % 10)))
print(" " * (SIM_W + 2) + "-" * (N + N // BLOCK))

# ── Data rows ─────────────────────────────────────────────────────────────────
for sh in sim_hashes:
    cells = []
    for i, mh in enumerate(method_hashes):
        if i and i % BLOCK == 0:
            cells.append(" ")
        cells.append(_cell(sh, mh))
    print(f"{sim_to_name[sh]:<{SIM_W}}  {''.join(cells)}")

# ── Legend ────────────────────────────────────────────────────────────────────
print()
print("Method legend:")
for i, label in enumerate(method_labels):
    print(f"  {i:3}: {label}")

# ── Optional: list missing pairs ──────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--missing", action="store_true")
args, _ = parser.parse_known_args()

if args.missing:
    print()
    print("Missing (sim_name, method_label):")
    missing = expected - complete
    for bh, mh in sorted(missing, key=lambda p: (batches[p[0]]["simulation_spec"]["fields"]["name"], _method_label(p[1]))):
        sim_name = batches[bh]["simulation_spec"]["fields"]["name"]
        print(f"  {sim_name}  {_method_label(mh)}  batch={bh[:8]}")
