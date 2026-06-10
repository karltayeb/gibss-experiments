#!/usr/bin/env python3
"""Delete simulation parquet files for batches absent from the cached manifest.

The script prints a dry-run summary first. It deletes files only after you type
exactly "yes" at the prompt.

Usage:
    uv run python scripts/purge_stale_simulations.py
    uv run python scripts/purge_stale_simulations.py --manifest results/manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _manifest_batch_hashes(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text())
    batches = manifest.get("batches")
    if not isinstance(batches, dict):
        raise ValueError(f"{manifest_path} does not contain a 'batches' object")
    return set(batches)


def _stale_simulation_paths(results: Path, valid_hashes: set[str]) -> list[Path]:
    by_batch = results / "by_batch"
    if not by_batch.exists():
        raise FileNotFoundError(f"Missing results by_batch directory: {by_batch}")

    stale: list[Path] = []
    for batch_dir in sorted(path for path in by_batch.iterdir() if path.is_dir()):
        if batch_dir.name in valid_hashes:
            continue
        sim_path = batch_dir / "simulations.parquet"
        if sim_path.exists():
            stale.append(sim_path)
    return stale


def purge_stale_simulations(
    *,
    results: Path,
    manifest: Path,
    limit: int | None,
) -> int:
    valid_hashes = _manifest_batch_hashes(manifest)
    stale = _stale_simulation_paths(results, valid_hashes)

    print(f"Manifest: {manifest}")
    print(f"Valid manifest batches: {len(valid_hashes)}")
    print(f"Stale simulation files: {len(stale)}")

    shown = stale if limit is None else stale[:limit]
    for path in shown:
        print(f"[dry] {path}")
    if limit is not None and len(stale) > limit:
        print(f"... {len(stale) - limit} more not shown")

    if not stale:
        print("Nothing to remove")
        return 0

    response = input(f"Type yes to remove {len(stale)} simulation files: ")
    if response != "yes":
        print("Aborted; no files removed")
        return 0

    for path in stale:
        path.unlink()

    print(f"Removed {len(stale)} simulation files")
    return len(stale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=Path("results"), type=Path)
    parser.add_argument(
        "--manifest",
        default=Path("results") / "manifest_cache.json",
        type=Path,
        help="Manifest JSON to compare against; defaults to results/manifest_cache.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum paths to print; use --limit 0 to print none",
    )
    args = parser.parse_args()

    limit = None if args.limit < 0 else args.limit
    purge_stale_simulations(
        results=args.results,
        manifest=args.manifest,
        limit=limit,
    )


if __name__ == "__main__":
    main()
