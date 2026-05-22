#!/usr/bin/env python3
"""Delete fit outputs for methods matching a name prefix.

Forces Snakemake to rerun fit_twogroup_experiment_batch_method for those methods.

Usage:
    uv run python scripts/invalidate_fits.py twogroup
    uv run python scripts/invalidate_fits.py twogroup --dry-run
    uv run python scripts/invalidate_fits.py --results /path/to/results twogroup
"""
import argparse
import json
import shutil
from pathlib import Path


def invalidate_fits(results: Path, prefix: str, dry_run: bool) -> None:
    manifest = json.loads((results / "manifest.json").read_text())
    hashes = [
        (h, v["fields"]["name"])
        for h, v in manifest["method_specs"].items()
        if v["fields"]["name"].startswith(prefix)
    ]

    if not hashes:
        print(f"No methods match prefix {prefix!r}")
        return

    print(f"Methods matched ({len(hashes)}):")
    for _, name in sorted(hashes, key=lambda x: x[1]):
        print(f"  {name}")

    count = 0
    for h, _ in hashes:
        for fit_dir in (results / "by_batch").glob(f"*/fits/{h}"):
            print(f"{'[dry]' if dry_run else 'rm'} {fit_dir}")
            if not dry_run:
                shutil.rmtree(fit_dir)
            count += 1

    print(f"\n{'Would remove' if dry_run else 'Removed'} {count} fit directories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Method name prefix (e.g. 'twogroup')")
    parser.add_argument("--results", default="results", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    invalidate_fits(args.results, args.prefix, args.dry_run)
