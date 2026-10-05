#!/usr/bin/env python
"""Bin the lambda phage genome into GC counts and write the CDS table.

Writes data/processed/lambda/bins_{width}.csv (position = bin index, start bp, k = GC
count, m = bases) and data/processed/lambda/cds.csv.
Run from changepoints/:  uv run python scripts/01_build_lambda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import lam


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["lambda"]
    seq = lam.load_sequence()
    bits = lam.gc_bits(seq)
    print(f"{cfg['accession']}: {len(seq)} bp, GC fraction {bits.mean():.4f}, "
          f"A/C/G/T = {[seq.count(b) for b in 'ACGT']}")
    # lag autocorrelation of the GC indicator (dependence that a binomial ignores)
    c = bits - bits.mean()
    ac = [float(np.dot(c[:-l], c[l:]) / np.dot(c, c)) for l in (1, 2, 3, 6)]
    print("GC indicator autocorrelation at lags 1,2,3,6:", np.round(ac, 4))
    lam.PROC.mkdir(parents=True, exist_ok=True)
    for w in cfg["bin_widths"]:
        start, k, m = lam.bin_counts(bits, w)
        pl.DataFrame({"position": np.arange(len(k)), "start": start, "k": k, "m": m}) \
            .write_csv(lam.PROC / f"bins_{w}.csv")
        print(f"width {w:5d}: {len(k)} bins, last bin {m[-1]} bp, "
              f"GC range {k[:-1].min() / w:.2f}-{k[:-1].max() / w:.2f}")
    cds = lam.cds_table()
    cds.write_csv(lam.PROC / "cds.csv")
    print(f"CDS features: {cds.height}")


if __name__ == "__main__":
    main()
