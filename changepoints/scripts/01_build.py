#!/usr/bin/env python
"""Build the coal-mining count series and record the cross-checks.

Writes data/processed/coal/{year,month}.csv (position, label, count) and prints the
comparisons that go into docs/data_notes.md.
Run from changepoints/:  uv run python scripts/01_build.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from coal import (
    CARLIN_1992_TRANSCRIBED,
    PROC,
    load_dates,
    monthly_counts,
    yearly_counts,
)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["coal"]
    y0, y1 = cfg["year_start"], cfg["year_end"]
    boot = load_dates("boot")
    pymc = load_dates("pymc")
    print(f"boot::coal: {len(boot)} events, {boot.min():.3f} .. {boot.max():.3f}")
    print(f"pymc copy : {len(pymc)} events, {pymc.min():.3f} .. {pymc.max():.3f}")
    print(f"max |boot - pymc| = {np.max(np.abs(np.sort(boot) - np.sort(pymc))):.2e}")

    years, counts = yearly_counts(boot, y0, y1)
    print(f"yearly series: n={len(years)} total={counts.sum()}")
    diff = np.flatnonzero(counts != CARLIN_1992_TRANSCRIBED)
    print(f"vs Carlin et al. 1992 series (transcribed from memory; total {CARLIN_1992_TRANSCRIBED.sum()}): "
          f"{len(diff)} differing years")
    for i in diff:
        print(f"  {years[i]}: boot={counts[i]} carlin={CARLIN_1992_TRANSCRIBED[i]}")

    PROC.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"position": np.arange(len(years)), "label": years, "count": counts}) \
        .write_csv(PROC / "year.csv")
    labels, mcounts = monthly_counts(boot, y0, y1)
    assert mcounts.sum() == len(boot)
    pl.DataFrame({"position": np.arange(len(labels)), "label": labels, "count": mcounts}) \
        .write_csv(PROC / "month.csv")
    print(f"monthly series: n={len(labels)} total={mcounts.sum()} max={mcounts.max()}")
    # decade summary for the notes
    dec = pl.DataFrame({"decade": (years // 10) * 10, "count": counts}) \
        .group_by("decade").agg(pl.col("count").sum(), pl.len().alias("years")).sort("decade")
    print(dec)


if __name__ == "__main__":
    main()
