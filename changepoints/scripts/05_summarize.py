#!/usr/bin/env python
"""Write the coal figures and comparison tables (drawn by src/coal_report.py).
Run from changepoints/:  uv run python scripts/05_summarize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import coal_report as cr


def save(fig, name: str) -> None:
    fig.savefig(cr.FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["coal"]
    cr.FIG.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    save(cr.fig_main(cfg), "coal_year_main.png")
    fig, single = cr.fig_single(cfg)
    save(fig, "coal_year_L1_vs_exact_single.png")
    print("L=1 vs exact single changepoint:", single)
    with open(cr.RESULTS / "single_changepoint_comparison.yaml", "w") as fh:
        yaml.safe_dump({k: (v if not isinstance(v, list) else [int(i) for i in v])
                        for k, v in single.items()}, fh)
    fig, kt = cr.fig_k_posterior(cfg)
    save(fig, "coal_year_k_posterior.png")
    print(kt)
    fig, rt = cr.fig_resolution(cfg)
    save(fig, "coal_resolution.png")
    rt.write_csv(cr.RESULTS / "resolution_table.csv")
    print(rt)
    cal = cr.RESULTS / "year/calibration"
    if (cal / "design_a_components.csv").exists():
        save(cr.fig_calibration_a(), "calibration_design_a.png")
    if (cal / "design_b_summary.csv").exists():
        save(cr.fig_calibration_b(), "calibration_design_b.png")


if __name__ == "__main__":
    main()
