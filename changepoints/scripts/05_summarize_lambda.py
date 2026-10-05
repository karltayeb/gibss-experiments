#!/usr/bin/env python
"""Write the lambda figures and tables (drawn by src/lambda_report.py).
Run from changepoints/:  uv run python scripts/05_summarize_lambda.py
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
import lambda_report as lr


def save(fig, name: str) -> None:
    fig.savefig(lr.FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["lambda"]
    lr.FIG.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_tbl_cols(25)
    pl.Config.set_tbl_rows(60)
    fig, tbl = lr.fig_main(cfg)
    save(fig, "lambda_main.png")
    tbl.write_csv(lr.RESULTS / f"w{cfg['primary_width']}" / "declared_cs_annotated.csv")
    print(tbl)
    scale = lr.table_scale(cfg)
    scale.write_csv(lr.RESULTS / "scale_table.csv")
    print(scale)
    save(lr.fig_scale(cfg), "lambda_scale.png")
    cal = lr.RESULTS / "calibration" / "summary.csv"
    if cal.exists():
        print(pl.read_csv(cal))


if __name__ == "__main__":
    main()
