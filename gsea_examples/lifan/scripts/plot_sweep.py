#!/usr/bin/env python
"""Plot the L=1 threshold sweep: method robustness vs selection fraction."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_SETS = [
    "REACTOME_RIBOSOME_ASSOCIATED_QUALITY_CONTROL",
    "REACTOME_AEROBIC_RESPIRATION_AND_RESPIRATORY_ELECTRON_TRANSPORT",
    "REACTOME_REGULATION_OF_EXPRESSION_OF_SLITS_AND_ROBOS",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", default="")
    args = ap.parse_args()

    sw = pl.read_parquet(args.sweep)
    # center each fit by its median; the raw feature_log_evidence carries a huge
    # fit-specific constant, so only the within-fit spread is comparable.
    sw = sw.with_columns(
        (pl.col("logbf") - pl.col("logbf").median().over(["method", "fraction"])).alias("logbf_c")
    )

    spread = (
        sw.group_by("method", "fraction")
        .agg((pl.col("logbf").max() - pl.col("logbf").median()).alias("spread"))
    )
    swept = spread.filter(pl.col("method").is_in(["logistic", "cox"])).sort("fraction")
    ref_full = spread.filter(pl.col("method") == "cox_full")["spread"][0]
    ref_rev = spread.filter(pl.col("method") == "cox_reversed")["spread"][0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for method, color in [("logistic", "tab:blue"), ("cox", "tab:red")]:
        d = swept.filter(pl.col("method") == method).sort("fraction")
        ax.plot(d["fraction"], d["spread"], "o-", color=color, label=method)
    ax.axhline(ref_full, ls="--", color="tab:green", label="cox_full (no censor)")
    ax.axhline(ref_rev, ls="--", color="tab:purple", label="cox_reversed (no censor)")
    ax.set_xscale("log")
    ax.set_xlabel("selection fraction f (top-f most significant genes)")
    ax.set_ylabel("signal spread  (max − median logBF)")
    ax.set_title("Method power vs threshold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for sid, color in zip(REF_SETS, ["tab:orange", "tab:red", "tab:brown"]):
        for method, ls in [("logistic", "-"), ("cox", "--")]:
            d = sw.filter((pl.col("set_id") == sid) & (pl.col("method") == method)).sort("fraction")
            if d.height:
                ax.plot(d["fraction"], d["logbf_c"], ls, color=color, alpha=0.8)
        for method, marker in [("cox_full", "s"), ("cox_reversed", "*")]:
            v = sw.filter((pl.col("set_id") == sid) & (pl.col("method") == method))
            if v.height:
                ax.scatter([0.7], v["logbf_c"], color=color, marker=marker, s=80, zorder=5)
    ax.set_xscale("log")
    ax.set_ylim(-30, 140)  # clip the heavy-censoring strict-cox blowups
    ax.set_xlabel("selection fraction f")
    ax.set_ylabel("centered logBF  (logBF − fit median)")
    ax.set_title("Reference sets: solid=logistic, dashed=cox\nsquare=cox_full, star=cox_reversed")
    ax.grid(alpha=0.3)
    # legend for the reference-set colors
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=3, label=s.replace("REACTOME_", "")[:34])
               for s, c in zip(REF_SETS, ["tab:orange", "tab:red", "tab:brown"])]
    ax.legend(handles=handles, fontsize=7, loc="lower left")

    fig.suptitle(f"L=1 threshold sweep - {args.factor}", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
