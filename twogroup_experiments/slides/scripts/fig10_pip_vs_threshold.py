"""Slide: the cost of discretizing - mean causal PIP vs threshold (c2 pipeline).

Reads figures/cod_data.json (extracted from the results pipeline by cod_extract.py:
the 010-c2-cost-of-discretizing supercollection, 200 reps/cell). Renders the four
difficulty-matched cells as a 2x2 grid - small/large gene set x loc/scale signal.

Per panel: cox and logistic swept over the threshold tau (keep |z|>tau), plus five
threshold-free reference lines (two-group oracle, linear on z, linear on |z|,
cox-full, cox-reversed). This is the artifact's PIP figure, deck-styled.
"""
import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _common import save, MUT

DATA = pathlib.Path(__file__).resolve().parents[1] / "figures" / "cod_data.json"

# Method palette - matches the artifact (Okabe-Ito based).
COL = {"twogroup": "#D55E00", "linear": "#8B4513", "linear_abs": "#CC79A7",
       "coxfull": "#117733", "coxrev": "#E69F00", "cox": "#009E73", "logistic": "#0072B2"}
CELL_LABEL = {"small-loc": "small · loc", "large-loc": "large · loc",
              "small-scale": "small · scale", "large-scale": "large · scale"}
GRID = [["small-loc", "large-loc"], ["small-scale", "large-scale"]]
# threshold-free references: (label, key, linestyle)
REFS = [("two-group", "twogroup", "--"), ("linear (z)", "linear", ":"),
        ("linear (|z|)", "linear_abs", (0, (1, 1))), ("cox-full", "coxfull", (0, (5, 1))),
        ("cox-reversed", "coxrev", "-.")]


def main(out: str) -> None:
    d = json.loads(DATA.read_text())
    pipd = d["pip"]
    taus = d["thresholds"]
    n = pipd[GRID[0][0]]["n"]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 4.7), sharex=True, sharey=True)
    for r in range(2):
        for c in range(2):
            cell = GRID[r][c]
            ax = axes[r][c]
            cell_d = pipd[cell]
            for key, ls in [(k, l) for _, k, l in REFS]:
                if key in cell_d["refs"]:
                    ax.axhline(cell_d["refs"][key], color=COL[key], ls=ls, lw=1.6)
            for kind, mk in [("logistic", "o"), ("cox", "s")]:
                ts = sorted(float(t) for t in cell_d[kind])
                ys = [cell_d[kind][f"{t:.1f}"] for t in ts]
                if ts:
                    ax.plot(ts, ys, "-" + mk, color=COL[kind], lw=1.8, ms=5)
            ax.set_title(CELL_LABEL[cell], fontsize=10, fontweight="bold")
            ax.set_xticks(taus)
            ax.set_ylim(-0.02, 0.72)
            ax.grid(True, color="#e9edf2", lw=0.7); ax.set_axisbelow(True)
            if r == 1:
                ax.set_xlabel(r"threshold $|z|>\tau$", fontsize=9)
            if c == 0:
                ax.set_ylabel("mean causal PIP", fontsize=9)

    handles = [
        Line2D([0], [0], color=COL["cox"], lw=1.8, marker="s", label=r"cox (censored ranks @ $\tau$)"),
        Line2D([0], [0], color=COL["logistic"], lw=1.8, marker="o", label=r"logistic (binary @ $\tau$)"),
        Line2D([0], [0], color=COL["twogroup"], lw=1.6, ls="--", label="two-group oracle"),
        Line2D([0], [0], color=COL["linear"], lw=1.6, ls=":", label="linear on z"),
        Line2D([0], [0], color=COL["linear_abs"], lw=1.6, ls=(0, (1, 1)), label="linear on |z|"),
        Line2D([0], [0], color=COL["coxfull"], lw=1.6, ls=(0, (5, 1)), label="cox-full (all ranks)"),
        Line2D([0], [0], color=COL["coxrev"], lw=1.6, ls="-.", label="cox-reversed (all ranks)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.995, 0.5, f"c2 · {n} reps/cell · refs are τ-free",
             rotation=90, ha="right", va="center", fontsize=6.5, color=MUT)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, out, {"source": "cod_data.json", "sc": d.get("sc"),
                    "git_of_data": d.get("git_commit")}, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
