"""Slide: logBF detection ROC (c2 pipeline) - can the SER logBF tell an enriched
draw from a null one?

Reads figures/cod_data.json (extracted by cod_extract.py from the
010-c2-cost-of-discretizing pipeline: 200 enrichment + 200 null reps per cell).
Renders the four difficulty-matched cells as a 2x2 grid. Positives are enrichment
replicates, negatives the paired null; score = SER log Bayes factor. Curves:
two-group / linear (z) / linear (|z|) / cox-full / cox-reversed (threshold-free)
plus cox and logistic at their best-AUC threshold. This is the artifact's ROC figure.
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt

from _common import save, MUT

DATA = pathlib.Path(__file__).resolve().parents[1] / "figures" / "cod_data.json"

COL = {"twogroup": "#D55E00", "linear": "#8B4513", "linear_abs": "#CC79A7",
       "coxfull": "#117733", "coxrev": "#E69F00", "cox": "#009E73", "logistic": "#0072B2"}
CELL_LABEL = {"small-loc": "small · loc", "large-loc": "large · loc",
              "small-scale": "small · scale", "large-scale": "large · scale"}
GRID = [["small-loc", "large-loc"], ["small-scale", "large-scale"]]
REF_ORDER = ["twogroup", "linear", "linear_abs", "coxfull", "coxrev"]
NAME = {"twogroup": "two-group", "linear": "linear (z)", "linear_abs": "linear (|z|)",
        "coxfull": "cox-full", "coxrev": "cox-reversed", "cox": "cox", "logistic": "logistic"}


def by_key(curves):
    refs = {c["key"]: c for c in curves if c["tau"] is None}
    best = {}
    for c in curves:
        if c["tau"] is None:
            continue
        if c["key"] not in best or c["auc"] > best[c["key"]]["auc"]:
            best[c["key"]] = c
    return refs, best


def main(out: str) -> None:
    d = json.loads(DATA.read_text())
    rocd = d["roc"]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.7), sharex=True, sharey=True)
    for r in range(2):
        for c in range(2):
            cell = GRID[r][c]
            ax = axes[r][c]
            refs, best = by_key(rocd[cell])
            ax.plot([0, 1], [0, 1], color="#c2ccd6", ls="--", lw=0.9, zorder=0)
            series = [(k, refs[k], True) for k in REF_ORDER if k in refs]
            series += [(k, best[k], False) for k in ("cox", "logistic") if k in best]
            for key, cv, is_ref in series:
                lbl = NAME[key] + (f" (τ={cv['tau']:g})" if not is_ref else "")
                ax.plot(cv["fpr"], cv["tpr"], color=COL[key], lw=1.6,
                        ls="--" if is_ref else "-", label=f"{lbl} {cv['auc']:.2f}")
            ax.set_title(CELL_LABEL[cell], fontsize=10, fontweight="bold")
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.grid(True, color="#e9edf2", lw=0.7); ax.set_axisbelow(True)
            ax.legend(loc="lower right", fontsize=5.6, frameon=True, framealpha=0.9,
                      edgecolor="#dde3ea", handlelength=1.4, borderpad=0.3, labelspacing=0.25)
            if r == 1:
                ax.set_xlabel("false positive rate", fontsize=9)
            if c == 0:
                ax.set_ylabel("true positive rate", fontsize=9)

    npos = rocd[GRID[0][0]][0]["npos"]
    fig.text(0.995, 0.5, f"c2 · {npos} enrich / {npos} null per cell · legend = AUC",
             rotation=90, ha="right", va="center", fontsize=6.5, color=MUT)
    fig.tight_layout()
    save(fig, out, {"source": "cod_data.json", "sc": d.get("sc"),
                    "git_of_data": d.get("git_commit")}, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
