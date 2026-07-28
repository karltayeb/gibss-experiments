"""Null / non-null colour menu for slide 6 (and 5/6-family schematics).

A menu of candidate (null, non-null) colour pairs, each shown in context on a
mini "scores" panel, so the null/non-null scheme can be chosen. Not a slide -
a decision aid for the next round. Avoids the repo METHOD colours (which are
reserved for cox/twogroup/logistic) to prevent confusion.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _common import save, INK, MUT

# (name, null_color, nonnull_color)
OPTIONS = [
    ("1  teal / amber (current)", "#0B6E75", "#CC7A34"),
    ("2  slate / coral", "#4C72B0", "#DD6E42"),
    ("3  cool grey / crimson", "#7D8CA3", "#B0322B"),
    ("4  navy / gold", "#2B3A67", "#C9A227"),
    ("5  graphite / teal-green", "#4A4E57", "#2A9D8F"),
    ("6  muted blue / rust", "#5B7DB1", "#A6511F"),
]


def mini_scores(ax, null_c, nonnull_c, title):
    rng = np.random.default_rng(7)
    n = 16
    nn = rng.random(n) < 0.35
    w = np.where(nn, rng.normal(3.0, 0.6, n), rng.normal(0.4, 0.3, n)) ** 2
    order = np.argsort(-w)
    ax.bar(np.arange(n), w[order], color=np.where(nn[order], nonnull_c, null_c), width=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_box_aspect(0.5)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")


def main(out: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 5.0))
    for ax, (name, nc, xc) in zip(axes.ravel(), OPTIONS):
        mini_scores(ax, nc, xc, name)
        ax.legend(handles=[Patch(color=xc, label="non-null"), Patch(color=nc, label="null")],
                  loc="upper right", frameon=False, fontsize=7, handlelength=1.0)
    fig.suptitle("null / non-null colour menu  -  pick one for the resolution figures",
                 fontsize=10, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, out, {"options": [{"name": n, "null": a, "nonnull": b} for n, a, b in OPTIONS]},
         __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
