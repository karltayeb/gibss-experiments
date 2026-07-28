"""Slide 6 - one dataset, three resolutions.  [spotlight, local]

Simulate ONE two-group dataset and show the SAME genes at three resolutions,
ranked by the Wald statistic z^2 (consistent with the chi-square / inv-chi-square
misspecification analysis): scores (magnitude + order), ranks (order only), binary
(above/below a threshold). Schematic; three square panels in a row.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _common import save, INK, NULL_C, NONNULL_C
PARAMS = {"n_genes": 22, "pi1": 0.3, "mu": 2.8, "wald_threshold": 4.0, "seed": 5}


def main(out: str) -> None:
    p = PARAMS
    rng = np.random.default_rng(p["seed"])
    n = p["n_genes"]
    non_null = rng.random(n) < p["pi1"]
    z = np.where(non_null, rng.normal(p["mu"], 1.0, n), rng.normal(0.0, 1.0, n))
    wald = z ** 2                                   # Wald statistic

    order = np.argsort(-wald)                        # most significant first
    w_s, nn_s = wald[order], non_null[order]
    col = np.where(nn_s, NONNULL_C, NULL_C)
    x = np.arange(n)
    thr = p["wald_threshold"]

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))

    axes[0].bar(x, w_s, color=col, width=0.86)
    axes[0].axhline(thr, color=INK, ls="--", lw=1)
    axes[0].set_ylabel(r"Wald stat  $z^2$")
    axes[0].set_title("(a) scores", fontsize=10)

    axes[1].bar(x, n - x, color=col, width=1.0)     # descending -> staircase (order only)
    axes[1].set_yticks([])
    axes[1].set_title("(b) ranks", fontsize=10)

    hit = (w_s > thr).astype(float)
    axes[2].bar(x, hit, color=col, width=0.86)
    axes[2].set_yticks([0, 1])
    axes[2].set_title(r"(c) binary  $\mathbb{1}(z^2>\tau)$", fontsize=10)

    for ax in axes:
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_xlabel("genes, ranked by $z^2$", fontsize=8)
    axes[0].legend(
        handles=[Patch(color=NONNULL_C, label="non-null"), Patch(color=NULL_C, label="null")],
        loc="upper right", frameon=False, fontsize=7.5, handlelength=1.0)
    fig.suptitle("same genes, three resolutions: magnitude+order  →  order only  →  a cut",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, out, PARAMS, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
