"""Motivation - two jobs for enrichment, and where the 2x2 fails.  [spotlight, local]

Enrichment analysis is used for two very different things:
  A. characterize a STRONG result - a few large-effect genes in a small set;
  B. detect a small COORDINATED shift - many genes each nudged slightly.

The threshold-based 2x2 / hit-list handles A (the big genes cross the threshold)
but is blind to B: no single gene crosses tau, so the hit count looks null even
though the whole set is shifted. That is exactly the coordinated signal we wanted
to aggregate - and the motivation for modeling richer resolutions.
"""
import argparse

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from _common import save, INK, MUT, NULL_C, NONNULL_C

PARAMS = {"tau": 2.0, "set_size": 50,
          "A": {"pi_big": 0.12, "mu_big": 4.5},   # sparse-strong
          "B": {"shift": 0.45}}                     # coordinated-weak


def hits_row(ax, dens, mean_shift, tau, n, title, verdict, verdict_color,
             verdict_xy, show_legend=False):
    from scipy.integrate import quad
    z = np.linspace(-3.5, 7.5, 600)
    ax.plot(z, norm.pdf(z, 0, 1), color=NULL_C, lw=1.6, ls="--", label="genome-wide null")
    ax.plot(z, dens(z), color=NONNULL_C, lw=2.4, label="gene set")
    ax.fill_between(z, 0, dens(z), color=NONNULL_C, alpha=0.10)
    over = z >= tau
    ax.fill_between(z[over], 0, dens(z)[over], color=NONNULL_C, alpha=0.35)
    ax.axvline(tau, color=INK, ls=":", lw=1.2)
    ax.set_ylim(0, 0.46)
    ax.text(tau + 0.1, 0.43, r"$\tau$", color=INK, fontsize=10)

    exp_null = n * norm.sf(tau)
    hits = n * quad(lambda x: dens(np.array([x]))[0], tau, 12)[0]
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_yticks([])
    ax.set_ylabel("density")
    # stats box upper-left (empty in both panels)
    ax.text(0.02, 0.94,
            f"hits > $\\tau$: {hits:.0f}  (null exp. {exp_null:.1f})\n"
            f"mean shift: {mean_shift:+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK,
            family="monospace")
    ax.text(*verdict_xy, verdict, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=verdict_color, fontweight="bold")
    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=8, bbox_to_anchor=(1.0, 0.72))


def main(out):
    p = PARAMS
    tau, n = p["tau"], p["set_size"]

    def densA(z):
        return (1 - p["A"]["pi_big"]) * norm.pdf(z, 0, 1) + p["A"]["pi_big"] * norm.pdf(z, p["A"]["mu_big"], 1)

    def densB(z):
        return norm.pdf(z, p["B"]["shift"], 1)

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.3), sharex=True)
    hits_row(axes[0], densA, p["A"]["pi_big"] * p["A"]["mu_big"], tau, n,
             "A.  strong result: a few large effects in a small set",
             "2$\\times$2 detects it", "#2E8B62", verdict_xy=(0.55, 0.94), show_legend=True)
    hits_row(axes[1], densB, p["B"]["shift"], tau, n,
             "B.  coordinated shift: many genes nudged a little",
             "2$\\times$2 misses it -\nbut the set is really shifted", NONNULL_C,
             verdict_xy=(0.55, 0.80))
    axes[1].set_xlabel("gene-level z-score")
    fig.suptitle("Two jobs for enrichment - the threshold 2$\\times$2 only does the first",
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    save(fig, out, PARAMS, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
