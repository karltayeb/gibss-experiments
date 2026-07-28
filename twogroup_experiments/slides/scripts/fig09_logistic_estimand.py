"""Slide 9 - what logistic actually estimates.  [content, local]

Binarizing at a threshold tau changes the estimand: the logistic coefficient is
the log-odds of EXCEEDING tau, not of being differentially expressed. When the
null f0 and non-null f1 overlap, thresholding mislabels genes - false positives
(null above tau) and false negatives (non-null below tau).
"""
import argparse

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _common import save, INK, NULL_C, NONNULL_C

PARAMS = {"mu0": 0.0, "sd0": 1.0, "mu1": 2.3, "sd1": 1.0, "threshold": 2.0}


def main(out: str) -> None:
    p = PARAMS
    tau = p["threshold"]
    x = np.linspace(-4, 7, 600)
    f0 = norm.pdf(x, p["mu0"], p["sd0"])
    f1 = norm.pdf(x, p["mu1"], p["sd1"])

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(x, f0, color=NULL_C, lw=2, label=r"$f_0$ (null)")
    ax.plot(x, f1, color=NONNULL_C, lw=2, label=r"$f_1$ (non-null)")

    fp = x > tau   # null mass above threshold -> called a hit but is null
    fn = x < tau   # non-null mass below threshold -> missed
    ax.fill_between(x[fp], 0, f0[fp], color=NULL_C, alpha=0.30)
    ax.fill_between(x[fn], 0, f1[fn], color=NONNULL_C, alpha=0.30)

    ax.set_ylim(0, 0.45)
    ax.axvline(tau, color=INK, ls="--", lw=1.2)
    ax.text(tau - 0.1, 0.44, r"threshold $\tau$", ha="right", va="top",
            fontsize=9, color=INK)

    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.set_title(r"logistic estimand: $\log$-odds of $z>\tau$, not of being DE",
                 loc="left", fontsize=10)
    handles = [
        plt.Line2D([], [], color=NULL_C, lw=2, label=r"$f_0$ (null)"),
        plt.Line2D([], [], color=NONNULL_C, lw=2, label=r"$f_1$ (non-null)"),
        Patch(facecolor=NULL_C, alpha=0.30, label=r"false pos. (null $>\tau$)"),
        Patch(facecolor=NONNULL_C, alpha=0.30, label=r"false neg. (non-null $<\tau$)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, PARAMS, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
