"""Slide 14 - the PH assumption breaks (location vs scale).  [spotlight, local]

Closed-form hazard ratios for the Gaussian two-group model, from
`Misspecification of the Gaussian two group model.md`. With Z~N(0,1) the null
z-scores, X = sigma Z + mu the non-null z-scores:
  forward  ordering  T = 1/z^2  (most significant arrive earliest) -> InvChi2-type
  reverse  ordering  S =   z^2  (most significant arrive latest)   -> Chi2-type
We plot log R_T (forward) and -log R_S (reverse; reciprocal so both are >1) vs
arrival time, for a location-driven and a scale-driven non-null. Neither respects
proportional hazards, but the forward ordering's violation is far more violent -
and worst exactly where the most informative genes live.
"""
import argparse

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from _common import save, M_COX, M_COX_REVERSED, MUT

# Parameterized by f1 = N(f1_loc, f1_scale^2) on the TRUE effect. The observed
# non-null z-score is N(f1_loc, 1 + f1_scale^2), so the hazard uses
# mu = f1_loc and sigma = sqrt(1 + f1_scale^2).
PARAMS = {
    "location": {"f1_loc": 3.0, "f1_scale": 0.01},   # pure location (near point mass)
    "scale": {"f1_loc": 0.0, "f1_scale": 2.0},        # pure scale
    "t_min": 0.05, "t_max": 40.0, "n_grid": 400,
}


def hazard_S(t, mu, sigma):
    """Hazard of S = X^2, X = sigma Z + mu (reverse ordering: z^2)."""
    rt = np.sqrt(t)
    z1 = (rt - mu) / sigma
    z2 = -(rt + mu) / sigma
    surv = 1 - norm.cdf(z1) + norm.cdf(z2)
    dens = (norm.pdf(z1) + norm.pdf(z2)) / (2 * sigma * rt)
    return dens / surv


def hazard_T(t, mu, sigma):
    """Hazard of T = 1 / X^2 (forward ordering: 1/z^2)."""
    rt = np.sqrt(t)
    z1 = (1 - mu * rt) / (sigma * rt)
    z2 = -(1 + mu * rt) / (sigma * rt)
    surv = norm.cdf(z1) - norm.cdf(z2)
    dens = (norm.pdf(z1) + norm.pdf(z2)) / (2 * sigma * t ** 1.5)
    return dens / surv


def panel(ax, t, sig, title):
    mu = sig["f1_loc"]
    sigma = np.sqrt(1.0 + sig["f1_scale"] ** 2)      # observed non-null SD
    logRT = np.log(hazard_T(t, mu, sigma) / hazard_T(t, 0.0, 1.0))
    negLogRS = -np.log(hazard_S(t, mu, sigma) / hazard_S(t, 0.0, 1.0))
    ax.axhline(0, color=MUT, lw=0.8, ls=":")
    ax.plot(t, logRT, color=M_COX, lw=2.2, label=r"forward (cox)  $\log R_T$ (InvChi$^2$)")
    ax.plot(t, negLogRS, color=M_COX_REVERSED, lw=2.2,
            label=r"reverse (cox-rev)  $-\log R_S$ (Chi$^2$)")
    ax.set_xscale("log")
    ax.set_xlabel("arrival time  $t$")
    ax.set_title(title, loc="left", fontsize=10)


def main(out: str) -> None:
    p = PARAMS
    t = np.logspace(np.log10(p["t_min"]), np.log10(p["t_max"]), p["n_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    panel(axes[0], t, p["location"],
          r"pure location: $f_1=N(%.0f,\ %.2f^2)$" % (p["location"]["f1_loc"], p["location"]["f1_scale"]))
    panel(axes[1], t, p["scale"],
          r"pure scale: $f_1=N(0,\ %.0f^2)$" % p["scale"]["f1_scale"])
    axes[0].set_ylabel("log hazard ratio")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, PARAMS, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
