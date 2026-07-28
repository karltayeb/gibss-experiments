"""Slide 8 - f1 is barely identifiable.  [spotlight, local]

Two panels:
  LEFT  - density decompositions for two scenarios that fit the data almost
          equally well. For each scenario s we draw the null component
          (1-pi_s) f0 (faint dotted), the non-null component pi_s f1_s (dashed),
          and the marginal ftilde_s (solid). Colour encodes the scenario (a
          scheme distinct from the null/non-null teal/amber used elsewhere).
  RIGHT - level curves of the marginal log-likelihood of the data over
          (pi, theta), where theta is a SINGLE f1 parameter. The elongated ridge
          shows a wide range of (pi, theta) fit nearly as well - f1 is barely
          identified. The two scenarios are the endpoints of that ridge.

f1 is boiled down to one parameter, selectable with --param:
  loc   : f1 shifts location, f1_obs = N(theta, 1 + s0^2), s0 fixed small.
  scale : f1 spreads,        f1_obs = N(0, 1 + theta^2).
(The location(t)/scale(t) path variant is a natural third option.)
"""
import argparse

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from _common import save, INK, MUT

# scenario colours - deliberately different from null/non-null (teal/amber)
SC1, SC2 = "#5E3C99", "#2C7FB8"      # violet, blue
PARAMS = {
    "param": "scale", "pi_true": 0.30, "s0": 0.1,
    "n_data": 4000, "grid": 70, "seed": 4,
}
RANGES = {"loc": (0.4, 4.0), "scale": (0.4, 4.0), "moment": (0.2, 9.0)}
# true theta per parameterization (theta = loc / scale / second moment)
THETA_TRUE = {"loc": 1.8, "scale": 1.6, "moment": 2.6}
# axis label per parameterization
THETA_LABEL = {"loc": r"location", "scale": r"scale",
               "moment": r"2nd moment $t=\ell^2+s^2$ ($\ell{=}s$)"}


def f1_obs(z, theta, param, s0):
    """Observed non-null density, single f1 parameter theta."""
    if param == "loc":
        return norm.pdf(z, theta, np.sqrt(1.0 + s0 ** 2))
    if param == "scale":
        return norm.pdf(z, 0.0, np.sqrt(1.0 + theta ** 2))
    # moment path: t = loc^2 + scale^2 with loc = scale = sqrt(t/2)
    m = np.sqrt(np.clip(theta, 0, None) / 2.0)
    return norm.pdf(z, m, np.sqrt(1.0 + m ** 2))


def marginal(z, pi, theta, param, s0):
    return (1 - pi) * norm.pdf(z, 0, 1) + pi * f1_obs(z, theta, param, s0)


def main(out: str) -> None:
    p = PARAMS
    param, s0 = p["param"], p["s0"]
    rng = np.random.default_rng(p["seed"])

    # sample data from the true model
    theta_true = THETA_TRUE[param]
    z = np.linspace(-5, 8, 500)
    nn = rng.random(p["n_data"]) < p["pi_true"]
    if param == "loc":
        b = np.where(nn, rng.normal(theta_true, s0, p["n_data"]), 0.0)
    elif param == "scale":
        b = np.where(nn, rng.normal(0.0, theta_true, p["n_data"]), 0.0)
    else:  # moment path, loc = scale = sqrt(t/2)
        m = np.sqrt(theta_true / 2.0)
        b = np.where(nn, rng.normal(m, m, p["n_data"]), 0.0)
    data = b + rng.normal(0, 1, p["n_data"])

    # log-likelihood surface over (pi, theta)
    pis = np.linspace(0.03, 0.6, p["grid"])
    lo, hi = RANGES[param]
    thetas = np.linspace(lo, hi, p["grid"])
    LL = np.empty((len(pis), len(thetas)))
    for i, pi in enumerate(pis):
        for j, th in enumerate(thetas):
            LL[i, j] = np.sum(np.log(np.clip(marginal(data, pi, th, param, s0), 1e-300, None)))
    LL -= LL.max()

    # two scenarios = ends of the ridge (grid points within 2 log-lik of the max,
    # smallest and largest pi)
    ridge = np.argwhere(LL >= -2.0)
    imin = ridge[np.argmin(ridge[:, 0])]
    imax = ridge[np.argmax(ridge[:, 0])]
    scen = [
        {"pi": pis[imax[0]], "theta": thetas[imax[1]], "c": SC1, "tag": "A"},
        {"pi": pis[imin[0]], "theta": thetas[imin[1]], "c": SC2, "tag": "B"},
    ]

    fig, (axd, axc) = plt.subplots(1, 2, figsize=(7.8, 3.5))

    # LEFT: decompositions
    for s in scen:
        null = (1 - s["pi"]) * norm.pdf(z, 0, 1)
        slab = s["pi"] * f1_obs(z, s["theta"], param, s0)
        axd.plot(z, null + slab, color=s["c"], lw=2.2, ls="-")
        axd.plot(z, slab, color=s["c"], lw=1.6, ls="--")
        axd.plot(z, null, color=s["c"], lw=1.1, ls=":", alpha=0.7)
    axd.plot([], [], color=INK, lw=2.0, ls="-", label=r"marginal $\tilde f$")
    axd.plot([], [], color=INK, lw=1.6, ls="--", label=r"non-null $\pi f_1$")
    axd.plot([], [], color=INK, lw=1.1, ls=":", label=r"null $(1-\pi)f_0$")
    axd.set_xlabel("z-score")
    axd.set_ylabel("density")
    axd.set_ylim(bottom=0)
    axd.set_title("two fits, ~one marginal", loc="left", fontsize=10)
    axd.legend(loc="upper right", frameon=False, fontsize=8)
    for s in scen:
        axd.text(0.02, 0.96 if s["tag"] == "A" else 0.88,
                 r"%s: $\pi{=}%.2f,\ \theta{=}%.1f$" % (s["tag"], s["pi"], s["theta"]),
                 transform=axd.transAxes, color=s["c"], fontsize=8.5, va="top")

    # RIGHT: log-likelihood contours
    T, P = np.meshgrid(thetas, pis)
    cs = axc.contourf(T, P, LL, levels=[-30, -15, -7, -3, -1, 0], cmap="Greys_r", alpha=0.9)
    axc.contour(T, P, LL, levels=[-7, -3, -1], colors="w", linewidths=0.6)
    for s in scen:
        axc.scatter([s["theta"]], [s["pi"]], color=s["c"], s=55, zorder=5,
                    edgecolor="w", linewidth=1.2)
        axc.text(s["theta"], s["pi"], "  " + s["tag"], color=s["c"], fontsize=10,
                 fontweight="bold", va="center")
    axc.set_xlabel(r"$f_1$ parameter: %s" % THETA_LABEL[param])
    axc.set_ylabel(r"$\pi$")
    axc.set_title("marginal log-likelihood: a ridge, not a peak", loc="left", fontsize=10)
    cbar = fig.colorbar(cs, ax=axc, fraction=0.046, pad=0.04)
    cbar.set_label("log-lik $-$ max", fontsize=8)

    fig.suptitle(f"f$_1$ barely identified ({param}-parameterized)", fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, out, {**PARAMS, "scenarios": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                             for k, v in s.items()} for s in scen]}, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--param", choices=["loc", "scale", "moment"], default=PARAMS["param"])
    a = ap.parse_args()
    PARAMS["param"] = a.param
    main(a.out)
