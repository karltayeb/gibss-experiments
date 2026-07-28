"""Slide 8 - misspecifying f1 (not identifiability) is the real problem.  [spotlight]

Two panels:
  LEFT  - population marginal misfit. The data are scale-driven (non-null
          N(0, 1+sigma^2), symmetric heavy tails). A LOCATION-family two-group
          (non-null N(loc, 1+s0^2), s0 fixed small) is fit to that marginal by
          maximizing the expected log-likelihood; it cannot represent the excess
          variance, so it systematically misses the tails (shaded).
  RIGHT - the enrichment consequence (real SER fits, ~150 sims, from
          fig08_data.json): mean causal PIP for the oracle, the correctly
          specified scale-family, the MISSPECIFIED location-family, and
          cox-reversed (a rank method that models no f1 at all).

Message: with a parametric f1 the model is identifiable - the practical risk is
assuming the WRONG f1 family. That tanks the two-group, while the rank method,
which never models f1, is unaffected.
"""
import argparse
import json
import pathlib

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from _common import (save, INK, MUT, M_ORACLE, M_ESTIMATED, M_TWOGROUP_LOC,
                     M_TWOGROUP_SCALE, M_COX_REVERSED)

DATA = pathlib.Path(__file__).resolve().parents[1] / "figures" / "fig08_data.json"
PARAMS = {"pi": 0.30, "sigma": 1.6, "s0": 0.1, "grid": 500}


def scale_marginal(z, pi, sigma):
    return (1 - pi) * norm.pdf(z, 0, 1) + pi * norm.pdf(z, 0, np.sqrt(1 + sigma ** 2))


def loc_marginal(z, pi, loc, s0):
    return (1 - pi) * norm.pdf(z, 0, 1) + pi * norm.pdf(z, loc, np.sqrt(1 + s0 ** 2))


def best_loc_fit(z, mstar, s0):
    """Best location-family fit to the true marginal: maximize E_mstar[log m_loc]."""
    best, arg = -np.inf, (0.0, 0.0)
    dz = z[1] - z[0]
    for pi in np.linspace(0.02, 0.8, 40):
        for loc in np.linspace(0.0, 4.0, 40):
            ml = loc_marginal(z, pi, loc, s0)
            val = np.sum(mstar * np.log(np.clip(ml, 1e-300, None))) * dz
            if val > best:
                best, arg = val, (pi, loc)
    return arg


def left_panel(ax, p):
    z = np.linspace(-6, 6, p["grid"])
    mstar = scale_marginal(z, p["pi"], p["sigma"])
    pi_l, loc_l = best_loc_fit(z, mstar, p["s0"])
    mloc = loc_marginal(z, pi_l, loc_l, p["s0"])

    ax.plot(z, mstar, color=INK, lw=2.4, label="true marginal (scale-driven)")
    ax.plot(z, mloc, color=M_TWOGROUP_LOC, lw=2.0, ls="--",
            label="estimated location family")
    miss = mstar > mloc
    ax.fill_between(z, mloc, mstar, where=miss, color=M_TWOGROUP_LOC, alpha=0.15)
    zt = -3.0
    ax.annotate("location family can't\nreach the scale tails",
                xy=(zt, scale_marginal(np.array([zt]), p["pi"], p["sigma"])[0]),
                xytext=(-5.8, 0.22), fontsize=8.5, color=M_TWOGROUP_LOC, ha="left",
                arrowprops=dict(arrowstyle="->", color=M_TWOGROUP_LOC, lw=1))
    ax.set_xlabel("z-score")
    ax.set_ylabel("density")
    ax.set_ylim(bottom=0)
    ax.set_title("wrong family can't fit the data", loc="left", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def right_panel(ax):
    d = json.loads(DATA.read_text())
    pip = d["pip"]
    order = [("oracle", "oracle", M_ORACLE),
             ("scale_fam", "scale-fam\n(correct)", M_TWOGROUP_SCALE),
             ("loc_scale_fam", "loc-scale\n(both est.)", M_ESTIMATED),
             ("loc_fam", "loc-fam\n(misspec.)", M_TWOGROUP_LOC),
             ("cox_reversed", "cox-rev\n(no f$_1$)", M_COX_REVERSED)]
    order = [o for o in order if o[0] in pip]
    xs = np.arange(len(order))
    means = [pip[k]["mean"] for k, _, _ in order]
    ses = [pip[k]["se"] for k, _, _ in order]
    cols = [c for _, _, c in order]
    ax.bar(xs, means, yerr=ses, color=cols, capsize=3, width=0.72)
    for x, m in zip(xs, means):
        ax.text(x, m + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab, _ in order], fontsize=7.5)
    ax.set_ylabel("mean causal PIP")
    ax.set_ylim(0, 1.0)
    n = d["config"]["n_rep"]
    ax.set_title(f"enrichment collapses under misspecification  ({n} sims)",
                 loc="left", fontsize=9.5)
    ax.text(0.98, 0.96, f"scale-driven data\n{d['config']['signal']}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUT, family="monospace")


def main(out):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.0, 3.6))
    left_panel(axl, PARAMS)
    right_panel(axr)
    fig.suptitle("f$_1$ is identifiable - misspecifying its family is the real risk",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    d = json.loads(DATA.read_text())
    save(fig, out, {**PARAMS, "pip_source": "fig08_data.json", "spec": d.get("spec")}, __file__)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
