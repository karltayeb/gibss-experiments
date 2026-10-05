#!/usr/bin/env python
"""Exact Bayesian changepoint posterior (Fearnhead 2006 recursions) for the coal series.

Poisson-Gamma segments, geometric segment lengths. For each (Gamma shape, p) in the
config grid: marginal boundary probabilities, distribution of the number of
changepoints, posterior mean rate profile; plus the exact single-changepoint posterior.
Monthly also gets the Beta-Bernoulli posterior for the indicator "at least one disaster
this month", with Beta(a, b) at prior mean = overall fraction of months with a disaster
and a = the default Gamma shape (the Bernoulli analogue of the default Poisson-Gamma).
Run from changepoints/:  uv run python scripts/02_exact_posterior.py [year|month ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from coal import PROC
from exact import (
    BetaBinomial,
    PoissonGamma,
    exact_posterior,
    single_changepoint_posterior,
)

RESULTS = ROOT / "results/coal"


def hpd_interval(prob: np.ndarray, mass: float = 0.95) -> tuple[int, int]:
    """Smallest set of highest-probability positions reaching `mass`; returns its hull."""
    idx = np.argsort(prob)[::-1]
    k = int(np.searchsorted(np.cumsum(prob[idx]), mass)) + 1
    sel = idx[:k]
    return int(sel.min()), int(sel.max())


def run(res: str, cfg: dict) -> None:
    d = pl.read_csv(PROC / f"{res}.csv")
    y = d["count"].to_numpy().astype(float)
    labels = d["label"].to_numpy()
    n = len(y)
    per_year = 12 if res == "month" else 1
    exposure = np.full(n, 1.0 / per_year)  # rates in events per year at any resolution
    out = RESULTS / res / "exact"
    out.mkdir(parents=True, exist_ok=True)
    ex = cfg["exact"]
    mean_rate = y.sum() / exposure.sum()
    rows = []
    for shape in ex["shape"]:
        model = PoissonGamma(shape=shape, rate=shape / mean_rate)  # prior mean = overall rate
        M = model.segment_logml(y, exposure)
        S = model.segment_mean(y, exposure)
        for p_year in ex["p"]:
            p = p_year / per_year  # same expected number of changepoints per year
            post = exact_posterior(M, S, p, kmax=ex["kmax"])
            bp = post.boundary_prob
            kp = post.k_prob
            np.savez(out / f"exact_shape{shape}_p{p_year}.npz", boundary_prob=bp,
                     k_prob=kp, mean_profile=post.mean_profile,
                     log_evidence=post.log_evidence, labels=labels, p=p, shape=shape)
            top = np.argsort(bp)[::-1][:3]
            rows.append({
                "resolution": res, "shape": shape, "p_per_year": p_year, "p": p,
                "log_evidence": post.log_evidence,
                "prior_k_mean": p * (n - 1), "k_mean": post.k_mean,
                "k_mode": int(np.argmax(kp)), "P_k0": kp[0], "P_k1": kp[1],
                "P_k2": kp[2], "P_k_ge3": kp[3:].sum(),
                "boundary_mass_1885_1900": float(bp[(labels[1:] >= 1885) & (labels[1:] <= 1900)].sum()),
                "boundary_mass_1925_1960": float(bp[(labels[1:] >= 1925) & (labels[1:] <= 1960)].sum()),
                "top1": float(labels[top[0] + 1]), "top1_prob": float(bp[top[0]]),
                "top2": float(labels[top[1] + 1]), "top2_prob": float(bp[top[1]]),
                "top3": float(labels[top[2] + 1]), "top3_prob": float(bp[top[2]]),
            })
        if shape == ex["shape_default"]:
            prob, log_bf = single_changepoint_posterior(M)
            lo, hi = hpd_interval(prob)
            np.savez(out / "single_changepoint.npz", prob=prob, log_bf10=log_bf,
                     labels=labels, shape=shape)
            print(f"[{res}] single changepoint (shape={shape}): mode new regime starts "
                  f"{labels[prob.argmax() + 1]:.2f}, 95% HPD hull {labels[lo + 1]:.2f}-"
                  f"{labels[hi + 1]:.2f}, log BF(1 vs 0 changepoints) = {log_bf:.1f}")
    if res == "month":
        z = (y > 0).astype(float)
        a = ex["shape_default"]
        pbar = z.mean()
        model = BetaBinomial(a=a, b=a * (1 - pbar) / pbar)
        M = model.segment_logml(z, np.ones(n))
        S = model.segment_mean(z, np.ones(n))
        for p_year in ex["p"]:
            post = exact_posterior(M, S, p_year / per_year, kmax=ex["kmax"])
            np.savez(out / f"exact_bernoulli_a{a}_p{p_year}.npz",
                     boundary_prob=post.boundary_prob, k_prob=post.k_prob,
                     mean_profile=post.mean_profile, log_evidence=post.log_evidence,
                     labels=labels, p=p_year / per_year, a=model.a, b=model.b)
            print(f"[{res}] Beta-Bernoulli Beta({model.a:.2f}, {model.b:.2f}) p/yr={p_year}: "
                  f"E[k]={post.k_mean:.2f}, P(k=0,1,2)={np.round(post.k_prob[:3], 3)}")
    tbl = pl.DataFrame(rows)
    tbl.write_csv(out / "grid_summary.csv")
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(40)
    print(tbl.select(["resolution", "shape", "p_per_year", "prior_k_mean", "k_mean", "k_mode",
                      "P_k0", "P_k1", "P_k2", "P_k_ge3", "boundary_mass_1885_1900",
                      "boundary_mass_1925_1960", "top1", "top1_prob", "top2", "top2_prob"]))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["coal"]
    for res in sys.argv[1:] or cfg["resolutions"]:
        run(res, cfg)


if __name__ == "__main__":
    main()
