#!/usr/bin/env python
"""Calibration of binomial SuSiE changepoint CSs on lambda-like data (primary bin width).

Truth = the piecewise-constant GC profile from the declared components of the L=20 fit at
the primary width (segment means from the data). Three generative models at the same
truth:
  binomial    : k ~ Binomial(m, theta_seg), independent bins (the fitted model);
Fits use calibration.fit_L from the config (L=20 at 30 s a fit is too slow for replicates;
L=10 reproduces the same declared CSs on the data).
  betabinom   : k ~ BetaBinomial with the Pearson overdispersion estimated from the data
                residuals at the primary width (extra-binomial variation, independent bins);
  autocorr    : per-base Bernoulli with a 2-state within-segment Markov dependence tuned
                to the lag-1 autocorrelation of the real GC indicator, then binned.
For each: coverage of true changepoints by declared CSs, false CSs, CS width.
Run from changepoints/:  uv run python scripts/04_calibration_lambda.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import itertools

from fitlib import fit_step_susie
from lam import PROC, gc_bits, load_sequence
from stepbasis import cs_interval

RESULTS = ROOT / "results/lambda/calibration"
MIN_LOG_BF = 2.0


def truth_from_fit(fit: dict, k: np.ndarray, m: np.ndarray) -> tuple[list[int], np.ndarray]:
    cps = sorted({int(np.argmax(fit["alpha"][l])) for l in range(fit["L"])
                  if fit["ser_log_bf"][l] >= MIN_LOG_BF})
    bounds = [0] + [c + 1 for c in cps] + [len(k)]
    theta = np.empty(len(k))
    for s, u in itertools.pairwise(bounds):
        theta[s:u] = k[s:u].sum() / m[s:u].sum()
    return cps, theta


def pearson_dispersion(k, m, theta, n_params: int) -> float:
    mu = m * theta
    chi2 = np.sum((k - mu) ** 2 / (mu * (1 - theta)))
    return float(chi2 / (len(k) - n_params))


def simulate_autocorr(theta_bins: np.ndarray, m: np.ndarray, rho: float, rng) -> np.ndarray:
    """Per-base Bernoulli chain with stationary mean theta and lag-1 correlation rho
    within each bin (P(1->1) = theta + rho(1-theta), P(0->1) = theta(1-rho)), binned."""
    k = np.empty(len(m), dtype=float)
    for i, (th, mi) in enumerate(zip(theta_bins, m.astype(int))):
        x = np.empty(mi, dtype=np.int8)
        x[0] = rng.random() < th
        u = rng.random(mi)
        p11 = th + rho * (1 - th)
        p01 = th * (1 - rho)
        for t in range(1, mi):
            x[t] = u[t] < (p11 if x[t - 1] else p01)
        k[i] = x.sum()
    return k


def score(fit: dict, true_cols: list[int], tol: int = 0) -> dict:
    declared = [l for l in range(fit["L"]) if fit["ser_log_bf"][l] >= MIN_LOG_BF]
    rows = []
    for l in declared:
        c = fit["cs"][l]
        _lo, _hi, width = cs_interval(c)
        covered = any(any(abs(t - cc) <= tol for cc in c) for t in true_cols)
        rows.append({"component": l + 1, "ser_log_bf": float(fit["ser_log_bf"][l]),
                     "cs_size": len(c), "cs_hull_width": width, "covered": covered,
                     "top": int(np.argmax(fit["alpha"][l]))})
    hit = [any(any(abs(t - cc) <= tol for cc in fit["cs"][l]) for l in declared) for t in true_cols]
    return {"components": rows, "n_declared": len(declared),
            "n_false": sum(not r["covered"] for r in rows), "recall": float(np.mean(hit)) if hit else np.nan}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["lambda"]
    w = cfg["primary_width"]
    L = cfg["fit"]["L"][-1]
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = pl.read_csv(PROC / f"bins_{w}.csv")
    k = d["k"].to_numpy().astype(float)
    m = d["m"].to_numpy().astype(float)
    with open(ROOT / f"results/lambda/w{w}/fits/binomial_L{L}.pkl", "rb") as fh:
        fit = pickle.load(fh)
    cps, theta = truth_from_fit(fit, k, m)
    phi = pearson_dispersion(k, m, theta, n_params=len(cps) + 1)
    bits = gc_bits(load_sequence())
    c = bits - bits.mean()
    rho = float(np.dot(c[:-1], c[1:]) / np.dot(c, c))
    print(f"truth: {len(cps)} changepoints at columns {cps}; segment GC "
          f"{np.round(np.unique(theta), 3)}; Pearson dispersion phi={phi:.3f}; "
          f"lag-1 GC autocorrelation rho={rho:.4f}")
    # beta-binomial with the same mean and variance inflation phi: var = m th(1-th) phi
    # => rho_bb = (phi - 1)/(m - 1), a = th (1-rho)/rho, b = (1-th)(1-rho)/rho
    rng = np.random.default_rng(cfg["calibration"]["seed"])
    n_reps = cfg["calibration"]["n_reps"]
    rows = []
    comp_rows = []
    for design in ("binomial", "betabinom", "autocorr"):
        for rep in range(n_reps):
            if design == "binomial":
                ksim = rng.binomial(m.astype(int), theta).astype(float)
            elif design == "betabinom":
                rbb = max(phi - 1.0, 1e-6) / (m - 1.0)
                a = theta * (1 - rbb) / rbb
                b = (1 - theta) * (1 - rbb) / rbb
                pth = rng.beta(a, b)
                ksim = rng.binomial(m.astype(int), pth).astype(float)
            else:
                ksim = simulate_autocorr(theta, m, rho, rng)
            for Lf in cfg["calibration"]["fit_L"]:
                f = fit_step_susie(ksim, L=Lf, family="binomial", trials=m)
                sc = score(f, cps)
                rows.append({"design": design, "rep": rep, "L": Lf, "n_declared": sc["n_declared"],
                             "n_false": sc["n_false"], "recall": sc["recall"],
                             "phi_hat": pearson_dispersion(ksim, m, np.clip(
                                 1 / (1 + np.exp(-f["eta"])), 1e-6, 1 - 1e-6), sc["n_declared"] + 1)})
                for r in sc["components"]:
                    comp_rows.append({"design": design, "rep": rep, "L": Lf, **r})
            if (rep + 1) % 25 == 0:
                print(f"  {design} rep {rep + 1}/{n_reps}", flush=True)
    tbl = pl.DataFrame(rows)
    tbl.write_csv(RESULTS / "reps.csv")
    comp = pl.DataFrame(comp_rows)
    comp.write_csv(RESULTS / "components.csv")
    summary = tbl.group_by(["design", "L"]).agg(
        pl.col("n_declared").mean().alias("mean_declared"),
        pl.col("n_false").mean().alias("mean_false"),
        pl.col("recall").mean().alias("recall"),
        pl.col("phi_hat").median().alias("median_phi_hat"),
    ).join(
        comp.group_by(["design", "L"]).agg(
            pl.col("covered").mean().alias("cs_coverage"),
            pl.col("cs_size").median().alias("median_cs_size"),
            pl.col("cs_hull_width").median().alias("median_hull_width"),
        ), on=["design", "L"]).sort(["design", "L"])
    summary = summary.with_columns(pl.lit(len(cps)).alias("true_k"), pl.lit(phi).alias("phi_data"),
                                   pl.lit(rho).alias("rho_data"))
    summary.write_csv(RESULTS / "summary.csv")
    pl.Config.set_tbl_width_chars(200)
    print(summary)


if __name__ == "__main__":
    main()
