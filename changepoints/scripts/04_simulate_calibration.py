#!/usr/bin/env python
"""Calibration of SuSiE changepoint credible sets by simulation (coal, yearly scale).

Design A ("fitted"): piecewise-constant Poisson truth with the two changepoints the
  data support (new regimes from 1892 and 1948) and segment MLE rates; many replicates;
  fit L in {1, 3, 5}. Coverage = a declared CS (log BF >= 2) contains a true changepoint
  column; false CS = a declared CS containing none; recall = true changepoints covered.
Design B ("sweep"): one changepoint at varying position and jump size; L=1 CS coverage
  and width against the exact single-changepoint 95% HPD set on the same data; L=3 for
  false CSs.
Run from changepoints/:  uv run python scripts/04_simulate_calibration.py [A|B]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import itertools

from coal import PROC
from exact import (
    PoissonGamma,
    exact_posterior,
    single_changepoint_posterior,
)
from fitlib import fit_step_susie
from stepbasis import cs_interval

RESULTS = ROOT / "results/coal/year/calibration"
MIN_LOG_BF = 2.0


def hpd_set(prob: np.ndarray, mass: float = 0.95) -> np.ndarray:
    idx = np.argsort(prob)[::-1]
    k = int(np.searchsorted(np.cumsum(prob[idx]), mass)) + 1
    return idx[:k]


def piecewise_rates(y: np.ndarray, cps: list[int]) -> np.ndarray:
    """Segment MLE rates; cps are step columns (jump between t=cp and t=cp+1)."""
    bounds = [0] + [c + 1 for c in cps] + [len(y)]
    rate = np.empty(len(y))
    for s, u in itertools.pairwise(bounds):
        rate[s:u] = y[s:u].mean()
    return rate


def score_fit(fit: dict, true_cols: list[int]) -> dict:
    declared = [l for l in range(fit["L"]) if fit["ser_log_bf"][l] >= MIN_LOG_BF]
    covered = [any(c in fit["cs"][l] for c in true_cols) for l in declared]
    hit = [any(c in fit["cs"][l] for l in declared) for c in true_cols]
    sizes = [len(fit["cs"][l]) for l in declared]
    widths = [cs_interval(fit["cs"][l])[2] for l in declared]
    return {
        "n_declared": len(declared), "n_covered": int(sum(covered)),
        "n_false": int(len(declared) - sum(covered)), "n_true_hit": int(sum(hit)),
        "cs_sizes": sizes, "cs_widths": widths,
    }


def design_a(cfg: dict, rng: np.random.Generator) -> None:
    d = pl.read_csv(PROC / "year.csv")
    y = d["count"].to_numpy().astype(float)
    labels = d["label"].to_numpy()
    true_cols = [int(np.flatnonzero(labels == 1892)[0]) - 1, int(np.flatnonzero(labels == 1948)[0]) - 1]
    rate = piecewise_rates(y, true_cols)
    print("design A truth: changepoint columns", true_cols, "segment rates",
          np.unique(rate, return_index=True)[0][np.argsort(np.unique(rate, return_index=True)[1])])
    rows = []
    n_reps = cfg["calibration"]["n_reps"]
    ex = cfg["exact"]
    for rep in range(n_reps):
        ysim = rng.poisson(rate).astype(float)
        model = PoissonGamma(shape=ex["shape_default"], rate=ex["shape_default"] / ysim.mean())
        post = exact_posterior(model.segment_logml(ysim), model.segment_mean(ysim), ex["p_default"],
                               kmax=ex["kmax"])
        for L in cfg["fit"]["L"]:
            fit = fit_step_susie(ysim, L=L, method="poisson")
            sc = score_fit(fit, true_cols)
            for l in range(L):
                c = fit["cs"][l]
                lo, hi, width = cs_interval(c)
                rows.append({
                    "rep": rep, "L": L, "component": l + 1,
                    "ser_log_bf": float(fit["ser_log_bf"][l]),
                    "declared": bool(fit["ser_log_bf"][l] >= MIN_LOG_BF),
                    "cs_size": len(c), "cs_hull_width": width,
                    "cs_lo": lo, "cs_hi": hi, "top": int(np.argmax(fit["alpha"][l])),
                    "covers_1892": true_cols[0] in c, "covers_1948": true_cols[1] in c,
                    "exact_mass_in_hull": float(post.boundary_prob[lo:hi + 1].sum()),
                    "exact_k_mean": post.k_mean, "n_declared": sc["n_declared"],
                    "n_false": sc["n_false"], "n_true_hit": sc["n_true_hit"],
                })
        if (rep + 1) % 25 == 0:
            print(f"  rep {rep + 1}/{n_reps}", flush=True)
    tbl = pl.DataFrame(rows)
    tbl.write_csv(RESULTS / "design_a_components.csv")
    decl = tbl.filter(pl.col("declared"))
    summary = decl.group_by("L").agg(
        pl.len().alias("declared_cs"),
        (pl.col("covers_1892") | pl.col("covers_1948")).mean().alias("cs_coverage"),
        (~(pl.col("covers_1892") | pl.col("covers_1948"))).sum().alias("false_cs"),
        pl.col("cs_size").median().alias("median_cs_size"),
        pl.col("cs_hull_width").median().alias("median_hull_width"),
        pl.col("exact_mass_in_hull").median().alias("median_exact_mass_in_hull"),
    ).sort("L")
    per_rep = tbl.group_by(["rep", "L"]).agg(
        pl.col("n_declared").first(), pl.col("n_false").first(), pl.col("n_true_hit").first(),
        pl.col("covers_1892").filter(pl.col("declared")).any().alias("hit_1892"),
        pl.col("covers_1948").filter(pl.col("declared")).any().alias("hit_1948"),
        pl.col("exact_k_mean").first(),
    ).group_by("L").agg(
        pl.col("n_declared").mean().alias("mean_declared"),
        pl.col("n_false").mean().alias("mean_false_per_rep"),
        pl.col("hit_1892").mean().alias("recall_1892"),
        pl.col("hit_1948").mean().alias("recall_1948"),
        pl.col("exact_k_mean").mean().alias("exact_k_mean"),
    ).sort("L")
    out = summary.join(per_rep, on="L")
    out.write_csv(RESULTS / "design_a_summary.csv")
    print(out)


def design_b(cfg: dict, rng: np.random.Generator) -> None:
    n = 112
    base = 3.0
    positions = [5, 15, 30, 55, 80, 95, 105]  # step column of the jump
    ratios = [0.5, 0.33, 0.25]
    n_reps = cfg["calibration"]["n_reps"] // 2
    rows = []
    for col in positions:
        for rr in ratios:
            rate = np.full(n, base)
            rate[col + 1:] = base * rr
            for rep in range(n_reps):
                ysim = rng.poisson(rate).astype(float)
                model = PoissonGamma(shape=1.0, rate=1.0 / ysim.mean())
                prob, log_bf = single_changepoint_posterior(model.segment_logml(ysim))
                hs = hpd_set(prob)
                fit1 = fit_step_susie(ysim, L=1, method="poisson")
                fit3 = fit_step_susie(ysim, L=3, method="poisson")
                cs1 = fit1["cs"][0]
                sc3 = score_fit(fit3, [col])
                rows.append({
                    "position": col, "rate_ratio": rr, "rep": rep,
                    "dist_to_edge": min(col + 1, n - 1 - col),
                    "L1_log_bf": float(fit1["ser_log_bf"][0]),
                    "L1_declared": bool(fit1["ser_log_bf"][0] >= MIN_LOG_BF),
                    "L1_covers": col in cs1, "L1_cs_size": len(cs1),
                    "L1_hull_width": cs_interval(cs1)[2],
                    "L1_exact_mass_in_cs": float(prob[list(cs1)].sum()),
                    "exact_log_bf10": log_bf, "exact_covers": col in hs,
                    "exact_hpd_size": len(hs), "exact_hpd_width": int(hs.max() - hs.min() + 1),
                    "L3_n_declared": sc3["n_declared"], "L3_n_false": sc3["n_false"],
                    "L3_hit": sc3["n_true_hit"] > 0,
                })
            print(f"  position {col} ratio {rr} done", flush=True)
    tbl = pl.DataFrame(rows)
    tbl.write_csv(RESULTS / "design_b_reps.csv")
    summary = tbl.group_by(["position", "rate_ratio"]).agg(
        pl.col("dist_to_edge").first(),
        pl.col("L1_declared").mean().alias("L1_power"),
        pl.col("L1_covers").mean().alias("L1_coverage"),
        pl.col("L1_cs_size").median().alias("L1_median_cs_size"),
        pl.col("L1_hull_width").median().alias("L1_median_hull_width"),
        pl.col("L1_exact_mass_in_cs").median().alias("median_exact_mass_in_L1_cs"),
        pl.col("exact_covers").mean().alias("exact_hpd_coverage"),
        pl.col("exact_hpd_size").median().alias("exact_median_hpd_size"),
        pl.col("L3_n_declared").mean().alias("L3_mean_declared"),
        pl.col("L3_n_false").mean().alias("L3_mean_false"),
        pl.col("L3_hit").mean().alias("L3_recall"),
    ).sort(["rate_ratio", "position"])
    summary.write_csv(RESULTS / "design_b_summary.csv")
    print(summary)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["coal"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    # each design draws from its own generator seeded identically, so a design's results
    # do not depend on whether the other design ran first in the same process
    seed = cfg["calibration"]["seed"]
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(40)
    which = sys.argv[1:] or ["A", "B"]
    if "A" in which:
        design_a(cfg, np.random.default_rng(seed))
    if "B" in which:
        design_b(cfg, np.random.default_rng(seed))


if __name__ == "__main__":
    main()
