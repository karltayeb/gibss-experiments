#!/usr/bin/env python
"""Exact Beta-Binomial changepoint posterior and a 2-state binomial HMM on the lambda
GC bins, for every bin width in the config.
Run from changepoints/:  uv run python scripts/02_exact_lambda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from exact import BetaBinomial, exact_posterior
from hmm import fit_hmm
from lam import PROC

RESULTS = ROOT / "results/lambda"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["lambda"]
    ex = cfg["exact"]
    model = BetaBinomial(ex["beta_a"], ex["beta_b"])
    rows = []
    hrows = []
    for w in cfg["bin_widths"]:
        d = pl.read_csv(PROC / f"bins_{w}.csv")
        k = d["k"].to_numpy().astype(float)
        m = d["m"].to_numpy().astype(float)
        start = d["start"].to_numpy()
        out = RESULTS / f"w{w}" / "exact"
        out.mkdir(parents=True, exist_ok=True)
        M = model.segment_logml(k, m)
        S = model.segment_mean(k, m)
        for p100 in ex["p_per_100bp"]:
            p = min(p100 * w / 100.0, 0.9)
            post = exact_posterior(M, S, p, kmax=ex["kmax"])
            bp = post.boundary_prob
            np.savez(out / f"exact_p{p100}.npz", boundary_prob=bp, k_prob=post.k_prob,
                     mean_profile=post.mean_profile, start=start, p=p, width=w)
            kp = post.k_prob
            rows.append({"width": w, "n": len(k), "p_per_100bp": p100, "p": p,
                         "prior_k_mean": p * (len(k) - 1), "k_mean": post.k_mean,
                         "k_mode": int(np.argmax(kp)), "k_q05": int(np.searchsorted(np.cumsum(kp), 0.05)),
                         "k_q95": int(np.searchsorted(np.cumsum(kp), 0.95)),
                         "log_evidence": post.log_evidence,
                         "n_boundaries_gt_0.5": int((bp > 0.5).sum())})
        h = fit_hmm(k, m, n_states=cfg["hmm"]["n_states"])
        np.savez(RESULTS / f"w{w}" / "hmm.npz", gamma=h["gamma"], switch_prob=h["switch_prob"],
                 theta=h["theta"], A=h["A"], start=start)
        hrows.append({"width": w, "n": len(k), "theta_low": float(h["theta"].min()),
                      "theta_high": float(h["theta"].max()),
                      "stay_prob_min": float(np.diag(h["A"]).min()),
                      "expected_switches": h["expected_switches"],
                      "n_switch_gt_0.5": int((h["switch_prob"] > 0.5).sum()),
                      "loglik": h["loglik"], "em_iters": h["n_iter"]})
        print(f"width {w}: exact done ({len(ex['p_per_100bp'])} priors); HMM theta="
              f"{np.round(h['theta'], 3)} expected switches {h['expected_switches']:.2f}")
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_rows(40)
    t = pl.DataFrame(rows)
    t.write_csv(RESULTS / "exact_grid_summary.csv")
    print(t)
    ht = pl.DataFrame(hrows)
    ht.write_csv(RESULTS / "hmm_summary.csv")
    print(ht)


if __name__ == "__main__":
    main()
