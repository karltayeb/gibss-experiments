#!/usr/bin/env python
"""Poisson SuSiE (gibss) on the step basis for the coal series.

Primary: yearly counts, raw step basis, estimated prior variance, the default
"poisson" method (free-form quadrature SER, shared estimated intercept, centered).
Sensitivities: L, standardized basis, fixed prior variance, implementation
(irls = plug-in IRLS), monthly resolution. cf_cavi (exact CAVI in Q2) runs at
every resolution next to the default gIBSS fit. Monthly also fits a Bernoulli model to the
indicator "at least one disaster this month" (gIBSS and exact CAVI), the binomial
counterpart of the Poisson count model.
Run from changepoints/:  uv run python scripts/03_fit_susie.py [year|month ...]
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
from coal import PROC
from fitlib import cs_table, fit_step_susie

RESULTS = ROOT / "results/coal"


def variants(cfg: dict, res: str):
    f = cfg["fit"]
    for L in f["L"]:
        yield f"poisson_L{L}", {"L": L, "method": "poisson"}
        # exact CAVI (Gaussian q, closed-form Poisson log-normal offset) at every resolution,
        # so gIBSS and CAVI can be compared yearly and monthly
        yield f"cf_cavi_L{L}", {"L": L, "method": "cf_cavi", "family": "poisson"}
    if res == "month":
        for L in f["L"]:
            yield f"bernoulli_L{L}", {"L": L, "method": None, "family": "binomial",
                                      "response": "any"}
            yield f"cf_cavi_bernoulli_L{L}", {"L": L, "method": "cf_cavi",
                                              "family": "binomial", "response": "any"}
    if res == "year":
        for L in f["L"]:
            yield f"poisson_std_L{L}", {"L": L, "method": "poisson", "standardize": True}
            for pv in f["fixed_prior_variance"]:
                yield f"poisson_pv{pv}_L{L}", {"L": L, "method": "poisson",
                                                   "estimate_prior_variance": False,
                                                   "prior_variance": pv}
            yield f"irls_L{L}", {"L": L, "method": "irls", "family": "poisson"}


def run(res: str, cfg: dict) -> None:
    d = pl.read_csv(PROC / f"{res}.csv")
    y = d["count"].to_numpy().astype(float)
    labels = d["label"].to_numpy()
    out = RESULTS / res / "fits"
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for tag, kw in variants(cfg, res):
        kw.setdefault("family", "poisson")
        if kw.pop("response", "count") == "any":
            yk, kw["trials"] = (y > 0).astype(float), np.ones_like(y)
        else:
            yk = y
        fit = fit_step_susie(yk, coverage=cfg["fit"]["coverage"], **kw)
        with open(out / f"{tag}.pkl", "wb") as fh:
            pickle.dump(fit, fh)
        tbl = cs_table(fit, labels, tag=tag)
        tables.append(tbl)
        n_decl = int(tbl["declared"].sum())
        print(f"[{res}] {tag:22s} {fit['seconds']:6.1f}s iters={fit['n_iter']:3d} "
              f"converged={fit['converged']} declared={n_decl} "
              f"intercept={fit['intercept']:.3f}")
        print(tbl.filter(pl.col("declared")).select(
            ["component", "ser_log_bf", "cs_size", "cs_start", "cs_end", "top", "max_pip",
             "beta", "beta_sd", "rate_ratio", "prior_variance"]))
    pl.concat(tables).write_csv(RESULTS / res / "cs_table.csv")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["coal"]
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)
    for res in sys.argv[1:] or cfg["resolutions"]:
        run(res, cfg)


if __name__ == "__main__":
    main()
