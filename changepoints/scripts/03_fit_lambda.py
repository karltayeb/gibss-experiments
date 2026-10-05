#!/usr/bin/env python
"""Binomial SuSiE (logit link, step basis) on the lambda GC bins: every bin width x L.
Run from changepoints/:  uv run python scripts/03_fit_lambda.py [width ...]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fitlib import cs_table, fit_step_susie
from lam import PROC

RESULTS = ROOT / "results/lambda"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["lambda"]
    widths = [int(w) for w in sys.argv[1:]] or cfg["bin_widths"]
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(60)
    for w in widths:
        d = pl.read_csv(PROC / f"bins_{w}.csv")
        k = d["k"].to_numpy().astype(float)
        m = d["m"].to_numpy().astype(float)
        start = d["start"].to_numpy().astype(float)
        out = RESULTS / f"w{w}" / "fits"
        out.mkdir(parents=True, exist_ok=True)
        tables = []
        variants = [(f"binomial_L{L}", {"L": L}) for L in cfg["fit"]["L"]]
        if w == cfg["primary_width"]:
            # sensitivity: fixed prior variance (no ARD pruning) at the largest L
            L = cfg["fit"]["L"][-1]
            variants += [(f"binomial_pv{pv}_L{L}", {"L": L, "estimate_prior_variance": False,
                                                        "prior_variance": pv})
                         for pv in cfg["fit"]["fixed_prior_variance"]]
        for tag, kw in variants:
            fit = fit_step_susie(k, family="binomial", trials=m, method=None,
                                 coverage=cfg["fit"]["coverage"], **kw)
            fit["trials"] = m
            with open(out / f"{tag}.pkl", "wb") as fh:
                pickle.dump(fit, fh)
            tbl = cs_table(fit, start, tag=tag)
            tables.append(tbl)
            decl = tbl.filter(pl.col("declared"))
            print(f"[w{w}] {tag:22s} {fit['seconds']:6.1f}s iters={fit['n_iter']:3d} "
                  f"converged={fit['converged']} declared={decl.height}")
            print(decl.select(["component", "ser_log_bf", "cs_size", "cs_hull_width",
                               "cs_start", "cs_end", "top", "max_pip", "beta", "rate_ratio"]))
        pl.concat(tables).write_csv(RESULTS / f"w{w}" / "cs_table.csv")


if __name__ == "__main__":
    main()
