"""Load the 021 Poisson-SuSiE fit results into tidy frames for analysis.

The Poisson companion to ``analysis/logistic_susie_simulations/load_results.py``. Reads the
local ``results/by_batch/.../fits/.../{pip,cs}.parquet`` (rsynced from Midway) for the
``021-poisson`` supercollection and attaches each simulation cell's coordinates.

Two differences from the logistic loader:
  * the intercept ``b0`` is the baseline LOG-RATE, so we also carry ``lambda0 = e^{b0}`` (the
    mean count of a null row) -- the experiment's rate axis;
  * the method name keeps its q-level (``q1``/``q2`` are BOTH run for Poisson), so the
    ``method`` column is e.g. ``q2_gibss`` / ``q1_cavi`` / ``q2_score`` (+ ``_pv1`` twins).

Cell T is mapped from (design, lambda0, beta) via betas.json. Run under the repo-root uv env.
"""
from __future__ import annotations

import json
import math
import os
import sys

import polars as pl

_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _TG_ROOT not in sys.path:
    sys.path.insert(0, _TG_ROOT)
import experiments.loader as loader  # noqa: E402

_BETAS = json.load(open(os.path.join(_HERE, "betas.json")))["betas"]
FREEZE_THR = 0.01


def _profile(n):
    return {500: "gaussian_n500", 1000: "binary_n1000_q50", 10000: "binary_n10000_q05"}.get(int(n))


def _lambda0_key(profile: str, b0: float):
    """Closest betas.json lambda0 key to e^{b0} (robust to the intercept's 4-dp rounding),
    returning (clean_lambda0_float, key_str)."""
    lam = math.exp(float(b0))
    keys = list(_BETAS[profile].keys())               # e.g. ["0.1","1","10"]
    key = min(keys, key=lambda k: abs(float(k) - lam))
    return float(key), key


def cell_meta(scoord: dict) -> dict:
    """Design / enrichment coordinates of a simulation cell -> flat metadata."""
    dargs = (scoord.get("design") or {}).get("arguments", {})
    enr = scoord.get("enrichment") or {}
    a = enr.get("arguments") or {}
    b0 = enr.get("intercept")
    if "causal_effects" in a:
        effs = [e for e in a["causal_effects"] if e != 0]
        lstar, beta, gap = len(effs), (effs[0] if effs else None), a.get("gap")
    else:
        beta = a.get("causal_effect")
        lstar, gap = (0 if beta in (None, 0, 0.0) else 1), None
    lambda0, T = None, None
    prof = _profile(dargs.get("n"))
    if b0 is not None and prof:
        lambda0, key = _lambda0_key(prof, b0)
        if beta is not None:
            row = _BETAS[prof][key]                    # {"4": beta, "8": ...}
            T = int(min(row, key=lambda t: abs(beta - row[t])))
    return {
        "design_n": dargs.get("n"), "design_p": dargs.get("p"),
        "density": dargs.get("density"), "corr": dargs.get("corr"), "rho": dargs.get("rho"),
        "b0": b0, "lambda0": lambda0, "b": beta, "Lstar": lstar, "T": T, "gap": gap,
    }


def _method_name(mname: str, L: int) -> str:
    """Method name -> comparison key, keeping the q-level and dropping the L tag (carried as
    fit_L). poisson_q2_ser_gibss -> q2_gibss; poisson_q1_L10_cavi_pv1 -> q1_cavi_pv1."""
    m = mname.replace("poisson_", "").replace(f"L{L}_", "").replace("ser_", "")
    return m


def reduction_frame(sc="021-poisson", which: str = "pip",
                    results_root: str = os.path.join(_TG_ROOT, "results")) -> pl.DataFrame:
    """Concatenate the `pip` or `cs` reduction parquets for a supercollection (or list), with
    cell metadata + method attached. Partial runs are fine: missing cells are skipped."""
    scs = [sc] if isinstance(sc, str) else list(sc)
    cfg = loader.load_config()
    frames, seen = [], set()
    for one in scs:
        for coll in loader.collection_method_pairs(cfg, one).values():
            for bh, mh, mname, mcoord, scoord in coll["pairs"]:
                if (bh, mh) in seen:
                    continue
                seen.add((bh, mh))
                p = f"{results_root}/by_batch/{bh}/fits/{mh}/reductions/{which}.parquet"
                if not os.path.exists(p):
                    continue
                L = int((mcoord.get("kwargs") or {}).get("L", 1) or 1)
                method = _method_name(mname, L)
                d = pl.read_parquet(p).with_columns(
                    pl.lit(method).alias("method"), pl.lit(L).alias("fit_L"),
                    **{k: pl.lit(v) for k, v in cell_meta(scoord).items()})
                frames.append(d)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def elbo_frame(sc="021-poisson", results_root: str = os.path.join(_TG_ROOT, "results")) -> pl.DataFrame:
    """Per-fit common-Q2 ELBO (`q2_elbo`), one row per (cell, method, fit_L, replicate), tagged
    with batch_hash + rep so methods can be PAIRED on the sim. Q1 arms have no q2_elbo (None)."""
    cfg = loader.load_config()
    rows, seen = [], set()
    for coll in loader.collection_method_pairs(cfg, sc).values():
        for bh, mh, mname, mcoord, scoord in coll["pairs"]:
            if (bh, mh) in seen:
                continue
            seen.add((bh, mh))
            f = f"{results_root}/by_batch/{bh}/fits/{mh}/fits.parquet"
            if not os.path.exists(f):
                continue
            df = pl.read_parquet(f)
            if "q2_elbo" not in df.columns:
                continue
            meta = cell_meta(scoord)
            L = int((mcoord.get("kwargs") or {}).get("L", 1) or 1)
            method = _method_name(mname, L)
            for r, elbo in enumerate(df["q2_elbo"].to_list()):
                rows.append({**meta, "method": method, "fit_L": L, "batch_hash": bh,
                             "rep": r, "q2_elbo": None if elbo is None else float(elbo)})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
