"""Produce the artifact `adjusted_vs_gibss.qmd` renders from (see `adjusted_lib.py`).

Three blocks, every dataset fit by every arm:

  exo_ser   exogenous known V, L=1 SER.  family x structure(homo/hetero) x V x b0 x signal
            (signal / null mean) x seed; arms gibss / adjusted / adjusted_nr / exact.
  exo_multi exogenous known V, L=3 (3 causal columns), homoskedastic V in {0, 1, 4}.
  endo      endogenous V, L=5 front-door fits. family x n x rho x seed; arms gibss /
            adjusted / cavi (cf_cavi_q1). Realized message variance recorded per fit.

Per-dataset content-addressed cache under data/cache/adjusted/ (a rerun resumes); the
artifact collects per-arm gap metrics for every dataset plus the full per-fit summaries of
seed 0 of every cell (for the worked examples). Bump VERSION on a logic change.

    uv run python analysis/gibss_vs_cavi/generate_adjusted.py --workers 8
    uv run python analysis/gibss_vs_cavi/generate_adjusted.py --smoke     # 1 seed, tiny grid
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import itertools
import multiprocessing as mp
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

VERSION = 1
CACHE_DIR = os.path.join(HERE, "data", "cache", "adjusted")
ARTIFACT = os.path.join(HERE, "data", "adjusted_artifact.pkl")

# --------------------------------------------------------------------------- grids
EXO = dict(
    n=800, p=20, causal=(7,), seeds=tuple(range(20)), n_sweeps=20, pv=1.0,
    V=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    b0={"logistic": (0.0, -1.0, -2.0, -3.0), "poisson": (-1.5, -0.5, 0.5)},
    beta={"logistic": 1.0, "poisson": 0.6},
    structures=("homo", "hetero"),
    signals=("signal", "null"),  # null = beta 0: any declared CS is a false positive
)
EXO_MULTI = dict(
    n=800, p=20, causal=(3, 10, 17), L=3, seeds=tuple(range(20)), n_sweeps=20, pv=1.0,
    V=(0.0, 1.0, 4.0), b0={"logistic": -1.0, "poisson": -0.5},
    beta={"logistic": 1.0, "poisson": 0.6}, structure="homo",
)
ENDO = dict(
    p=50, causal=(10, 25, 40), L=5, seeds=tuple(range(10)), n_sweeps=30, pv=1.0,
    n=(200, 500, 2000), rho=(0.5, 0.9, 0.99),
    b0={"logistic": -2.0, "poisson": -0.7}, beta={"logistic": 0.8, "poisson": 0.45},
)
FAMILIES = ("logistic", "poisson")


def cells():
    """Every (block, cell dict) in the experiment. The cell dict is the cache key and the
    simulate/fit arguments; `seed` is part of it."""
    out = []
    for fam in FAMILIES:
        for st, V, b0, sig, seed in itertools.product(
            EXO["structures"], EXO["V"], EXO["b0"][fam], EXO["signals"], EXO["seeds"]
        ):
            if sig == "null" and st == "homo":
                continue  # the null-mean question is the heteroskedastic one
            out.append(("exo_ser", dict(
                family=fam, structure=st, V=V, b0=b0, signal=sig, seed=seed, L=1,
                beta=EXO["beta"][fam] if sig == "signal" else 0.0, causal=EXO["causal"],
                n=EXO["n"], p=EXO["p"], n_sweeps=EXO["n_sweeps"], pv=EXO["pv"])))
        for V, seed in itertools.product(EXO_MULTI["V"], EXO_MULTI["seeds"]):
            out.append(("exo_multi", dict(
                family=fam, structure=EXO_MULTI["structure"], V=V, b0=EXO_MULTI["b0"][fam],
                signal="signal", seed=seed, L=EXO_MULTI["L"], beta=EXO_MULTI["beta"][fam],
                causal=EXO_MULTI["causal"], n=EXO_MULTI["n"], p=EXO_MULTI["p"],
                n_sweeps=EXO_MULTI["n_sweeps"], pv=EXO_MULTI["pv"])))
        for n, rho, seed in itertools.product(ENDO["n"], ENDO["rho"], ENDO["seeds"]):
            out.append(("endo", dict(
                family=fam, n=n, rho=rho, seed=seed, L=ENDO["L"], b0=ENDO["b0"][fam],
                beta=ENDO["beta"][fam], causal=ENDO["causal"], p=ENDO["p"],
                n_sweeps=ENDO["n_sweeps"], pv=ENDO["pv"])))
    return out


def smoke_cells():
    allc = cells()
    keep = []
    for block, c in allc:
        if c["seed"] != 0:
            continue
        if block == "exo_ser" and not (c["V"] in (0.0, 4.0) and c["b0"] in (-1.0, -0.5)):
            continue
        if block == "exo_multi" and c["V"] != 4.0:
            continue
        if block == "endo" and not (c["n"] == 500 and c["rho"] == 0.9):
            continue
        keep.append((block, c))
    return keep


# --------------------------------------------------------------------------- one cell
def _run_cell(job):
    block, cell = job
    from simcache import cached
    import adjusted_lib as al

    def compute():
        t0 = time.time()
        if block == "endo":
            X, y = al.simulate_endo(cell["family"], cell["n"], cell["rho"], cell["b0"],
                                    cell["beta"], cell["causal"], cell["p"], cell["seed"])
            recs, times = {}, {}
            for arm in al.ENDO_ARMS:
                t = time.time()
                st = al.fit_endo(X, y, cell["family"], arm, L=cell["L"], pv=cell["pv"],
                                 n_sweeps=cell["n_sweeps"])
                recs[arm], times[arm] = al.summarize(st), time.time() - t
            ref = "cavi"
        else:
            X, y, v = al.simulate_exo(cell["family"], cell["structure"], cell["V"], cell["b0"],
                                      cell["beta"], cell["causal"], cell["n"], cell["p"],
                                      cell["seed"])
            recs, times = {}, {}
            for arm in al.EXO_ARMS:
                t = time.time()
                st = al.fit_exo(X, y, v, cell["family"], arm, L=cell["L"], pv=cell["pv"],
                                n_sweeps=cell["n_sweeps"])
                recs[arm], times[arm] = al.summarize(st), time.time() - t
            ref = "exact"
        gaps = {arm: al.gap(recs[arm], recs[ref]) for arm in recs if arm != ref}
        return dict(block=block, cell=cell, recs=recs, gaps=gaps, times=times,
                    wall=time.time() - t0)

    from simcache import key_of
    key = key_of(dict(VERSION=VERSION, block=block, **cell))
    return cached(dict(VERSION=VERSION, block=block, **cell), compute,
                  os.path.join(CACHE_DIR, f"{block}_{key}.pkl"))


# --------------------------------------------------------------------------- driver
def build(workers: int, smoke: bool):
    jobs = smoke_cells() if smoke else cells()
    print(f"{len(jobs)} datasets ({'smoke' if smoke else 'full'}), {workers} workers", flush=True)
    t0 = time.time()
    results = []
    if workers <= 1:
        for i, job in enumerate(jobs):
            results.append(_run_cell(job))
            if i % 50 == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:6.0f}s", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_cell, jobs, chunksize=4)):
                results.append(r)
                if i % 100 == 0:
                    print(f"  {i+1}/{len(jobs)}  {time.time()-t0:6.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)

    # flatten: one row per (dataset, arm) with the gap metrics + per-row V summary;
    # keep the full summaries for seed 0 as worked examples.
    rows, examples = [], []
    for r in results:
        c = r["cell"]
        for arm, g in r["gaps"].items():
            rows.append(dict(block=r["block"], arm=arm, **c, **g,
                             logbf=float(r["recs"][arm]["ser_log_bf"].sum()),
                             logbf_ref=float(r["recs"][
                                 "cavi" if r["block"] == "endo" else "exact"]["ser_log_bf"].sum()),
                             msg_var_mean=r["recs"][arm]["msg_var_mean"],
                             msg_var_p95=r["recs"][arm]["msg_var_p95"],
                             msg_var_max=r["recs"][arm]["msg_var_max"],
                             secs=r["times"][arm]))
        ref = "cavi" if r["block"] == "endo" else "exact"
        rows.append(dict(block=r["block"], arm=ref, **c, alpha_gap=0.0, pip_gap=0.0,
                         logbf_gap=0.0, eb_gap=0.0, sd_ratio=1.0, icpt_gap=0.0,
                         top_feature=int(np.argmax(r["recs"][ref]["pip"])),
                         logbf=float(r["recs"][ref]["ser_log_bf"].sum()),
                         logbf_ref=float(r["recs"][ref]["ser_log_bf"].sum()),
                         msg_var_mean=r["recs"][ref]["msg_var_mean"],
                         msg_var_p95=r["recs"][ref]["msg_var_p95"],
                         msg_var_max=r["recs"][ref]["msg_var_max"], secs=r["times"][ref]))
        if c["seed"] == 0:
            examples.append(dict(block=r["block"], cell=c, recs=r["recs"]))
    art = dict(version=VERSION, grids=dict(EXO=EXO, EXO_MULTI=EXO_MULTI, ENDO=ENDO),
               rows=rows, examples=examples, smoke=smoke)
    path = ARTIFACT.replace(".pkl", "_smoke.pkl") if smoke else ARTIFACT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(art, f)
    print(f"wrote {path}: {len(rows)} rows, {len(examples)} examples")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    build(a.workers, a.smoke)
