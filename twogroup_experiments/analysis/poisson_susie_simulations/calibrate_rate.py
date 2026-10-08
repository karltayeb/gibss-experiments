"""Calibrate betas for 025 (Poisson background-rate grid): E[LRT] = T per (design, lambda0, T).

One causal column, profiled intercept, y ~ Poisson(exp(b0 + beta x)) with b0 = log(lambda0):

* gaussian  N(0,1) column, n=500 (the AR1 design's marginal is N(0,1) whatever rho):
            Monte Carlo Newton LRT (calibration.e_lrt_gaussian_mc).
* binary    0/1 column of binary_markov_X(n=1000, density=0.1): set size ~ Binomial(n, 0.1),
            two-group Poisson deviance (calibration.e_lrt_binary).
* block     a column at depth DEPTH of binary_attrition_X(n=1000, density=0.5, drop=0.45):
            density 0.5 * 0.55**DEPTH, same two-group deviance.

Writes betas_rate.json keyed betas[design][lambda0][T]. Entries already in the file are kept, so
adding a lambda0 or T only calibrates the new cells (the MC seeds are deterministic either way).

    uv run python analysis/poisson_susie_simulations/calibrate_rate.py
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from calibration import e_lrt_binary, e_lrt_gaussian_mc

HERE = Path(__file__).resolve().parent
OUT = HERE / "betas_rate.json"

LAMBDA0 = [0.01, 0.1, 1.0, 10.0, 100.0]
TARGETS = [8, 12, 16, 20, 24]
GAUSS_N = 500
BIN_N, BIN_DENSITY = 1000, 0.1
BLOCK_N, BLOCK_ROOT, BLOCK_DROP, DEPTH = 1000, 0.5, 0.45, 3
BLOCK_DENSITY = BLOCK_ROOT * (1 - BLOCK_DROP) ** DEPTH
BETA_HI = 12.0


def e_lrt(design: str, b0: float, beta: float) -> float:
    if design == "gaussian":
        return e_lrt_gaussian_mc(GAUSS_N, b0, beta, n_rep=2000)
    density = BIN_DENSITY if design == "binary" else BLOCK_DENSITY
    return e_lrt_binary(BIN_N, density, b0, beta)


def invert(args) -> tuple[str, float, int, float]:
    """Smallest beta > 0 with E[LRT] = T, by bisection to 1e-3 (E[LRT] increases in beta)."""
    design, lam0, t = args
    b0 = math.log(lam0)
    if e_lrt(design, b0, BETA_HI) < t:
        raise ValueError(f"T={t} unreachable for {design} lambda0={lam0} at beta<={BETA_HI}")
    lo, hi = 0.0, BETA_HI
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if e_lrt(design, b0, mid) < t else (lo, mid)
    return design, lam0, t, round(0.5 * (lo + hi), 4)


def main() -> None:
    betas: dict = json.loads(OUT.read_text())["betas"] if OUT.exists() else {}
    jobs = [(d, l, t) for d in ("gaussian", "binary", "block") for l in LAMBDA0 for t in TARGETS
            if str(t) not in betas.get(d, {}).get(f"{l:g}", {})]
    with ProcessPoolExecutor(max_workers=10) as ex:
        for design, lam0, t, beta in ex.map(invert, jobs):
            betas.setdefault(design, {}).setdefault(f"{lam0:g}", {})[str(t)] = beta
    for design, tab in betas.items():
        betas[design] = tab = dict(sorted(tab.items(), key=lambda kv: float(kv[0])))
        for lam0, row in tab.items():
            print(f"{design:9s} lambda0={lam0:>5s} " + " ".join(f"T{t}={b:.4f}" for t, b in row.items()))
    meta = {"lambda0": LAMBDA0, "targets": TARGETS, "gauss_n": GAUSS_N, "bin_n": BIN_N,
            "bin_density": BIN_DENSITY, "block_depth": DEPTH, "block_density": BLOCK_DENSITY}
    OUT.write_text(json.dumps({"_meta": meta, "betas": betas}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
