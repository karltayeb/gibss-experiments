"""Map effect size <-> expected univariate logBF via the exact expected LRT
(Λ = n·MI), monotone in b. glmbf-free (numpy + scipy only).

Λ(b) = n·[H(p̄) − E_x H(p_x)],  p_x = σ(b0 + b·x),  expectations over the causal
feature's marginal. Monotone in b, saturating at the deviance n·H(p̄_∞), so every
grid rung is reachable by a 1-D root find — no Wald ceiling (see
notes/2026-06-27-univariate-logbf-z-grid.md).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_GRID = (4, 8, 16, 32, 64)


def _gaussian():
    x = np.linspace(-8.0, 8.0, 1200)
    w = np.exp(-x * x / 2.0)
    return x, w / w.sum()


def _uniform():
    x = np.linspace(-1.0, 1.0, 1200)
    return x, np.full(1200, 1.0 / 1200)


def _binary(q):
    return np.array([-1.0, 1.0]), np.array([1.0 - q, q])


MARGINALS = {
    "gaussian": _gaussian(),
    "uniform": _uniform(),
    "binary q=0.5": _binary(0.5),
    "binary q=0.1": _binary(0.1),
}


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _H(p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def expected_lrt(design: str, b0: float, b: float, n: int) -> float:
    """Λ = n·MI(x;y). Monotone in |b|."""
    x, pw = MARGINALS[design]
    px = _sigmoid(b0 + b * x)
    pbar = float(np.sum(pw * px))
    return float(n) * (_H(pbar) - float(np.sum(pw * _H(px))))


def solve_b(design: str, target_logbf: float, b0: float, n: int) -> float:
    """Effect size with Λ(b) = target_logbf (positive-b branch)."""
    if expected_lrt(design, b0, 60.0, n) < target_logbf:
        raise ValueError(
            f"target logBF {target_logbf} exceeds deviance ceiling for "
            f"{design!r}, b0={b0}, n={n}"
        )
    return float(brentq(lambda bb: expected_lrt(design, b0, bb, n) - target_logbf, 1e-6, 60.0))


def nearest_logbf(design: str, b0: float, b: float, n: int, grid=_GRID) -> int:
    """The grid rung whose Λ is closest to Λ(b). 0 for the null (b==0)."""
    if b == 0.0:
        return 0
    lam = expected_lrt(design, b0, abs(b), n)
    return int(min(grid, key=lambda g: abs(g - lam)))


def frozen_table(n: int = 1000, b0s=(0.0, -2.0, -4.0), grid=_GRID) -> list[dict]:
    """Generator for the values baked into library.yaml."""
    rows = []
    for design in MARGINALS:
        for b0 in b0s:
            rec = {"design": design, "b0": b0}
            for g in grid:
                rec[f"L{g}"] = round(solve_b(design, g, b0, n), 3)
            rows.append(rec)
    return rows


DESIGN_KEY = {
    ("gaussian_markov_X", None): "gaussian",
    ("uniform_markov_X", None): "uniform",
    ("binary_markov_X", 0.5): "binary q=0.5",
    ("binary_markov_X", 0.1): "binary q=0.1",
}

# Frozen exact-LRT effect sizes at n=1000 (regenerate with frozen_table()).
FROZEN_B = {
    "gaussian": {
        0.0:  {4: 0.180, 8: 0.256, 16: 0.366, 32: 0.531, 64: 0.789},
        -2.0: {4: 0.275, 8: 0.387, 16: 0.545, 32: 0.766, 64: 1.091},
        -4.0: {4: 0.621, 8: 0.830, 16: 1.086, 32: 1.405, 64: 1.829},
    },
    "uniform": {
        0.0:  {4: 0.311, 8: 0.441, 16: 0.628, 32: 0.902, 64: 1.316},
        -2.0: {4: 0.477, 8: 0.675, 16: 0.953, 32: 1.347, 64: 1.910},
        -4.0: {4: 1.124, 8: 1.543, 16: 2.077, 32: 2.735, 64: 3.537},
    },
    "binary q=0.5": {
        0.0:  {4: 0.179, 8: 0.254, 16: 0.361, 32: 0.514, 64: 0.740},
        -2.0: {4: 0.277, 8: 0.392, 16: 0.556, 32: 0.793, 64: 1.137},
        -4.0: {4: 0.674, 8: 0.951, 16: 1.337, 32: 1.853, 64: 2.502},
    },
    "binary q=0.1": {
        0.0:  {4: 0.299, 8: 0.425, 16: 0.607, 32: 0.874, 64: 1.286},
        -2.0: {4: 0.440, 8: 0.613, 16: 0.851, 32: 1.180, 64: 1.649},
        -4.0: {4: 0.974, 8: 1.313, 16: 1.752, 32: 2.316, 64: 3.039},
    },
}


if __name__ == "__main__":
    for r in frozen_table():
        print(r)
