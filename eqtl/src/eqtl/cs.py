"""Credible sets straight from a fitted state's alpha, for every arm alike (gibss's
`summarize_fit` covers the GLM stack only; the linear stack has no response object).

Same definitions as SuSiE / `summarize_fit`: a level-`coverage` set is the smallest
prefix of features in decreasing alpha whose cumulative alpha reaches the level; purity is
the min absolute pairwise correlation among the set's columns of X (1 for a singleton).
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["credible_set", "purity", "component_table"]


def credible_set(alpha: np.ndarray, coverage: float = 0.95) -> tuple[np.ndarray, float]:
    order = np.argsort(-alpha, kind="stable")
    cum = np.cumsum(alpha[order])
    k = int(np.searchsorted(cum, coverage, side="left")) + 1
    k = min(k, alpha.shape[0])
    return order[:k], float(cum[k - 1])


def purity(X: np.ndarray, members: np.ndarray) -> float:
    if members.shape[0] <= 1:
        return 1.0
    c = np.corrcoef(X[:, members], rowvar=False)
    off = np.abs(c[np.triu_indices(members.shape[0], k=1)])
    off = off[np.isfinite(off)]
    return float(off.min()) if off.size else 1.0


def component_table(state, X: np.ndarray, *, coverage: float = 0.95) -> pl.DataFrame:
    """One row per single effect: component (1-based), cs_size, top, max_pip, cs_coverage,
    ser_log_bf, beta / beta_sd (top feature's posterior mean / sd), beta_ser (sum_j alpha_j
    mu_j), prior_variance, purity, cs (member list)."""
    rows = []
    for l, e in enumerate(state.single_effects, start=1):
        alpha = np.asarray(e.alpha, dtype=float)
        mu = np.asarray(e.mu, dtype=float)
        var = np.asarray(e.var, dtype=float)
        members, cov = credible_set(alpha, coverage)
        top = int(members[0])
        rows.append({
            "component": l,
            "cs_size": int(members.shape[0]),
            "top": top,
            "max_pip": float(alpha[top]),
            "cs_coverage": cov,
            "ser_log_bf": float(e.ser_log_bf),
            "beta": float(mu[top]),
            "beta_sd": float(np.sqrt(var[top])),
            "beta_ser": float(np.sum(alpha * mu)),
            "prior_variance": float(e.prior_variance),
            "purity": purity(X, members),
            "cs": [int(j) for j in members],
        })
    return pl.DataFrame(rows, schema_overrides={"cs": pl.List(pl.Int64)})
