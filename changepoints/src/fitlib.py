"""gibss GLM SuSiE on a step basis, reduced to numpy + a credible-set table."""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from stepbasis import cs_interval, standardize_columns, step_basis


def fit_step_susie(y: np.ndarray, *, L: int, family: str = "poisson",
                   method: str | None = None, standardize: bool = False,
                   estimate_prior_variance: bool = True, prior_variance: float = 1.0,
                   offset=0.0, coverage: float = 0.95, max_iter: int = 100,
                   X: np.ndarray | None = None, trials=None) -> dict:
    """Fit and return a host-numpy record (no jax objects, picklable)."""
    from gibss.methods import fit_glm_susie

    n = len(y)
    X = step_basis(n) if X is None else X
    scale = np.ones(X.shape[1])
    if standardize:
        X, scale = standardize_columns(X)
    kwargs = {"L": L, "method": method, "family": family, "trials": trials, "offset": offset, "center": True,
                  "estimate_intercept": True, "estimate_prior_variance": estimate_prior_variance,
                  "prior_variance": prior_variance, "max_iter": max_iter}
    t0 = time.time()
    state = fit_glm_susie(X, np.asarray(y, dtype=float), **kwargs)
    elapsed = time.time() - t0
    fs = state.family_state
    alpha = np.asarray(state.alpha)
    mu = np.asarray(state.mu)
    var = np.asarray(state.var) if hasattr(state, "var") else np.stack(
        [np.asarray(e.var) for e in state.single_effects])
    post_mean = np.sum(alpha * mu, axis=0)  # on the (possibly scaled) columns
    Xc = X - X.mean(axis=0)
    eta = float(fs.intercept_value) + Xc @ post_mean
    cs = state.get_credible_sets(coverage)
    return {
        "L": L, "family": family, "method": method or family, "standardize": standardize,
        "estimate_prior_variance": estimate_prior_variance,
        "prior_variance_arg": prior_variance, "coverage": coverage,
        "n": n, "p": X.shape[1], "seconds": elapsed,
        "converged": bool(state.converged), "n_iter": int(state.n_iter),
        "alpha": alpha, "mu": mu / scale, "var": var / scale**2, "scale": scale,
        "ser_log_bf": np.asarray(state.ser_log_bf, dtype=float),
        "prior_variance": np.array([float(e.prior_variance) for e in state.single_effects]),
        "intercept": float(fs.intercept_value), "eta": np.asarray(eta),
        "pip": np.asarray(state.pip), "cs": [tuple(int(i) for i in c) for c in cs],
    }


def cs_table(fit: dict, labels: np.ndarray, *, min_log_bf: float | None = None,
             tag: str = "") -> pl.DataFrame:
    """One row per component. A step column j is the jump between labels[j] and
    labels[j+1]; the CS hull is reported in those label units."""
    rows = []
    for l in range(fit["L"]):
        c = fit["cs"][l]
        a = fit["alpha"][l]
        lo, hi, width = cs_interval(c)
        top = int(np.argmax(a))
        lbf = float(fit["ser_log_bf"][l])
        if min_log_bf is not None and lbf < min_log_bf:
            continue
        rows.append({
            "tag": tag, "L": fit["L"], "component": l + 1, "ser_log_bf": lbf,
            "declared": lbf >= 2.0,
            "cs_size": len(c), "cs_hull_width": width,
            "cs_start": float(labels[lo + 1]), "cs_end": float(labels[hi + 1]),
            "top": float(labels[top + 1]), "max_pip": float(a[top]),
            "cs_mass": float(a[list(c)].sum()),
            "beta": float(fit["mu"][l, top]), "beta_sd": float(np.sqrt(fit["var"][l, top])),
            "beta_ser": float(np.sum(a * fit["mu"][l])),
            "rate_ratio": float(np.exp(fit["mu"][l, top])),
            "prior_variance": float(fit["prior_variance"][l]),
        })
    return pl.DataFrame(rows)
