"""Helpers for `split_component.qmd`: three independent effects fit with L = 2.

One SuSiE component sits on effect A (column 0); the other has to split its alpha
between B and C (columns 1, 2). The split component's leave-one-out offset seen by the
clean component is then a two-point mixture, ``mu * x_B`` or ``mu * x_C`` with weight 1/2
each. CAVI-Q2 integrates that mixture exactly (``offset_integration="cf"``); gIBSS-Q2
plugs in its mean (``"none"``); gIBSS-GO replaces it with a moment-matched Gaussian
(``"lognormal"`` for Poisson, ``"probit_mixture"`` for logistic).

On iid data a strong B/C pair rarely splits: noise breaks the tie and the component
commits. ``simulate`` forces the split by swap-symmetrizing the design (n/2 rows, then the
same rows with columns 1 and 2 swapped, y duplicated), which makes lbf_B == lbf_C exactly.

Convergence: at pin b8f5a7b, `fit_glm_susie` checks GLM convergence on the alpha SKL
alone. Here alpha saturates in the first sweep, so every arm stops after 2 sweeps with
the ELBO hundreds of nats short of its fixed point. Every fit here therefore runs a fixed
``SWEEPS`` sweeps (``tol=-1`` never passes the check) and records the end-of-sweep states.
"""
from __future__ import annotations

import numpy as np
from gibss import glm
from gibss.elbo import compute_elbo_gaussian
from gibss.history import History
from gibss.methods import fit_glm_susie

ARMS = ["cavi", "gibss", "go"]
LABEL = {"cavi": "CAVI-Q2", "gibss": "gIBSS-Q2", "go": "gIBSS-GO"}
COLOR = {"cavi": "#D55E00", "gibss": "#0072B2", "go": "#000000"}
OFFSET = {
    "poisson": {"cavi": "cf", "gibss": "none", "go": "lognormal"},
    "logistic": {"cavi": "cf", "gibss": "none", "go": "probit_mixture"},
}
COMMON = dict(center=True, estimate_prior_variance=True, max_prior_variance=100.0,
              variational_family="gaussian")
SWEEPS = 100
A, B, C = 0, 1, 2


def simulate(rng: np.random.Generator, family: str, *, n: int, p: int, beta: float, b0: float):
    """Swap-symmetric design with causal columns A, B, C (effect ``beta`` each)."""
    half = rng.standard_normal((n // 2, p))
    swapped = half.copy()
    swapped[:, [B, C]] = half[:, [C, B]]
    X = np.vstack([half, swapped])
    eta = b0 + beta * half[:, [A, B, C]].sum(1)
    if family == "poisson":
        y = rng.poisson(np.exp(eta))
    else:
        y = rng.uniform(size=n // 2) < 1.0 / (1.0 + np.exp(-eta))
    return X, np.concatenate([y, y]).astype(float)


def fit(arm: str, family: str, X, y, *, L: int = 2, sweeps: int = SWEEPS):
    """Fixed-sweep fit; returns (final state, list of end-of-sweep states)."""
    hist = History(granularity="sweep")
    state = fit_glm_susie(X, y, family=family, L=L, offset_integration=OFFSET[family][arm],
                          tol=-1.0, max_iter=sweeps, record=hist, **COMMON)
    states = [r.state for r in hist.records if r.phase == "sweep"]
    return state, states


def elbo(X, y, state) -> float:
    return float(compute_elbo_gaussian(glm.prep_data(X, y, center=True), state,
                                       score_intercept="shared"))


def components(state) -> dict:
    """Label the two components clean (most alpha on A) and split (the other one)."""
    alphas = [np.asarray(e.alpha) for e in state.single_effects]
    clean = int(np.argmax([a[A] for a in alphas]))
    out = {}
    for role, l in (("clean", clean), ("split", 1 - clean)):
        e = state.single_effects[l]
        a, mu, var = alphas[l], np.asarray(e.mu), np.asarray(e.var)
        j = A if role == "clean" else (B if a[B] >= a[C] else C)
        out[role] = dict(alpha_A=float(a[A]), alpha_B=float(a[B]), alpha_C=float(a[C]),
                         mu=float(mu[j]), sd=float(np.sqrt(var[j])),
                         lbf=float(e.ser_log_bf), pv=float(e.prior_variance))
    return out


def trace(X, y, states) -> dict:
    """Per-sweep ELBO and the clean / split components' posterior mean at their top column."""
    rows = [components(s) for s in states]
    return dict(elbo=np.array([elbo(X, y, s) for s in states]),
                mu_clean=np.array([r["clean"]["mu"] for r in rows]),
                mu_split=np.array([r["split"]["mu"] for r in rows]))


def stable(elbos: np.ndarray, window: int = 20, tol: float = 1.0, head: np.ndarray | None = None,
           drop: float = 0.1) -> bool:
    """Did the ELBO settle (range under ``tol`` nats over the last ``window`` sweeps)? With
    ``head`` (the ELBO over the first sweeps), also require the final ELBO to be no more than
    a fraction ``drop`` of |best| below the best ELBO in ``head``: a gIBSS-GO fit that ran
    away and froze at a far worse state is not settled. The tolerance is relative because
    gIBSS is not an ELBO ascent: a converged gIBSS-Q2 fit can end 1-30 nats below an early
    peak."""
    tail = elbos[-window:]
    ok = bool(np.all(np.isfinite(tail)) and tail.max() - tail.min() < tol)
    if head is not None:
        best = np.nanmax(head)
        ok = ok and bool(tail[-1] >= best - drop * abs(best))
    return ok
