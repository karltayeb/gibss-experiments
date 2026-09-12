"""Effect-size calibration for the Poisson-SuSiE simulations (021).

The Poisson companion to ``analysis/logistic_susie_simulations/calibration.py``. The
simulations regress well-specified counts ``y_i ~ Poisson(exp(b0 + beta * x_i))`` onto a
design column ``x``. We pick the causal coefficient ``beta`` so the UNIVARIATE
detectability of the causal column hits a target ``T = E[LRT]`` (expected likelihood-ratio
statistic for ``H0: beta = 0`` vs ``H1: beta`` free). ``T`` on the LRT scale is roughly
``2 x E[log BF]`` (Wakefield), so ``T in {4, 8, 16, 32}`` targets ``log BF ~ {2, 4, 8, 16}``.

BASELINE RATE AXIS. The intercept is the baseline log-rate ``b0 = log(lambda0)``, so the
mean count of a null row is ``lambda0``. We sweep ``lambda0 in {0.1, 1, 10}`` (evenly
spaced in ``b0``: {-2.30, 0, +2.30}). At MATCHED T the leading-order information
``T ~ beta^2 * lambda0 * n`` means rate and effect trade off, so this axis is a DISCRETENESS
axis, not a power axis: low ``lambda0`` = skewed, mostly-zero counts (where the SER
approximations are stressed), high ``lambda0`` = near-Gaussian by the CLT.

INTERCEPT CONVENTION (profiled): H0 estimates the intercept (the null log-rate MLE
``log(ybar)``), matching a real ``y ~ 1 + x`` marginal regression and the fitted SER, whose
``feature_log_bf`` is taken against an estimated (shared) intercept.

Two design profiles enter only through the causal column's marginal law:

* ``gaussian`` (AR1 ``gaussian_markov_X``): stationary N(0,1) column, ``n`` rows. No closed
  form -> Monte-Carlo (draw x, y, fit the profiled Poisson MLE by Newton). ``rho`` is
  irrelevant to a single column (stationary marginal).
* ``binary`` (``binary_markov_X``): a 0/1 indicator at membership rate ``density``. The data
  collapse to the two group totals ``S1 = sum_{x=1} y ~ Poisson(m * exp(b0+beta))`` and
  ``S0 = sum_{x=0} y ~ Poisson((n-m) * exp(b0))``, and the profiled LRT is the Poisson
  two-group deviance (the count analogue of the 2x2 G-test). E[LRT] is Monte-Carlo over
  ``(m, S1, S0)`` -- m ~ Binomial(n, density) (a random causal column), then the two Poisson
  totals. ``corr`` is irrelevant to a single column.

Pure NumPy / SciPy - no jax, no gibss, no design generation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from scipy.special import xlogy


# --------------------------------------------------------------------------- #
# Binary design: profiled two-group Poisson deviance, Monte-Carlo over totals   #
# --------------------------------------------------------------------------- #
def _poisson_deviance_2group(S1, m, S0, M):
    """Profiled LRT (deviance) for a 2-group Poisson table: in-set (m rows, total S1) vs
    out-set (M rows, total S0). H1 fits a rate per group (S1/m, S0/M); H0 one common rate
    ((S1+S0)/n). ``log y!`` cancels; the ``-lambda`` terms cancel because both fits match the
    total count. Additive ``xlogy`` form so empty cells / all-zero (LRT=0) are exact.

        LRT = 2[ S1 log(S1/m) + S0 log(S0/M) - (S1+S0) log((S1+S0)/n) ].
    """
    n = m + M
    tot = S1 + S0
    return 2.0 * (xlogy(S1, S1 / m) + xlogy(S0, S0 / M) - xlogy(tot, tot / n))


def e_lrt_binary(n, density, b0, beta, *, n_rep=40000, seed=0):
    """E[LRT] for a binary column at baseline log-rate ``b0`` and effect ``beta``. Monte-Carlo
    over a random causal column: per rep draw the set size ``m ~ Binomial(n, density)`` (>=1),
    then the two group totals ``S1 ~ Poisson(m e^{b0+beta})``, ``S0 ~ Poisson((n-m) e^{b0})``,
    and average the profiled deviance. Deterministic given (b0, beta) via ``seed`` so the
    curve is smooth for bisection."""
    if beta == 0.0:
        return 0.0
    rng = np.random.default_rng(_seed_for("binary", b0, beta, seed))
    m = rng.binomial(n, density, size=n_rep)
    m = np.clip(m, 1, n - 1)                       # keep both groups non-empty
    M = n - m
    lam_in = np.exp(b0 + beta)
    lam_out = np.exp(b0)
    S1 = rng.poisson(m * lam_in)
    S0 = rng.poisson(M * lam_out)
    return float(np.mean(_poisson_deviance_2group(S1, m, S0, M)))


# --------------------------------------------------------------------------- #
# Gaussian AR1 design: Monte-Carlo (profiled Poisson MLE by Newton)            #
# --------------------------------------------------------------------------- #
def _seed_for(profile, b0, beta, salt=0):
    """Deterministic per-point seed (no global RNG)."""
    return int(hash((profile, round(float(b0), 6), round(float(beta), 6), int(salt))) & 0xFFFFFFFF)


def _poisson_ll(eta, y):
    return float(np.sum(y * eta - np.exp(eta)))    # drops log y! (cancels in the LRT)


def _newton_poisson(y, X, n_iter=50, tol=1e-9, eta_clip=30.0):
    """MLE for Poisson log-lik with design columns X (n x d). Concave; damped Newton (IRLS
    weight ``lambda = e^eta``) converges in a few steps. ``eta`` clipped to guard overflow."""
    d = X.shape[1]
    b = np.zeros(d)
    b[0] = np.log(max(y.mean(), 1e-6))             # warm-start intercept at log(ybar)
    for _ in range(n_iter):
        eta = np.clip(X @ b, -eta_clip, eta_clip)
        lam = np.exp(eta)
        g = X.T @ (y - lam)
        H = (X * lam[:, None]).T @ X + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b, _poisson_ll(np.clip(X @ b, -eta_clip, eta_clip), y)


def e_lrt_gaussian_mc(n, b0, beta, *, n_rep=3000, seed=None):
    """E[LRT] for a Gaussian N(0,1) column. Draw x ~ N(0,1)^n, y ~ Poisson(exp(b0+beta x)),
    profiled LRT = 2[max_{b0',b'} ll - max_{b0'} ll] with H0 the intercept-only MLE
    (log ybar). Deterministic given (b0, beta)."""
    if beta == 0.0:
        return 0.0
    rng = np.random.default_rng(_seed_for("gaussian", b0, beta) if seed is None else seed)
    lrt = np.empty(n_rep)
    for r in range(n_rep):
        x = rng.standard_normal(n)
        y = rng.poisson(np.exp(np.clip(b0 + beta * x, -50, 30))).astype(float)
        ybar = max(y.mean(), 1e-12)
        ll0 = _poisson_ll(np.full(n, np.log(ybar)), y)
        _, ll1 = _newton_poisson(y, np.column_stack([np.ones(n), x]))
        lrt[r] = 2.0 * (ll1 - ll0)
    return float(lrt.mean())


# --------------------------------------------------------------------------- #
# Unified interface + inversion                                               #
# --------------------------------------------------------------------------- #
PROFILES = {
    "gaussian_n500":     {"kind": "gaussian", "n": 500,   "label": "AR1 Gaussian (n=500)"},
    "binary_n1000_q50":  {"kind": "binary",   "n": 1000,  "density": 0.5,  "label": "Bin-AR (n=1000, q=0.5)"},
    "binary_n10000_q05": {"kind": "binary",   "n": 10000, "density": 0.05, "label": "Bin-AR (n=10000, q=0.05)"},
}
LAMBDA0 = [0.1, 1.0, 10.0]           # baseline mean count; b0 = log(lambda0)
TARGETS = [4, 8, 16, 32]


def e_lrt(profile, b0, beta):
    p = PROFILES[profile]
    if p["kind"] == "binary":
        return e_lrt_binary(p["n"], p["density"], b0, beta)
    return e_lrt_gaussian_mc(p["n"], b0, beta)


def invert_beta_for_lrt(profile, b0, target, *, beta_hi=10.0, tol=1e-3):
    """Smallest beta > 0 with E[LRT] = target, by bisection (E[LRT] increases in beta)."""
    f_hi = e_lrt(profile, b0, beta_hi)
    if f_hi < target:
        raise ValueError(
            f"target {target} unreachable at beta<={beta_hi} for {profile} b0={b0:.3f} "
            f"(max {f_hi:.1f})."
        )
    lo, hi = 0.0, beta_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if e_lrt(profile, b0, mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def calibrate():
    """Return {profile: {lambda0: {T: beta}}} over PROFILES x LAMBDA0 x TARGETS."""
    out = {}
    for prof in PROFILES:
        out[prof] = {}
        for lam0 in LAMBDA0:
            b0 = float(np.log(lam0))
            out[prof][lam0] = {t: invert_beta_for_lrt(prof, b0, t) for t in TARGETS}
    return out


if __name__ == "__main__":
    import json
    import os

    tab = calibrate()
    hdr = f"{'profile':20s} {'lambda0':>7s} {'b0':>6s} " + " ".join(f"T={t:<2d}" for t in TARGETS)
    print(hdr)
    print("-" * len(hdr))
    for prof in PROFILES:
        for lam0 in LAMBDA0:
            b0 = np.log(lam0)
            betas = [tab[prof][lam0][t] for t in TARGETS]
            print(f"{prof:20s} {lam0:>7.2f} {b0:>6.2f} " + " ".join(f"{b:6.3f}" for b in betas))

    path = os.path.join(os.path.dirname(__file__), "betas.json")
    payload = {
        prof: {f"{lam0:g}": {str(t): tab[prof][lam0][t] for t in TARGETS} for lam0 in LAMBDA0}
        for prof in PROFILES
    }
    with open(path, "w") as f:
        json.dump({"lambda0": LAMBDA0, "targets": TARGETS, "betas": payload}, f, indent=2)
    print(f"\nwrote {path}")
