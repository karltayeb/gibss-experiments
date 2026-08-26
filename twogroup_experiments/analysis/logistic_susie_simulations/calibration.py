"""Analytic / semi-analytic effect-size calibration for the logistic-SuSiE simulations.

The simulations regress a binary response ``z_i ~ Bernoulli(sigmoid(b0 + beta * x_i))``
onto a design column ``x``. We pick the causal coefficient ``beta`` so that the
UNIVARIATE detectability of the causal column hits a target ``T = E[LRT]`` (expected
likelihood-ratio statistic for ``H0: beta = 0`` vs ``H1: beta`` free). ``T`` on the LRT
scale is roughly ``2 x E[log BF]`` (Wakefield), so ``T in {4, 8, 16, 32}`` targets
Bayesian evidence ``log BF ~ {2, 4, 8, 16}`` nats.

Why LRT and not the Bayes factor: the LRT is monotone in ``beta`` and cheap. The
binary designs admit an EXACT expectation (below); the Wald/ABF route is avoided
because the logistic Wald statistic has Hauck-Donner turnover in the rare+strong
corner, which breaks a monotone inversion.

INTERCEPT CONVENTION (``intercept="profiled"``, the default): the univariate model
ESTIMATES the intercept under both hypotheses, matching a real ``z ~ 1 + x`` marginal
regression and the fitted SER (whose ``feature_log_bf`` is taken against an estimated
intercept). This matters: with the intercept fixed at the true ``b0`` instead, the
dense ``q=0.5`` design reads ~2x higher LRT because a free intercept otherwise soaks up
the average elevation. ``intercept="fixed"`` keeps ``b0`` known (the in-set-only
reduction) and is retained only as a reference.

Three design "profiles" enter only through the causal column's marginal law:

* ``gaussian`` (AR1 ``gaussian_markov_X``): stationary N(0,1) marginal, ``n`` rows.
  No closed form for E[LRT]; use Monte-Carlo (exact in expectation). rho is
  irrelevant to a single column (stationary marginal).
* ``binary`` (``binary_markov_X``): column is a 0/1 indicator with membership rate
  ``density``. The response reduces to the 2x2 table (in/out set x hit/miss), so
  E[LRT] is an EXACT finite sum over the set size ``m ~ Binomial(n, density)``, the
  in-set hits ``a ~ Binomial(m, p1)`` and out-set hits ``c ~ Binomial(n-m, p0)``
  (profiled: LRT = G-test of the 2x2; fixed: in-set Binomial deviance vs known p0).
  ``corr`` is irrelevant to a single column.

Everything here is pure NumPy / SciPy - no jax, no gibss, no design generation.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom
from scipy.special import roots_hermitenorm, xlogy, logsumexp


def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


# --------------------------------------------------------------------------- #
# Binary design: exact expectation (in-set Binomial deviance, b0 fixed)        #
# --------------------------------------------------------------------------- #
def _lrt_binomial_fixed(a, m, p0):
    """LRT of an observed in-set rate a/m against the KNOWN null rate p0 (b0 fixed).

    Binomial deviance ``2[a log(a/(m p0)) + (m-a) log((m-a)/(m(1-p0)))]``. ``xlogy``
    handles the a=0 and a=m endpoints (0 log 0 -> 0) exactly.
    """
    return 2.0 * (xlogy(a, a / (m * p0)) + xlogy(m - a, (m - a) / (m * (1.0 - p0))))


def _g_test_2x2(a, m, c, M):
    """LRT (G-test) for a 2x2 table: in-set (m genes, a hits) vs out-set (M genes, c hits).

    The 2-parameter logistic z ~ 1 + x saturates the table; H0 (slope 0) fits one common
    rate. LRT = 2(ll_saturated - ll_common). ``a``/``c`` are integer grids (broadcastable);
    ``xlogy`` handles empty cells. This is the profiled-intercept univariate LRT.
    """
    n = m + M
    H = a + c                                   # total hits
    ll1 = xlogy(a, a / m) + xlogy(m - a, (m - a) / m) + xlogy(c, c / M) + xlogy(M - c, (M - c) / M)
    ll0 = xlogy(H, H / n) + xlogy(n - H, (n - H) / n)
    return 2.0 * (ll1 - ll0)


def _e_lrt_binary_fixed_m(m, M, b0, beta, intercept, *, c_sd=7.0):
    """E[LRT | set size m], exact over in-set hits a; out-set hits c only for profiled."""
    p0, p1 = sigmoid(b0), sigmoid(b0 + beta)
    a = np.arange(0, m + 1)
    wa = binom.pmf(a, m, p1)
    if intercept == "fixed":
        return float(np.sum(wa * _lrt_binomial_fixed(a, m, p0)))
    # profiled: also sum over out-set hits c ~ Binomial(M, p0), truncated to +-c_sd sd
    mu, sd = M * p0, np.sqrt(M * p0 * (1.0 - p0))
    c_lo = max(0, int(mu - c_sd * sd))
    c_hi = min(M, int(mu + c_sd * sd) + 1)
    c = np.arange(c_lo, c_hi + 1)
    wc = binom.pmf(c, M, p0)
    wc = wc / wc.sum()
    G = _g_test_2x2(a[:, None], m, c[None, :], M)          # (len a, len c)
    return float(wa @ G @ wc)


def e_lrt_binary(n, density, b0, beta, *, intercept="profiled", m_tail=1e-9):
    """E[LRT] for a binary column: average E[LRT|m] over m ~ Binomial(n, density).

    The set size is a design draw; averaging over its law is more honest than fixing
    m = n*density (the enrichment picks a random column). ``intercept="profiled"``
    (default) estimates the intercept (2x2 G-test); ``"fixed"`` knows the true b0
    (in-set deviance).
    """
    if beta == 0.0:
        return 0.0
    m_lo = int(binom.ppf(m_tail, n, density))
    m_hi = int(binom.ppf(1.0 - m_tail, n, density))
    ms = np.arange(max(1, m_lo), m_hi + 1)
    w = binom.pmf(ms, n, density)
    w = w / w.sum()
    return float(np.sum([wi * _e_lrt_binary_fixed_m(int(m), n - int(m), b0, beta, intercept)
                         for wi, m in zip(w, ms)]))


def e_logbf_binary_fixed_m(m, b0, beta, *, prior_sd=1.0, n_gh=64):
    """E[log BF | m] for a binary column, prior beta ~ N(0, prior_sd^2).

    log BF(k, m) = log integral over beta' of the in-set Bernoulli likelihood ratio
    against the null, times N(beta'; 0, prior_sd^2). The 1-D integral is Gauss-Hermite.
    Reference quantity for the E[log BF] figure; prior_sd=1 matches the fixed-variance
    reference (the fits estimate the prior variance, capped at 100).
    """
    nodes, wts = roots_hermitenorm(n_gh)  # weight exp(-x^2/2), sum(wts)=sqrt(2pi)
    wts = wts / np.sqrt(2.0 * np.pi)
    betas = prior_sd * nodes  # change of variables to N(0, prior_sd^2)
    p0 = sigmoid(b0)
    k = np.arange(0, m + 1)[:, None]
    # log-likelihood-ratio of in-set data (k hits of m) at each quadrature beta'
    eta = b0 + betas[None, :]
    loglik = k * eta - m * np.logaddexp(0.0, eta)        # up to const in k
    loglik0 = k * b0 - m * np.logaddexp(0.0, b0)
    # log BF(k) = log integral w_j exp(loglik_j - loglik0); stable via logsumexp.
    logbf_grid = logsumexp((loglik - loglik0) + np.log(wts)[None, :], axis=1)
    return float(np.sum(binom.pmf(np.arange(0, m + 1), m, sigmoid(b0 + beta)) * logbf_grid))


def e_logbf_binary(n, density, b0, beta, *, prior_sd=1.0, m_tail=1e-9):
    if beta == 0.0:
        # null: E[log BF] over noise; small negative (Occam). Cheap to include.
        pass
    m_lo = int(binom.ppf(m_tail, n, density))
    m_hi = int(binom.ppf(1.0 - m_tail, n, density))
    ms = np.arange(max(1, m_lo), m_hi + 1)
    w = binom.pmf(ms, n, density)
    w = w / w.sum()
    return float(np.sum([wi * e_logbf_binary_fixed_m(int(m), b0, beta, prior_sd=prior_sd)
                         for wi, m in zip(w, ms)]))


# --------------------------------------------------------------------------- #
# Gaussian AR1 design: Monte-Carlo (asymptotic 1+lambda is too loose)          #
# --------------------------------------------------------------------------- #
def _seed_for(profile, b0, beta):
    """Deterministic per-point seed (no global RNG / Date.now)."""
    key = (hash((profile, round(float(b0), 6), round(float(beta), 6))) & 0xFFFFFFFF)
    return int(key)


def _ll(eta, z):
    return float(np.sum(z * eta - np.logaddexp(0.0, eta)))


def _newton_logistic(z, X, n_iter=25, tol=1e-9):
    """MLE for logistic ll with design columns X (n x d), including intercept if present.

    Concave; damped Newton converges in a handful of steps. Returns (coef, loglik).
    """
    d = X.shape[1]
    b = np.zeros(d)
    for _ in range(n_iter):
        eta = X @ b
        p = sigmoid(eta)
        g = X.T @ (z - p)
        w = p * (1.0 - p)
        H = (X * w[:, None]).T @ X + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    return b, _ll(X @ b, z)


def e_lrt_gaussian_mc(n, b0, beta, *, intercept="profiled", n_rep=3000, seed=None,
                      want_logbf=False, prior_sd=1.0, n_gh=48):
    """E[LRT] (and optionally E[log BF]) for a Gaussian N(0,1) column.

    Draw x ~ N(0,1)^n, z ~ Bernoulli(sigmoid(b0+beta x)). ``intercept="profiled"``:
    LRT = 2[max_{b0',b'} ll - max_{b0'} ll] (H0 is intercept-only). ``"fixed"``: b0
    known, H0 = ll(b0). For log BF, integrate the 1-D marginal over b' ~ N(0, prior_sd^2)
    by Gauss-Hermite (intercept at its null MLE for the profiled reference).
    """
    rng = np.random.default_rng(_seed_for("gaussian" + intercept, b0, beta) if seed is None else seed)
    if want_logbf:
        gh_nodes, gh_wts = roots_hermitenorm(n_gh)
        gh_wts = gh_wts / np.sqrt(2.0 * np.pi)
        bprime = prior_sd * gh_nodes
    lrt = np.empty(n_rep)
    logbf = np.empty(n_rep) if want_logbf else None
    for r in range(n_rep):
        x = rng.standard_normal(n)
        z = rng.binomial(1, sigmoid(b0 + beta * x)).astype(float)
        if intercept == "fixed":
            b0_null = b0
            ll0 = _ll(np.full(n, b0), z)
            _, ll1 = _fit_slope_fixed_intercept(z, x, b0)
        else:
            zbar = np.clip(z.mean(), 1e-12, 1 - 1e-12)
            b0_null = float(np.log(zbar / (1 - zbar)))
            ll0 = _ll(np.full(n, b0_null), z)
            _, ll1 = _newton_logistic(z, np.column_stack([np.ones(n), x]))
        lrt[r] = 2.0 * (ll1 - ll0)
        if want_logbf:
            eta = b0_null + np.outer(x, bprime)               # intercept at null MLE
            ll = (z[:, None] * eta - np.logaddexp(0.0, eta)).sum(0)
            logbf[r] = float(np.log(np.clip(np.exp(ll - ll0) @ gh_wts, 1e-300, None)))
    out = {"lrt": float(lrt.mean()), "lrt_se": float(lrt.std() / np.sqrt(n_rep))}
    if want_logbf:
        out["logbf"] = float(logbf.mean())
        out["logbf_se"] = float(logbf.std() / np.sqrt(n_rep))
    return out


def _fit_slope_fixed_intercept(z, x, b0, n_iter=25, tol=1e-9):
    """1-D Newton for the slope with the intercept held at b0 (an offset)."""
    b = 0.0
    for _ in range(n_iter):
        eta = b0 + b * x
        p = sigmoid(eta)
        g = float(x @ (z - p))
        h = float((x * x) @ (p * (1 - p))) + 1e-12
        step = g / h
        b += step
        if abs(step) < tol:
            break
    return b, _ll(b0 + b * x, z)


# --------------------------------------------------------------------------- #
# Unified interface + inversion                                                #
# --------------------------------------------------------------------------- #
PROFILES = {
    "gaussian_n500":   {"kind": "gaussian", "n": 500,   "label": "AR1 Gaussian (n=500)"},
    "binary_n1000_q50": {"kind": "binary", "n": 1000,  "density": 0.5,  "label": "Bin-AR (n=1000, q=0.5)"},
    "binary_n10000_q05": {"kind": "binary", "n": 10000, "density": 0.05, "label": "Bin-AR (n=10000, q=0.05)"},
}


def e_lrt(profile, b0, beta, *, intercept="profiled", **mc_kw):
    p = PROFILES[profile]
    if p["kind"] == "binary":
        return e_lrt_binary(p["n"], p["density"], b0, beta, intercept=intercept)
    return e_lrt_gaussian_mc(p["n"], b0, beta, intercept=intercept, **mc_kw)["lrt"]


def e_logbf(profile, b0, beta, *, prior_sd=1.0, **mc_kw):
    """E[log BF] at (profile, b0, beta): binary exact (GH over the prior), gaussian MC."""
    p = PROFILES[profile]
    if p["kind"] == "binary":
        return e_logbf_binary(p["n"], p["density"], b0, beta, prior_sd=prior_sd)
    return e_lrt_gaussian_mc(p["n"], b0, beta, want_logbf=True, prior_sd=prior_sd, **mc_kw)["logbf"]


def lrt_curve(profile, b0, betas, **mc_kw):
    """E[LRT] over a grid of betas (for the calibration figure)."""
    return np.array([e_lrt(profile, b0, float(b), **mc_kw) for b in betas])


def logbf_curve(profile, b0, betas, **mc_kw):
    """E[log BF] over a grid of betas."""
    return np.array([e_logbf(profile, b0, float(b), **mc_kw) for b in betas])


def invert_beta_for_lrt(profile, b0, target, *, intercept="profiled", beta_hi=8.0, tol=1e-3, **mc_kw):
    """Smallest beta > 0 with E[LRT] = target, by bisection (E[LRT] increases in beta)."""
    lo, hi = 0.0, beta_hi
    f_hi = e_lrt(profile, b0, hi, intercept=intercept, **mc_kw)
    if f_hi < target:
        raise ValueError(f"target {target} unreachable at beta<={beta_hi} for {profile} b0={b0} (max {f_hi:.1f}).")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if e_lrt(profile, b0, mid, intercept=intercept, **mc_kw) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    TARGETS = [4, 8, 16, 32]
    INTERCEPTS = [-3, -2, -1]
    print(f"{'profile':22s} {'b0':>4s} " + " ".join(f"T={t:<2d}->beta" for t in TARGETS))
    for prof in PROFILES:
        for b0 in INTERCEPTS:
            betas = [invert_beta_for_lrt(prof, b0, t) for t in TARGETS]
            print(f"{prof:22s} {b0:>4d} " + " ".join(f"{b:10.4f}" for b in betas))
