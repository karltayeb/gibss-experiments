from __future__ import annotations

import math

import numpy as np


def gaussian_markov_X(
    rng: np.random.Generator, *, n: int, p: int, rho: float
) -> np.ndarray:
    """
    Generate ``n`` independent Gaussian Markov chains of length ``p``.

    Each row is a stationary AR(1) process across columns with
    ``X[i, j + 1] | X[i, j] ~ N(rho * X[i, j], 1 - rho**2)``.
    """
    if n < 0 or p < 0:
        raise ValueError("n and p must be non-negative.")
    if abs(rho) > 1:
        raise ValueError("gaussian_markov_X requires |rho| <= 1.")
    X = np.empty((n, p), dtype=float)
    if n == 0 or p == 0:
        return X
    X[:, 0] = rng.normal(size=n)
    innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho**2)))
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + innovation_scale * rng.normal(size=n)
    return X


def uniform_markov_X(
    rng: np.random.Generator, *, n: int, p: int, rho: float
) -> np.ndarray:
    """
    Generate ``n`` independent uniform Markov chains of length ``p``, CENTERED.

    Each row applies the Gaussian CDF coordinatewise to a stationary Gaussian
    AR(1) chain (Gaussian-copula dependence, latent adjacent correlation
    ``rho``), then maps Uniform(0, 1) -> 2U - 1 so marginals are Uniform(-1, 1)
    (mean 0, Var 1/3). Centering removes the intercept/effect identifiability
    confound an uncentered [0, 1] design would create; the gaussian-vs-uniform
    contrast then isolates marginal SHAPE, not centering.
    """
    gaussian_X = gaussian_markov_X(rng, n=n, p=p, rho=rho)
    gaussian_cdf = np.vectorize(
        lambda x: 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))),
        otypes=[float],
    )
    return 2.0 * gaussian_cdf(gaussian_X) - 1.0


def _latent_rho_for_binary(rho_x: float, q: float) -> float:
    """Latent Gaussian AR(1) correlation ``rho_u`` such that thresholding to
    +/-1 binary (P(X=+1)=q) yields adjacent binary correlation ``rho_x``
    (tetrachoric inversion).

    Symmetric q=0.5: closed form ``rho_u = sin(pi * rho_x / 2)`` (Sheppard).
    General q: solve ``Phi2(-t,-t; rho_u) = q^2 + rho_x*q*(1-q)`` numerically,
    with ``t = Phi^{-1}(1-q)``.
    """
    if abs(q - 0.5) < 1e-12:
        return math.sin(math.pi * rho_x / 2.0)
    from scipy.stats import norm, multivariate_normal
    from scipy.optimize import brentq

    t = float(norm.ppf(1.0 - q))
    target = q * q + rho_x * q * (1.0 - q)

    def gap(rho_u: float) -> float:
        cdf = multivariate_normal.cdf(
            [-t, -t], mean=[0.0, 0.0], cov=[[1.0, rho_u], [rho_u, 1.0]]
        )
        return float(cdf) - target

    return float(brentq(gap, -0.999999, 0.999999))


def binary_markov_X(
    rng: np.random.Generator, *, n: int, p: int, rho: float, freq: float = 0.5
) -> np.ndarray:
    """
    Generate ``n`` independent binary (+/-1) Markov chains of length ``p``.

    Threshold a stationary Gaussian AR(1) latent chain: ``X = 2*I(U > t) - 1``
    with ``t = Phi^{-1}(1 - freq)`` so ``P(X=+1) = freq``. ``rho`` is the TARGET
    adjacent correlation OF X (binary), achieved by setting the latent
    adjacent correlation via tetrachoric inversion (see
    ``_latent_rho_for_binary``) -- so ``rho`` means the same thing as for the
    gaussian/uniform designs.

    freq=0.5 -> centered symmetric +/-1 (mean 0, Var 1): a "binary
    gaussian-markov" isolating discreteness. freq!=0.5 -> rare-binary (mean
    2*freq-1, uncentered), a tunable-correlation analogue of the gene sets.

    NOTE: thresholding loses correlation, so high ``rho`` needs a near-collinear
    latent (rho_x=0.99 -> rho_u~0.9999); extreme binary correlation is limited.
    """
    q = float(freq)
    if not 0.0 < q < 1.0:
        raise ValueError("binary_markov_X requires 0 < freq < 1.")
    if abs(rho) > 1:
        raise ValueError("binary_markov_X requires |rho| <= 1.")
    rho_u = _latent_rho_for_binary(float(rho), q)
    latent = gaussian_markov_X(rng, n=n, p=p, rho=rho_u)
    threshold = 0.0 if abs(q - 0.5) < 1e-12 else float(
        math.sqrt(2.0) * _erfinv(2.0 * (1.0 - q) - 1.0)
    )
    return np.where(latent > threshold, 1.0, -1.0)


def _erfinv(y: float) -> float:
    """Inverse error function (so threshold = Phi^{-1}(1-q) without scipy in the
    common path). Uses scipy when available; falls back to a Newton refine."""
    try:
        from scipy.special import erfinv
        return float(erfinv(y))
    except Exception:
        x = 0.0
        for _ in range(50):
            x -= (math.erf(x) - y) / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
        return x
