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
    Generate ``n`` independent uniform Markov chains of length ``p``.

    Each row is formed by applying the Gaussian CDF coordinatewise to a
    stationary Gaussian AR(1) chain. Marginals are Uniform(0, 1), and the
    within-row dependence is induced by a Gaussian copula with latent
    adjacent-column correlation ``rho``.
    """
    gaussian_X = gaussian_markov_X(rng, n=n, p=p, rho=rho)
    gaussian_cdf = np.vectorize(
        lambda x: 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))),
        otypes=[float],
    )
    return gaussian_cdf(gaussian_X)
