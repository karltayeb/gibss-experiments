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


def _norm_ppf(q: float) -> float:
    """Inverse standard-normal CDF (stdlib, no SciPy/NumPy-erfinv dependency)."""
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(q))


def binary_corr_from_latent(rho: float, density: float) -> float:
    """Pearson correlation of two thresholded columns from their LATENT Gaussian ``rho``.

    Two standard-normal variables with correlation ``rho`` are each thresholded at
    ``t = norm_ppf(1 - density)`` to give Bernoulli(``density``) memberships
    (``X = 1`` iff ``Z > t``). Their phi/Pearson correlation is

    ``corr = (P11 - d**2) / (d * (1 - d))``,  ``P11 = P(Z1 > t, Z2 > t; rho)``.

    Using the classic tetrachoric identity ``dP11/dr = phi_2(t, t; r)`` and the
    substitution ``r = sin(theta)`` (which cancels the ``1/sqrt(1 - r**2)`` factor and
    its endpoint singularity), the excess joint probability is

    ``P11 - d**2 = (1 / 2pi) * integral_0^{arcsin(rho)} exp(-t**2 / (1 + sin theta)) d(theta)``.

    Monotone increasing in ``rho``: ``corr -> 1`` as ``rho -> 1`` and
    ``corr -> -density / (1 - density)`` as ``rho -> -1``.
    """
    if not (0.0 < density < 1.0):
        raise ValueError("binary_corr_from_latent requires 0 < density < 1.")
    rho = float(np.clip(rho, -1.0, 1.0))
    if rho == 0.0:
        return 0.0
    t = _norm_ppf(1.0 - density)
    theta = np.linspace(0.0, math.asin(rho), 2001)
    integrand = np.exp(-t * t / (1.0 + np.sin(theta)))
    excess = float(np.trapezoid(integrand, theta)) / (2.0 * math.pi)
    return excess / (density * (1.0 - density))


def latent_from_binary_corr(corr: float, density: float) -> float:
    """Invert :func:`binary_corr_from_latent`: latent Gaussian ``rho`` for a target
    binary column correlation ``corr`` at membership rate ``density``.

    ``binary_corr_from_latent`` is strictly increasing in ``rho``, so a bisection on
    ``rho in (-1, 1)`` recovers the unique latent correlation. Raises if ``corr`` is
    outside the feasible range ``(-density / (1 - density), 1)`` set by the marginals.
    """
    if not (0.0 < density < 1.0):
        raise ValueError("latent_from_binary_corr requires 0 < density < 1.")
    corr = float(corr)
    lo_corr = -density / (1.0 - density)
    if not (lo_corr < corr < 1.0):
        raise ValueError(
            f"binary corr={corr} infeasible at density={density}; "
            f"achievable range is ({lo_corr:.4g}, 1)."
        )
    if corr == 0.0:
        return 0.0
    lo, hi = -1.0 + 1e-9, 1.0 - 1e-9
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if binary_corr_from_latent(mid, density) < corr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binary_markov_X(
    rng: np.random.Generator, *, n: int, p: int, corr: float, density: float
):
    """Binary (0/1) set-membership design with a controllable column-correlation chain.

    Threshold a stationary AR(1) Gaussian-copula chain (``gaussian_markov_X``) at the
    upper ``density`` quantile, so every column (a gene set) has marginal membership
    rate ``density`` (expected set size ``density * n``). The design is parameterized by
    ``corr``, the Pearson (phi) correlation between ADJACENT binary columns -- the
    directly observable, unit-free overlap knob, not the latent Gaussian correlation.
    Internally ``corr`` is mapped to the latent adjacent-column correlation
    ``rho = latent_from_binary_corr(corr, density)`` (tetrachoric inverse), and the chain
    is built with that ``rho``.

    Because the copula is AR(1), columns ``i`` and ``j`` share LATENT correlation
    ``rho ** |i - j|``; their BINARY correlation is ``binary_corr_from_latent(rho ** |i - j|,
    density)`` -- decaying with the index gap but NOT as ``corr ** |i - j|`` (the latent
    power, not the binary correlation, is what compounds). Two causal columns placed a
    fixed ``gap`` apart therefore have a fixed overlap purely by ``gap`` at fixed ``corr``.

    Returned SPARSE (``jax BCOO``), like the real gene-set designs, so the fit stays on
    the sparse fast path: ``as_operator`` dispatches to the BCOO operator, the profiled
    background uses the Chebyshev surrogate, and -- critically for L>1 -- the compress
    offset fold takes ``build_aux_sequential_sparse`` instead of densifying the product
    mixture. (Membership is a rare 0/1 indicator, so BCOO is both faithful and fast.)
    """
    if not (0.0 < density < 1.0):
        raise ValueError("binary_markov_X requires 0 < density < 1.")
    from jax.experimental.sparse import BCOO
    rho = latent_from_binary_corr(corr, density)
    gaussian_X = gaussian_markov_X(rng, n=n, p=p, rho=rho)
    threshold = _norm_ppf(1.0 - float(density))  # upper-tail cut for P(X=1)=density
    dense = (gaussian_X > threshold).astype(float)
    return BCOO.fromdense(dense)
