"""Two-state hidden Markov model with binomial emissions (Churchill 1989 style) on GC
bins: EM for the transition matrix and the two GC fractions, forward-backward posterior
state probabilities, and the posterior probability of a state switch between adjacent
bins. Scaled recursions in probability space; n is a few hundred to a thousand."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def _log_emission(k: np.ndarray, m: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(n, S) binomial log pmf for each state's GC fraction."""
    k = k[:, None]
    m = m[:, None]
    t = theta[None, :]
    return gammaln(m + 1) - gammaln(k + 1) - gammaln(m - k + 1) + k * np.log(t) + (m - k) * np.log1p(-t)


def forward_backward(logB: np.ndarray, A: np.ndarray, pi: np.ndarray):
    n, S = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    alpha = np.zeros((n, S))
    c = np.zeros(n)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.zeros((n, S))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    # pairwise posteriors xi[t, i, j] = P(s_t = i, s_{t+1} = j | y)
    xi = alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :] / c[1:, None, None]
    loglik = float(np.sum(np.log(c)) + np.sum(logB.max(axis=1)))
    return gamma, xi, loglik


def fit_hmm(k: np.ndarray, m: np.ndarray, n_states: int = 2, n_iter: int = 500,
            tol: float = 1e-8, theta0=None, stay0: float = 0.98) -> dict:
    k = np.asarray(k, dtype=float)
    m = np.asarray(m, dtype=float)
    S = n_states
    frac = k / m
    theta = np.quantile(frac, np.linspace(0.25, 0.75, S)) if theta0 is None else np.asarray(theta0, float)
    A = np.full((S, S), (1 - stay0) / (S - 1))
    np.fill_diagonal(A, stay0)
    pi = np.full(S, 1.0 / S)
    prev = -np.inf
    for it in range(n_iter):
        logB = _log_emission(k, m, theta)
        gamma, xi, ll = forward_backward(logB, A, pi)
        A = xi.sum(axis=0)
        A /= A.sum(axis=1, keepdims=True)
        pi = gamma[0]
        theta = (gamma * k[:, None]).sum(axis=0) / (gamma * m[:, None]).sum(axis=0)
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        if ll - prev < tol:
            break
        prev = ll
    switch = 1.0 - np.einsum("tii->t", xi)  # P(s_t != s_{t+1} | y)
    return {"theta": theta, "A": A, "pi": pi, "gamma": gamma, "switch_prob": switch,
            "loglik": ll, "n_iter": it + 1, "expected_switches": float(switch.sum())}
