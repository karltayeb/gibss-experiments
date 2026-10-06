"""Exact Bayesian multiple-changepoint posterior for conjugate segment models.

Product-partition model with independent segment parameters (Barry & Hartigan 1992)
and the forward-backward recursions of Fearnhead (2006, Stat. Comput. 16:203-213).
Every position t in 1..n-1 is a changepoint (segment boundary after observation t)
independently with prior probability p, which is the geometric segment-length prior
with the last segment right-censored at n. The exact posterior over segmentations is
summed in O(n^2) for the boundary marginals and the mean parameter profile, and in
O(n^2 kmax) for the distribution of the number of changepoints.

Segment models (log marginal likelihood of y[s:u] as one segment):
  PoissonGamma: y_i ~ Poisson(lambda e_i), lambda ~ Gamma(shape a, rate b)
  BetaBinomial: y_i ~ Binomial(m_i, theta), theta ~ Beta(a, b)
  NormalNormal: y_i ~ N(mu, sigma2), mu ~ N(m0, tau2), sigma2 known and shared
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, gammaln, logsumexp

_NEG_INF = -np.inf


@dataclass(frozen=True)
class PoissonGamma:
    shape: float
    rate: float

    def segment_logml(self, y: np.ndarray, exposure: np.ndarray | None = None) -> np.ndarray:
        """(n+1, n+1) matrix M with M[s, u] = log P(y[s:u]) for s < u, -inf elsewhere."""
        y = np.asarray(y, dtype=float)
        n = y.shape[0]
        e = np.ones(n) if exposure is None else np.asarray(exposure, dtype=float)
        cy = np.concatenate([[0.0], np.cumsum(y)])
        ce = np.concatenate([[0.0], np.cumsum(e)])
        cb = np.concatenate([[0.0], np.cumsum(y * np.log(e) - gammaln(y + 1.0))])
        S = cy[None, :] - cy[:, None]
        E = ce[None, :] - ce[:, None]
        B = cb[None, :] - cb[:, None]
        a, b = self.shape, self.rate
        with np.errstate(invalid="ignore", divide="ignore"):
            M = gammaln(a + S) - gammaln(a) + a * np.log(b) - (a + S) * np.log(b + E) + B
        s, u = np.indices(M.shape)
        M[~(s < u)] = _NEG_INF
        return M

    def segment_mean(self, y: np.ndarray, exposure: np.ndarray | None = None) -> np.ndarray:
        """(n+1, n+1) posterior mean rate of segment y[s:u]."""
        y = np.asarray(y, dtype=float)
        n = y.shape[0]
        e = np.ones(n) if exposure is None else np.asarray(exposure, dtype=float)
        cy = np.concatenate([[0.0], np.cumsum(y)])
        ce = np.concatenate([[0.0], np.cumsum(e)])
        S = cy[None, :] - cy[:, None]
        E = ce[None, :] - ce[:, None]
        return (self.shape + S) / (self.rate + E)


@dataclass(frozen=True)
class BetaBinomial:
    a: float
    b: float

    def segment_logml(self, y: np.ndarray, trials: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        m = np.asarray(trials, dtype=float)
        ck = np.concatenate([[0.0], np.cumsum(y)])
        cm = np.concatenate([[0.0], np.cumsum(m)])
        cb = np.concatenate([[0.0], np.cumsum(gammaln(m + 1) - gammaln(y + 1) - gammaln(m - y + 1))])
        K = ck[None, :] - ck[:, None]
        Mt = cm[None, :] - cm[:, None]
        B = cb[None, :] - cb[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            M = betaln(self.a + K, self.b + Mt - K) - betaln(self.a, self.b) + B
        s, u = np.indices(M.shape)
        M[~(s < u)] = _NEG_INF
        return M

    def segment_mean(self, y: np.ndarray, trials: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        m = np.asarray(trials, dtype=float)
        ck = np.concatenate([[0.0], np.cumsum(y)])
        cm = np.concatenate([[0.0], np.cumsum(m)])
        K = ck[None, :] - ck[:, None]
        Mt = cm[None, :] - cm[:, None]
        return (self.a + K) / (self.a + self.b + Mt)


@dataclass(frozen=True)
class NormalNormal:
    m0: float
    tau2: float
    sigma2: float

    def _sums(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(y)])
        cq = np.concatenate([[0.0], np.cumsum(y * y)])
        n = y.shape[0]
        k = np.arange(n + 1)
        K = (k[None, :] - k[:, None]).astype(float)
        return K, cs[None, :] - cs[:, None], cq[None, :] - cq[:, None]

    def segment_logml(self, y: np.ndarray) -> np.ndarray:
        """log N(y[s:u]; m0 1, sigma2 I + tau2 J), from the segment's length, sum and
        sum of squares (Sherman-Morrison for the inverse, matrix determinant lemma)."""
        K, S, Q = self._sums(y)
        s2, t2, m0 = self.sigma2, self.tau2, self.m0
        r1 = S - K * m0  # sum of residuals about m0
        r2 = Q - 2 * m0 * S + K * m0**2  # sum of squared residuals about m0
        with np.errstate(invalid="ignore", divide="ignore"):
            quad = (r2 - t2 * r1**2 / (s2 + K * t2)) / s2
            M = -0.5 * (K * np.log(2 * np.pi * s2) + np.log1p(K * t2 / s2) + quad)
        s, u = np.indices(M.shape)
        M[~(s < u)] = _NEG_INF
        return M

    def segment_mean(self, y: np.ndarray) -> np.ndarray:
        K, S, _ = self._sums(y)
        return (self.m0 / self.tau2 + S / self.sigma2) / (1 / self.tau2 + K / self.sigma2)


@dataclass
class ExactPosterior:
    n: int
    p: float
    log_evidence: float
    boundary_prob: np.ndarray  # (n-1,), P(boundary after position t), t = 0..n-2
    k_prob: np.ndarray  # (kmax+1,), P(number of changepoints = k)
    mean_profile: np.ndarray  # (n,), posterior mean segment parameter at each position
    log_forward: np.ndarray  # (n+1,)
    log_backward: np.ndarray  # (n+1,)

    @property
    def k_mean(self) -> float:
        return float(np.sum(np.arange(len(self.k_prob)) * self.k_prob))


def _log_cohesion(n: int, p: float) -> np.ndarray:
    """(n+1, n+1) log prior weight of segment [s, u): interior non-boundaries and the
    boundary at u (none at u = n)."""
    s, u = np.indices((n + 1, n + 1))
    length = u - s
    with np.errstate(divide="ignore"):
        lc = (length - 1) * np.log1p(-p) + np.where(u < n, np.log(p), 0.0)
    lc[~(s < u)] = _NEG_INF
    return lc


def exact_posterior(logml: np.ndarray, seg_mean: np.ndarray, p: float,
                    kmax: int | None = None) -> ExactPosterior:
    """Forward-backward sums over all segmentations.

    logml[s, u] = log marginal likelihood of y[s:u] as one segment (s < u), and
    seg_mean[s, u] its posterior mean parameter. p is the per-position prior
    changepoint probability. kmax truncates the distribution over the number of
    changepoints (default min(n-1, 30)).
    """
    n = logml.shape[0] - 1
    W = logml + _log_cohesion(n, p)  # log weight of segment [s, u)

    # forward: F[u] = log P(y[:u], boundary at u)
    F = np.full(n + 1, _NEG_INF)
    F[0] = 0.0
    for u in range(1, n + 1):
        F[u] = logsumexp(F[:u] + W[:u, u])
    log_Z = F[n]

    # backward: B[s] = log P(y[s:] | boundary at s)
    B = np.full(n + 1, _NEG_INF)
    B[n] = 0.0
    for s in range(n - 1, -1, -1):
        B[s] = logsumexp(W[s, s + 1:] + B[s + 1:])

    boundary = np.exp(F[1:n] + B[1:n] - log_Z)

    # segment posterior probabilities and the mean profile by a difference array
    with np.errstate(invalid="ignore"):
        logseg = F[:, None] + W + B[None, :] - log_Z
    seg_prob = np.exp(logseg)
    seg_prob[~np.isfinite(logseg)] = 0.0
    contrib = seg_prob * np.nan_to_num(seg_mean)
    diff = np.zeros(n + 2)
    # segment [s, u) covers positions s..u-1
    np.add.at(diff, np.arange(n + 1), contrib.sum(axis=1))  # start at s
    np.add.at(diff, np.arange(n + 1), -contrib.sum(axis=0))  # end at u
    mean_profile = np.cumsum(diff)[:n]

    # number of changepoints: forward recursion carrying k (boundary at u < n adds one)
    if kmax is None:
        kmax = min(n - 1, 30)
    Fk = np.full((n + 1, kmax + 1), _NEG_INF)
    Fk[0, 0] = 0.0
    for u in range(1, n + 1):
        prev = Fk[:u] + W[:u, u][:, None]  # (u, kmax+1)
        col = logsumexp(prev, axis=0)
        if u < n:
            Fk[u, 1:] = col[:-1]
        else:
            Fk[u] = col
    k_prob = np.exp(Fk[n] - logsumexp(Fk[n]))

    return ExactPosterior(n=n, p=p, log_evidence=float(log_Z), boundary_prob=boundary,
                          k_prob=k_prob, mean_profile=mean_profile, log_forward=F,
                          log_backward=B)


def single_changepoint_posterior(logml: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact posterior over the location of exactly one changepoint (uniform prior on
    the n-1 positions), and the log Bayes factor of one changepoint vs none.

    Returns (prob over boundaries after position t = 0..n-2, log BF_10).
    """
    n = logml.shape[0] - 1
    t = np.arange(1, n)
    lw = logml[0, t] + logml[t, n]
    log_m1 = logsumexp(lw) - np.log(n - 1)
    return np.exp(lw - logsumexp(lw)), float(log_m1 - logml[0, n])


def sample_segmentations(logml: np.ndarray, p: float, n_samples: int,
                         rng: np.random.Generator) -> list[np.ndarray]:
    """Perfect samples of boundary positions from the exact posterior (backward
    simulation from the forward sums)."""
    n = logml.shape[0] - 1
    W = logml + _log_cohesion(n, p)
    F = np.full(n + 1, _NEG_INF)
    F[0] = 0.0
    for u in range(1, n + 1):
        F[u] = logsumexp(F[:u] + W[:u, u])
    out = []
    for _ in range(n_samples):
        u = n
        bounds = []
        while u > 0:
            lw = F[:u] + W[:u, u]
            pr = np.exp(lw - logsumexp(lw))
            s = rng.choice(u, p=pr)
            if s > 0:
                bounds.append(s)
            u = s
        out.append(np.array(bounds[::-1], dtype=int))
    return out
