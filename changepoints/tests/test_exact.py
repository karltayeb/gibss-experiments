"""Brute-force checks of the exact changepoint recursions on tiny sequences."""

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import nbinom, poisson

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from exact import (
    PoissonGamma,
    exact_posterior,
    single_changepoint_posterior,
)


def brute_force(y, model, p):
    n = len(y)
    M = model.segment_logml(y)
    mean = model.segment_mean(y)
    logw = []
    configs = []
    for bits in itertools.product([0, 1], repeat=n - 1):
        bounds = [0] + [t + 1 for t, b in enumerate(bits) if b] + [n]
        lw = sum(M[s, u] for s, u in itertools.pairwise(bounds))
        k = sum(bits)
        lw += k * np.log(p) + (n - 1 - k) * np.log1p(-p)
        logw.append(lw)
        configs.append((bits, bounds))
    logw = np.array(logw)
    log_Z = logsumexp(logw)
    post = np.exp(logw - log_Z)
    boundary = np.zeros(n - 1)
    kprob = np.zeros(n)
    profile = np.zeros(n)
    for w, (bits, bounds) in zip(post, configs):
        boundary += w * np.array(bits)
        kprob[sum(bits)] += w
        for s, u in itertools.pairwise(bounds):
            profile[s:u] += w * mean[s, u]
    return log_Z, boundary, kprob, profile


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.poisson(4.0, 4), rng.poisson(1.0, 5)])
    model = PoissonGamma(shape=1.0, rate=0.5)
    for p in (0.05, 0.3):
        post = exact_posterior(model.segment_logml(y), model.segment_mean(y), p)
        log_Z, boundary, kprob, profile = brute_force(y, model, p)
        assert np.isclose(post.log_evidence, log_Z)
        np.testing.assert_allclose(post.boundary_prob, boundary, atol=1e-12)
        np.testing.assert_allclose(post.k_prob[: len(kprob)], kprob, atol=1e-12)
        np.testing.assert_allclose(post.mean_profile, profile, atol=1e-12)


def test_segment_marginal_is_negative_binomial():
    # one observation: Poisson-Gamma marginal is NB(a, b/(b+1))
    model = PoissonGamma(shape=2.5, rate=0.7)
    y = np.array([3.0])
    M = model.segment_logml(y)
    assert np.isclose(M[0, 1], nbinom.logpmf(3, 2.5, 0.7 / 1.7))
    # exposure e: NB(a, b/(b+e)) for the count
    M2 = model.segment_logml(y, exposure=np.array([2.0]))
    assert np.isclose(M2[0, 1], nbinom.logpmf(3, 2.5, 0.7 / 2.7))
    # two observations sharing one rate: integrate Gamma by quadrature
    y2 = np.array([1.0, 4.0])
    from scipy.integrate import quad
    from scipy.stats import gamma

    def f(lam):
        return poisson.pmf(1, lam) * poisson.pmf(4, lam) * gamma.pdf(lam, 2.5, scale=1 / 0.7)

    val, _ = quad(f, 0, 60)
    assert np.isclose(model.segment_logml(y2)[0, 2], np.log(val), rtol=1e-6)


def test_single_changepoint_sums_to_one():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.poisson(5.0, 10), rng.poisson(1.0, 10)])
    model = PoissonGamma(shape=1.0, rate=0.3)
    prob, log_bf = single_changepoint_posterior(model.segment_logml(y))
    assert np.isclose(prob.sum(), 1.0)
    assert prob.argmax() in range(7, 12)
    assert log_bf > 0
