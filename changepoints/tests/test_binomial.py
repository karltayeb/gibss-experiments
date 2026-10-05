"""gibss Binomial (fit_glm_susie(..., trials=m)) as this pipeline uses it: the
single-feature log BF against numerical integration, and equality with the Bernoulli fit
on the expanded rows. gibss's own tests cover every preset; these pin the usage here."""

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm


def _data(seed=0, n=30):
    rng = np.random.default_rng(seed)
    m = rng.integers(20, 60, size=n)
    x = np.column_stack([np.linspace(-1, 1, n), rng.normal(size=n)])
    k = rng.binomial(m, expit(0.3 + 1.2 * x[:, 0]))
    return x, k.astype(float), m.astype(float)


def test_single_feature_log_bf_matches_quadrature():
    from gibss.methods import fit_glm_susie

    x, k, m = _data()
    pv = 0.5
    state = fit_glm_susie(x, k, L=1, trials=m, center=False, estimate_intercept=False,
                          estimate_prior_variance=False, prior_variance=pv, max_iter=1)
    got = np.asarray(state.single_effects[0].feature_log_bf)
    for j in range(x.shape[1]):
        def loglik(b, j=j):
            eta = x[:, j] * b
            return float(np.sum(k * eta - m * np.logaddexp(0.0, eta)))
        ll0 = loglik(0.0)
        val, _ = quad(lambda b, ll0=ll0: np.exp(loglik(b) - ll0) * norm.pdf(b, 0, np.sqrt(pv)),
                      -8, 8, limit=200)
        assert np.isclose(got[j], np.log(val), atol=2e-3), (j, got[j], np.log(val))


def test_equals_bernoulli_expansion():
    """center=False on both: gibss pre-centers by the UNWEIGHTED column mean, which on the
    expanded design is the trial-weighted mean of the binned one, so centered fits differ
    by that convention. Uncentered, they are the same model."""
    from gibss.methods import fit_glm_susie

    x, k, m = _data(seed=1, n=25)
    fit_b = fit_glm_susie(x, k, L=2, trials=m, center=False, tol=1e-9)
    rows = np.repeat(np.arange(len(k)), m.astype(int))
    y = np.concatenate([np.r_[np.ones(int(ki)), np.zeros(int(mi - ki))] for ki, mi in zip(k, m)])
    fit_e = fit_glm_susie(x[rows], y, L=2, center=False, tol=1e-9)
    np.testing.assert_allclose(np.asarray(fit_b.alpha), np.asarray(fit_e.alpha), atol=1e-5)
    np.testing.assert_allclose(np.asarray(fit_b.ser_log_bf), np.asarray(fit_e.ser_log_bf), atol=1e-4)
