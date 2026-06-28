"""Rigorous tests for the Markov design samplers (markov.py).

Focus: binary_markov_X's tetrachoric correlation map (the nontrivial part) plus
the centered uniform design and shared structural guarantees.
"""
import math

import numpy as np
import pytest

from simulations.design.markov import (
    binary_markov_X,
    gaussian_markov_X,
    uniform_markov_X,
    _latent_rho_for_binary,
)

# Large n so empirical correlations/frequencies are tight; tolerance ~3 SE.
_N = 400_000


def _rng():
    return np.random.default_rng(12345)


# --------------------------------------------------------------------------- #
# _latent_rho_for_binary: closed form vs numeric solver                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho_x", [0.0, 0.2, 0.5, 0.7, 0.9, 0.99, -0.3])
def test_latent_rho_symmetric_closed_form(rho_x):
    # freq=0.5 must use the Sheppard closed form rho_u = sin(pi*rho_x/2)
    assert _latent_rho_for_binary(rho_x, 0.5) == pytest.approx(
        math.sin(math.pi * rho_x / 2.0), abs=1e-12
    )


@pytest.mark.parametrize("rho_x", [0.2, 0.6, 0.9])
def test_latent_rho_solver_inverts_tetrachoric(rho_x):
    # For asymmetric freq, the solved rho_u must reproduce rho_x through the
    # tetrachoric forward map: rho_x = (Phi2(-t,-t;rho_u) - q^2)/(q(1-q)).
    from scipy.stats import norm, multivariate_normal

    q = 0.2
    rho_u = _latent_rho_for_binary(rho_x, q)
    t = float(norm.ppf(1.0 - q))
    p11 = float(
        multivariate_normal.cdf([-t, -t], mean=[0.0, 0.0],
                                cov=[[1.0, rho_u], [rho_u, 1.0]])
    )
    recovered = (p11 - q * q) / (q * (1.0 - q))
    assert recovered == pytest.approx(rho_x, abs=1e-6)


# --------------------------------------------------------------------------- #
# binary_markov_X: marginal, support, correlation                             #
# --------------------------------------------------------------------------- #
def test_binary_values_are_plus_minus_one():
    X = binary_markov_X(_rng(), n=1000, p=5, rho=0.7, freq=0.5)
    assert set(np.unique(X)).issubset({-1.0, 1.0})


@pytest.mark.parametrize("freq", [0.5, 0.3, 0.1, 0.8])
def test_binary_marginal_frequency(freq):
    X = binary_markov_X(_rng(), n=_N, p=3, rho=0.6, freq=freq)
    emp = float((X == 1.0).mean())
    assert emp == pytest.approx(freq, abs=0.005)


@pytest.mark.parametrize("freq", [0.5, 0.2])
@pytest.mark.parametrize("rho_x", [0.0, 0.3, 0.7, 0.9])
def test_binary_adjacent_correlation_matches_target(freq, rho_x):
    X = binary_markov_X(_rng(), n=_N, p=2, rho=rho_x, freq=freq)
    emp = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    assert emp == pytest.approx(rho_x, abs=0.01)


def test_binary_markov_decay_is_monotone():
    # Adjacent correlation should exceed lag-2, which should exceed lag-4.
    X = binary_markov_X(_rng(), n=_N, p=6, rho=0.8, freq=0.5)
    c = np.corrcoef(X, rowvar=False)
    lag1 = np.mean([c[i, i + 1] for i in range(5)])
    lag2 = np.mean([c[i, i + 2] for i in range(4)])
    lag4 = np.mean([c[i, i + 4] for i in range(2)])
    assert lag1 > lag2 > lag4 > 0.0


def test_binary_rho_zero_is_independent():
    X = binary_markov_X(_rng(), n=_N, p=3, rho=0.0, freq=0.5)
    assert abs(float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])) < 0.01


def test_binary_reproducible_with_seed():
    a = binary_markov_X(np.random.default_rng(7), n=500, p=4, rho=0.6, freq=0.3)
    b = binary_markov_X(np.random.default_rng(7), n=500, p=4, rho=0.6, freq=0.3)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# binary_markov_X: shape + validation                                         #
# --------------------------------------------------------------------------- #
def test_binary_shape():
    X = binary_markov_X(_rng(), n=37, p=11, rho=0.5, freq=0.5)
    assert X.shape == (37, 11)


@pytest.mark.parametrize("bad_freq", [0.0, 1.0, -0.1, 1.2])
def test_binary_bad_freq_raises(bad_freq):
    with pytest.raises(ValueError):
        binary_markov_X(_rng(), n=10, p=2, rho=0.5, freq=bad_freq)


def test_binary_bad_rho_raises():
    with pytest.raises(ValueError):
        binary_markov_X(_rng(), n=10, p=2, rho=1.5, freq=0.5)


# --------------------------------------------------------------------------- #
# uniform_markov_X: centered to [-1, 1]                                        #
# --------------------------------------------------------------------------- #
def test_uniform_is_centered_on_minus_one_to_one():
    U = uniform_markov_X(_rng(), n=_N, p=3, rho=0.5)
    assert U.min() >= -1.0 and U.max() <= 1.0
    assert float(U.mean()) == pytest.approx(0.0, abs=0.01)
    # Var of Uniform(-1,1) = 1/3.
    assert float(U.var()) == pytest.approx(1.0 / 3.0, abs=0.01)


def test_uniform_latent_correlation_positive():
    U = uniform_markov_X(_rng(), n=_N, p=2, rho=0.7)
    # Copula attenuates a little; just require strong positive dependence.
    assert 0.5 < float(np.corrcoef(U[:, 0], U[:, 1])[0, 1]) < 0.7


# --------------------------------------------------------------------------- #
# gaussian_markov_X: sanity (unchanged baseline)                              #
# --------------------------------------------------------------------------- #
def test_gaussian_adjacent_correlation():
    X = gaussian_markov_X(_rng(), n=_N, p=2, rho=0.6)
    assert float(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) == pytest.approx(0.6, abs=0.01)
