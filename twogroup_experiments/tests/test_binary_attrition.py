from __future__ import annotations

import numpy as np
import pytest

import core
from simulations.design.markov import binary_attrition_X

ARGS = dict(n=1000, p=256, corr=0.8, density=0.3, block_size=8, drop=0.3)


def _dense(rng, **kw):
    return np.asarray(binary_attrition_X(rng, **kw).todense())


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_core_reexports_binary_attrition_X():
    assert core.binary_attrition_X.__name__ == "binary_attrition_X"


def test_shape_sparse_binary_and_deterministic():
    X1 = binary_attrition_X(np.random.default_rng(0), **ARGS)
    X2 = binary_attrition_X(np.random.default_rng(0), **ARGS)
    assert X1.shape == (1000, 256) and type(X1).__name__ == "BCOO"
    D1, D2 = np.asarray(X1.todense()), np.asarray(X2.todense())
    np.testing.assert_array_equal(D1, D2)
    assert set(np.unique(D1)) <= {0.0, 1.0}


def test_roots_have_the_markov_density_and_children_shrink_geometrically():
    X = _dense(np.random.default_rng(3), **ARGS)
    sizes = X.sum(axis=0)
    depth = np.arange(256) % 8
    assert abs(sizes[depth == 0].mean() - 300) < 25
    # expected size at depth k is 300 * 0.7**k (add = 0): check the per-depth means
    for k in range(8):
        assert abs(sizes[depth == k].mean() - 300 * 0.7**k) < 0.25 * 300 * 0.7**k + 5
    assert sizes.min() >= 1
    assert sizes.max() / sizes.min() >= 5  # heterogeneous within one matrix


def test_children_are_exactly_nested_when_add_is_zero():
    X = _dense(np.random.default_rng(4), **ARGS)
    for j in range(256):
        if j % 8:
            assert not np.any((X[:, j] == 1) & (X[:, j - 1] == 0))


def test_add_creates_a_few_strays_only():
    X = _dense(np.random.default_rng(4), **{**ARGS, "add": 0.002})
    strays = [np.sum((X[:, j] == 1) & (X[:, j - 1] == 0)) for j in range(256) if j % 8]
    assert 0 < np.mean(strays) < 5


def test_within_block_phi_is_about_sqrt_one_minus_drop():
    for drop in (0.1, 0.36):
        X = _dense(np.random.default_rng(5), n=20000, p=64, corr=0.8, density=0.3,
                   block_size=4, drop=drop)
        phis = [_phi(X[:, j], X[:, j - 1]) for j in range(64) if j % 4]
        assert abs(np.mean(phis) - np.sqrt(1 - drop)) < 0.08


def test_adjacent_roots_keep_the_markov_correlation():
    X = _dense(np.random.default_rng(6), n=20000, p=256, corr=0.8, density=0.3,
               block_size=8, drop=0.3)
    roots = X[:, ::8]
    phis = [_phi(roots[:, i], roots[:, i + 1]) for i in range(roots.shape[1] - 1)]
    assert abs(np.mean(phis) - 0.8) < 0.05


def test_block_size_one_is_binary_markov():
    from simulations.design.markov import binary_markov_X
    X = _dense(np.random.default_rng(7), n=500, p=32, corr=0.8, density=0.3, block_size=1, drop=0.3)
    M = np.asarray(binary_markov_X(np.random.default_rng(7), n=500, p=32, corr=0.8, density=0.3).todense())
    np.testing.assert_array_equal(X, M)


def test_min_ones_floor_is_kept_by_redrawing_steps():
    # expected size at the bottom ~7.6; without the floor ~1% of draws contain an empty column
    for seed in range(40):
        X = _dense(np.random.default_rng(seed), n=1000, p=256, corr=0.8, density=0.5,
                   block_size=8, drop=0.45, min_ones=3)
        assert X.sum(axis=0).min() >= 3


def test_raises_when_the_configuration_is_too_sparse():
    # a floor no child can reach (the root has ~5 members): every redraw fails, then it raises
    with pytest.raises(ValueError, match="redraws"):
        binary_attrition_X(np.random.default_rng(8), n=50, p=64, corr=0.8, density=0.1,
                           block_size=8, drop=0.5, min_ones=40)


@pytest.mark.parametrize(
    "bad",
    [dict(density=0.0), dict(density=1.0), dict(drop=1.0), dict(add=-0.1),
     dict(block_size=0), dict(block_size=7), dict(n=0), dict(min_ones=0)],
)
def test_rejects_bad_arguments(bad):
    with pytest.raises(ValueError):
        binary_attrition_X(np.random.default_rng(9), **{**ARGS, **bad})
