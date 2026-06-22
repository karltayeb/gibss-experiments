from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import utils


def _pearson_ref(X: np.ndarray, c: int) -> np.ndarray:
    n, p = X.shape
    xc = X[:, c] - X[:, c].mean()
    out = np.zeros(p)
    for j in range(p):
        xj = X[:, j] - X[:, j].mean()
        denom = np.linalg.norm(xc) * np.linalg.norm(xj)
        out[j] = 0.0 if denom == 0 else float(xc @ xj / denom)
    out[c] = 1.0
    return out


def test_correlation_with_causal_matches_pearson():
    rng = np.random.default_rng(0)
    X = (rng.random((60, 8)) < 0.4).astype(float)
    causal = [2, 5]
    out = np.array(utils.correlation_with_causal(X, causal))
    assert out.shape == (2, 8)
    for row, c in zip(out, causal):
        np.testing.assert_allclose(row, _pearson_ref(X, c), atol=1e-10)


def test_correlation_with_causal_sparse_matches_dense():
    from jax.experimental import sparse as jsparse

    rng = np.random.default_rng(1)
    Xd = (rng.random((50, 6)) < 0.5).astype(float)
    Xs = jsparse.BCOO.fromdense(Xd)
    causal = [1, 4]
    dense = np.array(utils.correlation_with_causal(Xd, causal))
    sparse = np.array(utils.correlation_with_causal(Xs, causal))
    np.testing.assert_allclose(sparse, dense, atol=1e-10)


def test_correlation_with_causal_empty():
    X = np.zeros((5, 3))
    assert utils.correlation_with_causal(X, []) == []
