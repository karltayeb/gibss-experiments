"""Step basis for changepoint regression.

For an ordered sequence of n observations, column j (j = 0..n-2) is x_tj = 1(t > j), so
a nonzero coefficient b_j is a jump in the linear predictor between positions j and
j+1. The intercept is the level of the first segment. Columns are NOT standardized by
default (standardizing changes the implicit prior on jump size near the edges).

Products with the step basis are cumulative sums: X b = cumsum(b) shifted by one, and
X^T r = reverse cumsum of r dropping the first entry. Both are provided for large n.
"""

from __future__ import annotations

import numpy as np


def step_basis(n: int, dtype=np.float64) -> np.ndarray:
    """Dense (n, n-1) step basis: X[t, j] = 1 if t > j else 0."""
    t = np.arange(n)[:, None]
    j = np.arange(n - 1)[None, :]
    return (t > j).astype(dtype)


def step_matvec(b: np.ndarray) -> np.ndarray:
    """X @ b for the (n, n-1) step basis, O(n)."""
    out = np.zeros(b.shape[0] + 1, dtype=np.result_type(b, float))
    out[1:] = np.cumsum(b)
    return out


def step_rmatvec(r: np.ndarray) -> np.ndarray:
    """X^T @ r for the (n, n-1) step basis, O(n)."""
    return np.cumsum(r[::-1])[::-1][1:]


def standardize_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit-variance columns (not centered: gibss centers itself). Returns (Xs, scale)."""
    scale = X.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    return X / scale, scale


def cs_interval(cs: tuple[int, ...]) -> tuple[int, int, int]:
    """(min column, max column, hull width) of a credible set over step columns."""
    lo, hi = min(cs), max(cs)
    return lo, hi, hi - lo + 1
