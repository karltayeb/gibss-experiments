from __future__ import annotations

import numpy as np


def uniform_single_effect(
    X: np.ndarray,
    rng: np.random.Generator,
    causal_effect: float,
) -> tuple[list[int], list[float]]:
    if causal_effect == 0.0:
        return [], []
    index = int(rng.integers(0, X.shape[1]))
    return [index], [float(causal_effect)]


def _column_sizes(X) -> np.ndarray:
    """Per-column membership counts (set sizes), without densifying a sparse X.

    Handles a jax ``BCOO`` (via its ``indices``/``data``) and a dense ndarray.
    """
    n_cols = int(X.shape[1])
    indices = getattr(X, "indices", None)
    if indices is not None:  # jax BCOO: indices (nnz x ndim), data (nnz,)
        cols = np.asarray(indices)[:, 1]
        data = np.asarray(getattr(X, "data"))
        return np.bincount(cols, weights=data, minlength=n_cols).astype(float)
    return np.asarray(X, dtype=float).sum(axis=0).ravel()


def sized_single_effect(
    X: np.ndarray,
    rng: np.random.Generator,
    causal_effect: float,
    size_lo: int,
    size_hi: int,
) -> tuple[list[int], list[float]]:
    """Single causal set drawn uniformly among columns with size in [size_lo, size_hi].

    Same enrichment mechanics as ``uniform_single_effect`` but the causal set is
    restricted to a size window, so a simulation can target a nominal set size.
    """
    if causal_effect == 0.0:
        return [], []
    sizes = _column_sizes(X)
    eligible = np.nonzero((sizes >= float(size_lo)) & (sizes <= float(size_hi)))[0]
    if eligible.size == 0:
        raise ValueError(
            f"No gene set with size in [{size_lo}, {size_hi}] "
            f"(design has {X.shape[1]} sets, sizes {sizes.min():.0f}-{sizes.max():.0f})."
        )
    index = int(eligible[int(rng.integers(0, eligible.size))])
    return [index], [float(causal_effect)]
