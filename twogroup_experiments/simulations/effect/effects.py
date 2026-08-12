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


def paired_index_effect(
    X: np.ndarray,
    rng: np.random.Generator,
    causal_effect: float,
    gap: int,
    base_index: int = 0,
    sign_b: float = 1.0,
) -> tuple[list[int], list[float]]:
    """Two causal columns at FIXED indices ``base_index`` and ``base_index + gap``.

    Deterministic placement (ignores ``rng``) so the two causal columns' overlap is set
    purely by ``gap`` at the design's fixed ``rho`` (for a Markov design, latent
    correlation ``rho ** gap``) and is identical across replicates. Symmetric magnitude
    ``|causal_effect|``; the first effect is ``+causal_effect`` and the second is
    ``sign_b * causal_effect`` -- ``sign_b=+1`` gives a ``(+,+)`` pair, ``-1`` a
    ``(+,-)`` pair. ``causal_effect == 0`` yields the paired NULL (no causal sets).
    """
    if causal_effect == 0.0:
        return [], []
    i = int(base_index)
    j = int(base_index) + int(gap)
    if not (0 <= i < X.shape[1] and 0 <= j < X.shape[1]):
        raise ValueError(
            f"paired_index_effect indices ({i}, {j}) out of range for p={X.shape[1]} "
            f"(base_index={base_index}, gap={gap})."
        )
    return [i, j], [float(causal_effect), float(sign_b) * float(causal_effect)]


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


def sized_multi_effect(
    X: np.ndarray,
    rng: np.random.Generator,
    causal_effects: list[float],
    size_lo: int,
    size_hi: int,
) -> tuple[list[int], list[float]]:
    """K distinct causal sets in [size_lo, size_hi], one per entry of causal_effects.

    Multi-causal generalization of ``sized_single_effect``: K = len(causal_effects)
    distinct columns are drawn WITHOUT replacement from the size window, and the
    i-th drawn set is assigned effect ``causal_effects[i]``. Passing a graded list of
    effects (e.g. [2.5, 2.0, 1.5]) places the causals across the detectability
    spectrum within a single simulation, so a fitted L>1 SuSiE faces a mix of
    clearly-enriched and near-boundary components. Zero entries are dropped (their
    set is not spent) so a partially-null list stays honest about the causal count.
    """
    effects = [float(b) for b in causal_effects if float(b) != 0.0]
    if not effects:
        return [], []
    sizes = _column_sizes(X)
    eligible = np.nonzero((sizes >= float(size_lo)) & (sizes <= float(size_hi)))[0]
    if eligible.size < len(effects):
        raise ValueError(
            f"Need {len(effects)} gene sets with size in [{size_lo}, {size_hi}] "
            f"but only {eligible.size} are eligible "
            f"(design has {X.shape[1]} sets, sizes {sizes.min():.0f}-{sizes.max():.0f})."
        )
    chosen = rng.choice(eligible, size=len(effects), replace=False)
    return [int(c) for c in chosen], effects


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
