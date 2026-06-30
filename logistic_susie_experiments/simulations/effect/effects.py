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


def uniform_k_effects(
    X: np.ndarray,
    rng: np.random.Generator,
    causal_effect: float,
    k: int,
) -> tuple[list[int], list[float]]:
    """Pick ``k`` distinct causal features, each with the same ``causal_effect``."""
    if causal_effect == 0.0 or k == 0:
        return [], []
    indices = rng.choice(X.shape[1], size=int(k), replace=False)
    return [int(i) for i in indices], [float(causal_effect)] * int(k)


def logbf_single_effect(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    design: str,
    b0: float,
    target_logbf: float,
) -> tuple[list[int], list[float]]:
    """One causal feature sized to a target univariate logBF via the frozen
    exact-LRT table (lookup, not a solve). target_logbf==0 => null (no effect).
    `design` is the sizing key, `b0` the intercept — both injected by the loader's
    sweep expansion from the paired design."""
    if target_logbf == 0:
        return [], []
    from simulations.effect.logbf_sizing import FROZEN_B
    b = FROZEN_B[design][float(b0)][int(target_logbf)]
    index = int(rng.integers(0, X.shape[1]))
    return [index], [float(b)]


def logbf_k_effects(
    X: np.ndarray,
    rng: np.random.Generator,
    *,
    design: str,
    b0: float,
    target_logbf: float,
    k: int = 3,
) -> tuple[list[int], list[float]]:
    """``k`` causal features, each frozen-sized to ``target_logbf`` (single-effect
    sizing; the realized per-effect logBF is approximate once effects coexist).
    Causals are placed one per contiguous block of the design so they sit in
    distinct correlated neighborhoods (identifiable, mutually ~independent).
    target_logbf==0 => null."""
    if target_logbf == 0 or k == 0:
        return [], []
    from simulations.effect.logbf_sizing import FROZEN_B
    b = FROZEN_B[design][float(b0)][int(target_logbf)]
    p = X.shape[1]
    edges = np.linspace(0, p, int(k) + 1).astype(int)
    indices = [int(rng.integers(edges[i], edges[i + 1])) for i in range(int(k))]
    return indices, [float(b)] * int(k)
