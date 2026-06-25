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
