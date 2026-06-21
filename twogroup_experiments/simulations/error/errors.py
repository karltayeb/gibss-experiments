from __future__ import annotations

import numpy as np


def t_error_sampler(
    rng: np.random.Generator,
    se: np.ndarray,
    *,
    df: float,
) -> np.ndarray:
    """Standardized t-distributed error: unit variance regardless of df."""
    scale = se * np.sqrt((df - 2.0) / df)
    return rng.standard_t(df, size=len(se)) * scale


def noiseless_error_sampler(
    rng: np.random.Generator,
    se: np.ndarray,
) -> np.ndarray:
    del rng
    return np.zeros_like(se, dtype=float)
