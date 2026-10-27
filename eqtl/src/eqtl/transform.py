"""Count transforms for the transform-then-linear pipeline.

GTEx / jaxQTL-linear: normalize read counts between individuals (TMM), then across
individuals by a rank-based inverse normal transform. With one gene the TMM size factor
is the library size, so `inverse_normal(library_normalize(y, log_libsize))` is that
pipeline for one gene. `log_normalized` is the other common choice, log(normalized
count + 1), the single-gene analogue of log(CPM + 1) / log2(TPM + 1).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

__all__ = ["library_normalize", "inverse_normal", "log_normalized", "TRANSFORMS"]


def library_normalize(y: np.ndarray, log_libsize: np.ndarray) -> np.ndarray:
    """Counts rescaled to the reference library: y_i / l_i with l_i = exp(log_libsize_i)."""
    return np.asarray(y, dtype=float) / np.exp(np.asarray(log_libsize, dtype=float))


def inverse_normal(x: np.ndarray, *, offset: float = 0.5) -> np.ndarray:
    """Rank-based inverse normal transform, Phi^{-1}((rank - offset) / n); average ranks on
    ties (GTEx's `rankNorm`, jaxQTL's `inverse_normal_transform` with offset 0.5)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    return ndtri((rankdata(x, method="average") - offset) / n)


def log_normalized(y: np.ndarray, log_libsize: np.ndarray, *, pseudocount: float = 1.0) -> np.ndarray:
    return np.log(library_normalize(y, log_libsize) + pseudocount)


TRANSFORMS = {
    "int": lambda y, ll: inverse_normal(library_normalize(y, ll)),
    "log": lambda y, ll: log_normalized(y, ll),
}
