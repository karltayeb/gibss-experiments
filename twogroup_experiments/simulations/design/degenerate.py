"""Degenerate / null design samplers (no real covariate structure)."""
import numpy as np


def null_enrich_X(rng: np.random.Generator, *, n: int = 4384) -> np.ndarray:
    """A single all-zero covariate column: no set-membership signal (null enrichment)."""
    del rng
    return np.zeros((n, 1), dtype=float)
