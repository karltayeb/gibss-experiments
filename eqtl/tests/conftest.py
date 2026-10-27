import numpy as np
import pytest

from eqtl.genotypes import PopulationSpec, simulate_population
from eqtl.simulate import ExpressionSpec, simulate_rep


@pytest.fixture(scope="session")
def small_population():
    """A 400-diploid, 100 kb, flat-map population (no genetic-map download in tests)."""
    return simulate_population(PopulationSpec(left=30_000_000, right=30_100_000, genetic_map=None, n_diploids=400, seed=3))


@pytest.fixture(scope="session")
def strong_sim(small_population):
    spec = ExpressionSpec(n=200, mu0=200.0, phi=0.2, libsize_sd=0.3, k_causal=1, beta=1.0, alpha=0.0)  # per-allele, as before the alpha model
    return simulate_rep(small_population, spec, seed=11)
