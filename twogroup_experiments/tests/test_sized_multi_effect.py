from __future__ import annotations

import numpy as np
import pytest

import core
from simulations.effect.effects import sized_multi_effect


def _dense_design(n_obs: int = 400, n_sets: int = 60, set_size: int = 100, seed: int = 0):
    """Binary genes x sets design where every set has exactly ``set_size`` members."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_obs, n_sets), dtype=float)
    for j in range(n_sets):
        X[rng.choice(n_obs, size=set_size, replace=False), j] = 1.0
    return X


def test_core_reexports_sized_multi_effect():
    assert core.sized_multi_effect is sized_multi_effect


def test_returns_one_distinct_set_per_effect():
    X = _dense_design()
    rng = np.random.default_rng(1)
    effects = [2.5, 2.0, 1.5]

    idx, eff = sized_multi_effect(X, rng, causal_effects=effects, size_lo=80, size_hi=120)

    assert eff == effects
    assert len(idx) == len(effects)
    assert len(set(idx)) == len(idx)  # distinct columns, sampled without replacement


def test_empty_effect_list_yields_no_causals():
    X = _dense_design()
    rng = np.random.default_rng(1)

    idx, eff = sized_multi_effect(X, rng, causal_effects=[], size_lo=80, size_hi=120)

    assert idx == [] and eff == []


def test_zero_entries_are_dropped_without_spending_a_set():
    X = _dense_design()
    rng = np.random.default_rng(1)

    idx, eff = sized_multi_effect(X, rng, causal_effects=[2.0, 0.0, 1.5], size_lo=80, size_hi=120)

    assert eff == [2.0, 1.5]
    assert len(idx) == 2


def test_chosen_sets_respect_the_size_window():
    X = _dense_design(set_size=100)
    # make a handful of sets fall outside the window so eligibility actually filters
    X[:, :10] = 0.0
    X[:40, :10] = 1.0  # size-40 sets, outside [80, 120]
    rng = np.random.default_rng(3)

    idx, _ = sized_multi_effect(X, rng, causal_effects=[2.0, 2.0], size_lo=80, size_hi=120)

    sizes = X.sum(axis=0)
    assert all(80 <= sizes[i] <= 120 for i in idx)


def test_raises_when_too_few_eligible_sets():
    X = _dense_design(n_sets=2)
    rng = np.random.default_rng(1)

    with pytest.raises(ValueError, match="Need 3 gene sets"):
        sized_multi_effect(X, rng, causal_effects=[2.0, 2.0, 2.0], size_lo=80, size_hi=120)
