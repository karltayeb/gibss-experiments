import numpy as np
import pytest
from simulations.effect.effects import logbf_single_effect
from simulations.effect import logbf_sizing


def test_logbf_single_effect_looks_up_frozen_b():
    rng = np.random.default_rng(0)
    X = np.zeros((10, 7))
    idx, eff = logbf_single_effect(X, rng, design="gaussian", b0=-2.0, target_logbf=16)
    assert len(idx) == 1 and len(eff) == 1
    assert eff[0] == logbf_sizing.FROZEN_B["gaussian"][-2.0][16]
    assert eff[0] == 0.545
    assert 0 <= idx[0] < 7


def test_logbf_single_effect_null():
    rng = np.random.default_rng(0)
    assert logbf_single_effect(np.zeros((4, 3)), rng, design="uniform", b0=0.0, target_logbf=0) == ([], [])
