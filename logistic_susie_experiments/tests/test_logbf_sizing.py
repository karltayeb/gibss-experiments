import numpy as np
import pytest
from simulations.effect import logbf_sizing as S

DESIGNS = ["gaussian", "uniform", "binary q=0.5", "binary q=0.1"]


@pytest.mark.parametrize("design", DESIGNS)
@pytest.mark.parametrize("b0", [0.0, -2.0, -4.0])
@pytest.mark.parametrize("target", [4, 8, 16, 32, 64])
def test_solve_b_roundtrips(design, b0, target):
    b = S.solve_b(design, target, b0, 1000)
    assert S.expected_lrt(design, b0, b, 1000) == pytest.approx(target, abs=1e-3)


def test_expected_lrt_zero_at_null_and_monotone():
    assert S.expected_lrt("gaussian", -2.0, 0.0, 1000) == pytest.approx(0.0, abs=1e-9)
    vals = [S.expected_lrt("gaussian", -2.0, b, 1000) for b in [0.1, 0.5, 1.0, 2.0, 4.0]]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_nearest_logbf_recovers_grid():
    # the frozen b for L16 must map back to 16
    b = S.solve_b("binary q=0.1", 16, -2.0, 1000)
    assert S.nearest_logbf("binary q=0.1", -2.0, b, 1000) == 16
    assert S.nearest_logbf("gaussian", 0.0, 0.0, 1000) == 0


def test_frozen_b_matches_solver():
    from simulations.effect import logbf_sizing as S
    for design, b0d in S.FROZEN_B.items():
        for b0, row in b0d.items():
            for lbf, b in row.items():
                assert S.solve_b(design, lbf, b0, 1000) == pytest.approx(b, abs=2e-3), (design, b0, lbf)


def test_design_key_map():
    from simulations.effect import logbf_sizing as S
    assert S.DESIGN_KEY[("binary_markov_X", 0.1)] == "binary q=0.1"
    assert S.DESIGN_KEY[("gaussian_markov_X", None)] == "gaussian"
