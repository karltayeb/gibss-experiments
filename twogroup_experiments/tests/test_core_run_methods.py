from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import core


def _tiny_simulation():
    from core import SimulationSpec, simulate
    from functools import partial
    from gibss.distributions import Normal, PointMass
    spec = SimulationSpec(
        design_sampler=partial(core.gaussian_markov_X, n=30, p=8, rho=0.5),
        effect_sampler=partial(core.uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        error_sampler=None,
        base_seed=1,
        hash="tinyhash",
        name="tiny",
    )
    return simulate(spec, 0)


def test_run_cox_method_returns_summary_row():
    sim = _tiny_simulation()
    row = core.run_cox_method(sim, threshold=None, time_sign=1.0, L=1)
    assert row.get("method") is None or "single_effects" in row  # row is the summarize dict
    assert "single_effects" in row and "fit_summary" in row


def test_run_method_getfile():
    import inspect, core
    assert inspect.getfile(core.run_twogroup_method).endswith("fits/twogroup.py")


def test_run_method_executes_coord():
    from experiments import loader
    sim = _tiny_simulation()
    coord = {"name": "cox_reversed__L=1", "function": "run_cox_method",
             "kwargs": {"threshold": None, "time_sign": 1.0, "L": 1}}
    row = loader.run_method(coord, sim)
    assert row["method"] == "cox_reversed__L=1"
    assert "single_effects" in row


def test_noiseless_exponential_simulation_preserves_event_time_ranking():
    from functools import partial
    from core import SimulationSpec, simulate
    from simulations.distributions import Exponential

    spec = SimulationSpec(
        design_sampler=partial(core.gaussian_markov_X, n=30, p=8, rho=0.5),
        effect_sampler=partial(core.uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=Exponential(rate=1.0),
        f1=Exponential(rate=2.0),
        error_sampler=core.noiseless_error_sampler,
        base_seed=1,
        hash="tiny-exp",
        name="tiny-exp",
    )

    sim = simulate(spec, 0)
    score = core._score(sim)

    assert np.all(sim.theta >= 0)
    assert np.all(sim.se == 1.0)
    assert np.allclose(sim.thetahat, sim.theta)
    assert np.array_equal(np.argsort(score), np.argsort(sim.theta))


def test_deterministic_membership_pins_z_to_threshold():
    from core import SimulationSpec, simulate
    from simulations.distributions import Exponential

    # Fixed 6x3 binary design; deterministic effect sampler picks column 1.
    X = np.array(
        [[0, 0, 1],
         [0, 1, 0],
         [0, 1, 1],
         [0, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        dtype=float,
    )
    spec = SimulationSpec(
        design_sampler=lambda rng: X,
        effect_sampler=lambda Xarg, rng: ([1], [2.0]),  # causal column 1, effect 2
        intercept=-1.0,
        f0=Exponential(rate=1.0),
        f1=Exponential(rate=0.5),
        error_sampler=None,
        base_seed=1,
        hash="dethash",
        name="det",
        membership="deterministic",
    )
    sim = simulate(spec, 0)
    logits = sim.intercept + sim.X @ sim.b
    np.testing.assert_array_equal(sim.z, (logits > 0).astype(int))
    # effect=2, intercept=-1 -> z equals the binary causal column exactly
    np.testing.assert_array_equal(sim.z, X[:, 1].astype(int))


def test_membership_defaults_to_stochastic():
    from core import SimulationSpec
    from gibss.distributions import PointMass
    spec = SimulationSpec(
        design_sampler=lambda rng: np.zeros((2, 2)),
        effect_sampler=lambda Xarg, rng: ([], []),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=PointMass(1.0),
        error_sampler=None,
        base_seed=1,
        hash="h",
        name="n",
    )
    assert spec.membership == "stochastic"
