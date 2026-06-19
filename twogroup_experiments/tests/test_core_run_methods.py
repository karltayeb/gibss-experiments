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


def test_run_method_executes_coord():
    from experiments import loader
    sim = _tiny_simulation()
    coord = {"name": "cox_heavy__L=1", "function": "run_cox_method",
             "kwargs": {"threshold": None, "time_sign": 1.0, "L": 1}}
    row = loader.run_method(coord, sim)
    assert row["method"] == "cox_heavy__L=1"
    assert "single_effects" in row
