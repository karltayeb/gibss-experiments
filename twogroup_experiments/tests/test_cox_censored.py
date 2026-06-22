from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import core
from fits.cox import _right_censored_survival


def _tiny_simulation():
    from core import SimulationSpec, simulate
    from functools import partial
    from gibss.distributions import Normal, PointMass
    spec = SimulationSpec(
        design_sampler=partial(core.gaussian_markov_X, n=40, p=10, rho=0.5),
        effect_sampler=partial(core.uniform_single_effect, causal_effect=2.0),
        intercept=-1.0,
        f0=PointMass(0.0),
        f1=Normal(loc=2.0, scale=0.1, estimate_loc=False, estimate_scale=False),
        error_sampler=None,
        base_seed=1,
        hash="coxcensored",
        name="tiny",
    )
    return simulate(spec, 0)


def test_right_censored_survival_plus_one():
    score = np.array([0.2, 1.0, 2.5, 3.0])
    t = 1.5
    et, ev = _right_censored_survival(score, t, 1.0)
    # +1: event iff |z| <= t; censored arrivals clamped to t
    np.testing.assert_array_equal(ev, np.array([1, 1, 0, 0]))
    np.testing.assert_allclose(et, np.array([0.2, 1.0, 1.5, 1.5]))


def test_right_censored_survival_minus_one():
    score = np.array([0.2, 1.0, 2.5, 3.0])
    t = 1.5
    et, ev = _right_censored_survival(score, t, -1.0)
    # -1: raw=-score, T=-t; event iff -score <= -t  <=>  score >= t
    np.testing.assert_array_equal(ev, np.array([0, 0, 1, 1]))
    # event_time = min(-score, -t) = -max(score, t)
    np.testing.assert_allclose(et, np.array([-1.5, -1.5, -2.5, -3.0]))


def test_cox_minus_one_threshold_matches_pre_change_behavior():
    # No-regression guard: new censored cox (time_sign=-1) == a direct gibss
    # fit using the OLD construction (event_type = score>t, event_time = -score).
    from gibss import cox, engine
    sim = _tiny_simulation()
    score = np.abs(np.asarray(sim.thetahat) / np.asarray(sim.se))
    t = float(np.median(score))  # ensure both events and censored exist

    data_old = cox.prep_data(
        sim.X, event_time=-1.0 * score, event_type=(score > t).astype(int)
    )
    st_old = cox.initialize_state(
        data_old, L=1, family_state_kwargs={"estimate_prior_variance": False}
    )
    fit_old = engine.fit_ibss(data_old, st_old, cox.default_schedule())
    alpha_old = np.asarray(fit_old.single_effects[0].alpha)

    new = core.run_cox_method(sim, threshold=t, time_sign=-1.0, L=1)
    alpha_new = np.asarray(new["single_effects"][0]["alpha"])
    np.testing.assert_allclose(alpha_new, alpha_old, atol=1e-8)
