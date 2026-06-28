"""Linear SuSiE on the raw binary response (Gaussian/identity model).

The naive baseline: treat y in {0,1} as a continuous Gaussian response and run
linear SuSiE directly — NO working-response recode (unlike fits/score, which fits
a linear SER to z = working response), and with the covariates CENTERED.

Implemented by reusing gibss.irls's centered SER machinery with an identity model.
It is just ONE inner SuSiE fit (IBSS coordinate ascent to convergence): no GLM
reweighting ever, and no intercept estimation (centering absorbs the mean).
- y_work = y (raw); weights never recomputed.
- weighted column centering with uniform weights = ordinary column centering,
  computed ONCE (sparsity-preserving: cbar = colmeans, never densifies BCOO X).
- intercept fixed at ybar; centering makes its estimation moot.
- residual variance FIXED at 1/4. Under a logistic model Var(y_i)=p_i(1-p_i) <=
  1/4 = Var(Bernoulli(1/2)), so 1/4 is the worst-case noise — fixing there is the
  safe/conservative choice (avoids the anti-conservative inflation an EB residual
  variance would give by underestimating the noise).
- per-effect prior variance: EB or fixed.

gibss.linear is NOT used because it centers nothing (its univariate fit divides by
the uncentered sum x^2), which attenuates slopes on uncentered designs (c4). The
irls centered SER divides by the centered weighted variance, which is correct.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import jax.numpy as jnp

from gibss import engine
import gibss.irls as _irls
from gibss.engine import Schedule
from gibss.irls import (
    add_message_index_step,
    check_convergence_step,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
    update_centering_step,
    update_effect_index_step,
    update_prior_variance_index_step,
)

# Max Bernoulli variance: Var(y) <= Var(Bernoulli(1/2)) = 1/4 under any logistic model.
_RESIDUAL_VARIANCE = 0.25


def fit_linear_susie_method(
    simulation,
    *,
    L: int = 1,
    estimate_prior_variance: bool = True,
    prior_variance: float = 1.0,
    residual_variance: float = _RESIDUAL_VARIANCE,
    max_iter: int = 100,
):
    y = np.asarray(simulation.y, dtype=float)
    n = y.shape[0]
    ybar = float(y.mean())

    data = _irls.prep_data(simulation.X, y)  # X, y, X_sq (BCOO preserved)
    state = _irls.initialize_state(
        data,
        L=L,
        family_state_kwargs={
            "center": True,                          # implicit column centering (sparse-safe)
            "estimate_prior_variance": estimate_prior_variance,
            "intercept": ybar,                       # centering handles the response mean
            "estimate_intercept": False,
            "y_work": jnp.asarray(y),                # identity model: working response = y
            "v_work": jnp.full(n, float(residual_variance)),  # fixed residual variance, never updated
        },
    )
    if not estimate_prior_variance:
        state = replace(
            state,
            single_effects=tuple(
                replace(e, prior_variance=float(prior_variance)) for e in state.single_effects
            ),
        )
    state = update_centering_step(data, state)  # cbar = colmeans, weight_sum = sum tau (once)

    effect = [subtract_message_index_step, update_effect_index_step]
    if estimate_prior_variance:
        effect.append(update_prior_variance_index_step)
    effect.append(add_message_index_step)

    # Pure inner SuSiE: sweep effects (+EB prior var) to convergence. No reweight,
    # no intercept/residual-variance updates.
    sched = Schedule(
        before_sweep=(snapshot_state_step,),
        effect_update=tuple(effect),
        after_sweep=(check_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
    state = engine.fit_ibss(data, state, sched, max_iter=max_iter)
    return {"state": state}


def summarize_linear_susie_method(fit_obj, simulation, **kwargs):
    from core import _extract_ser_struct, _make_cs_struct, _make_fit_summary_struct
    state = fit_obj["state"]
    n_eff = len(state.single_effects)
    return {
        "impl": "linear_susie",
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_eff)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_eff)],
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_linear_susie_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_linear_susie_method(
        fit_linear_susie_method(simulation, **kwargs), simulation, **kwargs
    )
