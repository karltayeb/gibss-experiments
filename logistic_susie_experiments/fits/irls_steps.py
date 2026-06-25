"""IRLS logistic SuSiE with a capped number of outer steps.

Runs gibss.irls (intercept Newton + reweight + optional weighted centering +
effect update per sweep) for ``n_outer`` sweeps. ``n_outer=1`` is one outer
step; a large ``n_outer`` runs to convergence. Exposes the two controls that the
002_irls factorial varies:

- ``center``: weighted column centering = per-feature (local) intercept (FWL).
  False = single global intercept.
- ``estimate_prior_variance`` / ``prior_variance``: EB vs a fixed prior variance.

For L=1 the inner SER is non-iterative, so one ``fit_ibss`` sweep == one outer
IRLS step (block and interleaved coincide).
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
    update_working_data_step,
)


def _intercept_to_convergence_step(data, state, n_newton: int = 50, tol: float = 1e-12):
    """Run the intercept Newton update to convergence (vs gibss's single step),
    holding effects fixed. At effects=0 this yields the null MLE logit(ybar), so
    the n_outer=1 truncation matches score_null (the calibrated one-step)."""
    fs = state.family_state
    if not fs.estimate_intercept:
        return state
    b0 = float(fs.intercept)
    mean = jnp.asarray(state.total_message.mean)
    offset = jnp.asarray(fs.glm_offset)
    yv = jnp.asarray(data.y)
    for _ in range(n_newton):
        mu, w = fs.glm.mean_and_weight(offset + b0 + mean)
        step = float(jnp.sum(yv - mu) / jnp.maximum(jnp.sum(w), 1e-8))
        b0 += step
        if abs(step) < tol:
            break
    return replace(state, family_state=replace(fs, intercept=b0))


# Inner schedule: effect + EB to convergence at FIXED weights/intercept/centering
# (no reweight, no intercept update). This is the "block" inner SER fit.
_INNER = Schedule(
    before_sweep=(snapshot_state_step,),
    effect_update=(
        subtract_message_index_step,
        update_effect_index_step,
        update_prior_variance_index_step,
        add_message_index_step,
    ),
    after_sweep=(check_convergence_step,),
)


def fit_irls_steps_method(
    simulation,
    *,
    n_outer: int = 1,
    L: int = 1,
    center: bool = True,
    estimate_prior_variance: bool = True,
    prior_variance: float = 1.0,
):
    y = np.asarray(simulation.y, dtype=float)
    data = _irls.prep_data(simulation.X, y)  # pass X through (preserve sparsity)
    state = _irls.initialize_state(
        data,
        L=L,
        family_state_kwargs={
            "center": center,
            "estimate_prior_variance": estimate_prior_variance,
        },
    )
    if not estimate_prior_variance:
        # fix the slab variance: set every effect's prior_variance, no EB update
        state = replace(
            state,
            single_effects=tuple(
                replace(e, prior_variance=float(prior_variance))
                for e in state.single_effects
            ),
        )
    # Block outer loop: each step = converge intercept -> reweight -> center ->
    # inner SER (effect + EB) to convergence. n_outer=1 => one linearization with
    # everything converged at it == score_null (calibrated); large n_outer => the
    # Laplace fixed point.
    for _ in range(int(n_outer)):
        state = _intercept_to_convergence_step(data, state)
        state = update_working_data_step(data, state)
        state = update_centering_step(data, state)
        state = replace(state, converged=False)
        state = engine.fit_ibss(data, state, _INNER, max_iter=100)
    state = to_numpy_state_step(data, state)
    return {"state": state, "n_outer": int(n_outer)}


def summarize_irls_steps_method(fit_obj, simulation, **kwargs):
    from core import _extract_ser_struct, _make_cs_struct, _make_fit_summary_struct
    state = fit_obj["state"]
    n_eff = len(state.single_effects)
    return {
        "impl": "irls_steps",
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_eff)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_eff)],
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_irls_steps_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_irls_steps_method(
        fit_irls_steps_method(simulation, **kwargs), simulation, **kwargs
    )
