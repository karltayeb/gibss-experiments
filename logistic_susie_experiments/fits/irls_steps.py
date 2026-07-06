"""IRLS logistic SuSiE — four reweight cadences, one unified entry point.

The intercept (null model) is ALWAYS solved to convergence at each linearization
point (`converge_intercept_step`: Newton on a concave objective at the current
SuSiE offset, quadratic convergence). What varies is the **reweight cadence** —
how much SER/SuSiE work happens per GLM re-linearization (weight+response update),
coarse -> fine:

  block        weights <-> full SuSiE fit   reweight once per inner IBSS-to-
                                             convergence (n_outer outer steps).
                                             n_outer=1, profile=false == score_null.
  interleaved  weights <-> 1 sweep          reweight before each sweep (one pass
                                             over l=1..L).
  greedy       weights <-> 1 SER update     reweight before each single (closed-
                                             form) effect update, cycling l.
  thorough     weights <-> full SER fit     for each effect l, loop (reweight +
                                             refit l) until effect l self-converges
                                             before moving on (per-effect Laplace).

Reweights per outer pass: block 1 < interleaved (#sweeps) < greedy (L*#sweeps) <
thorough (sum of per-effect iters). For L=1 all four coincide (one effect: a sweep
= one SER update = converged inner); they diverge only at L>1.

Options (all): ``center`` (column pre-centering, prep_data preprocessing),
``profile`` (weighted FWL / intercept profiling, distinct from ``center``),
``estimate_prior_variance`` / ``prior_variance`` (EB vs fixed slab).
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
    default_schedule,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
    update_centering_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    update_working_data_step,
)


def converge_intercept_step(data, state, n_newton: int = 100, tol: float = 1e-12):
    """Solve the intercept (null model) to convergence at the current SuSiE offset.

    Newton on b0 with mu, w recomputed each step at eta = glm_offset + b0 +
    total_message.mean (effects held fixed). Quadratic convergence. At effects=0
    and zero offset this is exactly the null MLE logit(ybar); with a nonzero SER /
    covariate offset it is the correct conditional intercept MLE. Schedule step."""
    fs = state.family_state
    if not fs.estimate_intercept:
        return state
    b0 = float(fs.intercept)
    offset = jnp.asarray(fs.glm_offset)
    mean = jnp.asarray(state.total_message.mean)
    yv = jnp.asarray(data.y)
    for _ in range(n_newton):
        mu, w = fs.glm.mean_and_weight(offset + b0 + mean)
        step = float(jnp.sum(yv - mu) / jnp.maximum(jnp.sum(w), 1e-8))
        b0 += step
        if abs(step) < tol:
            break
    return replace(state, family_state=replace(fs, intercept=b0))


# The reweight = full intercept solve + GLM linearization (working data) + weighted
# centering, in that order (centering uses the new weights).
_REWEIGHT = (converge_intercept_step, update_working_data_step, update_centering_step)

_EFFECT = (
    subtract_message_index_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    add_message_index_step,
)


def _thorough_effect_step(data, l, state, max_iter: int = 50, tol: float = 1e-6):
    """THOROUGH: fully fit effect l's own logistic SER (Laplace) given the others.

    Loops (reweight at current eta -> refit effect l) until effect l's (alpha, mu)
    stop moving, i.e. effect l converges to its own fixed point under reweighting,
    before the sweep advances to l+1."""
    prev = None
    for _ in range(max_iter):
        for step in _REWEIGHT:
            state = step(data, state)
        for step in _EFFECT:
            state = step(data, l, state)
        e = state.single_effects[l]
        cur = (np.asarray(e.alpha), np.asarray(e.mu))
        if prev is not None:
            d = max(np.abs(cur[0] - prev[0]).max(), np.abs(cur[1] - prev[1]).max())
            if d < tol:
                break
        prev = cur
    return state


# Schedules per cadence (block runs its own outer loop; the rest fit_ibss to conv).
_INNER_BLOCK = Schedule(
    before_sweep=(snapshot_state_step,),
    effect_update=_EFFECT,
    after_sweep=(check_convergence_step,),
)
_SCHED_INTERLEAVED = Schedule(
    before_sweep=(snapshot_state_step, *_REWEIGHT),
    effect_update=_EFFECT,
    after_sweep=(check_convergence_step,),
    after_fit=(to_numpy_state_step,),
)
_SCHED_GREEDY = Schedule(
    before_sweep=(snapshot_state_step,),
    before_effect_update=_REWEIGHT,
    effect_update=_EFFECT,
    after_sweep=(check_convergence_step,),
    after_fit=(to_numpy_state_step,),
)
_SCHED_THOROUGH = Schedule(
    before_sweep=(snapshot_state_step,),
    effect_update=(_thorough_effect_step,),
    after_sweep=(check_convergence_step,),
    after_fit=(to_numpy_state_step,),
)

# "native" = gibss's own default_schedule (single intercept update + reweight per
# sweep, iterate to convergence). PREFERRED for converged IRLS: numerically robust
# under extreme class/feature imbalance, where the repo block cadence (over-solved
# intercept x many reweights) diverges. The block/interleaved/greedy/thorough
# cadences remain for the reweight-cadence comparison (002), not as the default.
_CADENCES = {"native", "block", "interleaved", "greedy", "thorough"}


def _init_state(simulation, *, L, center, profile, offset_integration,
                estimate_prior_variance, prior_variance):
    y = np.asarray(simulation.y, dtype=float)
    # `center`: column pre-centering (prep_data preprocessing). `profile`:
    # per-iteration WEIGHTED centering = intercept profiling (family flag).
    data = _irls.prep_data(simulation.X, y, center=center)
    fsk = {"profile": profile, "estimate_prior_variance": estimate_prior_variance}
    if offset_integration is not None:  # off/taylor/GH-k over leave-one-out offset var
        from fits.logistic import _offset_integration_value
        fsk["offset_integration"] = _offset_integration_value(offset_integration)
    state = _irls.initialize_state(data, L=L, family_state_kwargs=fsk)
    if not estimate_prior_variance:
        state = replace(
            state,
            single_effects=tuple(
                replace(e, prior_variance=float(prior_variance))
                for e in state.single_effects
            ),
        )
    return data, state


def fit_irls_method(
    simulation,
    *,
    ser_cadence: str = "block",
    n_outer: int = 50,
    L: int = 1,
    center: bool = True,
    profile: bool = False,
    offset_integration: str | int | bool = "taylor",
    estimate_prior_variance: bool = True,
    prior_variance: float = 1.0,
    max_iter: int = 200,
):
    if ser_cadence not in _CADENCES:
        raise ValueError(f"ser_cadence must be one of {sorted(_CADENCES)}; got {ser_cadence!r}")
    data, state = _init_state(
        simulation, L=L, center=center, profile=profile,
        offset_integration=offset_integration,
        estimate_prior_variance=estimate_prior_variance, prior_variance=prior_variance,
    )
    if ser_cadence == "block":
        # outer reweight loop; inner SuSiE to convergence at fixed linearization.
        # n_outer=1 (center=false) == score_null; large n_outer -> Laplace.
        for _ in range(int(n_outer)):
            for step in _REWEIGHT:
                state = step(data, state)
            state = replace(state, converged=False)
            state = engine.fit_ibss(data, state, _INNER_BLOCK, max_iter=100)
        state = to_numpy_state_step(data, state)
    else:
        sched = {"native": default_schedule(), "interleaved": _SCHED_INTERLEAVED,
                 "greedy": _SCHED_GREEDY, "thorough": _SCHED_THOROUGH}[ser_cadence]
        state = engine.fit_ibss(data, state, sched, max_iter=max_iter)
    return {"state": state, "ser_cadence": ser_cadence}


def summarize_irls_method(fit_obj, simulation, **kwargs):
    from core import _extract_ser_struct, _make_cs_struct, _make_fit_summary_struct
    state = fit_obj["state"]
    cad = fit_obj["ser_cadence"]
    n_eff = len(state.single_effects)
    return {
        "impl": "irls_block" if cad == "block" else f"irls_{cad}",
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_eff)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_eff)],
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_irls_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_irls_method(fit_irls_method(simulation, **kwargs), simulation, **kwargs)
