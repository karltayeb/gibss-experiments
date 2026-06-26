"""Global-JJ logistic SuSiE — four reweight cadences, one unified entry point.

Mirrors fits/irls_steps.py. The JJ "weights" are the tangent points xi; the
intercept + xi are mutually-dependent closed-form updates, so the null-model
solve is alternating ``estimate_intercept`` / ``update_xi`` to convergence at the
current SuSiE offset (`converge_jj_intercept_step`). The reweight here = that full
variational-null solve (which leaves xi consistent with the current eta).

Cadences (coarse -> fine), see irls_steps for the shared taxonomy:
  block        weights <-> full SuSiE fit   reweight per inner convergence
                                             (n_outer outer steps; n_outer=1 is
                                             GlobalJJBlock1).
  interleaved  weights <-> 1 sweep
  greedy       weights <-> 1 SER update
  thorough     weights <-> full SER fit      per-effect variational fit to conv.

For L=1 all four coincide; they diverge only at L>1. No ``center`` axis (global JJ
bound, single global intercept). Options: ``estimate_prior_variance`` /
``prior_variance``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from gibss import engine
import gibss.globaljj as _globaljj
from gibss.engine import Schedule
from gibss.globaljj import (
    add_message_index_step,
    check_elbo_convergence_step,
    compute_elbo_step,
    estimate_intercept_step,
    subtract_message_index_step,
    to_numpy_state_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    update_xi_step,
)


def converge_jj_intercept_step(data, state, max_iter: int = 100, tol: float = 1e-10):
    """Solve the JJ variational null (intercept + xi) to convergence at fixed
    effects: alternate the two closed-form coordinate updates until the intercept
    is stable. Leaves xi consistent with the current eta. Schedule step."""
    fs = state.family_state
    if not fs.estimate_intercept:
        # still refresh xi so the bound tracks the current eta
        return update_xi_step(data, state)
    prev = float(fs.intercept)
    for _ in range(max_iter):
        state = estimate_intercept_step(data, state)
        state = update_xi_step(data, state)
        cur = float(state.family_state.intercept)
        if abs(cur - prev) < tol:
            break
        prev = cur
    return state


_EFFECT = (
    subtract_message_index_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    add_message_index_step,
)


def _thorough_effect_step(data, l, state, max_iter: int = 50, tol: float = 1e-6):
    """THOROUGH: fully fit effect l's own variational SER given the others, looping
    (reweight -> refit l) until effect l's (alpha, mu) converge."""
    prev = None
    for _ in range(max_iter):
        state = converge_jj_intercept_step(data, state)
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


_INNER_BLOCK = Schedule(
    effect_update=_EFFECT,
    after_sweep=(compute_elbo_step, check_elbo_convergence_step),
)
# interleaved: reweight (converge intercept+xi) ONCE per sweep, weights fixed
# across the L effect updates in that sweep.
_SCHED_INTERLEAVED = Schedule(
    before_sweep=(converge_jj_intercept_step,),
    effect_update=_EFFECT,
    after_sweep=(compute_elbo_step, check_elbo_convergence_step),
    after_fit=(to_numpy_state_step,),
)
# greedy: reweight before EACH effect update (per l).
_SCHED_GREEDY = Schedule(
    before_effect_update=(converge_jj_intercept_step,),
    effect_update=_EFFECT,
    after_sweep=(compute_elbo_step, check_elbo_convergence_step),
    after_fit=(to_numpy_state_step,),
)
_SCHED_THOROUGH = Schedule(
    effect_update=(_thorough_effect_step,),
    after_sweep=(compute_elbo_step, check_elbo_convergence_step),
    after_fit=(to_numpy_state_step,),
)

_CADENCES = {"block", "interleaved", "greedy", "thorough"}


def _init_state(simulation, *, L, estimate_prior_variance, prior_variance):
    y = np.asarray(simulation.y, dtype=float)
    data = _globaljj.prep_data(simulation.X, y)  # pass X through (preserve sparsity)
    state = _globaljj.initialize_state(
        data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": estimate_prior_variance},
    )
    if not estimate_prior_variance:
        state = replace(
            state,
            single_effects=[
                replace(e, prior_variance=float(prior_variance))
                for e in state.single_effects
            ],
        )
    return data, state


def fit_globaljj_method(
    simulation,
    *,
    ser_cadence: str = "block",
    n_outer: int = 50,
    L: int = 1,
    estimate_prior_variance: bool = True,
    prior_variance: float = 1.0,
    max_iter: int = 200,
):
    if ser_cadence not in _CADENCES:
        raise ValueError(f"ser_cadence must be one of {sorted(_CADENCES)}; got {ser_cadence!r}")
    data, state = _init_state(
        simulation, L=L,
        estimate_prior_variance=estimate_prior_variance, prior_variance=prior_variance,
    )
    if ser_cadence == "block":
        for _ in range(int(n_outer)):
            state = converge_jj_intercept_step(data, state)
            state = replace(state, converged=False)
            state = engine.fit_ibss(data, state, _INNER_BLOCK, max_iter=100)
        state = to_numpy_state_step(data, state)
    else:
        sched = {"interleaved": _SCHED_INTERLEAVED, "greedy": _SCHED_GREEDY,
                 "thorough": _SCHED_THOROUGH}[ser_cadence]
        state = engine.fit_ibss(data, state, sched, max_iter=max_iter)
    return {"state": state, "ser_cadence": ser_cadence}


def summarize_globaljj_method(fit_obj, simulation, **kwargs):
    from core import _extract_ser_struct, _make_cs_struct, _make_fit_summary_struct
    state = fit_obj["state"]
    cad = fit_obj["ser_cadence"]
    n_eff = len(state.single_effects)
    return {
        "impl": "globaljj_block" if cad == "block" else f"globaljj_{cad}",
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_eff)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_eff)],
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_globaljj_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_globaljj_method(fit_globaljj_method(simulation, **kwargs), simulation, **kwargs)
