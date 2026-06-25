"""Block IRLS logistic SuSiE.

A "block" Fisher-scoring schedule that cleanly separates the IRLS reweighting
(outer loop) from the inner SuSiE fit, in contrast to gibss.irls's default
schedule which interleaves a reweight before every sweep.

Schedule (per the score_null <-> linear-SuSiE decomposition):
  - intercept fixed at the null-MLE logit(ybar) (estimated once, before the loop)
  - outer loop, repeated ``n_outer`` times:
      1. reweight: linearize the logistic GLM at the current eta (working
         response z + per-observation weights w)
      2. fit weighted linear SuSiE to convergence on (z, 1/w)

With effects initialized at 0, the first reweight is at the null, so
``n_outer=1`` is bit-for-bit identical to score_null. Increasing ``n_outer``
moves the expansion point toward the posterior mode (Fisher scoring); the fixed
point is the Laplace fit that gibss.irls computes via interleaved updates.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from gibss import engine
import gibss.irls as _irls
from gibss.engine import Schedule
from gibss.irls import (
    add_message_index_step,
    check_convergence_step,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    update_working_data_step,
)

# Inner schedule: reweight ONCE (before_fit), then pure weighted linear SuSiE
# sweeps to convergence — no reweight or intercept re-estimation in the loop.
_BLOCK_INNER = Schedule(
    before_fit=(update_working_data_step,),
    before_sweep=(snapshot_state_step,),
    effect_update=(
        subtract_message_index_step,
        update_effect_index_step,
        update_prior_variance_index_step,
        add_message_index_step,
    ),
    after_sweep=(check_convergence_step,),
)


def fit_block_irls_method(simulation, *, n_outer: int = 1, L: int = 1):
    y = np.asarray(simulation.y, dtype=float)
    n = y.shape[0]
    # null intercept-only logistic MLE, held fixed (clip for all-0/all-1 batches)
    ybar = float(np.clip(y.mean(), 1.0 / (n + 1), n / (n + 1)))
    eta0 = float(np.log(ybar / (1.0 - ybar)))

    data = _irls.prep_data(simulation.X, y)  # pass X through (preserve sparsity)
    state = _irls.initialize_state(
        data,
        L=L,
        # global intercept fixed at the null MLE, no weighted centering -> the
        # n_outer=1 truncation stays bit-identical to score_null.
        family_state_kwargs={"intercept": eta0, "estimate_intercept": False, "center": False},
    )
    for _ in range(int(n_outer)):
        # before_fit reweights at the current eta; reset converged so the inner
        # sweeps actually run (a converged state would skip the loop).
        state = replace(state, converged=False)
        state = engine.fit_ibss(data, state, _BLOCK_INNER, max_iter=200)
    state = to_numpy_state_step(data, state)
    return {"state": state, "n_outer": int(n_outer)}


def summarize_block_irls_method(fit_obj, simulation, *, n_outer: int = 1, L: int = 1):
    from core import _extract_ser_struct, _make_cs_struct, _make_fit_summary_struct
    del n_outer, L
    state = fit_obj["state"]
    n_eff = len(state.single_effects)
    return {
        "impl": "block_irls",
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_eff)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_eff)],
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_block_irls_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_block_irls_method(
        fit_block_irls_method(simulation, **kwargs), simulation, **kwargs
    )
