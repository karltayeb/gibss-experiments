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

from gibss import engine
import gibss.irls as _irls


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
    fitted = engine.fit_ibss(data, state, _irls.default_schedule(), max_iter=int(n_outer))
    return {"state": fitted, "n_outer": int(n_outer)}


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
