"""Linear-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import engine, linear


def fit_linear_method(
    simulation,
    *,
    estimate_residual_variance: bool,
    min_residual_variance: float = 0.0,
    abs_response: bool = False,
    response=None,
    L: int = 1,
) -> dict[str, Any]:
    # `response` (a resolved response transform, simulation -> Response) is the
    # preferred path; it subsumes `abs_response` (== response `[stat, standardize, abs]`)
    # and lets linear regress any observation (e.g. `z`). The legacy thetahat/abs_response
    # branch is kept for the existing 011/012 method entries.
    if response is not None:
        r = response(simulation)
        y = np.asarray(r.values, dtype=float)
        # se on the Response is the observation-noise sd; the fit weights by 1/se^2
        # (LinearData: Var(y_i) = residual_variance * obs_variance_i). `standardize`
        # resets se->1 (unweighted z-score), `abs` drops it (unweighted); a raw `stat`
        # response keeps se and is inverse-variance (GLS) weighted.
        obs_variance = None if r.se is None else np.asarray(r.se, dtype=float) ** 2
    elif abs_response:
        # legacy |z-score| path (linear_abs): unweighted, unchanged
        y = np.abs(
            np.asarray(simulation.thetahat, dtype=float)
            / np.asarray(simulation.se, dtype=float)
        )
        obs_variance = None
    else:
        # legacy raw-thetahat path (linear_fixed): unweighted, unchanged
        y = np.asarray(simulation.thetahat, dtype=float)
        obs_variance = None
    data = linear.prep_data(simulation.X, y, center=True, obs_variance=obs_variance)
    state = linear.initialize_state(
        data,
        L=L,
        family_state_kwargs={
            "estimate_residual_variance": estimate_residual_variance,
            "residual_variance": 1.0,
            "min_residual_variance": min_residual_variance,
        },
    )
    fitted = engine.fit_ibss(data, state, linear.default_schedule())
    return {"state": fitted}


def summarize_linear_method(
    fit_obj,
    simulation,
    *,
    estimate_residual_variance: bool,
    min_residual_variance: float = 0.0,
    abs_response: bool = False,
    response=None,
    L: int = 1,
) -> dict[str, Any]:
    from core import _extract_ser_struct, _extract_family_state_struct, _make_cs_struct, _make_fit_summary_struct
    del estimate_residual_variance, min_residual_variance, abs_response, response, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": None,
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_linear_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_linear_method(fit_linear_method(simulation, **kwargs), simulation, **kwargs)
