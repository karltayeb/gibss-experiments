"""Linear-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import engine, linear


def fit_linear_method(
    simulation,
    *,
    estimate_residual_variance: bool,
    abs_response: bool = False,
    L: int = 1,
) -> dict[str, Any]:
    if abs_response:
        response = np.abs(
            np.asarray(simulation.thetahat, dtype=float)
            / np.asarray(simulation.se, dtype=float)
        )
    else:
        response = simulation.thetahat
    data = linear.prep_data(simulation.X, response, center=True)
    state = linear.initialize_state(
        data,
        L=L,
        family_state_kwargs={
            "estimate_residual_variance": estimate_residual_variance,
            "residual_variance": 1.0,
        },
    )
    fitted = engine.fit_ibss(data, state, linear.default_schedule())
    return {"state": fitted}


def summarize_linear_method(
    fit_obj,
    simulation,
    *,
    estimate_residual_variance: bool,
    abs_response: bool = False,
    L: int = 1,
) -> dict[str, Any]:
    from core import _extract_ser_struct, _extract_family_state_struct, _make_cs_struct, _make_fit_summary_struct
    del estimate_residual_variance, abs_response, L
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
