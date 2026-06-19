"""Logistic-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import engine, localjj


def fit_logistic_method(
    simulation, *, response_source, threshold=None, L=1
):
    from core import _score
    if response_source == "z":
        y = np.asarray(simulation.z, dtype=float)
    elif response_source == "score_threshold":
        if threshold is None:
            raise ValueError("score_threshold logistic method requires a threshold.")
        y = (_score(simulation) > float(threshold)).astype(float)
    else:
        raise ValueError(f"Unsupported logistic response_source: {response_source}")

    data = localjj.prep_data(simulation.X, y)
    state = localjj.initialize_state(
        data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    fitted = engine.fit_ibss(data, state, localjj.default_schedule())
    return {
        "state": fitted,
        "threshold": threshold,
        "n_selected": int(np.asarray(y).sum()),
    }


def summarize_logistic_method(
    fit_obj,
    simulation,
    *,
    response_source,
    threshold=None,
    L=1,
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del response_source, threshold, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }


def run_logistic_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_logistic_method(fit_logistic_method(simulation, **kwargs), simulation, **kwargs)
