"""Twogroup-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import engine, localjj, twogroup, twogrouplocaljj
from gibss.distributions import Normal


def fit_twogroup_method(
    simulation,
    *,
    f1,
    L=1,
    oracle_init=False,
    n_null_iter=20,
    n_intercept_iter=20,
    em_update="local",
):
    if oracle_init:
        resolved_f1 = Normal(
            loc=simulation.f1.loc,
            scale=simulation.f1.scale,
            estimate_loc=f1.estimate_loc,
            estimate_scale=f1.estimate_scale,
        )
    else:
        resolved_f1 = simulation.f1 if f1 is None else f1
    y0 = np.full(simulation.X.shape[0], 0.5, dtype=float)
    if em_update == "local":
        inner_module = twogrouplocaljj
        schedule = twogroup.local_default_schedule(twogrouplocaljj.default_schedule())
    elif em_update == "global":
        inner_module = localjj
        schedule = twogroup.default_schedule(localjj.default_schedule())
    else:
        raise ValueError(f"Unknown twogroup em_update mode: {em_update!r}")

    inner_data = inner_module.prep_data(simulation.X, y0)
    inner_state = inner_module.initialize_state(
        inner_data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    data = twogroup.prep_data(simulation.X, bhat=simulation.thetahat, se=simulation.se)
    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=simulation.f0,
        f1=resolved_f1,
        n_null_iter=n_null_iter,
        n_intercept_iter=n_intercept_iter,
    )
    fitted = engine.fit_ibss(data, state, schedule)
    return {
        "state": fitted,
        "threshold": None,
        "n_selected": None,
    }


def summarize_twogroup_method(
    fit_obj,
    simulation,
    *,
    f1,
    L=1,
    oracle_init=False,
    n_null_iter=20,
    n_intercept_iter=20,
    em_update="local",
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del f1, L, oracle_init, n_null_iter, n_intercept_iter, em_update
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


def run_twogroup_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_twogroup_method(fit_twogroup_method(simulation, **kwargs), simulation, **kwargs)
