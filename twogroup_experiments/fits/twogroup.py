"""Twogroup-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

from gibss import engine, twogroup
from gibss.distributions import Normal
from gibss.response import GH, Smoothed, TwoGroupMarginal

# GH order for leave-one-out offset integration once L > 1 (SuSiE). At L=1 (a bare
# SER) the message is zero, so the fit is plain GH-quadrature on the exact
# z-marginal (TwoGroupMarginal); Smoothed(..., GH(k)) only matters across effects.
OFFSET_QUADRATURE_POINTS = 5


def fit_twogroup_method(
    simulation,
    *,
    f1,
    L=1,
    oracle_init=False,
    nullweight=1.0,
    # Legacy knobs of the old nested inner/outer EM implementation. The daf5a24
    # twogroup family marginalizes z analytically (no inner SER state, no per-sweep
    # inner-iteration counts, and only the "local" M-step survives), so these are
    # accepted and ignored to keep the shared library templates resolving.
    n_null_iter=20,
    n_intercept_iter=20,
    em_update="local",
):
    del n_null_iter, n_intercept_iter, em_update
    if oracle_init:
        resolved_f1 = Normal(
            loc=simulation.f1.loc,
            scale=simulation.f1.scale,
            estimate_loc=f1.estimate_loc,
            estimate_scale=f1.estimate_scale,
        )
    else:
        resolved_f1 = simulation.f1 if f1 is None else f1

    response = (
        TwoGroupMarginal()
        if L == 1
        else Smoothed(TwoGroupMarginal(), GH(OFFSET_QUADRATURE_POINTS))
    )
    data = twogroup.prep_data(
        simulation.X, bhat=simulation.thetahat, se=simulation.se, center=True
    )
    state = twogroup.initialize_state(
        data,
        L=L,
        f0=simulation.f0,
        f1=resolved_f1,
        response=response,
        family_state_kwargs={"estimate_prior_variance": False},
        nullweight=nullweight,
    )
    fitted = engine.fit_ibss(data, state, twogroup.default_schedule())
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
    nullweight=1.0,
    n_null_iter=20,
    n_intercept_iter=20,
    em_update="local",
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del f1, L, oracle_init, nullweight, n_null_iter, n_intercept_iter, em_update
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
