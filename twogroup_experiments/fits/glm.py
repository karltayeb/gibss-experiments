"""GLM-SuSiE fit over a Response (logistic / poisson).

One method per non-quadratic ``fit_glm_susie`` family. The response to regress is
supplied as a resolved response transform (``simulation -> Response``); the fit
applies it and never selects a simulation field itself. Linear (gaussian) stays in
``fits/linear.py`` (``gibss.linear`` estimates residual variance; the gaussian family
would pin it to 1).

See docs/superpowers/specs/2026-07-27-response-transform-design.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss.methods import fit_glm_susie

# GH order for leave-one-out offset integration once L > 1 (SuSiE). At L=1 (a bare
# SER) there is no offset, so the fit is plain GH-quadrature on the exact marginal.
OFFSET_QUADRATURE_POINTS = 5

_SUPPORTED_FAMILIES = ("logistic", "poisson")


def fit_glm_method(
    simulation,
    *,
    response,
    family,
    method=None,
    L=1,
    estimate_prior_variance=False,
) -> dict[str, Any]:
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"unsupported glm family {family!r}; options: {list(_SUPPORTED_FAMILIES)} "
            "(linear/gaussian goes through run_linear_method)."
        )
    y = np.asarray(response(simulation).values, dtype=float)
    kwargs = dict(
        L=L,
        family=family,
        center=True,
        estimate_prior_variance=estimate_prior_variance,
    )
    if method is not None:
        # A named preset (e.g. "score" = Taylor-fixed linearization anchored at the
        # null intercept - the logistic_score approximation) owns offset_integration /
        # anchor / intercept, so we pass it through and do NOT override those.
        kwargs["method"] = method
    else:
        kwargs["offset_integration"] = "none" if L == 1 else "gh"
        kwargs["offset_quadrature_points"] = OFFSET_QUADRATURE_POINTS
    fitted = fit_glm_susie(simulation.X, y, **kwargs)
    return {"state": fitted, "n_selected": int(y.sum())}


def summarize_glm_method(
    fit_obj,
    simulation,
    *,
    response,
    family,
    method=None,
    L=1,
    estimate_prior_variance=False,
) -> dict[str, Any]:
    from core import (
        _extract_ser_struct,
        _extract_family_state_struct,
        _extract_twogroup_state_struct,
        _make_cs_struct,
        _make_fit_summary_struct,
    )
    del response, family, method, L, estimate_prior_variance
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }


def run_glm_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_glm_method(fit_glm_method(simulation, **kwargs), simulation, **kwargs)
