"""Logistic-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss.methods import fit_glm_susie

# GH order for leave-one-out offset integration once L > 1 (SuSiE). At L=1 (a bare
# SER) there is no offset, so the fit is plain GH-quadrature on the exact marginal.
OFFSET_QUADRATURE_POINTS = 5


def fit_logistic_method(
    simulation, *, response_source, threshold=None, L=1,
    offset_integration=None, offset_quadrature_points=None,
    center=True, intercept=None,
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

    # How the leave-one-out (inter-component) offset is integrated once L > 1. Default
    # keeps the historical behaviour (no offset at L=1; Gauss-Hermite otherwise); an
    # explicit value pins the fidelity for a study: "none" (plug-in mean), "gh"
    # (moment-matched), "compress" (full mixture, exact sequential fold over components).
    integ = offset_integration if offset_integration is not None else ("none" if L == 1 else "gh")
    kwargs = dict(
        L=L,
        family="logistic",
        center=center,
        estimate_prior_variance=False,
        offset_integration=integ,
        offset_quadrature_points=(
            OFFSET_QUADRATURE_POINTS if offset_quadrature_points is None
            else int(offset_quadrature_points)
        ),
    )
    if intercept is not None:
        kwargs["intercept"] = intercept
    # NB: the Compress Chebyshev degree M (offset_integration="compress") is not exposed
    # by fit_glm_susie; it uses the engine default (M=48), which is past the interpolation
    # floor for this offset (the residual vs full MixtureGH is ~1e-4 on the log-BF).
    fitted = fit_glm_susie(simulation.X, y, **kwargs)
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
    offset_integration=None,
    offset_quadrature_points=None,
    center=True,
    intercept=None,
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del response_source, threshold, L, offset_integration, offset_quadrature_points, center, intercept
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
