"""Logistic-family fit/summarize/run methods."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from gibss.methods import fit_glm_susie

# GH order for leave-one-out offset integration once L > 1 (SuSiE). At L=1 (a bare
# SER) there is no offset, so the fit is plain GH-quadrature on the exact marginal.
OFFSET_QUADRATURE_POINTS = 5


def fit_logistic_method(
    simulation, *, response_source, threshold=None, L=1,
    offset_integration=None, offset_quadrature_points=None,
    variational_family=None, center=True, intercept=None, densify=False,
    estimate_prior_variance=False, max_prior_variance=None, max_iter=None,
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
    # (moment-matched), "compress_selfnorm" (exact free-form CAVI in Q1), or "compress"
    # with variational_family="gaussian" (exact CAVI in Q2). The old moment-projected
    # "compress" + unconstrained path was dropped upstream as dominated.
    integ = offset_integration if offset_integration is not None else ("none" if L == 1 else "gh")
    # estimate_prior_variance defaults False (historical behaviour: the effect prior
    # variance is fixed at 1, so a method comparison isolates the SER approximation).
    # A method may opt into EB estimation of the per-effect prior variance; the estimate
    # can be capped with max_prior_variance (a hard ceiling, applied only when set).
    kwargs = dict(
        L=L,
        family="logistic",
        center=center,
        estimate_prior_variance=bool(estimate_prior_variance),
        offset_integration=integ,
        offset_quadrature_points=(
            OFFSET_QUADRATURE_POINTS if offset_quadrature_points is None
            else int(offset_quadrature_points)
        ),
    )
    if max_prior_variance is not None:
        kwargs["max_prior_variance"] = float(max_prior_variance)
    # Cap the IBSS sweep count (fit_glm_susie default 100). Set for the expensive CAVI
    # arm so a slow-to-converge fit is bounded; unset methods keep the engine default.
    if max_iter is not None:
        kwargs["max_iter"] = int(max_iter)
    # The variational family over each effect: the default "unconstrained" (free-form q,
    # exact CAVI in Q1) or "gaussian" (Gaussian q, exact CAVI in Q2). Only set when pinned
    # by the method so the engine default is otherwise preserved.
    if variational_family is not None:
        kwargs["variational_family"] = variational_family
    if intercept is not None:
        kwargs["intercept"] = intercept
    # NB: the Compress Chebyshev degree M (offset_integration="compress") is not exposed
    # by fit_glm_susie; it uses the engine default (M=48), which is past the interpolation
    # floor for this offset (the residual vs full MixtureGH is ~1e-4 on the log-BF).
    #
    # `densify`: materialize a sparse (BCOO) design to dense before fitting. The conjugate
    # local-JJ kernel (variational_family="gaussian", offset_integration="jj") rejects a
    # centered BCOO design ("center=True not supported for kernel='jj' on a sparse design"),
    # so the local-JJ arm sets densify=true; gIBSS / CAVI / global-JJ stay on the BCOO fast
    # path. Densifying does not change the model, only the layout, so the fit is identical to
    # the sparse one where both are supported.
    X = simulation.X
    if densify:
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)
    t0 = time.perf_counter()
    fitted = fit_glm_susie(X, y, **kwargs)
    fit_seconds = time.perf_counter() - t0
    return {
        "state": fitted,
        "threshold": threshold,
        "n_selected": int(np.asarray(y).sum()),
        "fit_seconds": float(fit_seconds),
        "q2_elbo": _q2_elbo(X, y, fitted, center),
    }


def _q2_elbo(X, y, fitted, center):
    """Exact Q2 ELBO F(q) = E_q[log p(y|eta)] - KL of the fitted state, via the
    characteristic-function integrator (``compute_elbo_gaussian``).

    A common yardstick across the Q2 approximations: it scores the SAME F(q) for any
    Gaussian-effect state, however it was fit (gIBSS plug-in, CAVI-cf fold, global-JJ
    bound). Returns ``None`` for a state it cannot score -- a free-form Q1 effect
    (``b_nodes`` set) is rejected, so the legacy Q1 methods get ``None`` rather than a
    crash. The offset fold auto-sizes to the offset support (M=64 residual degree).
    """
    try:
        from gibss import glm
        from gibss.elbo import compute_elbo_gaussian
        data = glm.prep_data(X, y, center=center)
        return float(compute_elbo_gaussian(data, fitted))
    except Exception:
        return None


def summarize_logistic_method(
    fit_obj,
    simulation,
    *,
    response_source,
    threshold=None,
    L=1,
    offset_integration=None,
    offset_quadrature_points=None,
    variational_family=None,
    center=True,
    intercept=None,
    densify=False,
    estimate_prior_variance=False,
    max_prior_variance=None,
    max_iter=None,
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del response_source, threshold, L, offset_integration, offset_quadrature_points, variational_family, center, intercept, densify
    del estimate_prior_variance, max_prior_variance, max_iter
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
        "fit_seconds": fit_obj.get("fit_seconds"),
        "q2_elbo": fit_obj.get("q2_elbo"),
    }


def run_logistic_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_logistic_method(fit_logistic_method(simulation, **kwargs), simulation, **kwargs)
