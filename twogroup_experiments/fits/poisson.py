"""Poisson-family fit/summarize/run methods.

The Poisson companion to ``fits/logistic.py``, kept as a DEDICATED module (not folded into
the generic ``fits/glm.py``) so the logistic pipeline is untouched: new file, new library
entries, new content-addressed hashes. The Poisson SER approximations differ enough from
the logistic ones to warrant it - there is no Jaakkola-Jordan arm (JJ is a logistic bound),
the exact-offset CAVI folds are the closed-form Poisson ones (``cf`` -> PoissonLogNormalOffset
in Q2, ``compress_selfnorm`` -> PoissonSelfNormOffset in Q1), and the response is a count
vector rather than the logistic z / score-threshold branch.

Well-specified counts ``y ~ Poisson(exp(b0 + X beta))`` are read straight off
``simulation.y_count`` (drawn in ``core.simulate`` from the same linear predictor as z).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from gibss.methods import fit_glm_susie

# GH order for the leave-one-out offset when offset_integration="gh" (moment-matched plug-in).
# The exact folds (cf / compress_selfnorm) are closed-form for the Poisson log-link and ignore
# this; it only bites the "gh" fallback.
OFFSET_QUADRATURE_POINTS = 5


def fit_poisson_method(
    simulation, *, L=1, offset_integration=None, offset_quadrature_points=None,
    variational_family=None, center=True, intercept=None,
    estimate_prior_variance=False, prior_variance=None, max_prior_variance=None, max_iter=None,
    freeze_prior_variance=None, method=None,
):
    y = np.asarray(simulation.y_count, dtype=float)

    # A `method=` (a fit_glm_susie PRESET name, e.g. "score") drives offset_integration /
    # intercept / _anchor from the preset; in that mode we do NOT inject offset_integration
    # (it would override the preset) and leave intercept to the preset unless pinned. Otherwise
    # the offset fold follows the axes below: "none" (plug-in mean; gIBSS), "cf" (exact Gaussian
    # CAVI in Q2 -> PoissonLogNormalOffset), "compress_selfnorm" (exact free-form CAVI in Q1 ->
    # PoissonSelfNormOffset), or "gh" (moment-matched) at L>1.
    if method is not None:
        kwargs = dict(
            L=L,
            family="poisson",
            center=center,
            estimate_prior_variance=bool(estimate_prior_variance),
            method=method,
        )
    else:
        integ = offset_integration if offset_integration is not None else ("none" if L == 1 else "gh")
        kwargs = dict(
            L=L,
            family="poisson",
            center=center,
            estimate_prior_variance=bool(estimate_prior_variance),
            offset_integration=integ,
            offset_quadrature_points=(
                OFFSET_QUADRATURE_POINTS if offset_quadrature_points is None
                else int(offset_quadrature_points)
            ),
        )
    # Fixed-prior arm: pin the effect prior variance (with estimate_prior_variance=False this
    # is the whole prior). Default (None) leaves the engine default (1.0).
    if prior_variance is not None:
        kwargs["prior_variance"] = float(prior_variance)
    if max_prior_variance is not None:
        kwargs["max_prior_variance"] = float(max_prior_variance)
    # Cap the IBSS sweep count (fit_glm_susie default 100); set for the CAVI arm so a
    # slow-to-converge fit is bounded, unset methods keep the engine default.
    if max_iter is not None:
        kwargs["max_iter"] = int(max_iter)
    if freeze_prior_variance is not None:
        kwargs["freeze_prior_variance"] = float(freeze_prior_variance)
    # The variational family over each effect: "unconstrained" (free-form q, exact CAVI in Q1)
    # or "gaussian" (Gaussian q, exact CAVI in Q2). Only set when pinned by the method.
    if variational_family is not None:
        kwargs["variational_family"] = variational_family
    if intercept is not None:
        kwargs["intercept"] = intercept

    t0 = time.perf_counter()
    fitted = fit_glm_susie(simulation.X, y, **kwargs)
    fit_seconds = time.perf_counter() - t0
    return {
        "state": fitted,
        "n_selected": int(np.asarray(y).sum()),
        "fit_seconds": float(fit_seconds),
        "q2_elbo": _q2_elbo(simulation.X, y, fitted, center),
    }


def _q2_elbo(X, y, fitted, center):
    """The fitted state's mean-field ELBO F(q) = E_q[log p(y|eta)] - KL, a common yardstick
    across arms. Same functional for every arm; the integrator is chosen EXPLICITLY by state
    type (the same signal compute_elbo dispatches on: a free-form Q1 effect carries quadrature
    nodes ``b_nodes``, a Gaussian Q2 effect does not):
      * Q2 (Gaussian effect): the characteristic-function integrator ``compute_elbo_gaussian``
        (exact, analytic, quadrature-free -- fast and stable).
      * Q1 (free-form effect): the general self-normalized fold ``compute_elbo``, the only
        integrator that handles the free-form nodes. Same functional, on the same scale, so the
        ELBO table can reference the exact free-form CAVI arm (Q1 >= Q2 by nesting).
    Each state is scored by ITS OWN integrator only -- no cross-fallback (never re-score a
    Gaussian state with the free-form fold; a failure of the matching integrator propagates as
    an error rather than silently switching methods). The field is named ``q2_elbo`` for schema
    continuity, but for Q1 arms it holds the free-form F(q)."""
    from gibss import glm
    from gibss.elbo import compute_elbo, compute_elbo_gaussian
    data = glm.prep_data(X, y, center=center)
    free_form = any(getattr(e, "b_nodes", None) is not None for e in fitted.single_effects)
    if free_form:
        return float(compute_elbo(data, fitted, order=16, M=64))
    return float(compute_elbo_gaussian(data, fitted, score_intercept="shared"))


def summarize_poisson_method(
    fit_obj,
    simulation,
    *,
    L=1,
    offset_integration=None,
    offset_quadrature_points=None,
    variational_family=None,
    center=True,
    intercept=None,
    estimate_prior_variance=False,
    prior_variance=None,
    max_prior_variance=None,
    max_iter=None,
    freeze_prior_variance=None,
    method=None,
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del L, offset_integration, offset_quadrature_points, variational_family, center, intercept
    del estimate_prior_variance, prior_variance, max_prior_variance, max_iter, freeze_prior_variance, method
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "family_state": _extract_family_state_struct(state),
        "two_group_state": _extract_twogroup_state_struct(state),
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
        "fit_seconds": fit_obj.get("fit_seconds"),
        "q2_elbo": fit_obj.get("q2_elbo"),
    }


def run_poisson_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_poisson_method(fit_poisson_method(simulation, **kwargs), simulation, **kwargs)
