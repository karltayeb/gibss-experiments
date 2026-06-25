"""Score (one-step linearization) logistic SuSiE methods.

A second-order Taylor expansion of the logistic log-likelihood about a constant
linear predictor eta0 turns the SER into a Gaussian (linear) SER: one
Fisher-scoring / IRLS step from a constant null. Because the expansion point is
the same for every observation, the IRLS weight w0 = p0(1-p0) is constant, so
the working model is homoscedastic and reduces to a plain linear SER with a
fixed residual variance v = 1/w0 and pseudo-response z = eta0 + (y - p0)/w0.

Two variants:
- score:                eta0 = 0          (p0 = 1/2). Intercept is estimated.
- score_null_intercept: eta0 = logit(ybar) (p0 = ybar, the null-MLE intercept).
                        Intercept is held fixed at eta0; only effects are fit.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import engine, linear


def fit_score_method(simulation, *, null_intercept: bool, L: int = 1):
    y = np.asarray(simulation.y, dtype=float)
    # Pass X through (dense numpy or sparse BCOO); gibss.linear handles both, and
    # densifying a gene-set design is far slower. Only the response is recoded.
    X = simulation.X

    if null_intercept:
        n = y.shape[0]
        # clip the marginal rate so logit / weight stay finite even for an
        # all-zero (or all-one) replicate.
        ybar = float(np.clip(y.mean(), 1.0 / (n + 1), n / (n + 1)))
        w0 = ybar * (1.0 - ybar)
        eta0 = float(np.log(ybar / (1.0 - ybar)))
        v = 1.0 / w0
        z = eta0 + (y - ybar) / w0
        family_state_kwargs = {
            "residual_variance": v,
            "estimate_residual_variance": False,
            "intercept": eta0,
            "estimate_intercept": False,
        }
    else:
        # eta0 = 0, p0 = 1/2, w0 = 1/4  ->  z = 4y - 2, v = 4. Intercept free.
        z = 4.0 * y - 2.0
        family_state_kwargs = {
            "residual_variance": 4.0,
            "estimate_residual_variance": False,
        }

    data = linear.prep_data(X, z)
    state = linear.initialize_state(data, L=L, family_state_kwargs=family_state_kwargs)
    fitted = engine.fit_ibss(data, state, linear.default_schedule())
    return {"state": fitted, "null_intercept": null_intercept}


def summarize_score_method(fit_obj, simulation, *, null_intercept: bool, L: int = 1):
    from core import _extract_ser_struct, _make_cs_struct, _make_fit_summary_struct
    del null_intercept, L
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "impl": "score_null" if fit_obj["null_intercept"] else "score",
        "threshold": None,
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [_make_cs_struct(state, simulation, l) for l in range(n_effects)],
        "fit_summary": _make_fit_summary_struct(state, simulation, None),
    }


def run_score_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_score_method(fit_score_method(simulation, **kwargs), simulation, **kwargs)
