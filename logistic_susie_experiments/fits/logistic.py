"""Logistic SuSiE fit/summarize/run methods.

A single parametrized method dispatches over the gibss logistic SuSiE
implementations being compared. All three expose the same interface
(``prep_data`` / ``initialize_state`` / ``default_schedule``) and are driven by
the common ``engine.fit_ibss`` loop, so the only thing that varies is which
module is selected by ``impl``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import engine
import gibss.globaljj as _globaljj
import gibss.localjj as _localjj
import gibss.logistic_quadrature as _quadrature
import gibss.logistic_profile as _profile
import gibss.irls as _irls

_IMPLS = {
    "globaljj": _globaljj,
    "localjj": _localjj,
    "logistic_quadrature": _quadrature,
    # profile-likelihood SER; cheb vs newton selected via
    # family_state_kwargs={"background_mode": "chebyshev"|"exact"}
    "logistic_profile": _profile,
    # IRLS / Laplace logistic SuSiE (default Logistic GLM family)
    "irls": _irls,
}


def fit_logistic_method(
    simulation,
    *,
    impl: str,
    L: int = 1,
    family_state_kwargs: dict | None = None,
    estimate_prior_variance: bool = True,
    prior_variance: float = 1.0,
    center: bool | None = None,
):
    if impl not in _IMPLS:
        raise ValueError(
            f"Unknown logistic impl {impl!r}; expected one of {sorted(_IMPLS)}"
        )
    module = _IMPLS[impl]
    y = np.asarray(simulation.y, dtype=float)
    # Pass X through as-is: dense numpy for Markov designs, sparse BCOO for gene
    # sets. Densifying a sparse design is ~30-50x slower and, for the chebyshev
    # profile background, disables the sparse fast path entirely (it requires a
    # BCOO X), forcing a dense-quadrature fallback.
    X = simulation.X

    fsk = dict(family_state_kwargs or {})
    fsk.setdefault("estimate_prior_variance", estimate_prior_variance)
    if center is not None:  # weighted-centering axis (irls only)
        fsk["center"] = center
    data = module.prep_data(X, y)
    state = module.initialize_state(data, L=L, family_state_kwargs=fsk)
    if not estimate_prior_variance:
        # fix the slab variance: set every effect's prior_variance, no EB update
        from dataclasses import replace
        state = replace(
            state,
            single_effects=[
                replace(e, prior_variance=float(prior_variance))
                for e in state.single_effects
            ],
        )
    fitted = engine.fit_ibss(data, state, module.default_schedule())
    return {
        "state": fitted,
        "impl": impl,
        "threshold": None,
        "n_selected": int(y.sum()),
    }


def summarize_logistic_method(
    fit_obj,
    simulation,
    *,
    impl: str,
    L: int = 1,
    family_state_kwargs: dict | None = None,
    estimate_prior_variance: bool = True,
    prior_variance: float = 1.0,
    center: bool | None = None,
):
    from core import (
        _extract_ser_struct,
        _make_cs_struct,
        _make_fit_summary_struct,
    )
    del impl, L, family_state_kwargs, estimate_prior_variance, prior_variance, center
    state = fit_obj["state"]
    n_effects = len(state.single_effects)
    return {
        "impl": fit_obj["impl"],
        "threshold": fit_obj["threshold"],
        "single_effects": [_extract_ser_struct(state, l) for l in range(n_effects)],
        "credible_sets": [
            _make_cs_struct(state, simulation, l) for l in range(n_effects)
        ],
        "fit_summary": _make_fit_summary_struct(state, simulation, fit_obj["n_selected"]),
    }


def run_logistic_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_logistic_method(
        fit_logistic_method(simulation, **kwargs), simulation, **kwargs
    )
