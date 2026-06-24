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

_IMPLS = {
    "globaljj": _globaljj,
    "localjj": _localjj,
    "logistic_quadrature": _quadrature,
}


def _as_dense_array(X: Any) -> np.ndarray:
    """Densify to a float array, handling sparse designs (e.g. jax BCOO)."""
    if hasattr(X, "todense"):
        X = X.todense()
    return np.asarray(X, dtype=float)


def fit_logistic_method(
    simulation,
    *,
    impl: str,
    L: int = 1,
    family_state_kwargs: dict | None = None,
):
    if impl not in _IMPLS:
        raise ValueError(
            f"Unknown logistic impl {impl!r}; expected one of {sorted(_IMPLS)}"
        )
    module = _IMPLS[impl]
    y = np.asarray(simulation.y, dtype=float)
    X = _as_dense_array(simulation.X)

    data = module.prep_data(X, y)
    state = module.initialize_state(
        data, L=L, family_state_kwargs=dict(family_state_kwargs or {})
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
):
    from core import (
        _extract_ser_struct,
        _make_cs_struct,
        _make_fit_summary_struct,
    )
    del impl, L, family_state_kwargs
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
