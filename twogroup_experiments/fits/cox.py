"""Cox-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import cox, engine


def fit_cox_method(simulation, *, threshold, time_sign, L=1):
    from core import _score, _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    score = _score(simulation)
    if threshold is None:
        event_type = np.ones_like(score, dtype=int)
    else:
        event_type = (score > float(threshold)).astype(int)
    data = cox.prep_data(
        simulation.X,
        event_time=time_sign * score,
        event_type=event_type,
    )
    state = cox.initialize_state(
        data,
        L=L,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    fitted = engine.fit_ibss(data, state, cox.default_schedule())
    return {
        "state": fitted,
        "threshold": threshold,
        "n_selected": int(event_type.sum()),
    }


def summarize_cox_method(
    fit_obj,
    simulation,
    *,
    threshold,
    time_sign,
    L=1,
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del time_sign, threshold, L
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


def run_cox_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_cox_method(fit_cox_method(simulation, **kwargs), simulation, **kwargs)
