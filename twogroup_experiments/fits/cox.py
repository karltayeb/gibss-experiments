"""Cox-family fit/summarize/run methods."""
from __future__ import annotations

from typing import Any

import numpy as np

from gibss import cox, engine


def _right_censored_survival(score, threshold, time_sign):
    """Right-censor the transformed arrival time ``time_sign*score`` at
    ``time_sign*threshold``: arrivals past the threshold are clamped to it and
    marked censored (``event_type=0``); they stay in the risk set.
    """
    score = np.asarray(score, dtype=float)
    raw = float(time_sign) * score
    T = float(time_sign) * float(threshold)
    event_time = np.minimum(raw, T)
    event_type = (raw <= T).astype(int)
    return event_time, event_type


def _bin_event_times(event_time, n_bins):
    """Discretize event times into ``n_bins`` equal-frequency (rank) bins.

    The Cox partial likelihood is O(#distinct event times); on a GSEA design with
    ~20k genes every gene is its own event time, which is ruinously slow. Binning
    the (order-preserving) event times into ~1000 levels collapses them to ties -
    handled by the partial likelihood - for an orders-of-magnitude speedup with
    negligible effect on the ranking-based statistic. Returns the bin index as the
    new event time (monotone in the original), so ordering and censoring are kept.
    """
    et = np.asarray(event_time, dtype=float)
    n = et.shape[0]
    n_bins = int(n_bins)
    if n_bins <= 0 or n_bins >= n:
        return et
    ranks = np.argsort(np.argsort(et, kind="mergesort"), kind="mergesort")
    return np.minimum((ranks * n_bins) // n, n_bins - 1).astype(float)


def fit_cox_method(simulation, *, threshold, time_sign, L=1, time_bins=None):
    from core import _score, _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    score = _score(simulation)
    if threshold is None:
        event_time = time_sign * score
        event_type = np.ones_like(score, dtype=int)
    else:
        event_time, event_type = _right_censored_survival(score, threshold, time_sign)
    if time_bins is not None:
        event_time = _bin_event_times(event_time, time_bins)
    data = cox.prep_data(
        simulation.X,
        event_time=event_time,
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
    time_bins=None,
):
    from core import _extract_ser_struct, _extract_family_state_struct, _extract_twogroup_state_struct, _make_cs_struct, _make_fit_summary_struct
    del time_sign, threshold, L, time_bins
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
