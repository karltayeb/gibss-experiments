"""Logistic SER under a known random offset, fit three ways.

The simulation carries an ``OffsetLaw`` (the per-row offset distribution); the fit
regresses the binary membership ``z`` with a single-effect logistic SER and differs
only in how it treats that offset:

    quadrature="none"      point estimate: plug in the mean (fixed offset)
    quadrature="gaussian"  Gauss-Hermite over N(mean, variance) (moment-matched)
    quadrature="mixture"   MixtureGH over the full per-row Gaussian mixture
    quadrature="compress"  Compress: amortized Chebyshev compression of the MixtureGH
                           correction - matches "mixture" to interpolation accuracy but at
                           O(M) in-loop cost independent of the mixture's component count
                           (aux precomputed once, out of the SER loop)

All three route through the SER kernel ``glm_profile_ser`` (profiles its own per-feature
intercept, takes ``aux``/``offset`` directly), so no engine / front-door is involved.
The fit is law-agnostic: it only calls ``law.mean`` / ``law.variance`` / ``law.mixture()``.
See docs/superpowers/specs/2026-07-27-response-transform-design.md.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from gibss.response import Bernoulli, Compress, GH, MixtureGH, Smoothed
from gibss.operators import as_operator
from gibss.linear import is_bcoo
from gibss.response_ser import build_ser_state, glm_profile_ser, _profile_null


def _reduce(law, y, quadrature: str, offset_order: int, compress_M: int, compress_T: float):
    """Return (response, offset, aux) for the requested quadrature, reducing the law."""
    y = jnp.asarray(y)
    if quadrature == "none":
        return Bernoulli(), jnp.asarray(law.mean), y
    if quadrature == "gaussian":
        return (
            Smoothed(Bernoulli(), GH(offset_order)),
            jnp.asarray(law.mean),
            (y, jnp.asarray(law.variance)),
        )
    if quadrature == "mixture":
        weights, means, vars_ = law.mixture()
        aux = (y, jnp.asarray(means), jnp.asarray(vars_), jnp.log(jnp.asarray(weights)))
        return Smoothed(Bernoulli(), MixtureGH(offset_order)), jnp.zeros(y.shape[0]), aux
    if quadrature == "compress":
        # Same per-row Gaussian-mixture inputs as "mixture", but compress the MixtureGH
        # correction into a per-row Chebyshev series ONCE here (build_aux, out of the SER
        # loop), so the in-loop cost is O(compress_M) regardless of component count. The
        # mean shift is folded into aux, so the plug-in offset is zero (as for "mixture").
        weights, means, vars_ = law.mixture()
        log_pi = jnp.log(jnp.asarray(weights))
        base = Bernoulli()
        comp = Compress(inner=MixtureGH(offset_order), M=compress_M, T=compress_T)
        aux = comp.build_aux(base, y, jnp.asarray(means), jnp.asarray(vars_), log_pi)
        return Smoothed(base, comp), jnp.zeros(y.shape[0]), aux
    raise ValueError(
        f"unknown quadrature {quadrature!r}; "
        "options: 'none', 'gaussian', 'mixture', 'compress'."
    )


def fit_logistic_offset_method(
    simulation,
    *,
    quadrature: str,
    effect_order: int = 15,
    offset_order: int = 15,
    prior_variance: float = 1.0,
    compress_M: int = 64,
    compress_T: float = 10.0,
) -> dict[str, Any]:
    if getattr(simulation, "offset_law", None) is None:
        raise ValueError(
            "run_logistic_offset_method requires a simulation with an offset_law "
            "(an offset axis); none is set."
        )
    # Keep a sparse (BCOO) gene-set design sparse - never densify. as_operator dispatches
    # to BCOOOperator, and the profiled background must use the Chebyshev surrogate (the
    # exact per-column row-sum is O(n*p) and would defeat the sparse fast path).
    X = simulation.X
    y = np.asarray(simulation.z, dtype=float)
    op = as_operator(X)
    background = "chebyshev" if is_bcoo(X) else "exact"
    response, offset, aux = _reduce(
        simulation.offset_law, y, quadrature, offset_order, compress_M, compress_T
    )
    mu, var, log_bf, coefficient_kl, _b0, _prec = glm_profile_ser(
        op, aux, offset, prior_variance, response=response, order=effect_order, background=background
    )
    c0, ll_null = _profile_null(response, jnp.asarray(offset), aux)
    ser = build_ser_state(mu, var, log_bf, coefficient_kl, prior_variance, null_log_marginal=float(ll_null))
    return {
        "ser": ser,
        "intercept": float(c0),
        "n_selected": int(y.sum()),
    }


def _ser_struct(ser) -> dict[str, Any]:
    return {
        "mu": np.asarray(ser.mu, dtype=float).tolist(),
        "var": np.asarray(ser.var, dtype=float).tolist(),
        "alpha": np.asarray(ser.alpha, dtype=float).tolist(),
        "prior_variance": float(ser.prior_variance),
        "marginal_log_likelihood": float(ser.marginal_log_likelihood),
        "null_log_likelihood": float(ser.null_log_marginal),
        "ser_log_bf": float(ser.marginal_log_likelihood - ser.null_log_marginal),
        "kl": float(ser.kl),
    }


def _cs_struct(ser, simulation, coverage: float = 0.95) -> dict[str, Any]:
    alpha = np.asarray(ser.alpha, dtype=float)
    order = np.argsort(-alpha)
    cumulative = np.cumsum(alpha[order])
    k = int(np.searchsorted(cumulative, coverage)) + 1
    cs = sorted(int(i) for i in order[:k])
    causal_indices = [int(idx) for idx in simulation.causal_indices]
    top_feature = int(np.argmax(alpha))
    return {
        "cs": cs,
        "cs_size": len(cs),
        "causal_indices": causal_indices,
        "causal_in_cs": any(idx in cs for idx in causal_indices),
        "top_feature": top_feature,
        "top_feature_is_causal": top_feature in causal_indices,
    }


def summarize_logistic_offset_method(fit_obj, simulation, **kwargs) -> dict[str, Any]:
    del kwargs
    ser = fit_obj["ser"]
    return {
        "threshold": None,
        "single_effects": [_ser_struct(ser)],
        "credible_sets": [_cs_struct(ser, simulation)],
        "family_state": {"intercept": fit_obj["intercept"], "intercept_kind": "profiled"},
        "two_group_state": None,
        "fit_summary": {"n_selected": fit_obj["n_selected"], "n_iter": 1, "converged": True},
    }


def run_logistic_offset_method(simulation, **kwargs) -> dict[str, Any]:
    return summarize_logistic_offset_method(
        fit_logistic_offset_method(simulation, **kwargs), simulation, **kwargs
    )
