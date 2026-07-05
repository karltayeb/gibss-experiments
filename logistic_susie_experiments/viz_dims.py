"""Map raw MANIFEST coordinates to plot-facing dimensions.

Single source of truth for the dimension columns attached to every plot_data row.
Returns clean SEMANTIC dims plus flattened RAW args (m_*, d_*, e_*). Adding a knob
= one line here. No name/alias parsing.
"""
from __future__ import annotations

_DESIGN_NAMES = {"gaussian_markov_X": "gaussian", "uniform_markov_X": "uniform",
                 "binary_markov_X": "binary",
                 "c4_gene_sets_X": "c4", "hallmark_gene_sets_X": "hallmark",
                 "msigdb_gene_sets_X": "msigdb"}


def method_dims(coord: dict) -> dict:
    k = dict(coord.get("kwargs", {}))
    fn = coord.get("function", "")
    impl = k.get("impl", "")
    if "irls" in fn or impl == "irls":
        family = "irls"
    elif "globaljj" in fn or impl == "globaljj":
        family = "globaljj"
    elif "localjj" in fn or impl == "localjj":
        family = "localjj"
    elif impl == "logistic_quadrature" or "quadrature" in fn:
        family = "quadrature"
    elif "linear" in fn:
        family = "linear"
    elif "score" in fn:
        family = "score"
    else:
        family = impl or fn
    semantic = {
        # the method's declared name (config key, sans over-suffix) — the
        # alias is whatever the experiment named it; no inference here.
        "method": str(coord.get("name", "")).split("__", 1)[0],
        "family": family,
        "step": "one_step" if k.get("n_outer") == 1 else "converged",
        "prior": "fixed" if k.get("estimate_prior_variance") is False else "eb",
        # intercept-profiling axis (family `profile` flag / `_c` methods); distinct
        # from the `center` column pre-centering preprocessing (default on).
        "profile": bool(k.get("profile", False)),
        # offset integration ON/off over the leave-one-out offset variance. OFF =
        # {None, False, "none"}; anything else ("taylor", GH order int) = ON. Raw
        # value kept in m_offset_integration for order-level faceting.
        "offset_integration": k.get("offset_integration", "none") not in (None, False, "none"),
        "cadence": k.get("ser_cadence", "block"),
        "L": int(k.get("L", 1)),
        "function": fn,
    }
    raw = {f"m_{key}": v for key, v in k.items()}
    return {**semantic, **raw}


def sim_dims(coord: dict) -> dict:
    from simulations.effect import logbf_sizing
    d = coord["design"]
    e = coord["enrichment"]
    dargs = d.get("arguments") or {}
    b = float(e["arguments"].get("causal_effect", 0.0))
    b0 = float(e["intercept"])
    design = _DESIGN_NAMES.get(d["function"], d["function"])
    if d["function"] == "binary_markov_X":
        design = f"binary_q{dargs.get('freq', 0.5):g}"
    n = int(dargs.get("n", 0)) or None
    e_args = e.get("arguments") or {}
    if "target_logbf" in e_args:
        logbf = int(e_args["target_logbf"])
    else:
        sizing_key = {"binary_q0.5": "binary q=0.5", "binary_q0.1": "binary q=0.1"}.get(design, design)
        logbf = (logbf_sizing.nearest_logbf(sizing_key, b0, b, n)
                 if (n and sizing_key in logbf_sizing.MARGINALS) else 0)
    semantic = {
        "design": design,
        "intercept": b0,
        "b": b,
        "signal": (b != 0.0) or (int(e_args.get("target_logbf", 0)) != 0),
        "rho": float(dargs.get("rho", 0.0)),
        "logbf": int(logbf),
    }
    raw = {f"d_{key}": v for key, v in dargs.items()}
    raw |= {f"e_{key}": v for key, v in (e.get("arguments") or {}).items()}
    return {**semantic, **raw}
