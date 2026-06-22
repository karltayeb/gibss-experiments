"""Paired analyses: reciprocal-rate pairing wrappers around existing renderers.

Each paired renderer rewrites the bundle so reciprocal-rate collections (e.g.
lambda=2/3 and lambda=3/2) share a facet (max(rate,1/rate)) and are colored by
depletion (rate<1) vs enrichment (rate>1), then calls the existing analysis
renderer unchanged. Intended for single-method experiments (009 / cox_reversed).
"""
from fractions import Fraction

import polars as pl

from analyses import pip, cs, logbf

_SIGN_METHODS = ("depletion", "enrichment")


def _rate(alias: str) -> Fraction:
    if "=" not in alias:
        raise ValueError(f"Cannot parse rate from collection alias: {alias!r}")
    return Fraction(alias.split("=", 1)[1])


def _pair_label(fr: Fraction) -> str:
    hi = max(fr, 1 / fr)
    return f"{hi.numerator}/{hi.denominator}"


def _sign(fr: Fraction) -> str:
    return "depletion" if fr < 1 else "enrichment"


def _sign_method_metadata() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "method": list(_SIGN_METHODS),
            "method_family": list(_SIGN_METHODS),
            "L": [1, 1],
            "threshold": [None, None],
            "is_thresholded": [False, False],
            "is_oracle": [False, False],
            "method_label_base": ["Depletion", "Enrichment"],
            "method_display": ["Depletion", "Enrichment"],
            "method_display_base": ["Depletion", "Enrichment"],
        },
        schema_overrides={"threshold": pl.Float64},
    )


def pair_reciprocal(bundle: dict) -> dict:
    aliases = list(bundle.get("collection_names", []))
    fr = {a: _rate(a) for a in aliases}
    pair_of = {a: _pair_label(fr[a]) for a in aliases}
    sign_of = {a: _sign(fr[a]) for a in aliases}

    out = dict(bundle)
    for key, df in bundle.items():
        if key.endswith("_plot_data") and isinstance(df, pl.DataFrame) and not df.is_empty():
            out[key] = df.with_columns(
                pl.col("collection_name").replace(sign_of).alias("method"),
                pl.lit(None, dtype=pl.Float64).alias("threshold"),
                pl.col("collection_name").replace(pair_of).alias("collection_name"),
            )
    out["method_metadata"] = _sign_method_metadata()
    out["collection_names"] = sorted({pair_of[a] for a in aliases})
    return out


def _paired(make_fn):
    def renderer(bundle: dict, settings: dict):
        transformed = pair_reciprocal(bundle)
        merged_settings = {**settings, "method_filter": list(_SIGN_METHODS)}
        return make_fn(transformed, merged_settings)
    return renderer


_BASE_RENDERERS = {**pip.RENDERERS, **cs.RENDERERS, **logbf.RENDERERS}
RENDERERS = {f"{name}_paired": _paired(fn) for name, fn in _BASE_RENDERERS.items()}


if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    import generate_plots
    from experiments import loader as _loader
    _wc = snakemake.wildcards
    _analysis = snakemake.params.analysis
    _cfg_obj = _loader.load_config()
    _bundle = _loader.load_sc_bundle(
        _cfg_obj, _wc.supercollection,
        _loader.analysis_requires(_cfg_obj, _analysis),
        simulation_filter=_loader.analysis_simulation_filter(_cfg_obj["library"], _analysis),
    )
    _args = _loader.resolve_args(_cfg_obj, _wc.supercollection, _wc.args_name)
    generate_plots.render_analysis(_bundle, _args, _analysis, snakemake.output[0])
