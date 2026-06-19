"""F1 family renderers (f1_boxplot, f1_scatter, f1_enrich_scatter)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import viz_utils
from analyses._common import foreground_methods


_TG_FAMILIES = {"twogroup", "twogroup_oracle", "twogroup_oracle_init", "twogroup_scale_fam", "twogroup_loc_fam"}


def _tg_methods(method_meta: pl.DataFrame, settings: dict) -> set[str]:
    fg = foreground_methods(method_meta, settings)
    tg = set(method_meta.filter(pl.col("method_family").is_in(_TG_FAMILIES))["method"].to_list())
    return fg & tg


def _with_meta(data: pl.DataFrame, method_meta: pl.DataFrame, tg_fg: set[str]) -> pl.DataFrame:
    return (
        data.filter(pl.col("method").is_in(tg_fg))
        .join(
            method_meta.select("method", "method_display", "method_family").unique("method"),
            on="method",
            how="left",
        )
    )


_TG_FAMILY_ORDER = {
    "twogroup_oracle": 0,
    "twogroup_oracle_init": 1,
    "twogroup": 2,
    "twogroup_loc_fam": 3,
    "twogroup_scale_fam": 4,
}


def _tg_display_order(method_meta: pl.DataFrame, methods: set[str]) -> list[str]:
    return (
        method_meta.filter(pl.col("method").is_in(methods))
        .with_columns(
            pl.col("method_family").replace(_TG_FAMILY_ORDER, default=99).alias("_fam_rank")
        )
        .sort(["_fam_rank", "method"])["method_display"]
        .to_list()
    )


def _make_f1_boxplot(combined_data: dict, settings: dict) -> plt.Figure:
    method_meta = combined_data["method_metadata"]
    tg_fg = _tg_methods(method_meta, settings)
    f1_raw = combined_data.get("f1_plot_data", pl.DataFrame())
    enrich_raw = combined_data.get("enrich_plot_data", pl.DataFrame())
    if f1_raw.is_empty():
        return viz_utils.make_placeholder_chart("No f1 data")
    f1 = _with_meta(f1_raw, method_meta, tg_fg)
    enrich = _with_meta(enrich_raw, method_meta, tg_fg) if not enrich_raw.is_empty() else pl.DataFrame()
    if not enrich.is_empty():
        all_data = f1.join(
            enrich.select("sample_id", "method", "mu_at_causal", "true_intercept", "true_effect"),
            on=["sample_id", "method"],
            how="left",
        )
    else:
        all_data = f1.with_columns(
            pl.lit(None).cast(pl.Float64).alias("mu_at_causal"),
            pl.lit(None).cast(pl.Float64).alias("true_intercept"),
            pl.lit(None).cast(pl.Float64).alias("true_effect"),
        )
    return viz_utils.render_f1_boxplot(
        all_data,
        collection_names=combined_data["collection_names"],
        method_order=_tg_display_order(method_meta, tg_fg),
    )


def _make_f1_scatter(combined_data: dict, settings: dict) -> plt.Figure:
    method_meta = combined_data["method_metadata"]
    tg_fg = _tg_methods(method_meta, settings)
    f1_raw = combined_data.get("f1_plot_data", pl.DataFrame())
    if f1_raw.is_empty():
        return viz_utils.make_placeholder_chart("No f1 scatter data")
    f1 = _with_meta(f1_raw, method_meta, tg_fg)
    return viz_utils.render_f1_scatter_chart(
        f1,
        collection_names=combined_data["collection_names"],
        method_order=_tg_display_order(method_meta, tg_fg),
    )


def _make_f1_enrich_scatter(combined_data: dict, settings: dict) -> plt.Figure:
    method_meta = combined_data["method_metadata"]
    tg_fg = _tg_methods(method_meta, settings)
    enrich_raw = combined_data.get("enrich_plot_data", pl.DataFrame())
    if enrich_raw.is_empty():
        return viz_utils.make_placeholder_chart("No enrichment scatter data")
    enrich = _with_meta(enrich_raw, method_meta, tg_fg)
    return viz_utils.render_f1_enrich_scatter_chart(
        enrich,
        collection_names=combined_data["collection_names"],
        method_order=_tg_display_order(method_meta, tg_fg),
    )


RENDERERS = {
    "f1_boxplot": _make_f1_boxplot,
    "f1_scatter": _make_f1_scatter,
    "f1_enrich_scatter": _make_f1_enrich_scatter,
}
