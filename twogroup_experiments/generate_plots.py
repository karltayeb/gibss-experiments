from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

_parent = str(Path(__file__).parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import plot_ready
import viz_utils


_PLOT_CONFIG_PATH = Path(__file__).parent / "notebooks" / "plot_config.yaml"
_COLLECTION_ALIAS_ROOT = Path(__file__).parent / "results" / "collections"

PLOT_TYPES = [
    "pip_calibration", "power_fdp", "causal_pip", "causal_rank",
    "mass_above_causal", "cs_dot_summary", "cs_power_fdp", "cs_beta_trace",
    "agg_pip_calibration", "agg_power_fdp", "agg_causal_pip", "agg_causal_rank",
    "agg_mass_above_causal", "agg_cs_power_fdp", "agg_cs_beta_trace",
]


def make_plot(
    supercollection: str,
    plot_settings: str,
    plot_type: str,
    output_path: str,
) -> None:
    """Generate one plot-type PDF for a (supercollection, plot_settings) combo."""
    cfg = _load_plot_config()
    settings = _resolve_settings(cfg, supercollection, plot_settings)
    combined_data = _load_supercollection_data(cfg, supercollection)
    fig = _PLOT_DISPATCH[plot_type](combined_data, settings)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_plot_config() -> dict:
    return yaml.safe_load(_PLOT_CONFIG_PATH.read_text()) or {}


def _resolve_settings(cfg: dict, supercollection: str, plot_settings: str) -> dict:
    defaults = cfg["supercollections"][supercollection].get("default_settings", {})
    overrides = cfg["settings"].get(plot_settings, {})
    return {**defaults, **overrides}


def _load_supercollection_data(cfg: dict, supercollection: str) -> dict:
    coll_list = cfg["supercollections"][supercollection]["collections"]
    aliases = {item["name"]: item.get("alias", item["name"]) for item in coll_list}
    bundles = {
        item["name"]: plot_ready.load_plot_ready_collection(
            _COLLECTION_ALIAS_ROOT / item["name"]
        )
        for item in coll_list
    }
    combined_method_metadata = (
        pl.concat([b["method_metadata"] for b in bundles.values()])
        .unique(subset=["method", "threshold"])
    )

    def _tag(key: str) -> pl.DataFrame:
        return pl.concat([
            b[key].with_columns(pl.lit(aliases.get(name, name)).alias("collection_name"))
            for name, b in bundles.items()
        ])

    return {
        "method_metadata": combined_method_metadata,
        "collection_names": [aliases.get(item["name"], item["name"]) for item in coll_list],
        "pip_plot_data": _tag("pip_plot_data"),
        "cs_plot_data": _tag("cs_plot_data"),
    }


def _foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    threshold = settings.get("threshold", 2.0)
    L = settings.get("L", 1)
    method_families = settings.get("method_families", [])
    mask = (
        pl.col("method_family").is_in(method_families)
        & (pl.col("L") == L)
        & (
            ~pl.col("is_thresholded")
            | (pl.col("threshold") == threshold)
            | pl.col("threshold").is_null()
        )
    )
    return set(method_metadata.filter(mask)["method"].to_list())


def _method_order(method_metadata: pl.DataFrame, foreground: set[str]) -> list[str]:
    return (
        method_metadata.filter(pl.col("method").is_in(foreground))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )


def _make_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    fg = _foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_threshold=threshold,
    )
    if summary.is_empty():
        return viz_utils.make_placeholder_chart("No PIP calibration data")
    return viz_utils.render_pip_calibration(
        summary,
        facet_by_simulation=True,
        collection_names=combined_data["collection_names"],
    )


def _make_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        show_background_threshold_traces=False,
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    summary = viz_utils.make_power_fdp_summary(power_fdp)
    return viz_utils.render_power_fdp_chart(
        summary,
        facet=True,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
        collection_names=combined_data["collection_names"],
    )


def _make_causal_pip(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    causal_pip = viz_utils.expand_causal_pip_from_compact(pip_plot, method_meta)
    filtered = causal_pip.filter(pl.col("method").is_in(fg))
    if filtered.is_empty():
        return viz_utils.make_placeholder_chart("No causal PIP data")
    order = _method_order(method_meta, fg)
    summary = viz_utils.make_causal_pip_summary(filtered)
    return viz_utils.render_causal_pip_chart(
        summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_causal_rank(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = _method_order(method_meta, fg)
    rank_summary = viz_utils.make_causal_rank_summary(cs_data, method_meta, selected_methods=fg)
    if rank_summary.is_empty():
        return viz_utils.make_placeholder_chart("No causal rank data")
    return viz_utils.render_causal_rank_chart(
        rank_summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_mass_above_causal(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = _method_order(method_meta, fg)
    expanded = viz_utils.expand_mass_above_causal_from_compact(
        cs_data.filter(pl.col("method").is_in(fg)),
        method_meta,
    )
    if expanded.is_empty():
        return viz_utils.make_placeholder_chart("No mass above causal data")
    summary = viz_utils.make_mass_above_causal_summary(expanded)
    return viz_utils.render_mass_above_causal_chart(
        summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_cs_dot_summary(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    threshold = settings.get("threshold", 2.0)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    cs_beta = settings.get("cs_beta", 0.95)
    collection_names = combined_data["collection_names"]
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_dot_summary_chart(
        summary,
        collection_names=collection_names,
        selected_beta=round(cs_beta, 2),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_cs_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    _BETA_095_IDX = 45
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    threshold = settings.get("threshold", 2.0)
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)

    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")

    cs_raw = cs_data.with_columns(
        pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size"),
        pl.when(pl.col("rank_of_causal").list.len() > 0)
        .then(pl.col("rank_of_causal").list.min() < pl.col("cs_sizes").list.get(_BETA_095_IDX))
        .otherwise(False)
        .alias("causal_in_cs"),
    ).select(
        "collection_name", "sample_id", "method", "threshold",
        "l", "cs_size", "causal_in_cs", "ser_log_bf",
    )

    raw = (
        cs_raw.filter(
            pl.col("method").is_in(fg)
            & (pl.col("threshold").is_null() | (pl.col("threshold") == threshold))
        )
        .join(
            method_meta.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
    )

    if raw.is_empty():
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    lbf_lo = float(raw["ser_log_bf"].min())
    lbf_hi = float(raw["ser_log_bf"].max())
    lbf_grid = np.linspace(lbf_lo, lbf_hi, 60)[::-1]
    method_groups = (
        raw.select("method", "threshold", "method_display", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method_display"])
    )

    rows = []
    for coll_name in collection_names:
        coll_raw = raw.filter(pl.col("collection_name") == coll_name)
        for mg in method_groups.iter_rows(named=True):
            thresh_filter = (
                pl.col("threshold").is_null()
                if mg["threshold"] is None
                else (pl.col("threshold") == mg["threshold"])
            )
            m_data = coll_raw.filter((pl.col("method") == mg["method"]) & thresh_filter)
            if m_data.is_empty():
                continue
            n_total = m_data.height
            causal_arr = m_data["causal_in_cs"].to_numpy()
            lbf_arr = m_data["ser_log_bf"].to_numpy()
            for t in lbf_grid:
                disc = lbf_arr >= t
                hit = disc & causal_arr
                n_disc = int(disc.sum())
                n_hit = int(hit.sum())
                rows.append({
                    "collection_name": coll_name,
                    "method": mg["method"],
                    "threshold": mg["threshold"],
                    "method_display": mg["method_display"],
                    "is_thresholded": mg["is_thresholded"],
                    "pip_threshold": float(t),
                    "power": float(n_hit / max(n_total, 1)),
                    "fdp": float((n_disc - n_hit) / max(n_disc, 1)),
                })

    if not rows:
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    cs_pf = pl.from_dicts(
        rows,
        schema={
            "collection_name": pl.String,
            "method": pl.String,
            "threshold": pl.Float64,
            "method_display": pl.String,
            "is_thresholded": pl.Boolean,
            "pip_threshold": pl.Float64,
            "power": pl.Float64,
            "fdp": pl.Float64,
        },
    ).with_columns(
        pl.col("method_display").alias("trace_label"),
        pl.col("method_display").alias("legend_label"),
        pl.lit(True).alias("is_selected_threshold"),
        pl.col("collection_name").alias("simulation_name"),
    )

    return viz_utils.render_power_fdp_chart(
        cs_pf,
        facet=True,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
        collection_names=collection_names,
    )


def _make_cs_beta_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    threshold = settings.get("threshold", 2.0)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_beta_trace_chart(
        beta_summary,
        collection_names=collection_names,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _set_agg_facecolor(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.set_facecolor("#ddeeff")


def _make_agg_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    fg = _foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_threshold=threshold,
    )
    if summary.is_empty():
        return viz_utils.make_placeholder_chart("No PIP calibration data")
    agg = (
        summary
        .group_by("method", "method_display", "series_label", "method_family",
                  "pip_bin_index", "pip_left", "pip_right", "pip_mid")
        .agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
        .with_columns(
            pl.when(pl.col("n_total") > 0)
            .then(pl.col("n_causal") / pl.col("n_total"))
            .otherwise(None)
            .alias("empirical_rate")
        )
    )
    fig = viz_utils.render_pip_calibration(agg, facet_by_simulation=False)
    _set_agg_facecolor(fig)
    return fig


def _make_agg_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        show_background_threshold_traces=False,
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    summary = viz_utils.make_power_fdp_summary(power_fdp)
    agg = (
        summary
        .group_by("method", "method_display", "trace_label", "legend_label",
                  "is_selected_threshold", "pip_threshold")
        .agg(pl.col("power").mean(), pl.col("fdp").mean())
    )
    fig = viz_utils.render_power_fdp_chart(
        agg,
        facet=False,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
    )
    _set_agg_facecolor(fig)
    return fig


def _make_agg_causal_pip(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    causal_pip = viz_utils.expand_causal_pip_from_compact(pip_plot, method_meta)
    filtered = causal_pip.filter(pl.col("method").is_in(fg))
    if filtered.is_empty():
        return viz_utils.make_placeholder_chart("No causal PIP data")
    order = _method_order(method_meta, fg)
    summary = viz_utils.make_causal_pip_summary(filtered)
    agg = (
        summary
        .group_by("method", "method_display", "method_display_base", "threshold")
        .agg(pl.col("mean_causal_pip").mean())
    )
    fig = viz_utils.render_causal_pip_chart(
        agg,
        facet=False,
        legend_outside=True,
        square_axes=True,
        method_order=order,
    )
    _set_agg_facecolor(fig)
    return fig


def _make_agg_causal_rank(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = _method_order(method_meta, fg)
    rank_summary = viz_utils.make_causal_rank_summary(cs_data, method_meta, selected_methods=fg)
    if rank_summary.is_empty():
        return viz_utils.make_placeholder_chart("No causal rank data")
    agg = (
        rank_summary
        .group_by("method", "method_display", "method_display_base", "threshold")
        .agg(pl.col("mean_causal_rank").mean())
    )
    fig = viz_utils.render_causal_rank_chart(
        agg,
        facet=False,
        legend_outside=True,
        square_axes=True,
        method_order=order,
    )
    _set_agg_facecolor(fig)
    return fig


def _make_agg_mass_above_causal(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = _method_order(method_meta, fg)
    expanded = viz_utils.expand_mass_above_causal_from_compact(
        cs_data.filter(pl.col("method").is_in(fg)),
        method_meta,
    )
    if expanded.is_empty():
        return viz_utils.make_placeholder_chart("No mass above causal data")
    summary = viz_utils.make_mass_above_causal_summary(expanded)
    agg = (
        summary
        .group_by("method", "method_display", "method_display_base", "threshold")
        .agg(pl.col("mean_mass_above_causal").mean())
    )
    fig = viz_utils.render_mass_above_causal_chart(
        agg,
        facet=False,
        legend_outside=True,
        square_axes=True,
        method_order=order,
    )
    _set_agg_facecolor(fig)
    return fig


def _make_agg_cs_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    _BETA_095_IDX = 45
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    threshold = settings.get("threshold", 2.0)
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)

    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")

    cs_raw = cs_data.with_columns(
        pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size"),
        pl.when(pl.col("rank_of_causal").list.len() > 0)
        .then(pl.col("rank_of_causal").list.min() < pl.col("cs_sizes").list.get(_BETA_095_IDX))
        .otherwise(False)
        .alias("causal_in_cs"),
    ).select(
        "collection_name", "sample_id", "method", "threshold",
        "l", "cs_size", "causal_in_cs", "ser_log_bf",
    )

    raw = (
        cs_raw.filter(
            pl.col("method").is_in(fg)
            & (pl.col("threshold").is_null() | (pl.col("threshold") == threshold))
        )
        .join(
            method_meta.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
    )

    if raw.is_empty():
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    lbf_lo = float(raw["ser_log_bf"].min())
    lbf_hi = float(raw["ser_log_bf"].max())
    lbf_grid = np.linspace(lbf_lo, lbf_hi, 60)[::-1]
    method_groups = (
        raw.select("method", "threshold", "method_display", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method_display"])
    )

    rows = []
    for mg in method_groups.iter_rows(named=True):
        thresh_filter = (
            pl.col("threshold").is_null()
            if mg["threshold"] is None
            else (pl.col("threshold") == mg["threshold"])
        )
        m_data = raw.filter((pl.col("method") == mg["method"]) & thresh_filter)
        if m_data.is_empty():
            continue
        n_total = m_data.height
        causal_arr = m_data["causal_in_cs"].to_numpy()
        lbf_arr = m_data["ser_log_bf"].to_numpy()
        for t in lbf_grid:
            disc = lbf_arr >= t
            hit = disc & causal_arr
            n_disc = int(disc.sum())
            n_hit = int(hit.sum())
            rows.append({
                "method": mg["method"],
                "threshold": mg["threshold"],
                "method_display": mg["method_display"],
                "is_thresholded": mg["is_thresholded"],
                "pip_threshold": float(t),
                "power": float(n_hit / max(n_total, 1)),
                "fdp": float((n_disc - n_hit) / max(n_disc, 1)),
            })

    if not rows:
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    agg_pf = pl.from_dicts(
        rows,
        schema={
            "method": pl.String,
            "threshold": pl.Float64,
            "method_display": pl.String,
            "is_thresholded": pl.Boolean,
            "pip_threshold": pl.Float64,
            "power": pl.Float64,
            "fdp": pl.Float64,
        },
    ).with_columns(
        pl.col("method_display").alias("trace_label"),
        pl.col("method_display").alias("legend_label"),
        pl.lit(True).alias("is_selected_threshold"),
    )

    fig = viz_utils.render_power_fdp_chart(
        agg_pf,
        facet=False,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
    )
    _set_agg_facecolor(fig)
    return fig


def _make_agg_cs_beta_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    threshold = settings.get("threshold", 2.0)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    # collection_names=[] renders only the aggregate row (no per-collection rows)
    return viz_utils.render_cs_beta_trace_chart(
        beta_summary,
        collection_names=[],
        selected_threshold=threshold,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


_PLOT_DISPATCH = {
    "pip_calibration": _make_pip_calibration,
    "power_fdp": _make_power_fdp,
    "causal_pip": _make_causal_pip,
    "causal_rank": _make_causal_rank,
    "mass_above_causal": _make_mass_above_causal,
    "cs_dot_summary": _make_cs_dot_summary,
    "cs_power_fdp": _make_cs_power_fdp,
    "cs_beta_trace": _make_cs_beta_trace,
    "agg_pip_calibration": _make_agg_pip_calibration,
    "agg_power_fdp": _make_agg_power_fdp,
    "agg_causal_pip": _make_agg_causal_pip,
    "agg_causal_rank": _make_agg_causal_rank,
    "agg_mass_above_causal": _make_agg_mass_above_causal,
    "agg_cs_power_fdp": _make_agg_cs_power_fdp,
    "agg_cs_beta_trace": _make_agg_cs_beta_trace,
}
