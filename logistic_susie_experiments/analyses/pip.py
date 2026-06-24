"""PIP-family renderers (pip_calibration, power_fdp, causal_pip, mass_above_causal)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import viz_utils
from analyses._common import foreground_methods, method_order, set_agg_facecolor


def _make_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_thresholds=None,
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
    max_fdp = settings.get("max_fdp", 0.5)
    fg = foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    return viz_utils.render_power_fdp_chart(
        power_fdp,
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
    fg = foreground_methods(method_meta, settings)
    causal_pip = viz_utils.expand_causal_pip_from_compact(pip_plot, method_meta)
    filtered = causal_pip.filter(pl.col("method").is_in(fg))
    if filtered.is_empty():
        return viz_utils.make_placeholder_chart("No causal PIP data")
    order = method_order(method_meta, fg)
    summary = viz_utils.make_causal_pip_summary(filtered)
    return viz_utils.render_causal_pip_chart(
        summary,
        facet=True,
        legend_outside=True,
        square_axes=True,
        method_order=order,
        collection_names=combined_data["collection_names"],
    )


def _make_mass_above_causal(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = method_order(method_meta, fg)
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


def _make_agg_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_thresholds=None,
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
    set_agg_facecolor(fig)
    return fig


def _make_agg_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    max_fdp = settings.get("max_fdp", 0.5)
    fg = foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        aggregate_across_collections=True,
    )
    if power_fdp.is_empty():
        return viz_utils.make_placeholder_chart("No power/FDP data")
    fig = viz_utils.render_power_fdp_chart(
        power_fdp,
        facet=False,
        max_fdp=max_fdp,
        fixed_y_scale=True,
        legend_outside=True,
        square_axes=True,
    )
    set_agg_facecolor(fig)
    return fig


def _make_agg_causal_pip(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    causal_pip = viz_utils.expand_causal_pip_from_compact(pip_plot, method_meta)
    filtered = causal_pip.filter(pl.col("method").is_in(fg))
    if filtered.is_empty():
        return viz_utils.make_placeholder_chart("No causal PIP data")
    order = method_order(method_meta, fg)
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
    set_agg_facecolor(fig)
    return fig


def _make_agg_mass_above_causal(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = method_order(method_meta, fg)
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
    set_agg_facecolor(fig)
    return fig


RENDERERS = {
    "pip_calibration": _make_pip_calibration,
    "power_fdp": _make_power_fdp,
    "causal_pip": _make_causal_pip,
    "mass_above_causal": _make_mass_above_causal,
    "agg_pip_calibration": _make_agg_pip_calibration,
    "agg_power_fdp": _make_agg_power_fdp,
    "agg_causal_pip": _make_agg_causal_pip,
    "agg_mass_above_causal": _make_agg_mass_above_causal,
}

if "snakemake" in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _parent = str(_Path(__file__).parent.parent)
    if _parent not in _sys.path:
        _sys.path.insert(0, _parent)
    import generate_plots
    from experiments import loader as _loader
    _wc = snakemake.wildcards
    _analysis = snakemake.params.analysis  # passed via params since analysis is baked into path
    _cfg_obj = _loader.load_config()
    _bundle = _loader.load_sc_bundle(
        _cfg_obj, _wc.supercollection,
        _loader.analysis_requires(_cfg_obj, _analysis),
        simulation_filter=_loader.analysis_simulation_filter(_cfg_obj["library"], _analysis),
    )
    _args = _loader.resolve_args(_cfg_obj, _wc.supercollection, _wc.args_name)
    generate_plots.render_analysis(_bundle, _args, _analysis, snakemake.output[0])
