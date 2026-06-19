"""Log-BF family renderers (log_bf_roc, log_bf_ser_ecdf + agg variants)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import viz_utils
from analyses._common import foreground_methods, set_agg_facecolor


def _make_log_bf_roc(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    roc_curves = viz_utils.make_log_bf_roc_curves(
        cs_data, method_meta,
        selected_methods=fg,
    )
    return viz_utils.render_log_bf_roc_chart(roc_curves)


def _make_agg_log_bf_roc(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    roc_curves = viz_utils.make_log_bf_roc_curves(
        cs_data, method_meta,
        selected_methods=fg,
    )
    fig = viz_utils.render_log_bf_roc_chart(roc_curves)
    set_agg_facecolor(fig)
    return fig


def _make_log_bf_ser_ecdf(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    ecdf_data = viz_utils.make_log_bf_ser_ecdf(
        cs_data, method_meta,
        selected_methods=fg,
    )
    return viz_utils.render_log_bf_ser_ecdf_chart(ecdf_data, collection_names=combined_data["collection_names"])


def _make_agg_log_bf_ser_ecdf(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    ecdf_data = viz_utils.make_log_bf_ser_ecdf(
        cs_data, method_meta,
        selected_methods=fg,
    )
    fig = viz_utils.render_log_bf_ser_ecdf_chart(ecdf_data, collection_names=[])
    set_agg_facecolor(fig)
    return fig


RENDERERS = {
    "log_bf_roc": _make_log_bf_roc,
    "agg_log_bf_roc": _make_agg_log_bf_roc,
    "log_bf_ser_ecdf": _make_log_bf_ser_ecdf,
    "agg_log_bf_ser_ecdf": _make_agg_log_bf_ser_ecdf,
}
