"""CS-family renderers (cs_*, causal_rank, preceding_posterior_mass_ecdf + agg variants)."""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import viz_utils
from analyses._common import foreground_methods, method_order, set_agg_facecolor


_CS_POWER_FDP_BETAS = [1.0, 0.95, 0.50]
_CS_BETA_SIZE_INDICES = {0.95: 94, 0.50: 49}  # index into CS_BETA_GRID
_CS_LBF_MARKERS = [0.0, 2.0]
_CS_LBF_MARKER_STYLES = ["o", "s"]


def _cs_power_fdp_curves(
    raw: pl.DataFrame,
    *,
    collection_groups: list[str | None],
) -> list[dict]:
    """Compute power/FDP curves via cumsum for each (collection, method, cs_beta).

    Denominator for power = total non-null components (rank_of_causal non-empty),
    same across all cs_beta panels so curves are directly comparable.
    """
    method_groups = (
        raw.select("method", "threshold", "method_display", "is_thresholded")
        .unique().sort(["is_thresholded", "method_display"])
    )
    rows = []
    for coll in collection_groups:
        coll_raw = raw if coll is None else raw.filter(pl.col("collection_name") == coll)
        for mg in method_groups.iter_rows(named=True):
            thresh_filter = (
                pl.col("threshold").is_null() if mg["threshold"] is None
                else (pl.col("threshold") == mg["threshold"])
            )
            m_data = coll_raw.filter((pl.col("method") == mg["method"]) & thresh_filter)
            if m_data.is_empty():
                continue
            lbf_arr = m_data["ser_log_bf"].to_numpy()
            rank_causal = m_data["rank_of_causal"].to_list()
            cs_sizes_list = m_data["cs_sizes"].to_list()
            has_causal = np.array([len(r) > 0 for r in rank_causal])
            n_non_null = int(has_causal.sum())
            order = np.argsort(-lbf_arr)
            sorted_lbf = lbf_arr[order]

            for cs_beta in _CS_POWER_FDP_BETAS:
                if cs_beta == 1.0:
                    causal_arr = has_causal
                else:
                    idx = _CS_BETA_SIZE_INDICES[cs_beta]
                    causal_arr = np.array([
                        len(r) > 0 and min(r) < cs_sizes_list[i][idx]
                        for i, r in enumerate(rank_causal)
                    ])
                sorted_causal = causal_arr[order]
                cum_tp = np.cumsum(sorted_causal)
                cum_fp = np.cumsum(~sorted_causal)
                n_reported = cum_tp + cum_fp
                power = cum_tp / max(n_non_null, 1)
                fdp = cum_fp / np.maximum(n_reported, 1)
                for k in range(len(sorted_lbf)):
                    rows.append({
                        "collection_name": coll or "",
                        "method": mg["method"],
                        "threshold": mg["threshold"],
                        "method_display": mg["method_display"],
                        "is_thresholded": mg["is_thresholded"],
                        "cs_beta": float(cs_beta),
                        "ser_log_bf": float(sorted_lbf[k]),
                        "power": float(power[k]),
                        "fdp": float(fdp[k]),
                    })
    return rows


def _plot_cs_power_fdp_panel(
    ax: "plt.Axes",
    panel_rows: list[dict],
    method_meta: pl.DataFrame,
    *,
    max_fdp: float,
    title: str,
) -> None:
    by_method: dict[str, list] = {}
    for r in panel_rows:
        by_method.setdefault(r["method_display"], []).append(r)
    for method_display, pts in sorted(by_method.items()):
        method = pts[0]["method"]
        color = viz_utils.method_color(method)
        fdp_arr = [p["fdp"] for p in pts]
        pwr_arr = [p["power"] for p in pts]
        lbf_arr = np.array([p["ser_log_bf"] for p in pts])
        ax.plot(fdp_arr, pwr_arr, color=color, linewidth=1.5, label=method_display)
        for thresh, marker in zip(_CS_LBF_MARKERS, _CS_LBF_MARKER_STYLES):
            mask = lbf_arr >= thresh
            if mask.any():
                idx = int(np.where(mask)[0][-1])
                ax.plot(fdp_arr[idx], pwr_arr[idx], marker=marker, color=color,
                        markersize=6, zorder=5, linestyle="none")
    ax.set_xlabel("FDP")
    ax.set_ylabel("Power")
    ax.set_xlim(0.0, max_fdp)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=10)


def _cs_power_fdp_filtered_raw(
    combined_data: dict,
    settings: dict,
) -> pl.DataFrame | None:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return None
    raw = (
        cs_data
        .filter(pl.col("method").is_in(fg))
        .join(
            method_meta.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
    )
    return raw if not raw.is_empty() else None


def _make_causal_rank(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = method_order(method_meta, fg)
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


def _make_preceding_posterior_mass_ecdf(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_preceding_mass_ecdf_summary(
        cs_data, method_meta,
        selected_methods=fg,
    )
    return viz_utils.render_preceding_mass_ecdf_chart(
        summary, collection_names=combined_data["collection_names"]
    )


def _make_agg_preceding_posterior_mass_ecdf(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_preceding_mass_ecdf_summary(
        cs_data, method_meta,
        selected_methods=fg,
    )
    return viz_utils.render_preceding_mass_ecdf_chart(summary, collection_names=[])


def _make_cs_dot_summary(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    cs_beta = settings.get("cs_beta", 0.95)
    collection_names = combined_data["collection_names"]
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
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


def _make_cs_calibrated_dot(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, target_coverage=min_beta, min_ser_log_bf=min_log_bf)
    return viz_utils.render_adaptive_cs_dot_chart(
        calibrated,
        collection_names=collection_names,
        nominal_beta=min_beta,
        min_ser_log_bf=min_log_bf,
    )


def _make_agg_cs_calibrated_dot(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, target_coverage=min_beta, min_ser_log_bf=min_log_bf)
    return viz_utils.render_adaptive_cs_dot_chart(
        calibrated,
        collection_names=[],
        nominal_beta=min_beta,
        min_ser_log_bf=min_log_bf,
    )


def _make_cs_size_power(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data, method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    nominal = beta_summary.filter(pl.col("beta") == round(min_beta, 2))
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, target_coverage=min_beta, min_ser_log_bf=min_log_bf)
    return viz_utils.render_cs_size_power_chart(
        nominal,
        calibrated,
        collection_names=collection_names,
        min_beta=min_beta,
        min_ser_log_bf=min_log_bf,
        max_cs_size=max_cs_size,
    )


def _make_cs_radius_power(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS radius data")
    radius_summary = viz_utils.make_cs_radius_power_summary(
        cs_data, method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    nominal = radius_summary.filter(pl.col("beta") == round(min_beta, 2))
    calibrated = viz_utils.find_calibrated_radius_summary(
        radius_summary, cs_data, method_meta,
        selected_methods=fg,
        target_coverage=min_beta,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_radius_power_chart(
        nominal,
        calibrated,
        collection_names=collection_names,
        min_beta=min_beta,
        min_ser_log_bf=min_log_bf,
        max_cs_size=max_cs_size,
    )


def _make_cs_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    raw = _cs_power_fdp_filtered_raw(combined_data, settings)
    if raw is None:
        return viz_utils.make_placeholder_chart("No CS power/FDP data")
    collection_names = combined_data["collection_names"]
    method_meta = combined_data["method_metadata"]
    max_fdp = settings.get("max_fdp", 0.5)
    theme = viz_utils.base_chart_theme()

    colls = [c for c in collection_names if c in raw["collection_name"].unique().to_list()]
    rows = _cs_power_fdp_curves(raw, collection_groups=colls)
    if not rows:
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    n_cols = len(colls)
    n_rows = len(_CS_POWER_FDP_BETAS)
    beta_labels = {1.0: "100% CS", 0.95: "95% CS", 0.50: "50% CS"}
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(theme["width"] * n_cols, theme["height"] * n_rows),
        squeeze=False,
    )
    for row_idx, cs_beta in enumerate(_CS_POWER_FDP_BETAS):
        for col_idx, coll in enumerate(colls):
            ax = axes[row_idx, col_idx]
            panel = [r for r in rows if r["collection_name"] == coll and r["cs_beta"] == cs_beta]
            title = f"{coll}\n{beta_labels[cs_beta]}" if row_idx == 0 else beta_labels[cs_beta]
            _plot_cs_power_fdp_panel(ax, panel, method_meta, max_fdp=max_fdp, title=title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    return fig


def _make_cs_power_size_coverage_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_power_size_coverage_trace_chart(
        beta_summary,
        collection_names=collection_names,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_cs_coverage_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_coverage_trace_chart(
        beta_summary,
        collection_names=collection_names,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_cs_coverage_size(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    curves = viz_utils.make_cs_coverage_size_curves(cs_data, method_meta, selected_methods=fg)
    return viz_utils.render_cs_coverage_size_chart(curves, collection_names=collection_names)


def _make_cs_coverage_radius(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS radius data")
    curves = viz_utils.make_cs_radius_power_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_coverage_radius_chart(curves, collection_names=collection_names)


def _make_cs_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    curves = viz_utils.make_cs_coverage_size_curves(cs_data, method_meta, selected_methods=fg)
    return viz_utils.render_cs_calibration_chart(curves, collection_names=collection_names)


def _make_agg_causal_rank(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    order = method_order(method_meta, fg)
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
    set_agg_facecolor(fig)
    return fig


def _make_agg_cs_power_fdp(combined_data: dict, settings: dict) -> plt.Figure:
    raw = _cs_power_fdp_filtered_raw(combined_data, settings)
    if raw is None:
        return viz_utils.make_placeholder_chart("No CS power/FDP data")
    method_meta = combined_data["method_metadata"]
    max_fdp = settings.get("max_fdp", 0.5)
    theme = viz_utils.base_chart_theme()

    rows = _cs_power_fdp_curves(raw, collection_groups=[None])
    if not rows:
        return viz_utils.make_placeholder_chart("No CS power/FDP data")

    beta_labels = {1.0: "100% CS", 0.95: "95% CS", 0.50: "50% CS"}
    n_cols = len(_CS_POWER_FDP_BETAS)
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(theme["width"] * n_cols, theme["height"]),
        squeeze=False,
    )
    for col_idx, cs_beta in enumerate(_CS_POWER_FDP_BETAS):
        ax = axes[0, col_idx]
        panel = [r for r in rows if r["cs_beta"] == cs_beta]
        _plot_cs_power_fdp_panel(ax, panel, method_meta, max_fdp=max_fdp, title=beta_labels[cs_beta])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    fig.tight_layout()
    set_agg_facecolor(fig)
    return fig


def _make_agg_cs_power_size_coverage_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    # collection_names=[] renders only the aggregate row (no per-collection rows)
    return viz_utils.render_cs_power_size_coverage_trace_chart(
        beta_summary,
        collection_names=[],
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_agg_cs_coverage_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_coverage_trace_chart(
        beta_summary,
        collection_names=[],
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_agg_cs_coverage_size(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    curves = viz_utils.make_cs_coverage_size_curves(cs_data, method_meta, selected_methods=fg)
    fig = viz_utils.render_cs_coverage_size_chart(curves, collection_names=[])
    set_agg_facecolor(fig)
    return fig


def _make_agg_cs_coverage_radius(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS radius data")
    curves = viz_utils.make_cs_radius_power_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    fig = viz_utils.render_cs_coverage_radius_chart(curves, collection_names=[])
    set_agg_facecolor(fig)
    return fig


def _make_agg_cs_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    curves = viz_utils.make_cs_coverage_size_curves(cs_data, method_meta, selected_methods=fg)
    fig = viz_utils.render_cs_calibration_chart(curves, collection_names=[])
    set_agg_facecolor(fig)
    return fig


def _make_agg_cs_size_power(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_power_size_coverage_summary(
        cs_data, method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    nominal = beta_summary.filter(pl.col("beta") == round(min_beta, 2))
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, target_coverage=min_beta, min_ser_log_bf=min_log_bf)
    return viz_utils.render_cs_size_power_chart(
        nominal,
        calibrated,
        collection_names=[],
        min_beta=min_beta,
        min_ser_log_bf=min_log_bf,
        max_cs_size=max_cs_size,
    )


def _make_agg_cs_radius_power(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS radius data")
    radius_summary = viz_utils.make_cs_radius_power_summary(
        cs_data, method_meta,
        selected_methods=fg,
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    nominal = radius_summary.filter(pl.col("beta") == round(min_beta, 2))
    calibrated = viz_utils.find_calibrated_radius_summary(
        radius_summary, cs_data, method_meta,
        selected_methods=fg,
        target_coverage=min_beta,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_radius_power_chart(
        nominal,
        calibrated,
        collection_names=[],
        min_beta=min_beta,
        min_ser_log_bf=min_log_bf,
        max_cs_size=max_cs_size,
    )


RENDERERS = {
    "causal_rank": _make_causal_rank,
    "preceding_posterior_mass_ecdf": _make_preceding_posterior_mass_ecdf,
    "agg_preceding_posterior_mass_ecdf": _make_agg_preceding_posterior_mass_ecdf,
    "cs_dot_summary": _make_cs_dot_summary,
    "cs_calibrated_dot": _make_cs_calibrated_dot,
    "agg_cs_calibrated_dot": _make_agg_cs_calibrated_dot,
    "cs_size_power": _make_cs_size_power,
    "cs_radius_power": _make_cs_radius_power,
    "cs_power_fdp": _make_cs_power_fdp,
    "cs_power_size_coverage_trace": _make_cs_power_size_coverage_trace,
    "cs_coverage_trace": _make_cs_coverage_trace,
    "cs_coverage_size": _make_cs_coverage_size,
    "cs_coverage_radius": _make_cs_coverage_radius,
    "cs_calibration": _make_cs_calibration,
    "agg_causal_rank": _make_agg_causal_rank,
    "agg_cs_power_fdp": _make_agg_cs_power_fdp,
    "agg_cs_power_size_coverage_trace": _make_agg_cs_power_size_coverage_trace,
    "agg_cs_coverage_trace": _make_agg_cs_coverage_trace,
    "agg_cs_size_power": _make_agg_cs_size_power,
    "agg_cs_radius_power": _make_agg_cs_radius_power,
    "agg_cs_coverage_size": _make_agg_cs_coverage_size,
    "agg_cs_coverage_radius": _make_agg_cs_coverage_radius,
    "agg_cs_calibration": _make_agg_cs_calibration,
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
