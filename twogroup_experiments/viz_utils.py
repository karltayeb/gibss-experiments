from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


_NOTHRESH_LINESTYLES = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def method_family_label_map() -> dict[str, str]:
    return {
        "logistic_threshold":    "Logistic",
        "cox_light_threshold":   "Cox Light",
        "twogroup":              "Twogroup",
        "twogroup_oracle":       "Twogroup",
        "logistic_oracle":       "Logistic",
        "cox_heavy":             "Cox Heavy",
        "twogroup_oracle_init":  "TG Oracle Init",
        "twogroup_scale_fam":    "TG Scale",
        "twogroup_loc_fam":      "TG Loc",
    }


def method_family_oracle_label_map() -> dict[str, str]:
    return {}


def method_family_color_map() -> dict[str, str]:
    # Okabe-Ito colorblind-safe palette + reddish variants for twogroup family
    return {
        "logistic_threshold":    "#0072B2",
        "logistic_oracle":       "#56B4E9",
        "cox_light_threshold":   "#009E73",
        "cox_heavy":             "#E69F00",
        "twogroup":              "#D55E00",  # vermillion
        "twogroup_oracle":       "#CC79A7",  # rose/mauve
        "twogroup_oracle_init":  "#994F00",  # dark burnt orange
        "twogroup_scale_fam":    "#FF6347",  # tomato
        "twogroup_loc_fam":      "#C0392B",  # crimson
    }


def method_color(method: str) -> str:
    family = method.rsplit("_L", 1)[0]
    return method_family_color_map().get(family, "#888888")


def method_metadata_from_method_spec_json(method_spec_json: str) -> dict[str, object]:
    method_spec = json.loads(method_spec_json)
    name = str(method_spec["fields"]["name"])
    kwargs = dict(method_spec["fields"].get("kwargs", {}))
    L = int(kwargs.get("L", 1))
    method_family = name.rsplit("_L", 1)[0]
    is_thresholded = "threshold" in method_family
    is_oracle = "oracle" in method_family
    family_label = method_family_label_map().get(method_family, method_family)
    oracle_label = method_family_oracle_label_map().get(method_family, "Oracle")
    suffix = "SER" if L == 1 else f"SuSiE [L={L}]"
    return {
        "method_family": method_family,
        "L": L,
        "is_thresholded": is_thresholded,
        "is_oracle": is_oracle,
        "oracle_label": oracle_label,
        "method_label_base": f"{family_label} {suffix}",
    }


def make_method_display_label(
    *,
    method_label_base: str,
    threshold: float | None,
    is_thresholded: bool,
    is_oracle: bool,
    oracle_label: str = "Oracle",
) -> str:
    if is_oracle:
        return f"{method_label_base} ({oracle_label})"
    if is_thresholded and threshold is not None:
        return f"{method_label_base} (@{threshold:g})"
    return method_label_base


def base_chart_theme() -> dict[str, float]:
    return {
        "width": 3.5,
        "height": 3.5,
    }


def make_placeholder_chart(title: str):
    theme = base_chart_theme()
    fig, ax = plt.subplots(figsize=(theme["width"], theme["height"]))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def expand_pip_calibration_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_threshold: float,
) -> pl.DataFrame:
    """Expand pip_plot_data to per-bin rows for render_pip_calibration."""
    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "collection_name": pl.String, "simulation_name": pl.String,
            "method": pl.String, "method_display": pl.String,
            "method_family": pl.String, "series_label": pl.String,
            "pip_bin_index": pl.Int64, "pip_left": pl.Float64, "pip_right": pl.Float64,
            "pip_mid": pl.Float64, "n_total": pl.Int64, "n_causal": pl.Int64,
            "empirical_rate": pl.Float64,
        })
    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_display_base",
        "method_label_base", "is_thresholded", "is_oracle",
    ).with_columns(
        pl.col("method_display").alias("series_label"),
        pl.col("method_display_base").alias("method_family"),
    )
    rows = []
    for row in pip_plot_data.iter_rows(named=True):
        counts = row["pip_bin_counts"]
        causal_counts = row["pip_bin_causal_counts"]
        for b in range(20):
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "pip_bin_index": b,
                "pip_left": b * 0.05,
                "pip_right": (b + 1) * 0.05,
                "pip_mid": (b + 0.5) * 0.05,
                "n_total": counts[b],
                "n_causal": causal_counts[b],
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_bin_index": pl.Int64, "pip_left": pl.Float64, "pip_right": pl.Float64,
        "pip_mid": pl.Float64, "n_total": pl.Int64, "n_causal": pl.Int64,
    })
    return (
        expanded
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .filter(~pl.col("is_thresholded") | (pl.col("threshold") == selected_threshold))
        .group_by(
            "collection_name", "method", "method_display", "method_family",
            "series_label", "pip_bin_index", "pip_left", "pip_right", "pip_mid",
        )
        .agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
        .with_columns(
            pl.when(pl.col("n_total") > 0)
            .then(pl.col("n_causal") / pl.col("n_total"))
            .otherwise(None)
            .alias("empirical_rate"),
            pl.col("collection_name").alias("simulation_name"),
        )
        .sort("collection_name", "method_display", "pip_mid")
    )


def expand_power_fdp_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_threshold: float,
    show_background_threshold_traces: bool,
) -> pl.DataFrame:
    """Expand pip_plot_data to per-threshold rows for render_power_fdp_chart."""
    _GRID = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99]

    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "simulation_name": pl.String, "method": pl.String, "method_display": pl.String,
            "trace_label": pl.String, "legend_label": pl.String,
            "is_selected_threshold": pl.Boolean,
            "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
        })
    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_label_base", "is_thresholded",
    )
    rows = []
    for row in pip_plot_data.iter_rows(named=True):
        for t, power, fdp in zip(
            _GRID, row["power_at_threshold"], row["fdp_at_threshold"]
        ):
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "pip_threshold": float(t),
                "power": power,
                "fdp": fdp,
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
    })
    joined = (
        expanded
        .filter(pl.col("method").is_in(list(selected_methods)))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded") | (pl.col("threshold") == selected_threshold)
            ).alias("is_selected_threshold")
        )
    )
    if not show_background_threshold_traces:
        joined = joined.filter(pl.col("is_selected_threshold"))
    return (
        joined
        .group_by(
            "collection_name", "method", "method_display", "method_label_base",
            "is_thresholded", "is_selected_threshold", "threshold", "pip_threshold",
        )
        .agg(pl.col("power").mean(), pl.col("fdp").mean())
        .with_columns(
            pl.col("collection_name").alias("simulation_name"),
            pl.when(pl.col("is_thresholded"))
            .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
            .otherwise(pl.col("method_display"))
            .alias("trace_label"),
            pl.when(pl.col("is_selected_threshold"))
            .then(pl.col("method_display"))
            .otherwise(None)
            .alias("legend_label"),
        )
        .sort("simulation_name", "method_display", "pip_threshold")
    )


def expand_causal_pip_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """Expand pip_plot_data to per-causal rows for render_causal_pip_chart."""
    if pip_plot_data.is_empty():
        return pl.DataFrame(schema={
            "collection_name": pl.String, "simulation_name": pl.String,
            "method": pl.String, "method_display": pl.String,
            "method_display_base": pl.String, "causal_pip": pl.Float64,
        })
    meta = method_metadata.select("method", "threshold", "method_display", "method_display_base")
    rows = []
    for row in pip_plot_data.iter_rows(named=True):
        for pip in row["causal_pips"]:
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "causal_pip": float(pip),
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String,
        "threshold": pl.Float64, "causal_pip": pl.Float64,
    })
    return (
        expanded
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(pl.col("collection_name").alias("simulation_name"))
        .sort("simulation_name", "method_display")
    )


def _plot_calibration_on_ax(ax: "plt.Axes", panel_df: pl.DataFrame, color: str | None = None) -> None:
    for series_label in sorted(x for x in panel_df.get_column("series_label").unique().to_list() if x is not None):
        series_df = panel_df.filter(pl.col("series_label") == series_label).sort("pip_mid")
        x = series_df["pip_mid"].to_numpy()
        y = series_df["empirical_rate"].to_numpy()
        if {"ci_lower", "ci_upper"}.issubset(series_df.columns):
            lower = series_df["ci_lower"].to_numpy()
            upper = series_df["ci_upper"].to_numpy()
            yerr = np.vstack(
                [
                    np.clip(y - lower, a_min=0.0, a_max=None),
                    np.clip(upper - y, a_min=0.0, a_max=None),
                ]
            )
            ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=2, color=color)
        else:
            ax.plot(x, y, marker="o", color=color)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle=":", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("PIP bin midpoint")


def render_pip_calibration(
    calibration_summary: pl.DataFrame,
    *,
    facet_by_simulation: bool,
    collection_names: list[str] | None = None,
):
    if calibration_summary.is_empty():
        return make_placeholder_chart("No PIP calibration data")

    theme = base_chart_theme()
    methods = sorted(m for m in calibration_summary.get_column("method_display").unique().to_list() if m is not None)
    n_cols = len(methods)
    method_color_lookup = {
        row["method_display"]: method_color(row["method"])
        for row in calibration_summary.select("method", "method_display").unique().iter_rows(named=True)
    }

    if facet_by_simulation:
        _all_sims = set(x for x in calibration_summary.get_column("simulation_name").unique().to_list() if x is not None)
        simulations = [n for n in collection_names if n in _all_sims] if collection_names else sorted(_all_sims)
        n_rows = len(simulations)
        fig, axes = plt.subplots(
            n_rows + 1, n_cols,
            figsize=(theme["width"] * n_cols, theme["height"] * (n_rows + 1)),
            squeeze=False,
        )
        for col_idx, method_name in enumerate(methods):
            axes[0, col_idx].set_title(method_name, fontsize=9)
        for row_idx, sim_name in enumerate(simulations):
            for col_idx, method_name in enumerate(methods):
                panel_df = calibration_summary.filter(
                    (pl.col("simulation_name") == sim_name)
                    & (pl.col("method_display") == method_name)
                )
                _plot_calibration_on_ax(axes[row_idx, col_idx], panel_df, color=method_color_lookup.get(method_name))
            axes[row_idx, 0].set_ylabel(f"{sim_name}\nEmpirical causal freq.", fontsize=9)
        # aggregate row
        _agg = (
            calibration_summary
            .group_by("method", "method_display", "series_label", "method_family", "pip_bin_index", "pip_left", "pip_right", "pip_mid")
            .agg(pl.col("n_total").sum(), pl.col("n_causal").sum())
            .with_columns(
                pl.when(pl.col("n_total") > 0)
                .then(pl.col("n_causal") / pl.col("n_total"))
                .otherwise(None)
                .alias("empirical_rate")
            )
        )
        for col_idx, method_name in enumerate(methods):
            ax = axes[n_rows, col_idx]
            ax.set_facecolor("#ddeeff")
            _plot_calibration_on_ax(ax, _agg.filter(pl.col("method_display") == method_name), color=method_color_lookup.get(method_name))
        axes[n_rows, 0].set_ylabel("All\nEmpirical causal freq.", fontsize=9, fontweight="bold")
    else:
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(theme["width"] * n_cols, theme["height"]),
            squeeze=False,
        )
        for col_idx, method_name in enumerate(methods):
            panel_df = calibration_summary.filter(pl.col("method_display") == method_name)
            _plot_calibration_on_ax(axes[0, col_idx], panel_df, color=method_color_lookup.get(method_name))
            axes[0, col_idx].set_title(method_name)
        axes[0, 0].set_ylabel("Empirical causal frequency")

    fig.tight_layout()
    return fig


def make_power_fdp_summary(plot_data: pl.DataFrame) -> pl.DataFrame:
    return (
        plot_data.select(
            "simulation_name",
            "method",
            "method_display",
            "trace_label",
            "legend_label",
            "is_selected_threshold",
            "pip_threshold",
            "power",
            "fdp",
        )
        .group_by(
            "simulation_name",
            "method",
            "method_display",
            "trace_label",
            "legend_label",
            "is_selected_threshold",
            "pip_threshold",
        )
        .agg(
            pl.col("power").mean().alias("power"),
            pl.col("fdp").mean().alias("fdp"),
        )
        .sort("simulation_name", "method_display", "trace_label", "pip_threshold")
    )


_PIP_MARKER_THRESHOLDS = [0.5, 0.9, 0.99]
_PIP_MARKER_COLORS = ["#e377c2", "#ff7f0e", "#d62728"]  # pink, orange, red
_PIP_MARKER_STYLES = ["D", "s", "^"]  # diamond, square, triangle


def _plot_power_fdp_on_ax(
    ax: "plt.Axes",
    panel_df: pl.DataFrame,
    *,
    max_fdp: float,
    fixed_y_scale: bool,
    title: str,
) -> None:
    marker_legend_added: set[float] = set()
    for trace_label in sorted(x for x in panel_df.get_column("trace_label").unique().to_list() if x is not None):
        trace_df = panel_df.filter(pl.col("trace_label") == trace_label).sort("pip_threshold")
        is_selected = bool(trace_df["is_selected_threshold"][0])
        legend_label = trace_df["legend_label"][0]
        color = method_color(trace_df["method"][0])
        ax.plot(
            trace_df["fdp"].to_numpy(),
            trace_df["power"].to_numpy(),
            color=color,
            linewidth=2.0 if is_selected else 1.0,
            alpha=1.0 if is_selected else 0.2,
            label=legend_label if legend_label is not None else "_nolegend_",
        )
        if is_selected:
            pip_arr = trace_df["pip_threshold"].to_numpy()
            fdp_arr = trace_df["fdp"].to_numpy()
            pwr_arr = trace_df["power"].to_numpy()
            for thresh, mcolor, mstyle in zip(
                _PIP_MARKER_THRESHOLDS, _PIP_MARKER_COLORS, _PIP_MARKER_STYLES
            ):
                idx = int(round(thresh * 1000)) - 1  # threshold_grid[i] = (i+1)/1000
                if 0 <= idx < len(pip_arr):
                    mlabel = f"PIP={thresh:g}" if thresh not in marker_legend_added else "_nolegend_"
                    ax.scatter(
                        fdp_arr[idx],
                        pwr_arr[idx],
                        color=mcolor,
                        marker=mstyle,
                        s=60,
                        zorder=5,
                        label=mlabel,
                    )
                    marker_legend_added.add(thresh)
    ax.set_xlabel("FDP")
    ax.set_xlim(0.0, max_fdp)
    if fixed_y_scale:
        ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=11)


def render_power_fdp_chart(
    power_fdp_summary: pl.DataFrame,
    *,
    facet: bool,
    max_fdp: float,
    fixed_y_scale: bool,
    title: str | None = None,
    legend_outside: bool = False,
    square_axes: bool = False,
    collection_names: list[str] | None = None,
):
    if power_fdp_summary.is_empty():
        return make_placeholder_chart("No power vs FDP data")
    visible = power_fdp_summary
    theme = base_chart_theme()

    if facet:
        _all_sims = set(x for x in visible.get_column("simulation_name").unique().to_list() if x is not None)
        simulations = [n for n in collection_names if n in _all_sims] if collection_names else sorted(_all_sims)
        n_cols = len(simulations)
        _legend_w = 2.0
        _fig_w = theme["width"] * (n_cols + 1) + _legend_w
        _plot_frac = (theme["width"] * (n_cols + 1)) / _fig_w
        fig, axes = plt.subplots(
            1, n_cols + 1,
            figsize=(_fig_w, theme["height"]),
            squeeze=False,
        )
        for col_idx, sim_name in enumerate(simulations):
            ax = axes[0, col_idx]
            _plot_power_fdp_on_ax(
                ax,
                visible.filter(pl.col("simulation_name") == sim_name),
                max_fdp=max_fdp,
                fixed_y_scale=fixed_y_scale,
                title=sim_name,
            )
        # aggregate column
        _agg_pf = (
            visible
            .group_by("method", "method_display", "trace_label", "legend_label", "is_selected_threshold", "pip_threshold")
            .agg(pl.col("power").mean(), pl.col("fdp").mean())
        )
        agg_ax = axes[0, n_cols]
        agg_ax.set_facecolor("#ddeeff")
        _plot_power_fdp_on_ax(agg_ax, _agg_pf, max_fdp=max_fdp, fixed_y_scale=fixed_y_scale, title="All")
        axes[0, 0].set_ylabel("Power")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=8,
                       loc="center left", bbox_to_anchor=(_plot_frac + 0.02, 0.5))
            fig.tight_layout(rect=[0, 0, _plot_frac, 1])
        else:
            fig.tight_layout()
        return fig
    else:
        _h = theme["height"]
        _w = _h * 2.2 if square_axes else theme["width"] * 1.45
        fig, ax = plt.subplots(figsize=(_w, _h))
        if square_axes:
            ax.set_box_aspect(1)
        _plot_power_fdp_on_ax(
            ax, visible, max_fdp=max_fdp, fixed_y_scale=fixed_y_scale,
            title=title if title is not None else "Aggregate",
        )
        ax.set_ylabel("Power")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_outside:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False, fontsize=8)
            else:
                ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def make_causal_pip_summary(plot_data: pl.DataFrame) -> pl.DataFrame:
    return (
        plot_data.group_by("simulation_name", "method", "method_display", "method_display_base", "threshold")
        .agg(pl.col("causal_pip").mean().alias("mean_causal_pip"))
        .sort("simulation_name", "method_display", "threshold")
    )


def _plot_causal_pip_on_ax(
    ax: "plt.Axes",
    panel_df: pl.DataFrame,
    *,
    title: str,
    method_order: list[str] | None = None,
) -> None:
    _all = set(x for x in panel_df.get_column("method").unique().to_list() if x is not None)
    _methods = [m for m in method_order if m in _all] if method_order is not None else sorted(_all)
    _nothresh_idx = 0
    for method_name in _methods:
        method_df = panel_df.filter(pl.col("method") == method_name)
        color = method_color(method_name)
        label = method_df["method_display_base"][0]
        if method_df["threshold"].is_null().all():
            y_val = float(method_df["mean_causal_pip"].mean())
            ls = _NOTHRESH_LINESTYLES[_nothresh_idx % len(_NOTHRESH_LINESTYLES)]
            ax.axhline(y=y_val, color=color, linestyle=ls, linewidth=1.5, label=label)
            _nothresh_idx += 1
        else:
            method_df = method_df.sort("threshold")
            ax.plot(
                method_df["threshold"].drop_nulls().to_numpy(),
                method_df["mean_causal_pip"].to_numpy(),
                marker="o",
                color=color,
                label=label,
            )
    ax.set_xlabel("Threshold")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=11)


def render_causal_pip_chart(
    causal_pip_summary: pl.DataFrame,
    *,
    facet: bool,
    title: str | None = None,
    legend_outside: bool = False,
    square_axes: bool = False,
    method_order: list[str] | None = None,
    collection_names: list[str] | None = None,
):
    if causal_pip_summary.is_empty():
        return make_placeholder_chart("No causal PIP data")
    theme = base_chart_theme()

    if facet:
        _all_sims = set(x for x in causal_pip_summary.get_column("simulation_name").unique().to_list() if x is not None)
        simulations = [n for n in collection_names if n in _all_sims] if collection_names else sorted(_all_sims)
        n_cols = len(simulations)
        _legend_w = 2.0
        _fig_w = theme["width"] * (n_cols + 1) + _legend_w
        _plot_frac = (theme["width"] * (n_cols + 1)) / _fig_w
        fig, axes = plt.subplots(
            1, n_cols + 1,
            figsize=(_fig_w, theme["height"]),
            squeeze=False,
        )
        for col_idx, sim_name in enumerate(simulations):
            _plot_causal_pip_on_ax(
                axes[0, col_idx],
                causal_pip_summary.filter(pl.col("simulation_name") == sim_name),
                title=sim_name,
                method_order=method_order,
            )
        _agg_cp = (
            causal_pip_summary
            .group_by("method", "method_display", "method_display_base", "threshold")
            .agg(pl.col("mean_causal_pip").mean())
        )
        agg_ax = axes[0, n_cols]
        agg_ax.set_facecolor("#ddeeff")
        _plot_causal_pip_on_ax(agg_ax, _agg_cp, title="All", method_order=method_order)
        axes[0, 0].set_ylabel("Mean causal PIP")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=8,
                       loc="center left", bbox_to_anchor=(_plot_frac + 0.02, 0.5))
            fig.tight_layout(rect=[0, 0, _plot_frac, 1])
        else:
            fig.tight_layout()
        return fig
    else:
        _h = theme["height"]
        _w = _h * 2.2 if square_axes else theme["width"] * 1.6
        fig, ax = plt.subplots(figsize=(_w, _h))
        if square_axes:
            ax.set_box_aspect(1)
        _plot_causal_pip_on_ax(
            ax, causal_pip_summary,
            title=title if title is not None else "Aggregate",
            method_order=method_order,
        )
        ax.set_ylabel("Mean causal PIP")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_outside:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False, fontsize=8)
            else:
                ax.legend(frameon=False)

    fig.tight_layout()
    return fig


def make_causal_rank_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
) -> pl.DataFrame:
    """Mean causal rank per (collection_name, method, threshold).

    Causal rank = minimum cs_size required to include any causal = min(rank_of_causal) + 1.
    Takes the minimum across all L effects per sample, then averages across samples.
    """
    empty_schema = {
        "simulation_name": pl.String,
        "method": pl.String,
        "threshold": pl.Float64,
        "method_display": pl.String,
        "method_display_base": pl.String,
        "mean_causal_rank": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_display_base", "is_thresholded"
    )
    per_effect = (
        cs_plot_data
        .filter(pl.col("method").is_in(list(selected_methods)))
        .filter(pl.col("rank_of_causal").list.len() > 0)
        .with_columns(
            (pl.col("rank_of_causal").list.min() + 1).alias("causal_rank")
        )
    )
    if per_effect.is_empty():
        return pl.DataFrame(schema=empty_schema)
    per_sample = (
        per_effect
        .group_by("collection_name", "sample_id", "method", "threshold")
        .agg(pl.col("causal_rank").min())
    )
    return (
        per_sample
        .group_by("collection_name", "method", "threshold")
        .agg(pl.col("causal_rank").mean().alias("mean_causal_rank"))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(pl.col("collection_name").alias("simulation_name"))
        .sort("simulation_name", "method_display", "threshold")
    )


def _plot_causal_rank_on_ax(
    ax: "plt.Axes",
    panel_df: pl.DataFrame,
    *,
    title: str,
    method_order: list[str] | None = None,
) -> None:
    _all = set(x for x in panel_df.get_column("method").unique().to_list() if x is not None)
    _methods = [m for m in method_order if m in _all] if method_order is not None else sorted(_all)
    _nothresh_idx = 0
    for method_name in _methods:
        method_df = panel_df.filter(pl.col("method") == method_name)
        color = method_color(method_name)
        label = method_df["method_display_base"][0]
        if method_df["threshold"].is_null().all():
            y_val = float(method_df["mean_causal_rank"].mean())
            ls = _NOTHRESH_LINESTYLES[_nothresh_idx % len(_NOTHRESH_LINESTYLES)]
            ax.axhline(y=y_val, color=color, linestyle=ls, linewidth=1.5, label=label)
            _nothresh_idx += 1
        else:
            method_df = method_df.sort("threshold")
            ax.plot(
                method_df["threshold"].drop_nulls().to_numpy(),
                method_df["mean_causal_rank"].to_numpy(),
                marker="o",
                color=color,
                label=label,
            )
    ax.set_xlabel("Threshold")
    ax.set_ylim(bottom=1)
    ax.set_title(title, fontsize=11)


def render_causal_rank_chart(
    causal_rank_summary: pl.DataFrame,
    *,
    facet: bool,
    title: str | None = None,
    legend_outside: bool = False,
    square_axes: bool = False,
    method_order: list[str] | None = None,
    collection_names: list[str] | None = None,
):
    if causal_rank_summary.is_empty():
        return make_placeholder_chart("No causal rank data")
    theme = base_chart_theme()

    if facet:
        _all_sims = set(x for x in causal_rank_summary.get_column("simulation_name").unique().to_list() if x is not None)
        simulations = [n for n in collection_names if n in _all_sims] if collection_names else sorted(_all_sims)
        n_cols = len(simulations)
        _legend_w = 2.0
        _fig_w = theme["width"] * (n_cols + 1) + _legend_w
        _plot_frac = (theme["width"] * (n_cols + 1)) / _fig_w
        fig, axes = plt.subplots(1, n_cols + 1, figsize=(_fig_w, theme["height"]), squeeze=False)
        for col_idx, sim_name in enumerate(simulations):
            _plot_causal_rank_on_ax(
                axes[0, col_idx],
                causal_rank_summary.filter(pl.col("simulation_name") == sim_name),
                title=sim_name,
                method_order=method_order,
            )
        _agg_cr = (
            causal_rank_summary
            .group_by("method", "method_display", "method_display_base", "threshold")
            .agg(pl.col("mean_causal_rank").mean())
        )
        agg_ax = axes[0, n_cols]
        agg_ax.set_facecolor("#ddeeff")
        _plot_causal_rank_on_ax(agg_ax, _agg_cr, title="All", method_order=method_order)
        axes[0, 0].set_ylabel("Mean causal rank")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=8,
                       loc="center left", bbox_to_anchor=(_plot_frac + 0.02, 0.5))
            fig.tight_layout(rect=[0, 0, _plot_frac, 1])
        else:
            fig.tight_layout()
        return fig
    else:
        _h = theme["height"]
        _w = _h * 2.2 if square_axes else theme["width"] * 1.6
        fig, ax = plt.subplots(figsize=(_w, _h))
        if square_axes:
            ax.set_box_aspect(1)
        _plot_causal_rank_on_ax(
            ax, causal_rank_summary,
            title=title if title is not None else "Aggregate",
            method_order=method_order,
        )
        ax.set_ylabel("Mean causal rank")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_outside:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False, fontsize=8)
            else:
                ax.legend(frameon=False)
        fig.tight_layout()
        return fig


def expand_mass_above_causal_from_compact(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """One row per (sample_id, method, threshold, causal_idx).

    For each causal, picks the L effect with highest alpha[causal_idx], uses
    that effect's mass_above_causal.
    """
    empty_schema = {
        "collection_name": pl.String, "simulation_name": pl.String,
        "method": pl.String, "threshold": pl.Float64,
        "method_display": pl.String, "method_display_base": pl.String,
        "causal_idx": pl.Int64, "mass_above_causal": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)
    meta = method_metadata.select("method", "threshold", "method_display", "method_display_base")
    expanded = (
        cs_plot_data
        .select("collection_name", "sample_id", "method", "threshold",
                "causal_indices", "causal_alpha", "mass_above_causal")
        .explode("causal_indices", "causal_alpha", "mass_above_causal")
    )
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)
    per_causal = (
        expanded
        .group_by("collection_name", "sample_id", "method", "threshold", "causal_indices")
        .agg(
            pl.col("mass_above_causal").sort_by(pl.col("causal_alpha"), descending=True).first()
        )
        .rename({"causal_indices": "causal_idx"})
    )
    return (
        per_causal
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(pl.col("collection_name").alias("simulation_name"))
        .sort("simulation_name", "method_display")
    )


def make_mass_above_causal_summary(plot_data: pl.DataFrame) -> pl.DataFrame:
    return (
        plot_data.group_by("simulation_name", "method", "method_display", "method_display_base", "threshold")
        .agg(pl.col("mass_above_causal").mean().alias("mean_mass_above_causal"))
        .sort("simulation_name", "method_display", "threshold")
    )


def _plot_mass_above_causal_on_ax(
    ax: "plt.Axes",
    panel_df: pl.DataFrame,
    *,
    title: str,
    method_order: list[str] | None = None,
) -> None:
    _all = set(x for x in panel_df.get_column("method").unique().to_list() if x is not None)
    _methods = [m for m in method_order if m in _all] if method_order is not None else sorted(_all)
    _nothresh_idx = 0
    for method_name in _methods:
        method_df = panel_df.filter(pl.col("method") == method_name)
        color = method_color(method_name)
        label = method_df["method_display_base"][0]
        if method_df["threshold"].is_null().all():
            y_val = float(method_df["mean_mass_above_causal"].mean())
            ls = _NOTHRESH_LINESTYLES[_nothresh_idx % len(_NOTHRESH_LINESTYLES)]
            ax.axhline(y=y_val, color=color, linestyle=ls, linewidth=1.5, label=label)
            _nothresh_idx += 1
        else:
            method_df = method_df.sort("threshold")
            ax.plot(
                method_df["threshold"].drop_nulls().to_numpy(),
                method_df["mean_mass_above_causal"].to_numpy(),
                marker="o",
                color=color,
                label=label,
            )
    ax.set_xlabel("Threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=11)


def render_mass_above_causal_chart(
    summary: pl.DataFrame,
    *,
    facet: bool,
    title: str | None = None,
    legend_outside: bool = False,
    square_axes: bool = False,
    method_order: list[str] | None = None,
    collection_names: list[str] | None = None,
):
    if summary.is_empty():
        return make_placeholder_chart("No mass above causal data")
    theme = base_chart_theme()
    if facet:
        _all_sims = set(x for x in summary.get_column("simulation_name").unique().to_list() if x is not None)
        simulations = [n for n in collection_names if n in _all_sims] if collection_names else sorted(_all_sims)
        n_cols = len(simulations)
        _legend_w = 2.0
        _fig_w = theme["width"] * (n_cols + 1) + _legend_w
        _plot_frac = (theme["width"] * (n_cols + 1)) / _fig_w
        fig, axes = plt.subplots(1, n_cols + 1, figsize=(_fig_w, theme["height"]), squeeze=False)
        for col_idx, sim_name in enumerate(simulations):
            _plot_mass_above_causal_on_ax(
                axes[0, col_idx],
                summary.filter(pl.col("simulation_name") == sim_name),
                title=sim_name,
                method_order=method_order,
            )
        _agg_mac = (
            summary
            .group_by("method", "method_display", "method_display_base", "threshold")
            .agg(pl.col("mean_mass_above_causal").mean())
        )
        agg_ax = axes[0, n_cols]
        agg_ax.set_facecolor("#ddeeff")
        _plot_mass_above_causal_on_ax(agg_ax, _agg_mac, title="All", method_order=method_order)
        axes[0, 0].set_ylabel("Mean mass above causal")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, fontsize=8,
                       loc="center left", bbox_to_anchor=(_plot_frac + 0.02, 0.5))
            fig.tight_layout(rect=[0, 0, _plot_frac, 1])
        else:
            fig.tight_layout()
        return fig
    else:
        _h = theme["height"]
        _w = _h * 2.2 if square_axes else theme["width"] * 1.6
        fig, ax = plt.subplots(figsize=(_w, _h))
        if square_axes:
            ax.set_box_aspect(1)
        _plot_mass_above_causal_on_ax(
            ax, summary,
            title=title if title is not None else "Aggregate",
            method_order=method_order,
        )
        ax.set_ylabel("Mean mass above causal")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if legend_outside:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False, fontsize=8)
            else:
                ax.legend(frameon=False)
        fig.tight_layout()
        return fig


def _expand_cs_to_beta_rows(cs_plot_data: pl.DataFrame) -> pl.DataFrame:
    """Expand compact cs_plot_data to one row per (sample_id, method, threshold, l, beta).

    covered = any causal in CS_l at this beta.
    power = fraction of causals in CS_l at this beta.
    """
    from utils import CS_BETA_GRID

    rows: list[dict] = []
    for row in cs_plot_data.iter_rows(named=True):
        ranks = row["rank_of_causal"]
        n_causal = max(len(ranks), 1)
        for beta, cs_size in zip(CS_BETA_GRID.tolist(), row["cs_sizes"]):
            n_covered = sum(1 for r in ranks if r < cs_size)
            rows.append({
                "collection_name": row["collection_name"],
                "sample_id": row["sample_id"],
                "method": row["method"],
                "threshold": row["threshold"],
                "l": row["l"],
                "beta": float(beta),
                "cs_size": cs_size,
                "covered": n_covered > 0,
                "power": float(n_covered / n_causal),
                "ser_log_bf": row["ser_log_bf"],
            })
    if not rows:
        return pl.DataFrame(schema={
            "collection_name": pl.String, "sample_id": pl.String,
            "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
            "beta": pl.Float64, "cs_size": pl.Int64, "covered": pl.Boolean,
            "power": pl.Float64, "ser_log_bf": pl.Float64,
        })
    return pl.from_dicts(rows, schema={
        "collection_name": pl.String, "sample_id": pl.String,
        "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
        "beta": pl.Float64, "cs_size": pl.Int64, "covered": pl.Boolean,
        "power": pl.Float64, "ser_log_bf": pl.Float64,
    })


def make_cs_beta_trace_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_threshold: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Summarize cs_plot_data across all betas for each (collection_name, method, threshold, beta)."""
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean,
        "is_selected_threshold": pl.Boolean, "beta": pl.Float64,
        "power": pl.Float64, "coverage": pl.Float64, "cs_size": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    expanded = _expand_cs_to_beta_rows(
        cs_plot_data.filter(pl.col("method").is_in(list(selected_methods)))
    )
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)
    filtered = (
        expanded
        .with_columns(
            ((pl.col("cs_size") <= max_cs_size) & (pl.col("ser_log_bf") >= min_ser_log_bf)).alias("valid_cs")
        )
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded") | (pl.col("threshold") == selected_threshold)
            ).alias("is_selected_threshold")
        )
    )
    return (
        filtered
        .group_by("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(
            (pl.col("covered") & pl.col("valid_cs")).cast(pl.Float64).mean().alias("power"),
            pl.when(pl.col("valid_cs")).then(pl.col("covered").cast(pl.Float64)).mean().alias("coverage"),
            pl.when(pl.col("valid_cs")).then(pl.col("cs_size").cast(pl.Float64)).mean().alias("cs_size"),
        )
        .sort("collection_name", "method_display", "threshold", "beta")
    )


def render_cs_dot_summary_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
    selected_beta: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> "plt.Figure":
    """Grouped dot plot: X=collection, dots per method, 3 metric subplots in one row."""
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

    beta_df = summary.filter(pl.col("beta") == selected_beta)
    if beta_df.is_empty():
        return make_placeholder_chart(f"No data at β={selected_beta:.2f}")

    # only show selected threshold for thresholded methods
    beta_df = beta_df.filter(pl.col("is_selected_threshold"))

    theme = base_chart_theme()
    metrics = [("power", "Power"), ("coverage", "Coverage"), ("cs_size", "CS Size")]
    _legend_w = 2.5
    _plot_w = theme["width"] * len(metrics)
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w
    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(_fig_w, theme["height"]),
        squeeze=False,
    )
    axes = axes[0]

    method_order = (
        beta_df.select("method", "method_display", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method_display"])
        ["method_display"].to_list()
    )
    n_methods = len(method_order)
    offsets = np.linspace(-0.35, 0.35, n_methods) if n_methods > 1 else np.array([0.0])
    method_offset = {m: float(offsets[i]) for i, m in enumerate(method_order)}
    coll_to_x = {c: i for i, c in enumerate(collection_names)}
    agg_x = len(collection_names)
    _agg_dot = (
        beta_df
        .group_by("method", "method_display", "threshold", "is_thresholded", "is_selected_threshold")
        .agg(pl.col("power").mean(), pl.col("coverage").mean(), pl.col("cs_size").mean())
    )

    legend_handles = []
    legend_labels = []
    seen_labels: set[str] = set()

    for col_idx, (metric_col, metric_title) in enumerate(metrics):
        ax = axes[col_idx]
        for trow in (
            beta_df.select("method", "method_display", "threshold", "is_thresholded")
            .unique()
            .sort(["is_thresholded", "method_display"])
            .iter_rows(named=True)
        ):
            thresh_filter = (
                pl.col("threshold").is_null() if trow["threshold"] is None
                else (pl.col("threshold") == trow["threshold"])
            )
            m_df = beta_df.filter(
                (pl.col("method") == trow["method"]) & thresh_filter
            )
            color = method_color(trow["method"])
            offset = method_offset.get(trow["method_display"], 0.0)
            xs, ys = [], []
            for coll in collection_names:
                coll_row = m_df.filter(pl.col("collection_name") == coll)
                if coll_row.is_empty():
                    continue
                val = coll_row[metric_col][0]
                if val is not None:
                    xs.append(coll_to_x[coll] + offset)
                    ys.append(float(val))
            # aggregate dot
            agg_row = _agg_dot.filter((pl.col("method") == trow["method"]) & thresh_filter)
            if not agg_row.is_empty():
                agg_val = agg_row[metric_col][0]
                if agg_val is not None:
                    xs.append(agg_x + offset)
                    ys.append(float(agg_val))
            if xs:
                sc = ax.scatter(xs, ys, color=color, s=60, zorder=3)
                if trow["method_display"] not in seen_labels:
                    legend_handles.append(sc)
                    legend_labels.append(trow["method_display"])
                    seen_labels.add(trow["method_display"])

        for ci, _ in enumerate(collection_names):
            if ci % 2 == 1:
                ax.axvspan(ci - 0.5, ci + 0.5, color="lightgrey", alpha=0.35, zorder=0, linewidth=0)
        ax.axvspan(agg_x - 0.5, agg_x + 0.5, color="#ddeeff", zorder=0, linewidth=0)

        if metric_col == "coverage":
            ax.axhline(y=selected_beta, color="black", linestyle="--", linewidth=1.0)

        ax.set_title(metric_title)
        ax.set_xticks(list(range(agg_x + 1)))
        ax.set_xticklabels(list(collection_names) + ["All"], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, agg_x + 0.5)

    settings_text = f"β = {selected_beta:.2f}\nmax cs = {max_cs_size}\nmin log BF = {min_ser_log_bf:.1f}"
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.95),
        )
    fig.text(
        _plot_frac + 0.03, 0.30, settings_text,
        fontsize=7, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.8),
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_beta_trace_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
    selected_threshold: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> "plt.Figure":
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

    # only show selected threshold for thresholded methods
    summary = summary.filter(pl.col("is_selected_threshold"))

    metrics = [("power", "Power"), ("cs_size", "CS Size"), ("coverage", "Coverage")]
    theme = base_chart_theme()
    n_rows = len(collection_names)
    _legend_w = 2.5
    _plot_w = theme["width"] * len(metrics)
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w
    fig, axes = plt.subplots(
        n_rows + 1, len(metrics),
        figsize=(_fig_w, theme["height"] * (n_rows + 1)),
        squeeze=False,
    )

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    def _plot_beta_trace_row(row_idx: int, row_df: pl.DataFrame) -> None:
        trace_labels = (
            row_df.select("method", "threshold", "method_display", "is_thresholded")
            .unique()
            .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
        )
        for col_idx, (metric_col, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for trow in trace_labels.iter_rows(named=True):
                thresh_filter = (
                    pl.col("threshold").is_null() if trow["threshold"] is None
                    else (pl.col("threshold") == trow["threshold"])
                )
                trace_df = row_df.filter(
                    (pl.col("method") == trow["method"]) & thresh_filter
                ).sort("beta")
                if trace_df.is_empty():
                    continue
                color = method_color(trow["method"])
                if trow["is_thresholded"]:
                    label = f"{trow['method_display']} (@{trow['threshold']:g})"
                else:
                    label = trow["method_display"]
                line, = ax.plot(
                    trace_df["beta"].to_numpy(),
                    trace_df[metric_col].to_numpy(),
                    color=color,
                    linewidth=2.0,
                )
                if label not in seen_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)
                    seen_labels.add(label)
            if metric_col == "coverage":
                betas = row_df["beta"].unique().sort().to_numpy()
                if len(betas):
                    ax.plot(betas, betas, color="black", linestyle="--", linewidth=1.0)
            if row_idx == 0:
                ax.set_title(metric_title)
            ax.set_xlabel("Nominal coverage (β)")

    for row_idx, coll_name in enumerate(collection_names):
        _plot_beta_trace_row(row_idx, summary.filter(pl.col("collection_name") == coll_name))
        axes[row_idx, 0].set_ylabel(coll_name, fontsize=9, fontweight="bold")

    # aggregate row
    _agg_bt = (
        summary
        .group_by("method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(pl.col("power").mean(), pl.col("coverage").mean(), pl.col("cs_size").mean())
    )
    _plot_beta_trace_row(n_rows, _agg_bt)
    for col_idx in range(len(metrics)):
        axes[n_rows, col_idx].set_facecolor("#ddeeff")
    axes[n_rows, 0].set_ylabel("All", fontsize=9, fontweight="bold")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    settings_text = f"threshold = {selected_threshold:g}\nmax cs = {max_cs_size}\nmin log BF = {min_ser_log_bf:.1f}"
    fig.text(
        _plot_frac + 0.03, 0.35, settings_text,
        fontsize=7, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.8),
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig
