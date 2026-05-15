from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml


_NOTHRESH_LINESTYLES = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def method_family_label_map() -> dict[str, str]:
    return {
        "logistic_threshold": "Logistic",
        "cox_light_threshold": "Cox Light",
        "twogroup": "Twogroup",
        "twogroup_oracle": "Twogroup",
        "logistic_oracle": "Logistic",
        "cox_heavy": "Cox Heavy",
    }


def method_family_oracle_label_map() -> dict[str, str]:
    return {}


def method_family_color_map() -> dict[str, str]:
    # Okabe-Ito colorblind-safe palette
    return {
        "logistic_threshold":  "#0072B2",
        "logistic_oracle":     "#56B4E9",
        "cox_light_threshold": "#009E73",
        "cox_heavy":           "#E69F00",
        "twogroup":            "#D55E00",
        "twogroup_oracle":     "#CC79A7",
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


def add_plot_metadata_columns(plot_data: pl.DataFrame) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    metadata_rows = [
        {"method_spec": method_spec, **method_metadata_from_method_spec_json(method_spec)}
        for method_spec in plot_data.get_column("method_spec").unique().to_list()
    ]
    metadata_df = pl.from_dicts(metadata_rows)
    return plot_data.join(metadata_df, on="method_spec", how="left").with_columns(
        pl.when(pl.col("is_oracle"))
        .then(pl.format("{} ({})", pl.col("method_label_base"), pl.col("oracle_label")))
        .when(pl.col("is_thresholded") & pl.col("threshold").is_not_null())
        .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
        .otherwise(pl.col("method_label_base"))
        .alias("method_display")
    )


def empty_pip_threshold_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
            "method_spec": pl.String,
            "simulation_spec": pl.String,
            "pip_threshold": pl.Float64,
            "selected_total": pl.Int64,
            "selected_causal": pl.Int64,
            "power": pl.Float64,
            "fdp": pl.Float64,
            "n_exact": pl.Int64,
            "n_causal_exact": pl.Int64,
            "batch_hash": pl.String,
            "batch_name": pl.String,
            "simulation_name": pl.String,
        }
    )


def empty_causal_pip_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
            "method_spec": pl.String,
            "simulation_spec": pl.String,
            "causal_pip": pl.Float64,
            "max_pip": pl.Float64,
            "batch_hash": pl.String,
            "batch_name": pl.String,
            "simulation_name": pl.String,
        }
    )


def empty_cs_component_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
            "method_spec": pl.String,
            "simulation_spec": pl.String,
            "component": pl.Int64,
            "ordered_pips": pl.List(pl.Float64),
            "betas": pl.List(pl.Float64),
            "cs_sizes": pl.List(pl.Int64),
            "ser_log_bf": pl.Float64,
            "batch_hash": pl.String,
            "batch_name": pl.String,
            "simulation_name": pl.String,
        }
    )


def empty_cs_truth_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
            "method_spec": pl.String,
            "simulation_spec": pl.String,
            "causal_variable": pl.Int64,
            "component": pl.Int64,
            "causal_rank": pl.Int64,
            "betas": pl.List(pl.Float64),
            "covered": pl.List(pl.Boolean),
            "batch_hash": pl.String,
            "batch_name": pl.String,
            "simulation_name": pl.String,
        }
    )


@dataclass(frozen=True)
class CollectionBundle:
    collection_spec: dict[str, Any]
    pip_threshold_plot_data: pl.DataFrame
    causal_pip_plot_data: pl.DataFrame
    cs_component_plot_data: pl.DataFrame
    cs_truth_plot_data: pl.DataFrame


def read_collection_spec(collection_root: Path) -> dict[str, Any]:
    spec_path = collection_root / "collection_spec.yaml"
    return dict(yaml.safe_load(spec_path.read_text()) or {})


def load_collection_specs(alias_root: Path) -> dict[str, dict[str, Any]]:
    collections: dict[str, dict[str, Any]] = {}
    for spec_path in sorted(alias_root.glob("*/collection_spec.yaml")):
        collections[spec_path.parent.name] = dict(yaml.safe_load(spec_path.read_text()) or {})
    return collections


def load_collection_bundle(collection_root: Path) -> CollectionBundle:
    pip_threshold_plot_data = _load_plot_table(
        collection_root,
        "batches/*/fits/*/pip_threshold_plot_data.parquet",
        empty_pip_threshold_plot_data(),
    )
    causal_pip_plot_data = _load_plot_table(
        collection_root,
        "batches/*/fits/*/causal_pip_plot_data.parquet",
        empty_causal_pip_plot_data(),
    )
    cs_component_plot_data = _load_plot_table(
        collection_root,
        "batches/*/fits/*/cs_component_plot_data.parquet",
        empty_cs_component_plot_data(),
    )
    cs_truth_plot_data = _load_plot_table(
        collection_root,
        "batches/*/fits/*/cs_truth_plot_data.parquet",
        empty_cs_truth_plot_data(),
    )
    return CollectionBundle(
        collection_spec=read_collection_spec(collection_root),
        pip_threshold_plot_data=pip_threshold_plot_data,
        causal_pip_plot_data=causal_pip_plot_data,
        cs_component_plot_data=cs_component_plot_data,
        cs_truth_plot_data=cs_truth_plot_data,
    )


def _load_plot_table(collection_root: Path, relative_glob: str, empty_df: pl.DataFrame) -> pl.DataFrame:
    if not any(collection_root.glob(relative_glob)):
        return empty_df
    return add_plot_metadata_columns(
        pl.read_parquet(str(collection_root / relative_glob), glob=True)
    )


def available_method_families(plot_data: pl.DataFrame) -> list[str]:
    return sorted(plot_data.get_column("method_family").unique().to_list())


def available_L_values(plot_data: pl.DataFrame) -> list[int]:
    return sorted(int(value) for value in plot_data.get_column("L").unique().to_list())


def available_thresholds(plot_data: pl.DataFrame) -> list[float]:
    return sorted(float(v) for v in plot_data.get_column("threshold").drop_nulls().unique().to_list())


def selected_method_names(
    plot_data: pl.DataFrame, *, selected_method_families: list[str], selected_L: int
) -> set[str]:
    return set(
        plot_data.filter(
            pl.col("method_family").is_in(selected_method_families)
            & (pl.col("L") == selected_L)
        )
        .get_column("method")
        .unique()
        .to_list()
    )


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


def filter_thresholded_methods(
    plot_data: pl.DataFrame, selected_threshold: float
) -> pl.DataFrame:
    return plot_data.filter(
        ((pl.col("is_thresholded")) & (pl.col("threshold") == selected_threshold))
        | (~pl.col("is_thresholded"))
    )


def add_method_display_labels(
    plot_data: pl.DataFrame, selected_threshold: float
) -> pl.DataFrame:
    return plot_data.with_columns(
        pl.when(pl.col("is_thresholded"))
        .then(
            pl.format(
                "{} (@{})", pl.col("method_label_base"), pl.lit(f"{selected_threshold:g}")
            )
        )
        .otherwise(pl.col("method_display"))
        .alias("series_label")
    )


def filter_selected_methods(plot_data: pl.DataFrame, selected_methods: set[str]) -> pl.DataFrame:
    return plot_data.filter(pl.col("method").is_in(list(selected_methods)))


def summarize_pip_calibration(plot_data: pl.DataFrame) -> pl.DataFrame:
    calibration_df = plot_data.with_columns(
        (pl.col("pip_threshold") * 20).floor().clip(0, 19).cast(pl.Int64).alias("pip_bin_index")
    ).with_columns(
        (pl.col("pip_bin_index") * 0.05).alias("pip_left"),
        ((pl.col("pip_bin_index") + 1) * 0.05).alias("pip_right"),
        ((pl.col("pip_bin_index") + 0.5) * 0.05).alias("pip_mid"),
    )
    return (
        calibration_df.group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "method_family",
            "method_display",
            "series_label",
            "pip_bin_index",
            "pip_left",
            "pip_right",
            "pip_mid",
        )
        .agg(
            pl.col("n_exact").sum().alias("n_total"),
            pl.col("n_causal_exact").sum().alias("n_causal"),
        )
        .with_columns((pl.col("n_causal") / pl.col("n_total")).alias("empirical_rate"))
        .sort("simulation_name", "batch_hash", "replicate", "series_label", "pip_mid")
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
    calibration_summary: pl.DataFrame, *, facet_by_simulation: bool
):
    if calibration_summary.is_empty():
        return make_placeholder_chart("No PIP calibration data")

    theme = base_chart_theme()
    methods = sorted(m for m in calibration_summary.get_column("method_display").unique().to_list() if m is not None)
    n_cols = len(methods)

    method_color_lookup = {
        row["method_display"]: method_family_color_map().get(row["method_family"], "#888888")
        for row in calibration_summary.select("method_display", "method_family").unique().iter_rows(named=True)
    }

    if facet_by_simulation:
        simulations = sorted(x for x in calibration_summary.get_column("simulation_name").unique().to_list() if x is not None)
        n_rows = len(simulations)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(theme["width"] * n_cols, theme["height"] * n_rows),
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


def summarize_calibration_with_bootstrap(
    calibration_summary: pl.DataFrame,
    group_cols: list[str],
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    grouped = calibration_summary.group_by(
        *group_cols,
        "pip_bin_index",
        "pip_left",
        "pip_right",
        "pip_mid",
    ).agg(
        pl.col("n_total").alias("n_total_values"),
        pl.col("n_causal").alias("n_causal_values"),
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for row in grouped.iter_rows(named=True):
        totals = np.asarray(row["n_total_values"], dtype=float)
        causals = np.asarray(row["n_causal_values"], dtype=float)
        total_sum = totals.sum()
        empirical_rate = float(causals.sum() / total_sum)
        bootstrap_idx = rng.integers(0, len(totals), size=(n_bootstrap, len(totals)))
        bootstrap_totals = totals[bootstrap_idx].sum(axis=1)
        bootstrap_causals = causals[bootstrap_idx].sum(axis=1)
        bootstrap_rates = bootstrap_causals / bootstrap_totals
        record = {column: row[column] for column in group_cols}
        record.update(
            {
                "pip_bin_index": row["pip_bin_index"],
                "pip_left": row["pip_left"],
                "pip_right": row["pip_right"],
                "pip_mid": row["pip_mid"],
                "empirical_rate": empirical_rate,
                "ci_lower": float(np.quantile(bootstrap_rates, 0.025)),
                "ci_upper": float(np.quantile(bootstrap_rates, 0.975)),
            }
        )
        records.append(record)
    return pl.DataFrame(records).sort(*group_cols, "pip_mid")


def prepare_power_fdp_plot_data_frame(
    plot_data: pl.DataFrame,
    *,
    selected_threshold: float,
    selected_methods: set[str],
    show_background_threshold_traces: bool,
) -> pl.DataFrame:
    method_filtered = plot_data.filter(pl.col("method").is_in(list(selected_methods))).with_columns(
        (
            ~pl.col("is_thresholded") | (pl.col("threshold") == selected_threshold)
        ).alias("is_selected_threshold")
    )
    displayed = (
        method_filtered
        if show_background_threshold_traces
        else method_filtered.filter(pl.col("is_selected_threshold"))
    )
    return displayed.with_columns(
        pl.when(pl.col("is_thresholded"))
        .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
        .otherwise(pl.col("method_display"))
        .alias("trace_label"),
        pl.when(pl.col("is_selected_threshold"))
        .then(
            pl.when(pl.col("is_thresholded"))
            .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.lit(f"{selected_threshold:g}")))
            .otherwise(pl.col("method_display"))
        )
        .otherwise(None)
        .alias("legend_label"),
    )


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
):
    if power_fdp_summary.is_empty():
        return make_placeholder_chart("No power vs FDP data")
    visible = power_fdp_summary
    theme = base_chart_theme()

    if facet:
        simulations = sorted(x for x in visible.get_column("simulation_name").unique().to_list() if x is not None)
        n_cols = len(simulations)
        _legend_w = 2.0
        _fig_w = theme["width"] * n_cols + _legend_w
        _plot_frac = (theme["width"] * n_cols) / _fig_w
        fig, axes = plt.subplots(
            1, n_cols,
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
        plot_data.group_by("simulation_name", "method", "method_display", "threshold")
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
):
    if causal_pip_summary.is_empty():
        return make_placeholder_chart("No causal PIP data")
    theme = base_chart_theme()

    if facet:
        simulations = sorted(x for x in causal_pip_summary.get_column("simulation_name").unique().to_list() if x is not None)
        n_cols = len(simulations)
        _legend_w = 2.0
        _fig_w = theme["width"] * n_cols + _legend_w
        _plot_frac = (theme["width"] * n_cols) / _fig_w
        fig, axes = plt.subplots(
            1, n_cols,
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


def explode_cs_component_beta_rows(cs_component_plot_data: pl.DataFrame) -> pl.DataFrame:
    if cs_component_plot_data.is_empty():
        return cs_component_plot_data
    return (
        cs_component_plot_data.with_columns(
            pl.struct(["betas", "cs_sizes"]).map_elements(
                lambda row: [
                    {"beta": float(beta), "cs_size": int(cs_size)}
                    for beta, cs_size in zip(row["betas"], row["cs_sizes"], strict=True)
                ],
                return_dtype=pl.List(pl.Struct({"beta": pl.Float64, "cs_size": pl.Int64})),
            ).alias("beta_cs_pairs")
        )
        .explode("beta_cs_pairs")
        .unnest("beta_cs_pairs")
        .drop("betas", "cs_sizes", "ordered_pips")
    )


def explode_cs_truth_beta_rows(cs_truth_plot_data: pl.DataFrame) -> pl.DataFrame:
    if cs_truth_plot_data.is_empty():
        return cs_truth_plot_data
    return (
        cs_truth_plot_data.with_columns(
            pl.struct(["betas", "covered"]).map_elements(
                lambda row: [
                    {"beta": float(beta), "covered": bool(covered)}
                    for beta, covered in zip(row["betas"], row["covered"], strict=True)
                ],
                return_dtype=pl.List(pl.Struct({"beta": pl.Float64, "covered": pl.Boolean})),
            ).alias("beta_cover_pairs")
        )
        .drop("betas", "covered")
        .explode("beta_cover_pairs")
        .unnest("beta_cover_pairs")
    )


def _power_per_replicate(
    component_rows: pl.DataFrame,
    truth_rows: pl.DataFrame,
    *,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
    extra_group_cols: list[str],
) -> pl.DataFrame:
    """Power = fraction of causal vars covered by a qualifying CS, per replicate."""
    group_cols = ["simulation_name", "replicate", "method", "method_display", "threshold"] + extra_group_cols
    qualifying_keys = component_rows.filter(
        (pl.col("beta") == nominal_coverage)
        & (pl.col("cs_size") <= max_cs_size)
        & (pl.col("ser_log_bf") >= min_ser_log_bf)
    ).select("replicate", "method").unique()
    truth_at_nominal = truth_rows.filter(pl.col("beta") == nominal_coverage)
    total = truth_at_nominal.group_by(group_cols).agg(
        pl.col("causal_variable").n_unique().alias("n_causal")
    )
    detected = (
        truth_at_nominal
        .join(qualifying_keys, on=["replicate", "method"], how="inner")
        .filter(pl.col("covered"))
        .group_by(group_cols)
        .agg(pl.col("causal_variable").n_unique().alias("n_detected"))
    )
    return (
        total.join(detected, on=group_cols, how="left", join_nulls=True)
        .with_columns(
            (pl.col("n_detected").fill_null(0).cast(pl.Float64) / pl.col("n_causal")).alias("value")
        )
        .select(*group_cols, "value")
    )


def make_conditional_cs_summary(
    cs_component_plot_data: pl.DataFrame,
    cs_truth_plot_data: pl.DataFrame,
    *,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if cs_component_plot_data.is_empty() or cs_truth_plot_data.is_empty():
        empty = pl.DataFrame(
            schema={
                "simulation_name": pl.String,
                "method": pl.String,
                "method_display": pl.String,
                "threshold": pl.Float64,
                "metric": pl.String,
                "value": pl.Float64,
            }
        )
        return empty, empty

    component_rows = explode_cs_component_beta_rows(cs_component_plot_data)
    truth_rows = explode_cs_truth_beta_rows(cs_truth_plot_data)
    qualifying = component_rows.filter(
        (pl.col("beta") == nominal_coverage)
        & (pl.col("cs_size") <= max_cs_size)
        & (pl.col("ser_log_bf") >= min_ser_log_bf)
    )
    power_per_rep = _power_per_replicate(
        component_rows, truth_rows,
        nominal_coverage=nominal_coverage, max_cs_size=max_cs_size, min_ser_log_bf=min_ser_log_bf,
        extra_group_cols=[],
    )
    power_summary = (
        power_per_rep
        .group_by("simulation_name", "method", "method_display", "threshold")
        .agg(pl.col("value").mean())
        .with_columns(pl.lit("Power").alias("metric"))
    )
    size_summary = qualifying.group_by("simulation_name", "method", "method_display", "threshold").agg(
        pl.col("cs_size").mean().alias("value")
    ).with_columns(pl.lit("CS Size").alias("metric"))
    coverage_summary = truth_rows.filter(pl.col("beta") == nominal_coverage).group_by(
        "simulation_name", "method", "method_display", "threshold"
    ).agg(
        pl.col("covered").cast(pl.Float64).mean().alias("value")
    ).with_columns(pl.lit("Coverage").alias("metric"))
    by_sim = pl.concat([power_summary, size_summary, coverage_summary], how="diagonal_relaxed")
    aggregate = by_sim.group_by("method", "method_display", "threshold", "metric").agg(
        pl.col("value").mean().alias("value")
    ).with_columns(pl.lit("Aggregate").alias("simulation_name"))
    return (
        aggregate.sort("metric", "method", "threshold"),
        by_sim.sort("simulation_name", "metric", "method", "threshold"),
    )


def _plot_cs_summary_row(ax_row, sim_df, metrics, x_min, x_max, x_margin, nominal_coverage: float | None = None):
    for idx, metric_name in enumerate(metrics):
        ax = ax_row[idx]
        metric_df = sim_df.filter(pl.col("metric") == metric_name)
        for method_name in sorted(x for x in metric_df.get_column("method").unique().to_list() if x is not None):
            method_df = metric_df.filter(pl.col("method") == method_name)
            color = method_color(method_name)
            display_label = method_df["method_display"][0] if "method_display" in method_df.columns else method_name
            if method_df["threshold"].is_null().all():
                y_val = float(method_df["value"].mean())
                ax.axhline(y=y_val, color=color, linestyle="--", linewidth=1.5, label=display_label)
            else:
                method_df = method_df.sort("threshold")
                ax.plot(
                    method_df["threshold"].drop_nulls().to_numpy(),
                    method_df["value"].to_numpy(),
                    marker="o", color=color, label=display_label,
                )
        if metric_name == "Coverage" and nominal_coverage is not None:
            ax.axhline(y=nominal_coverage, color="black", linestyle="--", linewidth=1.2,
                       label=f"nominal ({nominal_coverage:.2f})")
        ax.set_title(metric_name)
        ax.set_xlabel("Threshold")
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax_row[0].set_ylabel("Value")


def render_conditional_cs_summary_chart(
    cs_summary: pl.DataFrame,
    *,
    facet: bool,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
):
    if cs_summary.is_empty():
        return make_placeholder_chart("No credible set summary data")
    theme = base_chart_theme()
    observed_thresholds = cs_summary["threshold"].drop_nulls().unique().to_list()
    x_min = min(observed_thresholds) if observed_thresholds else 0.0
    x_max = max(observed_thresholds) if observed_thresholds else 1.0
    x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 0.5
    metrics = ["Power", "CS Size", "Coverage"]

    if facet:
        simulations = sorted(x for x in cs_summary["simulation_name"].unique().to_list() if x is not None)
        n_sims = len(simulations)
        fig, axes = plt.subplots(
            n_sims, len(metrics),
            figsize=(theme["width"] * 3, theme["height"] * n_sims),
            squeeze=False,
        )
        for row_idx, sim_name in enumerate(simulations):
            sim_df = cs_summary.filter(pl.col("simulation_name") == sim_name)
            _plot_cs_summary_row(axes[row_idx], sim_df, metrics, x_min, x_max, x_margin, nominal_coverage=nominal_coverage)
            axes[row_idx, 0].set_ylabel(f"{sim_name}\nValue")
        fig.suptitle("By simulation scenario")
    else:
        fig, axes = plt.subplots(1, len(metrics), figsize=(theme["width"] * 3, theme["height"]), squeeze=False)
        _plot_cs_summary_row(axes[0], cs_summary, metrics, x_min, x_max, x_margin, nominal_coverage=nominal_coverage)
        fig.suptitle(
            f"Aggregate | nominal={nominal_coverage:.2f} | max size={max_cs_size} | min log BF={min_ser_log_bf:.1f}"
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
    fig.tight_layout()
    return fig


def make_conditional_cs_replicate_summary(
    cs_component_plot_data: pl.DataFrame,
    cs_truth_plot_data: pl.DataFrame,
    *,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    if cs_component_plot_data.is_empty() or cs_truth_plot_data.is_empty():
        return pl.DataFrame(
            schema={
                "simulation_name": pl.String,
                "replicate": pl.Int64,
                "method": pl.String,
                "method_display": pl.String,
                "threshold": pl.Float64,
                "is_thresholded": pl.Boolean,
                "metric": pl.String,
                "value": pl.Float64,
            }
        )
    component_rows = explode_cs_component_beta_rows(cs_component_plot_data)
    truth_rows = explode_cs_truth_beta_rows(cs_truth_plot_data)
    qualifying = component_rows.filter(
        (pl.col("beta") == nominal_coverage)
        & (pl.col("cs_size") <= max_cs_size)
        & (pl.col("ser_log_bf") >= min_ser_log_bf)
    )
    power_summary = _power_per_replicate(
        component_rows, truth_rows,
        nominal_coverage=nominal_coverage, max_cs_size=max_cs_size, min_ser_log_bf=min_ser_log_bf,
        extra_group_cols=["is_thresholded"],
    ).with_columns(pl.lit("Power").alias("metric"))
    size_summary = qualifying.group_by(
        "simulation_name", "replicate", "method", "method_display", "threshold", "is_thresholded"
    ).agg(pl.col("cs_size").mean().alias("value")).with_columns(pl.lit("CS Size").alias("metric"))
    coverage_summary = truth_rows.filter(pl.col("beta") == nominal_coverage).group_by(
        "simulation_name", "replicate", "method", "method_display", "threshold", "is_thresholded"
    ).agg(pl.col("covered").cast(pl.Float64).mean().alias("value")).with_columns(
        pl.lit("Coverage").alias("metric")
    )
    return pl.concat([power_summary, size_summary, coverage_summary], how="diagonal_relaxed").sort(
        "simulation_name", "replicate", "metric", "method", "threshold"
    )


def summarize_replicate_metric_with_bootstrap(
    replicate_metric_summary: pl.DataFrame,
    group_cols: list[str],
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    if replicate_metric_summary.is_empty():
        return pl.DataFrame(
            schema={
                **{column: replicate_metric_summary.schema[column] for column in group_cols},
                "value": pl.Float64,
                "ci_lower": pl.Float64,
                "ci_upper": pl.Float64,
            }
        )
    grouped = replicate_metric_summary.group_by(*group_cols).agg(pl.col("value").alias("value_samples"))
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for row in grouped.iter_rows(named=True):
        values = np.asarray(row["value_samples"], dtype=float)
        mean_value = float(np.mean(values)) if len(values) else np.nan
        ci_lower = mean_value
        ci_upper = mean_value
        if len(values):
            bootstrap_idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            bootstrap_means = values[bootstrap_idx].mean(axis=1)
            ci_lower = float(np.quantile(bootstrap_means, 0.025))
            ci_upper = float(np.quantile(bootstrap_means, 0.975))
        record = {column: row[column] for column in group_cols}
        record.update({"value": mean_value, "ci_lower": ci_lower, "ci_upper": ci_upper})
        records.append(record)
    return pl.DataFrame(records).sort(*group_cols)


def select_current_threshold_cs_rows(
    plot_data: pl.DataFrame, *, selected_threshold: float
) -> pl.DataFrame:
    return plot_data.filter(
        ((pl.col("is_thresholded")) & (pl.col("threshold") == selected_threshold))
        | (~pl.col("is_thresholded"))
    )


def prepare_cs_histogram_data(
    cs_component_plot_data: pl.DataFrame,
    *,
    nominal_coverage: float,
    selected_threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if cs_component_plot_data.is_empty():
        return (
            pl.DataFrame(schema={"method": pl.String, "method_display": pl.String, "simulation_name": pl.String, "threshold": pl.Float64, "cs_size": pl.Int64}),
            pl.DataFrame(schema={"method": pl.String, "method_display": pl.String, "simulation_name": pl.String, "threshold": pl.Float64, "ser_log_bf": pl.Float64}),
        )
    component_rows = explode_cs_component_beta_rows(cs_component_plot_data)
    selected_rows = filter_thresholded_methods(component_rows, selected_threshold=selected_threshold)
    size_df = selected_rows.filter(pl.col("beta") == nominal_coverage).select(
        "method", "method_display", "simulation_name", "threshold", "cs_size"
    )
    bf_df = selected_rows.select("method", "method_display", "simulation_name", "threshold", "ser_log_bf")
    return size_df, bf_df


def render_conditional_cs_scenario_points_chart(
    scenario_summary: pl.DataFrame,
    *,
    selected_threshold: float,
):
    if scenario_summary.is_empty():
        return make_placeholder_chart("No scenario-point data")
    theme = base_chart_theme()
    metrics = ["Power", "CS Size", "Coverage"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(theme["width"] * 5, theme["height"] * 1.35), squeeze=False)
    sim_names = sorted(x for x in scenario_summary.get_column("simulation_name").unique().to_list() if x is not None)
    scenario_spacing = 3.0
    x_positions = np.arange(len(sim_names)) * scenario_spacing
    sim_index = {name: i * scenario_spacing for i, name in enumerate(sim_names)}
    all_methods = sorted(x for x in scenario_summary.get_column("method").unique().to_list() if x is not None)
    n_methods = len(all_methods)
    jitter_step = 0.2
    method_offset = {m: (i - (n_methods - 1) / 2) * jitter_step for i, m in enumerate(all_methods)}
    for idx, metric_name in enumerate(metrics):
        ax = axes[0, idx]
        metric_df = scenario_summary.filter(pl.col("metric") == metric_name)
        for method_name in all_methods:
            method_df = metric_df.filter(pl.col("method") == method_name).sort("simulation_name")
            if method_df.is_empty():
                continue
            x = np.array([sim_index[s] for s in method_df["simulation_name"].to_list()]) + method_offset[method_name]
            y = method_df["value"].to_numpy()
            yerr = np.vstack(
                [
                    np.clip(y - method_df["ci_lower"].to_numpy(), a_min=0.0, a_max=None),
                    np.clip(method_df["ci_upper"].to_numpy() - y, a_min=0.0, a_max=None),
                ]
            )
            display_label = method_df["method_display"][0] if "method_display" in method_df.columns else method_name
            ax.errorbar(x, y, yerr=yerr, fmt="o", color=method_color(method_name), label=display_label)
        ax.set_title(metric_name)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(sim_names, rotation=45, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("Value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
    fig.tight_layout()
    return fig


def _plot_split_histogram(
    ax: "plt.Axes",
    values: "np.ndarray",
    *,
    threshold: float,
    good_side: str,
    color: str,
    n_bins: int = 20,
) -> None:
    if values.size == 0:
        return
    bins = np.histogram_bin_edges(values, bins=n_bins)
    good = values[values <= threshold] if good_side == "left" else values[values >= threshold]
    bad = values[values > threshold] if good_side == "left" else values[values < threshold]
    if good.size > 0:
        ax.hist(good, bins=bins, histtype="stepfilled", color=color, alpha=0.6)
    if bad.size > 0:
        ax.hist(bad, bins=bins, histtype="step", color=color)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.0)


def render_cs_histograms(
    cs_size_histogram_data: pl.DataFrame,
    ser_log_bf_histogram_data: pl.DataFrame,
    *,
    selected_threshold: float,
    max_cs_size: int,
    min_ser_log_bf: float,
):
    if cs_size_histogram_data.is_empty() and ser_log_bf_histogram_data.is_empty():
        return make_placeholder_chart("No histogram data")
    panel_size = base_chart_theme()["width"] * 0.75
    methods = sorted(
        set(cs_size_histogram_data.get_column("method").unique().to_list())
        | set(ser_log_bf_histogram_data.get_column("method").unique().to_list())
    )
    n_methods = len(methods)
    fig, axes = plt.subplots(
        2, n_methods,
        figsize=(panel_size * n_methods, panel_size * 2),
        squeeze=False,
    )
    for col_idx, method_name in enumerate(methods):
        color = method_color(method_name)
        cs_rows = cs_size_histogram_data.filter(pl.col("method") == method_name)
        bf_rows = ser_log_bf_histogram_data.filter(pl.col("method") == method_name)
        display = (
            cs_rows["method_display"][0] if not cs_rows.is_empty() and "method_display" in cs_rows.columns
            else bf_rows["method_display"][0] if not bf_rows.is_empty() and "method_display" in bf_rows.columns
            else method_name
        )
        ax_size = axes[0, col_idx]
        if not cs_rows.is_empty():
            _plot_split_histogram(
                ax_size, cs_rows["cs_size"].to_numpy().astype(float),
                threshold=float(max_cs_size), good_side="left", color=color,
            )
        ax_size.set_title(display, fontsize=8)
        if col_idx == 0:
            ax_size.set_ylabel("CS Size", fontsize=8)

        ax_bf = axes[1, col_idx]
        if not bf_rows.is_empty():
            _plot_split_histogram(
                ax_bf, bf_rows["ser_log_bf"].to_numpy(),
                threshold=min_ser_log_bf, good_side="right", color=color,
            )
        if col_idx == 0:
            ax_bf.set_ylabel("SER log BF", fontsize=8)

    fig.tight_layout()
    return fig


def make_cs_beta_trace_summary(
    cs_beta_trace: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_threshold: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Summarize cs_beta_trace across all betas for each (collection_name, method, threshold, beta)."""
    if cs_beta_trace.is_empty():
        return pl.DataFrame(schema={
            "collection_name": pl.String,
            "method": pl.String,
            "method_display": pl.String,
            "threshold": pl.Float64,
            "is_thresholded": pl.Boolean,
            "is_selected_threshold": pl.Boolean,
            "beta": pl.Float64,
            "power": pl.Float64,
            "coverage": pl.Float64,
            "cs_size": pl.Float64,
        })
    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    filtered = (
        cs_beta_trace
        .filter(pl.col("method").is_in(list(selected_methods)))
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


def render_cs_beta_trace_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
    selected_threshold: float,
) -> "plt.Figure":
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

    metrics = [("power", "Power"), ("cs_size", "CS Size"), ("coverage", "Coverage")]
    theme = base_chart_theme()
    n_rows = len(collection_names)
    fig, axes = plt.subplots(
        n_rows, len(metrics),
        figsize=(theme["width"] * len(metrics), theme["height"] * n_rows),
        squeeze=False,
    )

    for row_idx, coll_name in enumerate(collection_names):
        coll_df = summary.filter(pl.col("collection_name") == coll_name)
        for col_idx, (metric_col, metric_title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            trace_labels = (
                coll_df.select("method", "threshold", "method_display", "is_thresholded", "is_selected_threshold")
                .unique()
                .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
            )
            for trow in trace_labels.iter_rows(named=True):
                thresh_filter = (
                    pl.col("threshold").is_null() if trow["threshold"] is None
                    else (pl.col("threshold") == trow["threshold"])
                )
                trace_df = (
                    coll_df.filter(
                        (pl.col("method") == trow["method"]) & thresh_filter
                    ).sort("beta")
                )
                if trace_df.is_empty():
                    continue
                is_selected = bool(trow["is_selected_threshold"])
                color = method_color(trow["method"])
                if trow["is_thresholded"]:
                    label = f"{trow['method_display']} (@{trow['threshold']:g})" if is_selected else "_nolegend_"
                else:
                    label = trow["method_display"] if is_selected else "_nolegend_"
                ax.plot(
                    trace_df["beta"].to_numpy(),
                    trace_df[metric_col].to_numpy(),
                    color=color,
                    linewidth=2.0 if is_selected else 0.8,
                    alpha=1.0 if is_selected else 0.15,
                    label=label,
                )
            if metric_col == "coverage":
                betas = coll_df["beta"].unique().sort().to_numpy()
                if len(betas):
                    ax.plot(betas, betas, color="black", linestyle="--", linewidth=1.0, label="y=x (ideal)")
            if row_idx == 0:
                ax.set_title(metric_title)
            ax.set_xlabel("Nominal coverage (β)")
            if col_idx == 0:
                ax.set_ylabel(coll_name, fontsize=9, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
