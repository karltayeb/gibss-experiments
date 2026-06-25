from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


_NOTHRESH_LINESTYLES = ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


def method_family_label_map() -> dict[str, str]:
    return {
        "logistic_threshold":    "Logistic",
        "cox":                   "Cox",
        "twogroup":              "Twogroup",
        "twogroup_oracle":       "Twogroup",
        "twogroup_oracle_global": "Twogroup Global EM",
        "logistic_oracle":       "Logistic",
        "cox_reversed":          "Cox (reversed)",
        "cox_reversed_censored": "Cox reversed (censored)",
        "cox_uncensored":        "Cox (uncensored)",
        "twogroup_oracle_init":  "TG Oracle Init",
        "twogroup_scale_fam":    "Twogroup Scale",
        "twogroup_loc_fam":      "Twogroup Loc",
        "linear_fixed":          "Linear",
        "linear_estimated":      "Linear (est. var)",
        "depletion":             "Depletion",
        "enrichment":            "Enrichment",
        "globaljj":              "Global JJ",
        "localjj":               "Local JJ",
        "quadrature":            "Quadrature",
        "irls":                  "IRLS (Laplace)",
        "profile_cheb":          "Profile (Cheb)",
        "score":                 "Score (b0=0)",
        "score_null_intercept":  "Score (null b0)",
    }


def method_family_oracle_label_map() -> dict[str, str]:
    return {}


def method_family_color_map() -> dict[str, str]:
    # Okabe-Ito colorblind-safe palette + reddish variants for twogroup family
    return {
        "logistic_threshold":    "#0072B2",
        "logistic_oracle":       "#56B4E9",
        "cox":                   "#009E73",
        "cox_reversed":          "#E69F00",
        "cox_reversed_censored": "#E69F00",  # share cox_reversed color
        "cox_uncensored":        "#009E73",  # share cox color
        "twogroup":              "#D55E00",  # vermillion
        "twogroup_oracle":       "#CC79A7",  # rose/mauve
        "twogroup_oracle_global": "#AA4499",  # purple
        "twogroup_oracle_init":  "#994F00",  # dark burnt orange
        "twogroup_scale_fam":    "#FF6347",  # tomato
        "twogroup_loc_fam":      "#C0392B",  # crimson
        "linear_fixed":          "#8B4513",  # saddle brown
        "linear_estimated":      "#A0522D",  # sienna
        "depletion":             "#0072B2",
        "enrichment":            "#D55E00",
        "globaljj":              "#0072B2",  # blue
        "localjj":               "#E69F00",  # orange
        "quadrature":            "#009E73",  # green (reference)
        "irls":                  "#44AA99",  # teal
        "profile_cheb":          "#CC79A7",  # rose
        "score":                 "#999999",  # grey  (crude b0=0 baseline)
        "score_null_intercept":  "#882255",  # wine  (distinct from the grey baseline)
    }


def method_color(method: str) -> str:
    # v2 method names are "<family>__key=val__...,"; family is the part before "__".
    # (Legacy v1 names "<family>_L1" have no "__", so split is a no-op there and we
    # fall back to the historical "_L" strip.)
    family = method.split("__", 1)[0] if "__" in method else method.rsplit("_L", 1)[0]
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


_N_FINE_BINS = 200
_N_COARSE_BINS = 20
_FINE_PER_COARSE = _N_FINE_BINS // _N_COARSE_BINS  # 10


def expand_pip_calibration_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Expand pip_plot_data to 20 coarse-bin rows for render_pip_calibration. Aggregates 200 fine bins (width 0.005) into 20 coarse bins (width 0.05)."""
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
        for j in range(_N_COARSE_BINS):
            start = j * _FINE_PER_COARSE
            stop = start + _FINE_PER_COARSE
            rows.append({
                "collection_name": row.get("collection_name", ""),
                "method": row["method"],
                "threshold": row["threshold"],
                "pip_bin_index": j,
                "pip_left": j * 0.05,
                "pip_right": (j + 1) * 0.05,
                "pip_mid": (j + 0.5) * 0.05,
                "n_total": sum(counts[start:stop]),
                "n_causal": sum(causal_counts[start:stop]),
            })
    expanded = pl.from_dicts(rows, schema={
        "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "pip_bin_index": pl.Int64, "pip_left": pl.Float64, "pip_right": pl.Float64,
        "pip_mid": pl.Float64, "n_total": pl.Int64, "n_causal": pl.Int64,
    })
    return (
        expanded
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .filter(
            ~pl.col("is_thresholded")
            | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
        )
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


_N_PIP_BINS = 200
_PIP_BIN_WIDTH = 1.0 / _N_PIP_BINS
_PIP_THRESHOLD_GRID = np.arange(_N_PIP_BINS) * _PIP_BIN_WIDTH  # [0.000, 0.005, ..., 0.995]


def _bins_to_power_fdp(
    counts: np.ndarray,
    causal_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rev_cum_counts = np.cumsum(counts[::-1])[::-1]
    rev_cum_causal = np.cumsum(causal_counts[::-1])[::-1]
    total_causal = int(causal_counts.sum())
    power = rev_cum_causal / max(total_causal, 1)
    fdp = (rev_cum_counts - rev_cum_causal) / np.maximum(rev_cum_counts, 1)
    return power.astype(float), fdp.astype(float)


def expand_power_fdp_from_compact(
    pip_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
    aggregate_across_collections: bool = False,
) -> pl.DataFrame:
    """Derive per-threshold power/FDP rows from 200-bin arrays.

    Bins are summed per (collection_name, method, threshold) across replicates,
    then power/FDP are computed via reverse cumulative sums (200 threshold points).
    When aggregate_across_collections=True, bins are summed over all collections
    before computing — correct for aggregate plots.
    """
    empty = pl.DataFrame(schema={
        "simulation_name": pl.String, "method": pl.String, "method_display": pl.String,
        "trace_label": pl.String, "legend_label": pl.String,
        "is_selected_threshold": pl.Boolean,
        "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
    })
    if pip_plot_data.is_empty():
        return empty

    meta = method_metadata.select(
        "method", "threshold", "method_display", "method_label_base", "is_thresholded",
    )

    filtered = (
        pip_plot_data
        .filter(pl.col("method").is_in(list(selected_methods)))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
    )
    if filtered.is_empty():
        return empty

    # Accumulate bin arrays element-wise in Python.
    # (Polars list.sum() in agg() sums within each row → scalar, not element-wise across rows.)
    bin_acc: dict = {}
    for row in filtered.iter_rows(named=True):
        col_name = "" if aggregate_across_collections else row.get("collection_name", "")
        key = (col_name, row["method"], row["threshold"])
        if key not in bin_acc:
            bin_acc[key] = {
                "collection_name": col_name,
                "method": row["method"],
                "threshold": row["threshold"],
                "method_display": row["method_display"],
                "method_label_base": row["method_label_base"],
                "is_thresholded": row["is_thresholded"],
                "is_selected_threshold": row["is_selected_threshold"],
                "counts": np.zeros(_N_PIP_BINS),
                "causal": np.zeros(_N_PIP_BINS),
            }
        bin_acc[key]["counts"] += np.asarray(row["pip_bin_counts"], dtype=float)
        bin_acc[key]["causal"] += np.asarray(row["pip_bin_causal_counts"], dtype=float)

    rows = []
    for acc in bin_acc.values():
        power_arr, fdp_arr = _bins_to_power_fdp(acc["counts"], acc["causal"])
        trace_label = (
            f"{acc['method_label_base']} (@{acc['threshold']})"
            if acc["is_thresholded"] else acc["method_display"]
        )
        for k in range(_N_PIP_BINS):
            rows.append({
                "simulation_name": acc["collection_name"],
                "method": acc["method"],
                "threshold": acc["threshold"],
                "method_display": acc["method_display"],
                "is_thresholded": acc["is_thresholded"],
                "is_selected_threshold": acc["is_selected_threshold"],
                "trace_label": trace_label,
                "legend_label": acc["method_display"] if acc["is_selected_threshold"] else None,
                "pip_threshold": float(_PIP_THRESHOLD_GRID[k]),
                "power": float(power_arr[k]),
                "fdp": float(fdp_arr[k]),
            })

    return pl.from_dicts(rows, schema={
        "simulation_name": pl.String, "method": pl.String, "threshold": pl.Float64,
        "method_display": pl.String, "is_thresholded": pl.Boolean,
        "is_selected_threshold": pl.Boolean,
        "trace_label": pl.String, "legend_label": pl.String,
        "pip_threshold": pl.Float64, "power": pl.Float64, "fdp": pl.Float64,
    }).sort("simulation_name", "method_display", "pip_threshold")


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



_PIP_MARKER_THRESHOLDS = [0.5, 0.9]
_PIP_MARKER_STYLES = ["D", "s"]  # diamond, square


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
            for thresh, mstyle in zip(_PIP_MARKER_THRESHOLDS, _PIP_MARKER_STYLES):
                idx = int(thresh * _N_PIP_BINS)  # pip_threshold[k] = k * 0.005
                if 0 <= idx < len(pip_arr):
                    mlabel = f"PIP={thresh:g}" if thresh not in marker_legend_added else "_nolegend_"
                    ax.scatter(
                        fdp_arr[idx],
                        pwr_arr[idx],
                        color=color,
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


def _ordered_families(panel_df: pl.DataFrame, method_order: list[str] | None) -> list[str]:
    """Unique method families (name before "__") present in panel_df.

    Thresholded coordinates of one family (e.g. cox__threshold=*) collapse to a
    single family so they render as one line / one legend entry. Order follows
    ``method_order`` when given, else sorted.
    """
    def _family(name: str) -> str:
        return name.split("__", 1)[0]

    present = [m for m in panel_df.get_column("method").unique().to_list() if m is not None]
    if method_order is not None:
        present_set = set(present)
        ordered = [m for m in method_order if m in present_set]
    else:
        ordered = sorted(present)
    families: list[str] = []
    for m in ordered:
        fam = _family(m)
        if fam not in families:
            families.append(fam)
    return families


def _plot_causal_pip_on_ax(
    ax: "plt.Axes",
    panel_df: pl.DataFrame,
    *,
    title: str,
    method_order: list[str] | None = None,
) -> None:
    families = _ordered_families(panel_df, method_order)
    _nothresh_idx = 0
    for family in families:
        family_df = panel_df.filter(pl.col("method").str.split("__").list.first() == family)
        color = method_color(family_df["method"][0])
        label = family_df["method_display_base"][0]
        if family_df["threshold"].is_null().all():
            y_val = float(family_df["mean_causal_pip"].mean())
            ls = _NOTHRESH_LINESTYLES[_nothresh_idx % len(_NOTHRESH_LINESTYLES)]
            ax.axhline(y=y_val, color=color, linestyle=ls, linewidth=1.5, label=label)
            _nothresh_idx += 1
        else:
            curve = (
                family_df.drop_nulls("threshold")
                .group_by("threshold").agg(pl.col("mean_causal_pip").mean())
                .sort("threshold")
            )
            ax.plot(
                curve["threshold"].to_numpy(),
                curve["mean_causal_pip"].to_numpy(),
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


def make_preceding_mass_ecdf_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Expand mass_above_causal for empirical CDF plotting. No validity filters applied."""
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean, "is_selected_threshold": pl.Boolean,
        "mass_above_causal": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)
    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    return (
        cs_plot_data
        .filter(pl.col("method").is_in(list(selected_methods)))
        .filter(pl.col("rank_of_causal").list.len() > 0)
        .explode("mass_above_causal", "causal_indices")
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
        .select("collection_name", "method", "method_display", "threshold",
                "is_thresholded", "is_selected_threshold", "mass_above_causal")
    )


def render_preceding_mass_ecdf_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
) -> "plt.Figure":
    """Empirical CDF of mass_above_causal per method. x=mass above causal, y=fraction covered."""
    if summary.is_empty():
        return make_placeholder_chart("No causal resolution data")

    theme = base_chart_theme()
    all_colls = list(collection_names) + ["All"]
    n_panels = len(all_colls)
    _legend_w = 2.5
    _plot_w = theme["width"] * n_panels
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w

    method_order = (
        summary.select("method_display", "is_thresholded").unique()
        .sort(["is_thresholded", "method_display"])["method_display"].to_list()
    )

    fig, axes = plt.subplots(1, n_panels, figsize=(_fig_w, theme["height"]), squeeze=False)
    axes = axes[0]

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    for panel_idx, coll in enumerate(all_colls):
        ax = axes[panel_idx]
        if panel_idx == n_panels - 1:
            ax.set_facecolor("#ddeeff")
        coll_df = summary if coll == "All" else summary.filter(pl.col("collection_name") == coll)

        for trow in (
            summary.select("method", "method_display", "threshold", "is_thresholded")
            .unique().sort(["is_thresholded", "method_display"]).iter_rows(named=True)
        ):
            thresh_filter = (
                pl.col("threshold").is_null() if trow["threshold"] is None
                else pl.col("threshold") == trow["threshold"]
            )
            m_df = coll_df.filter((pl.col("method") == trow["method"]) & thresh_filter)
            if m_df.is_empty():
                continue
            vals = m_df["mass_above_causal"].sort().to_numpy()
            y = np.arange(1, len(vals) + 1) / len(vals)
            color = method_color(trow["method"])
            label = trow["method_display"]
            (line,) = ax.plot(vals, y, color=color, linewidth=1.2)
            if label not in seen_labels:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mass above causal")
        ax.set_title(coll, fontsize=8, fontweight="bold")
        if panel_idx == 0:
            ax.set_ylabel("coverage")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.95),
        )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
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
    families = _ordered_families(panel_df, method_order)
    _nothresh_idx = 0
    for family in families:
        family_df = panel_df.filter(pl.col("method").str.split("__").list.first() == family)
        color = method_color(family_df["method"][0])
        label = family_df["method_display_base"][0]
        if family_df["threshold"].is_null().all():
            y_val = float(family_df["mean_mass_above_causal"].mean())
            ls = _NOTHRESH_LINESTYLES[_nothresh_idx % len(_NOTHRESH_LINESTYLES)]
            ax.axhline(y=y_val, color=color, linestyle=ls, linewidth=1.5, label=label)
            _nothresh_idx += 1
        else:
            curve = (
                family_df.drop_nulls("threshold")
                .group_by("threshold").agg(pl.col("mean_mass_above_causal").mean())
                .sort("threshold")
            )
            ax.plot(
                curve["threshold"].to_numpy(),
                curve["mean_mass_above_causal"].to_numpy(),
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
        n_features = row["n_features"]
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
                "cs_size_frac": cs_size / n_features,
                "covered": n_covered > 0,
                "power": float(n_covered / n_causal),
                "ser_log_bf": row["ser_log_bf"],
            })
    if not rows:
        return pl.DataFrame(schema={
            "collection_name": pl.String, "sample_id": pl.String,
            "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
            "beta": pl.Float64, "cs_size": pl.Int64, "cs_size_frac": pl.Float64,
            "covered": pl.Boolean, "power": pl.Float64, "ser_log_bf": pl.Float64,
        })
    return pl.from_dicts(rows, schema={
        "collection_name": pl.String, "sample_id": pl.String,
        "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
        "beta": pl.Float64, "cs_size": pl.Int64, "cs_size_frac": pl.Float64,
        "covered": pl.Boolean, "power": pl.Float64, "ser_log_bf": pl.Float64,
    })


def _compute_causal_power_rows(
    cs_plot_data: pl.DataFrame,
    *,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Expand to one row per (sample_id, l, causal_idx, beta) with valid_covered flag.

    valid_covered = this specific causal's rank < cs_size AND cs_size <= max AND ser_log_bf >= min.
    Rows with empty causal_indices are skipped (no discovery target).
    """
    from utils import CS_BETA_GRID

    schema = {
        "collection_name": pl.String, "sample_id": pl.String,
        "method": pl.String, "threshold": pl.Float64, "l": pl.Int64,
        "causal_idx": pl.Int64, "beta": pl.Float64, "valid_covered": pl.Boolean,
        "cs_causal_radius": pl.Float64,
    }
    rows: list[dict] = []
    for row in cs_plot_data.iter_rows(named=True):
        sizes = row["cs_sizes"]
        radii = row.get("cs_causal_radius")
        ser_valid = row["ser_log_bf"] >= min_ser_log_bf
        for causal_pos, (causal_rank, causal_idx) in enumerate(zip(row["rank_of_causal"], row["causal_indices"])):
            radius_by_beta = radii[causal_pos] if radii is not None else [None] * len(sizes)
            for beta, cs_size, radius in zip(CS_BETA_GRID.tolist(), sizes, radius_by_beta):
                valid_covered = ser_valid and (cs_size <= max_cs_size) and (causal_rank < cs_size)
                rows.append({
                    "collection_name": row["collection_name"],
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "threshold": row["threshold"],
                    "l": row["l"],
                    "causal_idx": int(causal_idx),
                    "beta": float(beta),
                    "valid_covered": valid_covered,
                    "cs_causal_radius": float(radius) if valid_covered and radius is not None else None,
                })
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(rows, schema=schema)


def make_cs_coverage_size_curves(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Raw coverage vs CS size curves, no BF or size filtering.

    Returns one row per (collection_name, method, method_display, threshold,
    is_thresholded, beta) with mean empirical coverage and mean CS size across
    all CS instances at that beta.
    """
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean, "beta": pl.Float64,
        "coverage": pl.Float64, "cs_size": pl.Float64, "cs_size_frac": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    thresh_mask = (
        ~pl.col("is_thresholded")
        | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
    )
    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    valid_pairs = (
        meta
        .filter(pl.col("method").is_in(list(selected_methods)))
        .filter(thresh_mask)
        .select("method", "threshold")
    )
    filtered = cs_plot_data.join(valid_pairs, on=["method", "threshold"], how="inner", nulls_equal=True)
    expanded = _expand_cs_to_beta_rows(filtered)
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)

    return (
        expanded
        .group_by("collection_name", "method", "threshold", "beta")
        .agg(
            pl.col("covered").cast(pl.Float64).mean().alias("coverage"),
            pl.col("cs_size").cast(pl.Float64).mean().alias("cs_size"),
            pl.col("cs_size_frac").mean().alias("cs_size_frac"),
        )
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .sort("collection_name", "method_display", "threshold", "beta")
    )


def make_cs_power_size_coverage_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
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

    cs_filtered = cs_plot_data.filter(pl.col("method").is_in(list(selected_methods)))
    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")

    def _add_threshold_meta(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
            .with_columns(
                (
                    ~pl.col("is_thresholded")
                    | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
                ).alias("is_selected_threshold")
            )
            .filter(pl.col("is_selected_threshold"))
        )

    # --- coverage and cs_size: CS-level metrics (unchanged) ---
    expanded = _expand_cs_to_beta_rows(cs_filtered)
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)
    cov_cs = (
        _add_threshold_meta(
            expanded.with_columns(
                ((pl.col("cs_size") <= max_cs_size) & (pl.col("ser_log_bf") >= min_ser_log_bf)).alias("valid_cs")
            )
        )
        .group_by("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(
            pl.when(pl.col("valid_cs")).then(pl.col("covered").cast(pl.Float64)).mean().alias("coverage"),
            pl.when(pl.col("valid_cs")).then(pl.col("cs_size").cast(pl.Float64)).mean().alias("cs_size"),
        )
    )

    # --- power: per-causal metric ---
    # denominator = total (sample_id, causal_idx) pairs; a causal counts once even if found by multiple CSs
    causal_rows = _compute_causal_power_rows(cs_filtered, max_cs_size=max_cs_size, min_ser_log_bf=min_ser_log_bf)
    if causal_rows.is_empty():
        return pl.DataFrame(schema=empty_schema)
    per_causal_sample = (
        _add_threshold_meta(causal_rows)
        .group_by("collection_name", "sample_id", "causal_idx", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(pl.col("valid_covered").any().alias("discovered"))
    )
    power_df = (
        per_causal_sample
        .group_by("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(pl.col("discovered").cast(pl.Float64).mean().alias("power"))
    )

    return (
        cov_cs.join(power_df, on=["collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta"], how="left", nulls_equal=True)
        .sort("collection_name", "method_display", "threshold", "beta")
    )


make_cs_beta_trace_summary = make_cs_power_size_coverage_summary


def make_cs_radius_power_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Summarize power and mean causal radius across covered causal targets."""
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean,
        "is_selected_threshold": pl.Boolean, "beta": pl.Float64,
        "power": pl.Float64, "coverage": pl.Float64, "cs_causal_radius": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    causal_rows = _compute_causal_power_rows(
        cs_plot_data.filter(pl.col("method").is_in(list(selected_methods))),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_ser_log_bf,
    )
    if causal_rows.is_empty():
        return pl.DataFrame(schema=empty_schema)

    filtered = (
        causal_rows
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
    )
    if filtered.is_empty():
        return pl.DataFrame(schema=empty_schema)

    per_causal_sample = (
        filtered
        .group_by("collection_name", "sample_id", "causal_idx", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(
            pl.col("valid_covered").any().alias("discovered"),
            pl.col("cs_causal_radius").max().alias("cs_causal_radius"),
        )
    )
    power_radius = (
        per_causal_sample
        .group_by("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(
            pl.col("discovered").cast(pl.Float64).mean().alias("power"),
            pl.col("cs_causal_radius").mean().alias("cs_causal_radius"),
        )
    )
    # Empirical coverage is a CS-level metric, not power. Reuse the same
    # computation as the coverage/size chart so the calibration panels match.
    coverage = make_cs_coverage_size_curves(
        cs_plot_data,
        method_metadata,
        selected_methods=selected_methods,
        selected_thresholds=selected_thresholds,
    ).select("collection_name", "method", "threshold", "beta", "coverage")
    return (
        power_radius
        .join(coverage, on=["collection_name", "method", "threshold", "beta"],
              how="left", nulls_equal=True)
        .sort("collection_name", "method_display", "threshold", "beta")
    )


def find_calibrated_radius_summary(
    radius_trace: pl.DataFrame,
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
    target_coverage: float,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Calibrated beta lookup for radius/power summaries."""
    group_cols = [
        "collection_name", "method", "method_display", "threshold",
        "is_thresholded", "is_selected_threshold",
    ]
    group_cols_no_coll = ["method", "method_display", "threshold", "is_thresholded", "is_selected_threshold"]
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean, "is_selected_threshold": pl.Boolean,
        "calibrated_beta": pl.Float64, "power": pl.Float64, "coverage": pl.Float64,
        "cs_causal_radius": pl.Float64,
    }
    if radius_trace.is_empty() or cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    expanded = (
        cs_plot_data
        .filter(
            pl.col("method").is_in(list(selected_methods))
            & (pl.col("ser_log_bf") >= min_ser_log_bf)
        )
        .explode("mass_above_causal", "causal_indices")
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
    )
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)

    def _exact_betas(df: pl.DataFrame, grp: list[str]) -> pl.DataFrame:
        return (
            df.group_by(grp)
            .agg(
                pl.col("mass_above_causal")
                .quantile(target_coverage, interpolation="higher")
                .round(2)
                .clip(0.01, 0.99)
                .alias("calibrated_beta")
            )
        )

    def _lookup_metrics(exact_betas: pl.DataFrame, trace: pl.DataFrame) -> pl.DataFrame:
        return exact_betas.join(
            trace.rename({"beta": "calibrated_beta"}),
            on=group_cols + ["calibrated_beta"],
            how="left",
            nulls_equal=True,
        )

    per_coll = _lookup_metrics(_exact_betas(expanded, group_cols), radius_trace)
    pooled_trace = (
        radius_trace
        .group_by(group_cols_no_coll + ["beta"])
        .agg(pl.col("power").mean(), pl.col("coverage").mean(), pl.col("cs_causal_radius").mean())
        .with_columns(pl.lit("All").alias("collection_name"))
    )
    pooled = _lookup_metrics(
        _exact_betas(expanded.with_columns(pl.lit("All").alias("collection_name")), group_cols),
        pooled_trace,
    )
    return (
        pl.concat([per_coll, pooled], how="diagonal")
        .sort("collection_name", "method_display", "threshold")
    )


def find_calibrated_beta_summary(
    beta_trace: pl.DataFrame,
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
    target_coverage: float,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Compute calibrated beta' = target quantile of mass_above_causal (exact, no grid quantization).

    For each (collection_name, method, threshold): calibrated_beta = target-th percentile of the
    distribution of mass_above_causal over valid (ser_log_bf >= min) (sample, l, causal) triples.
    power/cs_size/coverage are looked up from beta_trace at the nearest grid beta.

    Also computes an "All" row pooling mass_above_causal across all collections before calibrating.
    """
    group_cols = [
        "collection_name", "method", "method_display", "threshold",
        "is_thresholded", "is_selected_threshold",
    ]
    group_cols_no_coll = ["method", "method_display", "threshold", "is_thresholded", "is_selected_threshold"]
    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean, "is_selected_threshold": pl.Boolean,
        "calibrated_beta": pl.Float64, "power": pl.Float64, "cs_size": pl.Float64, "coverage": pl.Float64,
    }
    if beta_trace.is_empty() or cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    expanded = (
        cs_plot_data
        .filter(
            pl.col("method").is_in(list(selected_methods))
            & (pl.col("ser_log_bf") >= min_ser_log_bf)
        )
        .explode("mass_above_causal", "causal_indices")
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
    )
    if expanded.is_empty():
        return pl.DataFrame(schema=empty_schema)

    def _exact_betas(df: pl.DataFrame, grp: list[str]) -> pl.DataFrame:
        return (
            df.group_by(grp)
            .agg(
                pl.col("mass_above_causal")
                .quantile(target_coverage, interpolation="higher")
                .round(2)
                .clip(0.01, 0.99)
                .alias("calibrated_beta")
            )
        )

    def _lookup_metrics(exact_betas: pl.DataFrame, trace: pl.DataFrame) -> pl.DataFrame:
        return exact_betas.join(
            trace.rename({"beta": "calibrated_beta"}),
            on=group_cols + ["calibrated_beta"],
            how="left",
            nulls_equal=True,
        )

    per_coll = _lookup_metrics(_exact_betas(expanded, group_cols), beta_trace)

    pooled_trace = (
        beta_trace
        .group_by(group_cols_no_coll + ["beta"])
        .agg(pl.col("coverage").mean(), pl.col("cs_size").mean(), pl.col("power").mean())
        .with_columns(pl.lit("All").alias("collection_name"))
    )
    pooled_expanded = expanded.with_columns(pl.lit("All").alias("collection_name"))
    pooled = _lookup_metrics(_exact_betas(pooled_expanded, group_cols), pooled_trace)

    return (
        pl.concat([per_coll, pooled], how="diagonal")
        .sort("collection_name", "method_display", "threshold")
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


def make_adaptive_cs_summary(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
    min_beta: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    """Per-CS: find smallest beta' >= min_beta where covered=True and cs_size <= max_cs_size.

    Power = mean(achieved) over ALL CS instances (same denominator as make_cs_power_size_coverage_summary).
    CSs failing ser_log_bf or max_cs_size filters contribute 0 (achieved=False) to power.
    """
    from utils import CS_BETA_GRID

    empty_schema = {
        "collection_name": pl.String, "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean, "is_selected_threshold": pl.Boolean,
        "power": pl.Float64, "cs_size": pl.Float64, "min_covering_beta": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    beta_indices = [i for i, b in enumerate(CS_BETA_GRID) if b >= min_beta]
    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")

    per_cs_rows: list[dict] = []
    for row in cs_plot_data.filter(pl.col("method").is_in(list(selected_methods))).iter_rows(named=True):
        sizes = row["cs_sizes"]
        ser_valid = row["ser_log_bf"] >= min_ser_log_bf
        for causal_rank, causal_idx in zip(row["rank_of_causal"], row["causal_indices"]):
            achieved_beta: float | None = None
            achieved_size: int | None = None
            if ser_valid:
                for i in beta_indices:
                    cs_size = sizes[i]
                    if cs_size <= max_cs_size and causal_rank < cs_size:
                        achieved_beta = float(CS_BETA_GRID[i])
                        achieved_size = cs_size
                        break
            per_cs_rows.append({
                "collection_name": row["collection_name"],
                "sample_id": row["sample_id"],
                "method": row["method"],
                "threshold": row["threshold"],
                "causal_idx": int(causal_idx),
                "achieved": achieved_beta is not None,
                "min_covering_beta": achieved_beta,
                "cs_size": float(achieved_size) if achieved_size is not None else None,
            })

    if not per_cs_rows:
        return pl.DataFrame(schema=empty_schema)

    per_cs = pl.from_dicts(per_cs_rows, schema={
        "collection_name": pl.String, "sample_id": pl.String, "method": pl.String,
        "threshold": pl.Float64, "causal_idx": pl.Int64, "achieved": pl.Boolean,
        "min_covering_beta": pl.Float64, "cs_size": pl.Float64,
    })

    # Per (sample_id, causal_idx): did any l achieve coverage for this specific causal?
    # denominator = total (sample_id, causal_idx) pairs
    per_causal_sample = (
        per_cs
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .with_columns(
            (
                ~pl.col("is_thresholded")
                | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
            ).alias("is_selected_threshold")
        )
        .filter(pl.col("is_selected_threshold"))
        .group_by("collection_name", "sample_id", "causal_idx", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold")
        .agg(
            pl.col("achieved").any().alias("any_achieved"),
            pl.when(pl.col("achieved")).then(pl.col("cs_size")).mean().alias("cs_size"),
            pl.when(pl.col("achieved")).then(pl.col("min_covering_beta")).mean().alias("min_covering_beta"),
        )
    )
    return (
        per_causal_sample
        .group_by("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold")
        .agg(
            pl.col("any_achieved").cast(pl.Float64).mean().alias("power"),
            pl.col("cs_size").mean(),
            pl.col("min_covering_beta").mean(),
        )
        .sort("collection_name", "method_display", "threshold")
    )


def render_adaptive_cs_dot_chart(
    calibrated: pl.DataFrame,
    *,
    collection_names: list[str],
    nominal_beta: float,
    min_ser_log_bf: float,
) -> "plt.Figure":
    """Dot plot: power, cs_size, and calibrated_beta at collection-level calibrated operating point."""
    if calibrated.is_empty():
        return make_placeholder_chart("No adaptive CS data")

    theme = base_chart_theme()
    metrics = [("power", "Power"), ("cs_size", "CS Size"), ("calibrated_beta", "calibrated β")]
    _legend_w = 2.5
    _plot_w = theme["width"] * len(metrics)
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w
    fig, axes = plt.subplots(1, len(metrics), figsize=(_fig_w, theme["height"]), squeeze=False)
    axes = axes[0]

    method_order = (
        calibrated.select("method_display", "is_thresholded").unique()
        .sort(["is_thresholded", "method_display"])
        ["method_display"].to_list()
    )
    n_methods = len(method_order)
    offsets = np.linspace(-0.35, 0.35, n_methods) if n_methods > 1 else np.array([0.0])
    method_offset = {m: float(offsets[i]) for i, m in enumerate(method_order)}
    coll_to_x = {c: i for i, c in enumerate(collection_names)}
    agg_x = len(collection_names)
    _agg_dot = calibrated.filter(pl.col("collection_name") == "All")

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    for col_idx, (metric_col, metric_title) in enumerate(metrics):
        ax = axes[col_idx]
        for trow in (
            calibrated.select("method", "method_display", "threshold", "is_thresholded")
            .unique()
            .sort(["is_thresholded", "method_display"])
            .iter_rows(named=True)
        ):
            thresh_filter = (
                pl.col("threshold").is_null() if trow["threshold"] is None
                else (pl.col("threshold") == trow["threshold"])
            )
            m_df = calibrated.filter((pl.col("method") == trow["method"]) & thresh_filter)
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
        if metric_col == "calibrated_beta":
            ax.axhline(y=nominal_beta, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(metric_title)
        ax.set_xticks(list(range(agg_x + 1)))
        ax.set_xticklabels(list(collection_names) + ["All"], rotation=45, ha="right", fontsize=7)
        ax.set_xlim(-0.5, agg_x + 0.5)

    settings_text = f"nominal β = {nominal_beta:.2f}\nmin log BF = {min_ser_log_bf:.1f}"
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


def render_cs_size_power_chart(
    nominal: pl.DataFrame,
    calibrated: pl.DataFrame,
    *,
    collection_names: list[str],
    min_beta: float,
    min_ser_log_bf: float,
    max_cs_size: int,
) -> "plt.Figure":
    """Scatter: x=mean cs_size, y=power. Filled=nominal beta, open=calibrated (adaptive) beta."""
    if nominal.is_empty() and calibrated.is_empty():
        return make_placeholder_chart("No CS data")

    theme = base_chart_theme()
    all_colls = list(collection_names) + ["All"]
    n_panels = len(all_colls)
    _legend_w = 2.5
    _plot_w = theme["width"] * n_panels
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w

    # build aggregate rows
    def _agg(df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by(
            "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold"
        ).agg(pl.col("power").mean(), pl.col("cs_size").mean()).with_columns(
            pl.lit("All").alias("collection_name")
        )

    nom_full = pl.concat([nominal, _agg(nominal)], how="diagonal")
    cal_full = calibrated  # already contains "All" row from find_calibrated_beta_summary

    method_order = (
        nominal.select("method_display", "is_thresholded").unique()
        .sort(["is_thresholded", "method_display"])["method_display"].to_list()
    )

    fig, axes = plt.subplots(1, n_panels, figsize=(_fig_w, theme["height"]), squeeze=False)
    axes = axes[0]

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    for panel_idx, coll in enumerate(all_colls):
        ax = axes[panel_idx]
        if panel_idx == n_panels - 1:
            ax.set_facecolor("#ddeeff")

        nom_coll = nom_full.filter(pl.col("collection_name") == coll)
        cal_coll = cal_full.filter(pl.col("collection_name") == coll)

        for trow in (
            nom_coll.select("method", "method_display", "threshold", "is_thresholded")
            .unique().sort(["is_thresholded", "method_display"]).iter_rows(named=True)
        ):
            thresh_filter = (
                pl.col("threshold").is_null() if trow["threshold"] is None
                else (pl.col("threshold") == trow["threshold"])
            )
            color = method_color(trow["method"])
            label = trow["method_display"]

            nom_row = nom_coll.filter((pl.col("method") == trow["method"]) & thresh_filter)
            cal_row = cal_coll.filter((pl.col("method") == trow["method"]) & thresh_filter)

            nom_x = nom_row["cs_size"][0] if not nom_row.is_empty() and nom_row["cs_size"][0] is not None else None
            nom_y = nom_row["power"][0] if not nom_row.is_empty() and nom_row["power"][0] is not None else None
            cal_x = cal_row["cs_size"][0] if not cal_row.is_empty() and cal_row["cs_size"][0] is not None else None
            cal_y = cal_row["power"][0] if not cal_row.is_empty() and cal_row["power"][0] is not None else None

            if nom_x is not None and nom_y is not None and cal_x is not None and cal_y is not None:
                ax.plot([nom_x, cal_x], [nom_y, cal_y], color=color, linewidth=0.8, zorder=2)

            if nom_x is not None and nom_y is not None:
                sc = ax.scatter([nom_x], [nom_y], color=color, s=60, zorder=3, marker="o")
                if label not in seen_labels:
                    legend_handles.append(sc)
                    legend_labels.append(label)
                    seen_labels.add(label)

            if cal_x is not None and cal_y is not None:
                ax.scatter([cal_x], [cal_y], color=color, s=60, zorder=3,
                           marker="o", facecolors="none", edgecolors=color, linewidths=1.5)

        ax.set_xlabel("Mean CS size")
        ax.set_title(coll, fontsize=8, fontweight="bold")
        if panel_idx == 0:
            ax.set_ylabel("Power")

    settings_text = f"min β = {min_beta:.2f}\nmax cs = {max_cs_size}\nmin log BF = {min_ser_log_bf:.1f}"
    if legend_handles:
        # add nominal/calibrated marker legend entries
        from matplotlib.lines import Line2D
        legend_handles += [
            Line2D([0], [0], marker="o", color="grey", markersize=6, linestyle="none", label="nominal"),
            Line2D([0], [0], marker="o", color="grey", markersize=6, linestyle="none",
                   markerfacecolor="none", markeredgewidth=1.5, label="calibrated"),
        ]
        legend_labels += ["nominal β", "calibrated β"]
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    fig.text(
        _plot_frac + 0.03, 0.35, settings_text,
        fontsize=7, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.8),
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_radius_power_chart(
    nominal: pl.DataFrame,
    calibrated: pl.DataFrame,
    *,
    collection_names: list[str],
    min_beta: float,
    min_ser_log_bf: float,
    max_cs_size: int,
) -> "plt.Figure":
    """Scatter: x=mean causal radius among discovered targets, y=power."""
    if nominal.is_empty() and calibrated.is_empty():
        return make_placeholder_chart("No CS radius data")

    theme = base_chart_theme()
    all_colls = list(collection_names) + ["All"]
    n_panels = len(all_colls)
    _legend_w = 2.5
    _plot_w = theme["width"] * n_panels
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w

    def _agg(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        return df.group_by(
            "method", "method_display", "threshold", "is_thresholded", "is_selected_threshold"
        ).agg(pl.col("power").mean(), pl.col("cs_causal_radius").mean()).with_columns(
            pl.lit("All").alias("collection_name")
        )

    nom_full = pl.concat([nominal, _agg(nominal)], how="diagonal") if not nominal.is_empty() else nominal
    cal_full = calibrated

    fig, axes = plt.subplots(1, n_panels, figsize=(_fig_w, theme["height"]), squeeze=False)
    axes = axes[0]
    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    for panel_idx, coll in enumerate(all_colls):
        ax = axes[panel_idx]
        if panel_idx == n_panels - 1:
            ax.set_facecolor("#ddeeff")

        nom_coll = nom_full.filter(pl.col("collection_name") == coll)
        cal_coll = cal_full.filter(pl.col("collection_name") == coll)
        labels = pl.concat([
            nom_coll.select("method", "method_display", "threshold", "is_thresholded"),
            cal_coll.select("method", "method_display", "threshold", "is_thresholded"),
        ], how="diagonal").unique().sort(["is_thresholded", "method_display"])

        for trow in labels.iter_rows(named=True):
            thresh_filter = (
                pl.col("threshold").is_null() if trow["threshold"] is None
                else (pl.col("threshold") == trow["threshold"])
            )
            color = method_color(trow["method"])
            label = trow["method_display"]
            nom_row = nom_coll.filter((pl.col("method") == trow["method"]) & thresh_filter)
            cal_row = cal_coll.filter((pl.col("method") == trow["method"]) & thresh_filter)

            nom_x = nom_row["cs_causal_radius"][0] if not nom_row.is_empty() else None
            nom_y = nom_row["power"][0] if not nom_row.is_empty() else None
            cal_x = cal_row["cs_causal_radius"][0] if not cal_row.is_empty() else None
            cal_y = cal_row["power"][0] if not cal_row.is_empty() else None

            if nom_x is not None and nom_y is not None and cal_x is not None and cal_y is not None:
                ax.plot([nom_x, cal_x], [nom_y, cal_y], color=color, linewidth=0.8, zorder=2)
            if nom_x is not None and nom_y is not None:
                sc = ax.scatter([nom_x], [nom_y], color=color, s=60, zorder=3, marker="o")
                if label not in seen_labels:
                    legend_handles.append(sc)
                    legend_labels.append(label)
                    seen_labels.add(label)
            if cal_x is not None and cal_y is not None:
                ax.scatter([cal_x], [cal_y], color=color, s=60, zorder=3,
                           marker="o", facecolors="none", edgecolors=color, linewidths=1.5)

        ax.set_xlabel("Mean causal radius")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(coll, fontsize=8, fontweight="bold")
        if panel_idx == 0:
            ax.set_ylabel("Power")

    settings_text = f"min β = {min_beta:.2f}\nmax cs = {max_cs_size}\nmin log BF = {min_ser_log_bf:.1f}"
    if legend_handles:
        from matplotlib.lines import Line2D
        legend_handles += [
            Line2D([0], [0], marker="o", color="grey", markersize=6, linestyle="none", label="nominal"),
            Line2D([0], [0], marker="o", color="grey", markersize=6, linestyle="none",
                   markerfacecolor="none", markeredgewidth=1.5, label="calibrated"),
        ]
        legend_labels += ["nominal β", "calibrated β"]
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    fig.text(
        _plot_frac + 0.03, 0.35, settings_text,
        fontsize=7, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.8),
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_power_size_coverage_trace_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
    selected_thresholds: list[float] | None = None,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> "plt.Figure":
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

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
            if metric_col == "cs_size":
                ax.set_yscale("log")
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
    _thresh_str = "all" if selected_thresholds is None else ", ".join(f"{t:g}" for t in selected_thresholds)
    settings_text = f"thresholds = {_thresh_str}\nmax cs = {max_cs_size}\nmin log BF = {min_ser_log_bf:.1f}"
    fig.text(
        _plot_frac + 0.03, 0.35, settings_text,
        fontsize=7, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.8),
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_coverage_trace_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
    selected_thresholds: list[float] | None = None,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> "plt.Figure":
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

    metrics = [("power", "Power"), ("cs_size", "CS Size")]
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

    def _plot_coverage_trace_row(row_idx: int, row_df: pl.DataFrame) -> None:
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
                ).sort("coverage")
                if trace_df.is_empty():
                    continue
                color = method_color(trow["method"])
                label = trow["method_display"]
                line, = ax.plot(
                    trace_df["coverage"].to_numpy(),
                    trace_df[metric_col].to_numpy(),
                    color=color,
                    linewidth=2.0,
                )
                marker_row = trace_df.filter(pl.col("beta") == 0.95)
                if not marker_row.is_empty():
                    ax.plot(
                        marker_row["coverage"].to_numpy(),
                        marker_row[metric_col].to_numpy(),
                        marker="o", markersize=6, color=color, linestyle="none",
                    )
                if label not in seen_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)
                    seen_labels.add(label)
            if row_idx == 0:
                ax.set_title(metric_title)
            ax.set_xlabel("Empirical coverage")

    for row_idx, coll_name in enumerate(collection_names):
        _plot_coverage_trace_row(row_idx, summary.filter(pl.col("collection_name") == coll_name))
        axes[row_idx, 0].set_ylabel(coll_name, fontsize=9, fontweight="bold")

    # aggregate row
    _agg_bt = (
        summary
        .group_by("method", "method_display", "threshold", "is_thresholded", "is_selected_threshold", "beta")
        .agg(pl.col("power").mean(), pl.col("coverage").mean(), pl.col("cs_size").mean())
    )
    _plot_coverage_trace_row(n_rows, _agg_bt)
    for col_idx in range(len(metrics)):
        axes[n_rows, col_idx].set_facecolor("#ddeeff")
    axes[n_rows, 0].set_ylabel("All", fontsize=9, fontweight="bold")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    _thresh_str = "all" if selected_thresholds is None else ", ".join(f"{t:g}" for t in selected_thresholds)
    settings_text = f"thresholds = {_thresh_str}\nmax cs = {max_cs_size}\nmin log BF = {min_ser_log_bf:.1f}"
    fig.text(
        _plot_frac + 0.03, 0.35, settings_text,
        fontsize=7, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="grey", alpha=0.8),
        transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_coverage_size_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
) -> "plt.Figure":
    """Two panels per row: left=empirical coverage vs CS size, right=nominal beta vs CS size."""
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

    theme = base_chart_theme()
    n_rows = len(collection_names)
    _legend_w = 2.5
    _plot_w = theme["width"] * 3
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w
    fig, axes = plt.subplots(
        n_rows + 1, 3,
        figsize=(_fig_w, theme["height"] * (n_rows + 1)),
        squeeze=False,
    )

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    def _plot_row(row_idx: int, row_df: pl.DataFrame) -> None:
        ax_cov = axes[row_idx, 0]
        ax_nom = axes[row_idx, 1]
        ax_cal = axes[row_idx, 2]
        betas = row_df["beta"].unique().sort().to_numpy()
        if len(betas):
            ax_cal.plot(betas, betas, color="black", linestyle="--", linewidth=1.0, zorder=0)
        trace_labels = (
            row_df.select("method", "threshold", "method_display", "is_thresholded")
            .unique()
            .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
        )
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
            label = trow["method_display"]
            line, = ax_cov.plot(
                trace_df["coverage"].to_numpy(),
                trace_df["cs_size_frac"].to_numpy(),
                color=color, linewidth=2.0,
            )
            marker_row = trace_df.filter(pl.col("beta") == 0.95)
            if not marker_row.is_empty():
                ax_cov.plot(
                    marker_row["coverage"].to_numpy(),
                    marker_row["cs_size_frac"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            ax_nom.plot(
                trace_df["beta"].to_numpy(),
                trace_df["cs_size_frac"].to_numpy(),
                color=color, linewidth=2.0,
            )
            if not marker_row.is_empty():
                ax_nom.plot(
                    marker_row["beta"].to_numpy(),
                    marker_row["cs_size_frac"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            ax_cal.plot(
                trace_df["beta"].to_numpy(),
                trace_df["coverage"].to_numpy(),
                color=color, linewidth=2.0,
            )
            if not marker_row.is_empty():
                ax_cal.plot(
                    marker_row["beta"].to_numpy(),
                    marker_row["coverage"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            if label not in seen_labels:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)
        ax_cov.set_xlim(0.0, 1.02)
        ax_cov.set_ylim(0.0, 1.02)
        ax_nom.set_xlim(0.0, 1.02)
        ax_nom.set_ylim(0.0, 1.02)
        ax_cal.set_xlim(0.0, 1.02)
        ax_cal.set_ylim(0.0, 1.02)
        if row_idx == 0:
            ax_cov.set_title("Empirical coverage vs CS size")
            ax_nom.set_title("Nominal level vs CS size")
            ax_cal.set_title("Calibration (nominal vs empirical)")
        ax_cov.set_xlabel("Empirical coverage")
        ax_nom.set_xlabel("Nominal level (β)")
        ax_cal.set_xlabel("Nominal level (β)")
        ax_cov.set_ylabel("Mean fraction of variables")
        ax_cal.set_ylabel("Empirical coverage")

    for row_idx, coll_name in enumerate(collection_names):
        _plot_row(row_idx, summary.filter(pl.col("collection_name") == coll_name))
        axes[row_idx, 0].set_title(coll_name, fontsize=9, fontweight="bold")

    _agg = (
        summary
        .group_by("method", "method_display", "threshold", "is_thresholded", "beta")
        .agg(pl.col("coverage").mean(), pl.col("cs_size").mean(), pl.col("cs_size_frac").mean())
    )
    _plot_row(n_rows, _agg)
    for col in range(3):
        axes[n_rows, col].set_facecolor("#ddeeff")
    if n_rows > 0:
        axes[n_rows, 0].set_title("All (aggregate)", fontsize=9, fontweight="bold")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_coverage_radius_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
) -> "plt.Figure":
    """Three panels per row using causal radius instead of CS size."""
    if summary.is_empty():
        return make_placeholder_chart("No CS radius data")

    theme = base_chart_theme()
    n_rows = len(collection_names)
    _legend_w = 2.5
    _plot_w = theme["width"] * 3
    _fig_w = _plot_w + _legend_w
    _plot_frac = _plot_w / _fig_w
    fig, axes = plt.subplots(
        n_rows + 1, 3,
        figsize=(_fig_w, theme["height"] * (n_rows + 1)),
        squeeze=False,
    )

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    def _plot_row(row_idx: int, row_df: pl.DataFrame) -> None:
        ax_cov = axes[row_idx, 0]
        ax_nom = axes[row_idx, 1]
        ax_cal = axes[row_idx, 2]
        betas = row_df["beta"].unique().sort().to_numpy()
        if len(betas):
            ax_cal.plot(betas, betas, color="black", linestyle="--", linewidth=1.0, zorder=0)
        trace_labels = (
            row_df.select("method", "threshold", "method_display", "is_thresholded")
            .unique()
            .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
        )
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
            label = trow["method_display"]
            line, = ax_cov.plot(
                trace_df["coverage"].to_numpy(),
                trace_df["cs_causal_radius"].to_numpy(),
                color=color, linewidth=2.0,
            )
            marker_row = trace_df.filter(pl.col("beta") == 0.95)
            if not marker_row.is_empty():
                ax_cov.plot(
                    marker_row["coverage"].to_numpy(),
                    marker_row["cs_causal_radius"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            ax_nom.plot(
                trace_df["beta"].to_numpy(),
                trace_df["cs_causal_radius"].to_numpy(),
                color=color, linewidth=2.0,
            )
            if not marker_row.is_empty():
                ax_nom.plot(
                    marker_row["beta"].to_numpy(),
                    marker_row["cs_causal_radius"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            ax_cal.plot(
                trace_df["beta"].to_numpy(),
                trace_df["coverage"].to_numpy(),
                color=color, linewidth=2.0,
            )
            if not marker_row.is_empty():
                ax_cal.plot(
                    marker_row["beta"].to_numpy(),
                    marker_row["coverage"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            if label not in seen_labels:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)
        ax_cov.set_xlim(0.0, 1.02)
        ax_cov.set_ylim(0.0, 1.02)
        ax_nom.set_xlim(0.0, 1.02)
        ax_nom.set_ylim(0.0, 1.02)
        ax_cal.set_xlim(0.0, 1.02)
        ax_cal.set_ylim(0.0, 1.02)
        if row_idx == 0:
            ax_cov.set_title("Empirical coverage vs causal radius")
            ax_nom.set_title("Nominal level vs causal radius")
            ax_cal.set_title("Calibration (nominal vs empirical)")
        ax_cov.set_xlabel("Empirical coverage")
        ax_nom.set_xlabel("Nominal level (β)")
        ax_cal.set_xlabel("Nominal level (β)")
        ax_cov.set_ylabel("Mean causal radius")
        ax_cal.set_ylabel("Empirical coverage")

    for row_idx, coll_name in enumerate(collection_names):
        _plot_row(row_idx, summary.filter(pl.col("collection_name") == coll_name))
        axes[row_idx, 0].set_title(coll_name, fontsize=9, fontweight="bold")

    _agg = (
        summary
        .group_by("method", "method_display", "threshold", "is_thresholded", "beta")
        .agg(pl.col("coverage").mean(), pl.col("power").mean(), pl.col("cs_causal_radius").mean())
    )
    _plot_row(n_rows, _agg)
    for col in range(3):
        axes[n_rows, col].set_facecolor("#ddeeff")
    if n_rows > 0:
        axes[n_rows, 0].set_title("All (aggregate)", fontsize=9, fontweight="bold")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def make_log_bf_ser_ecdf(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """Raw log BF observations with is_null flag. ECDF computed in render from these raw values."""
    empty_schema = {
        "collection_name": pl.String,
        "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean,
        "is_null": pl.Boolean, "log_bf": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    thresh_mask = (
        ~pl.col("is_thresholded")
        | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
    )
    valid_pairs = (
        meta.filter(pl.col("method").is_in(list(selected_methods))).filter(thresh_mask)
        .select("method", "threshold")
    )
    return (
        cs_plot_data
        .join(valid_pairs, on=["method", "threshold"], how="inner", nulls_equal=True)
        .with_columns((pl.col("causal_indices").list.len() == 0).alias("is_null"))
        .join(meta, on=["method", "threshold"], how="left", nulls_equal=True)
        .select("collection_name", "method", "method_display", "threshold", "is_thresholded", "is_null",
                (-pl.col("ser_log_bf")).exp().alias("log_bf"))
    )


def render_log_bf_ser_ecdf_chart(
    raw_data: pl.DataFrame,
    *,
    collection_names: list[str],
) -> "plt.Figure":
    """ECDF of SER log BF: null (dashed) vs non-null (solid), one row per collection + aggregate.

    ECDF is computed from raw_data within each panel so aggregate pools across collections.
    """
    import numpy as np
    import matplotlib.lines as mlines

    if raw_data.is_empty():
        return make_placeholder_chart("No log BF ECDF data")

    theme = base_chart_theme()
    n_rows = len(collection_names)
    _legend_w = 2.5
    _fig_w = theme["width"] + _legend_w
    _plot_frac = theme["width"] / _fig_w
    fig, axes = plt.subplots(
        n_rows + 1, 1,
        figsize=(_fig_w, theme["height"] * (n_rows + 1)),
        squeeze=False,
    )

    legend_handles: list = []
    legend_labels: list = []
    seen_method_labels: set[str] = set()

    def _ecdf(vals: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
        s = np.sort(vals)
        return s, np.arange(1, len(s) + 1) / len(s)

    def _plot_row(row_idx: int, row_df: pl.DataFrame) -> None:
        ax = axes[row_idx, 0]
        trace_labels = (
            row_df.select("method", "threshold", "method_display", "is_thresholded")
            .unique()
            .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
        )
        for trow in trace_labels.iter_rows(named=True):
            thresh_filter = (
                pl.col("threshold").is_null() if trow["threshold"] is None
                else pl.col("threshold") == trow["threshold"]
            )
            method_df = row_df.filter(
                (pl.col("method") == trow["method"]) & thresh_filter
            )
            color = method_color(trow["method"])
            label = trow["method_display"]
            for is_null, linestyle in ((False, "-"), (True, "--")):
                vals = method_df.filter(pl.col("is_null") == is_null)["log_bf"].to_numpy()
                if len(vals) == 0:
                    continue
                x, y = _ecdf(vals)
                line, = ax.plot(x, y, color=color, linewidth=2.0, linestyle=linestyle)
                if label not in seen_method_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)
                    seen_method_labels.add(label)
        ax.set_xlabel("BF₀₁ = 1/BF₁₀")
        ax.set_ylabel("Empirical CDF")

    for row_idx, coll_name in enumerate(collection_names):
        _plot_row(row_idx, raw_data.filter(pl.col("collection_name") == coll_name))
        axes[row_idx, 0].set_title(coll_name, fontsize=9, fontweight="bold")

    _plot_row(n_rows, raw_data)
    axes[n_rows, 0].set_facecolor("#ddeeff")
    axes[n_rows, 0].set_title("All (aggregate)" if n_rows > 0 else "SER log BF ECDF", fontsize=9, fontweight="bold")

    legend_handles += [
        mlines.Line2D([], [], color="black", linewidth=2.0, linestyle="-"),
        mlines.Line2D([], [], color="black", linewidth=2.0, linestyle="--"),
    ]
    legend_labels += ["non-null", "null"]

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.80),
        )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def make_log_bf_roc_curves(
    cs_plot_data: pl.DataFrame,
    method_metadata: pl.DataFrame,
    *,
    selected_methods: set[str],
    selected_thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """ROC curves for log BF discrimination: null (causal_indices empty) vs non-null.

    Returns one row per (method, threshold, point) with fpr, tpr, log_bf columns.
    Pools all collections.
    """
    import numpy as np

    empty_schema = {
        "method": pl.String, "method_display": pl.String,
        "threshold": pl.Float64, "is_thresholded": pl.Boolean,
        "log_bf": pl.Float64, "fpr": pl.Float64, "tpr": pl.Float64,
    }
    if cs_plot_data.is_empty():
        return pl.DataFrame(schema=empty_schema)

    meta = method_metadata.select("method", "threshold", "method_display", "is_thresholded")
    thresh_mask = (
        ~pl.col("is_thresholded")
        | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
    )
    valid_pairs = (
        meta.filter(pl.col("method").is_in(list(selected_methods))).filter(thresh_mask)
        .select("method", "threshold")
    )
    filtered = cs_plot_data.join(valid_pairs, on=["method", "threshold"], how="inner", nulls_equal=True)
    if filtered.is_empty():
        return pl.DataFrame(schema=empty_schema)

    flat = (
        filtered
        .with_columns(
            (pl.col("causal_indices").list.len() > 0).alias("is_positive")
        )
        .select("method", "threshold", "ser_log_bf", "is_positive")
    )

    rows: list[dict] = []
    for key, group in flat.group_by("method", "threshold"):
        method, threshold = key
        log_bfs = group["ser_log_bf"].to_numpy()
        labels = group["is_positive"].to_numpy().astype(float)
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        order = np.argsort(-log_bfs)
        labels_s = labels[order]
        lbf_s = log_bfs[order]
        cum_pos = np.cumsum(labels_s)
        cum_neg = np.cumsum(1 - labels_s)
        tpr = np.concatenate([[0.0], cum_pos / n_pos])
        fpr = np.concatenate([[0.0], cum_neg / n_neg])
        lbf_pts = np.concatenate([[np.inf], lbf_s])
        meta_row = meta.filter(
            (pl.col("method") == method) & (
                pl.col("threshold").is_null() if threshold is None
                else pl.col("threshold") == threshold
            )
        )
        method_display = meta_row["method_display"][0] if not meta_row.is_empty() else method
        is_thresholded = bool(meta_row["is_thresholded"][0]) if not meta_row.is_empty() else False
        for f, t, lb in zip(fpr.tolist(), tpr.tolist(), lbf_pts.tolist()):
            rows.append({
                "method": method, "method_display": method_display,
                "threshold": threshold, "is_thresholded": is_thresholded,
                "log_bf": lb, "fpr": f, "tpr": t,
            })

    if not rows:
        return pl.DataFrame(schema=empty_schema)
    return pl.from_dicts(rows, schema=empty_schema).sort("method_display", "threshold", "fpr")


def render_log_bf_roc_chart(
    roc_curves: pl.DataFrame,
) -> "plt.Figure":
    """Single-panel ROC: x=FPR, y=TPR, sweeping over log BF threshold."""
    if roc_curves.is_empty():
        return make_placeholder_chart("No ROC data (need null + non-null collections)")

    theme = base_chart_theme()
    _legend_w = 2.5
    _fig_w = theme["width"] + _legend_w
    _plot_frac = theme["width"] / _fig_w
    fig, ax = plt.subplots(1, 1, figsize=(_fig_w, theme["height"]))

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, zorder=0)

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    trace_labels = (
        roc_curves.select("method", "threshold", "method_display", "is_thresholded")
        .unique()
        .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
    )
    for trow in trace_labels.iter_rows(named=True):
        thresh_filter = (
            pl.col("threshold").is_null() if trow["threshold"] is None
            else (pl.col("threshold") == trow["threshold"])
        )
        curve = roc_curves.filter(
            (pl.col("method") == trow["method"]) & thresh_filter
        ).sort("fpr")
        if curve.is_empty():
            continue
        color = method_color(trow["method"])
        label = trow["method_display"]
        line, = ax.plot(
            curve["fpr"].to_numpy(),
            curve["tpr"].to_numpy(),
            color=color, linewidth=2.0,
        )
        for lbf_thresh, marker in [(0.0, "o"), (2.0, "s")]:
            pt = curve.filter(pl.col("log_bf") <= lbf_thresh).head(1)
            if not pt.is_empty():
                ax.plot(pt["fpr"][0], pt["tpr"][0],
                        marker=marker, markersize=6, color=color, linestyle="none", zorder=5)
        if label not in seen_labels:
            legend_handles.append(line)
            legend_labels.append(label)
            seen_labels.add(label)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Log BF ROC (null vs non-null)")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_cs_calibration_chart(
    summary: pl.DataFrame,
    *,
    collection_names: list[str],
) -> "plt.Figure":
    """CS calibration: x=nominal beta, y=empirical coverage. Diagonal = perfect calibration."""
    if summary.is_empty():
        return make_placeholder_chart("No CS beta trace data")

    theme = base_chart_theme()
    n_rows = len(collection_names)
    _legend_w = 2.5
    _fig_w = theme["width"] + _legend_w
    _plot_frac = theme["width"] / _fig_w
    fig, axes = plt.subplots(
        n_rows + 1, 1,
        figsize=(_fig_w, theme["height"] * (n_rows + 1)),
        squeeze=False,
    )

    legend_handles: list = []
    legend_labels: list = []
    seen_labels: set[str] = set()

    def _plot_row(row_idx: int, row_df: pl.DataFrame) -> None:
        ax = axes[row_idx, 0]
        betas = row_df["beta"].unique().sort().to_numpy()
        if len(betas):
            ax.plot(betas, betas, color="black", linestyle="--", linewidth=1.0, zorder=0)
        trace_labels = (
            row_df.select("method", "threshold", "method_display", "is_thresholded")
            .unique()
            .sort("is_thresholded", "method_display", "threshold", nulls_last=True)
        )
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
            label = trow["method_display"]
            line, = ax.plot(
                trace_df["beta"].to_numpy(),
                trace_df["coverage"].to_numpy(),
                color=color,
                linewidth=2.0,
            )
            marker_row = trace_df.filter(pl.col("beta") == 0.95)
            if not marker_row.is_empty():
                ax.plot(
                    marker_row["beta"].to_numpy(),
                    marker_row["coverage"].to_numpy(),
                    marker="o", markersize=6, color=color, linestyle="none",
                )
            if label not in seen_labels:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)
        if row_idx == 0:
            ax.set_title("CS Calibration")
        ax.set_xlabel("Nominal coverage (β)")
        ax.set_ylabel("Empirical coverage")

    for row_idx, coll_name in enumerate(collection_names):
        _plot_row(row_idx, summary.filter(pl.col("collection_name") == coll_name))
        axes[row_idx, 0].set_title(coll_name, fontsize=9, fontweight="bold")

    _agg = (
        summary
        .group_by("method", "method_display", "threshold", "is_thresholded", "beta")
        .agg(pl.col("coverage").mean())
    )
    _plot_row(n_rows, _agg)
    axes[n_rows, 0].set_facecolor("#ddeeff")
    if n_rows > 0:
        axes[n_rows, 0].set_title("All (aggregate)", fontsize=9, fontweight="bold")

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            frameon=False, fontsize=8,
            loc="upper left", bbox_to_anchor=(_plot_frac + 0.02, 0.98),
        )
    fig.tight_layout(rect=[0, 0, _plot_frac, 1])
    return fig


def render_f1_boxplot(
    all_data: pl.DataFrame,
    collection_names: list[str],
    method_order: list[str] | None = None,
) -> plt.Figure:
    """4 axes (f1_loc, f1_scale, est_intercept, mu_at_causal).
    X-axis = collections; grouped boxes at each tick, one box per method.
    """
    color_map = method_family_color_map()

    present = set(all_data["method_display"].to_list())

    if method_order is None:
        method_order = (
            all_data.select("method_display", "method_family")
            .unique()
            .sort("method_family")["method_display"]
            .to_list()
        )
    else:
        method_order = [m for m in method_order if m in present]

    family_of: dict[str, str] = dict(
        all_data.select("method_display", "method_family").unique().iter_rows()
    )

    params = [
        ("f1_loc",        "true_f1_loc",    "f₁ loc"),
        ("f1_scale",      "true_f1_scale",  "f₁ scale"),
        ("est_intercept", "true_intercept", "intercept"),
        ("mu_at_causal",  "true_effect",    "μ at causal"),
    ]

    n_colls = len(collection_names)
    n_methods = len(method_order)
    group_width = n_methods + 1

    cell_w = max(1.5 * n_colls, 6)
    cell_h = cell_w * 2 / 3
    tick_fs = 11
    fig, axes_2d = plt.subplots(
        2, 2,
        figsize=(2 * cell_w, 2 * cell_h),
        gridspec_kw={"hspace": 0.5, "wspace": 0.35},
        squeeze=False,
    )
    axes = [axes_2d[0, 0], axes_2d[0, 1], axes_2d[1, 0], axes_2d[1, 1]]

    for ax_idx, (param, true_col, label) in enumerate(params):
        ax = axes[ax_idx]

        for ci, cname in enumerate(collection_names):
            coll = all_data.filter(pl.col("collection_name") == cname)

            true_vals = coll[true_col].drop_nulls()
            true_val = float(true_vals[0]) if not true_vals.is_empty() else None

            x_left = ci * group_width + 0.5
            x_right = ci * group_width + n_methods + 0.5
            if true_val is not None:
                ax.hlines(true_val, x_left, x_right, colors="black",
                          linestyles="--", linewidth=1.5, zorder=5)

            for mi, m in enumerate(method_order):
                vals = coll.filter(pl.col("method_display") == m)[param].drop_nulls()
                if vals.is_empty():
                    continue
                x_pos = ci * group_width + mi + 1
                color = color_map.get(family_of.get(m, ""), "#888888")
                bp = ax.boxplot(
                    vals.to_numpy(),
                    positions=[x_pos],
                    patch_artist=True,
                    showfliers=False,
                    widths=0.7,
                    manage_ticks=False,
                )
                bp["boxes"][0].set_facecolor(color)
                bp["boxes"][0].set_alpha(0.7)
                for element in ("whiskers", "caps", "medians"):
                    for line in bp[element]:
                        line.set_color("#333333")

        group_centers = [ci * group_width + (n_methods - 1) / 2 + 1 for ci in range(n_colls)]
        ax.set_xlim(0.5, n_colls * group_width - 0.5)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(collection_names, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_title(label, fontsize=tick_fs + 2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=color_map.get(family_of.get(m, ""), "#888888"), alpha=0.7)
        for m in method_order
    ]
    axes[3].legend(handles, method_order, loc="upper right", fontsize=tick_fs, framealpha=0.5,
                   bbox_to_anchor=(1.0, 1.0))
    return fig


def _scatter_with_marginals(
    fig: plt.Figure,
    outer_spec,
    x: np.ndarray,
    y: np.ndarray,
    true_x: float | None,
    true_y: float | None,
    color: str,
    *,
    row_ref: dict,
    ri: int,
    ci: int,
    n_rows: int,
    xlabel: str,
    ylabel: str,
    ylabel_prefix: str,
    col_label: str | None,
) -> None:
    import matplotlib.gridspec as gridspec

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_spec,
        width_ratios=[3, 1], height_ratios=[1, 3],
        hspace=0.05, wspace=0.05,
    )
    share_kw = {"sharex": row_ref[ri], "sharey": row_ref[ri]} if ri in row_ref else {}
    ax_s = fig.add_subplot(inner[1, 0], **share_kw)
    if ri not in row_ref:
        row_ref[ri] = ax_s
    ax_t = fig.add_subplot(inner[0, 0], sharex=ax_s)
    ax_r = fig.add_subplot(inner[1, 1], sharey=ax_s)
    ax_corner = fig.add_subplot(inner[0, 1])
    ax_corner.axis("off")

    if len(x):
        ax_s.scatter(x, y, alpha=0.4, color=color, s=12, linewidths=0)
        ax_t.hist(x, bins=20, color=color, alpha=0.75)
        ax_r.hist(y, bins=20, color=color, alpha=0.75, orientation="horizontal")

    if true_x is not None and true_y is not None:
        ax_s.scatter([true_x], [true_y], color="black", marker="*", s=80, zorder=6)
        ax_s.axvline(true_x, color="black", linewidth=0.7, alpha=0.4)
        ax_s.axhline(true_y, color="black", linewidth=0.7, alpha=0.4)
        ax_t.axvline(true_x, color="black", linewidth=1.0, alpha=0.7)
        ax_r.axhline(true_y, color="black", linewidth=1.0, alpha=0.7)

    plt.setp(ax_t.get_xticklabels(), visible=False)
    plt.setp(ax_r.get_yticklabels(), visible=False)
    ax_t.tick_params(axis="x", which="both", bottom=False)
    ax_r.tick_params(axis="y", which="both", left=False)
    for ax in (ax_s, ax_t, ax_r):
        ax.tick_params(labelsize=6)

    if ci == 0:
        ax_s.set_ylabel(f"{ylabel_prefix}\n{ylabel}", fontsize=7)
    else:
        ax_s.set_ylabel("")
        plt.setp(ax_s.get_yticklabels(), visible=False)
    if ri == n_rows - 1:
        ax_s.set_xlabel(xlabel, fontsize=7)
    if col_label is not None:
        ax_t.set_title(col_label, fontsize=8, pad=3)


def render_f1_scatter_chart(
    f1_data: pl.DataFrame,
    collection_names: list[str],
    method_order: list[str] | None = None,
) -> plt.Figure:
    """N_collections rows × N_methods columns.
    Each cell: scatter(est_loc, est_scale) + marginal histograms.
    f1_data: collection_name, method_display, method_family,
             f1_loc, f1_scale, true_f1_loc, true_f1_scale.
    """
    color_map = method_family_color_map()
    if method_order is None:
        method_order = (
            f1_data.select("method_display", "method_family")
            .unique().sort("method_family")["method_display"].to_list()
        )
    family_of: dict[str, str] = dict(
        f1_data.select("method_display", "method_family").unique().iter_rows()
    )

    n_rows, n_cols = len(collection_names), len(method_order)
    cell = 2.8
    fig = plt.figure(figsize=(n_cols * cell, n_rows * cell))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.45)
    row_ref: dict = {}

    for ri, cname in enumerate(collection_names):
        coll = f1_data.filter(pl.col("collection_name") == cname)
        true_loc_s = coll["true_f1_loc"].drop_nulls()
        true_scale_s = coll["true_f1_scale"].drop_nulls()
        true_x = float(true_loc_s[0]) if not true_loc_s.is_empty() else None
        true_y = float(true_scale_s[0]) if not true_scale_s.is_empty() else None

        for ci, m in enumerate(method_order):
            fd = coll.filter(pl.col("method_display") == m)
            fam = family_of.get(m)
            color = color_map.get(fam, "#888888")
            x = fd["f1_loc"].drop_nulls().to_numpy() if not fd.is_empty() else np.array([])
            y = fd["f1_scale"].drop_nulls().to_numpy() if not fd.is_empty() else np.array([])
            col_label = m if ri == 0 else None

            _scatter_with_marginals(
                fig, outer[ri, ci], x, y, true_x, true_y, color,
                row_ref=row_ref, ri=ri, ci=ci, n_rows=n_rows,
                xlabel="loc", ylabel="scale", ylabel_prefix=cname,
                col_label=col_label,
            )

    return fig


def render_f1_enrich_scatter_chart(
    enrich_data: pl.DataFrame,
    collection_names: list[str],
    method_order: list[str] | None = None,
) -> plt.Figure:
    """N_collections rows × N_methods columns.
    Each cell: scatter(est_intercept, mu_at_causal) + marginal histograms.
    enrich_data: collection_name, method_display, method_family,
                 est_intercept, mu_at_causal, true_intercept, true_effect.
    """
    color_map = method_family_color_map()
    if method_order is None:
        method_order = (
            enrich_data.select("method_display", "method_family")
            .unique().sort("method_family")["method_display"].to_list()
        )
    family_of: dict[str, str] = dict(
        enrich_data.select("method_display", "method_family").unique().iter_rows()
    )

    n_rows, n_cols = len(collection_names), len(method_order)
    cell = 2.8
    fig = plt.figure(figsize=(n_cols * cell, n_rows * cell))
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.45)
    row_ref: dict = {}

    for ri, cname in enumerate(collection_names):
        for ci, m in enumerate(method_order):
            fd = enrich_data.filter(
                (pl.col("collection_name") == cname) & (pl.col("method_display") == m)
            )
            fam = family_of.get(m)
            color = color_map.get(fam, "#888888")

            if not fd.is_empty():
                x = fd["est_intercept"].drop_nulls().to_numpy()
                y = fd["mu_at_causal"].drop_nulls().to_numpy()
                true_x = float(fd["true_intercept"].drop_nulls()[0])
                true_y = float(fd["true_effect"].drop_nulls()[0])
            else:
                x, y = np.array([]), np.array([])
                true_x, true_y = None, None

            col_label = m if ri == 0 else None

            _scatter_with_marginals(
                fig, outer[ri, ci], x, y, true_x, true_y, color,
                row_ref=row_ref, ri=ri, ci=ci, n_rows=n_rows,
                xlabel="intercept", ylabel="μ_causal", ylabel_prefix=cname,
                col_label=col_label,
            )

    return fig
