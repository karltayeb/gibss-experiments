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
    "mass_above_causal", "cs_dot_summary", "cs_calibrated_dot", "cs_size_power", "cs_power_fdp", "cs_beta_trace", "cs_coverage_trace", "preceding_posterior_mass_ecdf",
    "agg_pip_calibration", "agg_power_fdp", "agg_causal_pip", "agg_causal_rank",
    "agg_mass_above_causal", "agg_cs_power_fdp", "agg_cs_beta_trace", "agg_cs_coverage_trace", "agg_cs_size_power", "agg_cs_calibrated_dot", "agg_preceding_posterior_mass_ecdf",
    "f1_boxplot", "f1_scatter", "f1_enrich_scatter",
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
    overrides = cfg["settings"].get(plot_settings) or {}
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

    def _tag_optional(key: str) -> pl.DataFrame:
        frames = [
            b[key].with_columns(pl.lit(aliases.get(name, name)).alias("collection_name"))
            for name, b in bundles.items()
            if key in b and not b[key].is_empty()
        ]
        return pl.concat(frames) if frames else pl.DataFrame()

    return {
        "method_metadata": combined_method_metadata,
        "collection_names": [aliases.get(item["name"], item["name"]) for item in coll_list],
        "pip_plot_data": _tag("pip_plot_data"),
        "cs_plot_data": _tag("cs_plot_data"),
        "f1_plot_data": _tag_optional("f1_plot_data"),
        "enrich_plot_data": _tag_optional("enrich_plot_data"),
    }


def _selected_thresholds(settings: dict) -> list[float] | None:
    if "thresholds" in settings:
        val = settings["thresholds"]
        return [float(t) for t in val] if val else None
    if "threshold" in settings:
        return [float(settings["threshold"])]
    return None


def _foreground_methods(method_metadata: pl.DataFrame, settings: dict) -> set[str]:
    thresholds = _selected_thresholds(settings)
    method_families = settings.get("method_families")
    threshold_mask = (
        ~pl.col("is_thresholded")
        | (pl.lit(True) if thresholds is None else pl.col("threshold").is_in(thresholds))
    )
    df = method_metadata.filter(threshold_mask)
    if method_families is not None:
        df = df.filter(pl.col("method_family").is_in(method_families))
    return set(df["method"].to_list())


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
    fg = _foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_thresholds=_selected_thresholds(settings),
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
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
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


def _make_preceding_posterior_mass_ecdf(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_preceding_mass_ecdf_summary(
        cs_data, method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
    )
    return viz_utils.render_preceding_mass_ecdf_chart(
        summary, collection_names=combined_data["collection_names"]
    )


def _make_agg_preceding_posterior_mass_ecdf(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    summary = viz_utils.make_preceding_mass_ecdf_summary(
        cs_data, method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
    )
    return viz_utils.render_preceding_mass_ecdf_chart(summary, collection_names=[])


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
        selected_thresholds=_selected_thresholds(settings),
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
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, selected_thresholds=_selected_thresholds(settings), target_coverage=min_beta, min_ser_log_bf=min_log_bf)
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
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, selected_thresholds=_selected_thresholds(settings), target_coverage=min_beta, min_ser_log_bf=min_log_bf)
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
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data, method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    nominal = beta_summary.filter(pl.col("beta") == round(min_beta, 2))
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, selected_thresholds=_selected_thresholds(settings), target_coverage=min_beta, min_ser_log_bf=min_log_bf)
    return viz_utils.render_cs_size_power_chart(
        nominal,
        calibrated,
        collection_names=collection_names,
        min_beta=min_beta,
        min_ser_log_bf=min_log_bf,
        max_cs_size=max_cs_size,
    )


_CS_POWER_FDP_BETAS = [1.0, 0.95, 0.50]
_CS_BETA_SIZE_INDICES = {0.95: 94, 0.50: 49}  # index into CS_BETA_GRID


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
        ax.plot(fdp_arr, pwr_arr, color=color, linewidth=1.5, label=method_display)
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
    thresholds = _selected_thresholds(settings)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return None
    _thresh_mask = (
        pl.col("threshold").is_null()
        | (pl.lit(True) if thresholds is None else pl.col("threshold").is_in(thresholds))
    )
    raw = (
        cs_data
        .filter(pl.col("method").is_in(fg) & _thresh_mask)
        .join(
            method_meta.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
    )
    return raw if not raw.is_empty() else None


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


def _make_cs_beta_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_beta_trace_chart(
        beta_summary,
        collection_names=collection_names,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _set_agg_facecolor(fig: plt.Figure) -> None:
    for ax in fig.axes:
        ax.set_facecolor("#ddeeff")


def _make_agg_pip_calibration(combined_data: dict, settings: dict) -> plt.Figure:
    pip_plot = combined_data["pip_plot_data"]
    method_meta = combined_data["method_metadata"]
    fg = _foreground_methods(method_meta, settings)
    summary = viz_utils.expand_pip_calibration_from_compact(
        pip_plot.filter(pl.col("method").is_in(fg)),
        method_meta,
        selected_thresholds=_selected_thresholds(settings),
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
    max_fdp = settings.get("max_fdp", 0.5)
    fg = _foreground_methods(method_meta, settings)
    power_fdp = viz_utils.expand_power_fdp_from_compact(
        pip_plot,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
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
    _set_agg_facecolor(fig)
    return fig


def _make_agg_cs_beta_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    # collection_names=[] renders only the aggregate row (no per-collection rows)
    return viz_utils.render_cs_beta_trace_chart(
        beta_summary,
        collection_names=[],
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_cs_coverage_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    collection_names = combined_data["collection_names"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_coverage_trace_chart(
        beta_summary,
        collection_names=collection_names,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_agg_cs_coverage_trace(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS beta trace data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data,
        method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    return viz_utils.render_cs_coverage_trace_chart(
        beta_summary,
        collection_names=[],
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )


def _make_agg_cs_size_power(combined_data: dict, settings: dict) -> plt.Figure:
    cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    method_meta = combined_data["method_metadata"]
    min_beta = settings.get("cs_beta", 0.95)
    max_cs_size = settings.get("max_cs_size", 10000)
    min_log_bf = settings.get("min_log_bf", 2.0)
    fg = _foreground_methods(method_meta, settings)
    if cs_data.is_empty():
        return viz_utils.make_placeholder_chart("No CS data")
    beta_summary = viz_utils.make_cs_beta_trace_summary(
        cs_data, method_meta,
        selected_methods=fg,
        selected_thresholds=_selected_thresholds(settings),
        max_cs_size=max_cs_size,
        min_ser_log_bf=min_log_bf,
    )
    nominal = beta_summary.filter(pl.col("beta") == round(min_beta, 2))
    calibrated = viz_utils.find_calibrated_beta_summary(beta_summary, cs_data, method_meta, selected_methods=fg, selected_thresholds=_selected_thresholds(settings), target_coverage=min_beta, min_ser_log_bf=min_log_bf)
    return viz_utils.render_cs_size_power_chart(
        nominal,
        calibrated,
        collection_names=[],
        min_beta=min_beta,
        min_ser_log_bf=min_log_bf,
        max_cs_size=max_cs_size,
    )


_TG_FAMILIES = {"twogroup", "twogroup_oracle", "twogroup_oracle_init", "twogroup_scale_fam", "twogroup_loc_fam"}


def _tg_methods(method_meta: pl.DataFrame, settings: dict) -> set[str]:
    fg = _foreground_methods(method_meta, settings)
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


_PLOT_DISPATCH = {
    "pip_calibration": _make_pip_calibration,
    "power_fdp": _make_power_fdp,
    "causal_pip": _make_causal_pip,
    "causal_rank": _make_causal_rank,
    "mass_above_causal": _make_mass_above_causal,
    "cs_dot_summary": _make_cs_dot_summary,
    "cs_calibrated_dot": _make_cs_calibrated_dot,
    "cs_size_power": _make_cs_size_power,
    "cs_power_fdp": _make_cs_power_fdp,
    "cs_beta_trace": _make_cs_beta_trace,
    "cs_coverage_trace": _make_cs_coverage_trace,
    "agg_pip_calibration": _make_agg_pip_calibration,
    "agg_power_fdp": _make_agg_power_fdp,
    "agg_causal_pip": _make_agg_causal_pip,
    "agg_causal_rank": _make_agg_causal_rank,
    "agg_mass_above_causal": _make_agg_mass_above_causal,
    "agg_cs_power_fdp": _make_agg_cs_power_fdp,
    "agg_cs_beta_trace": _make_agg_cs_beta_trace,
    "agg_cs_coverage_trace": _make_agg_cs_coverage_trace,
    "agg_cs_size_power": _make_agg_cs_size_power,
    "agg_cs_calibrated_dot": _make_agg_cs_calibrated_dot,
    "preceding_posterior_mass_ecdf": _make_preceding_posterior_mass_ecdf,
    "agg_preceding_posterior_mass_ecdf": _make_agg_preceding_posterior_mass_ecdf,
    "f1_boxplot": _make_f1_boxplot,
    "f1_scatter": _make_f1_scatter,
    "f1_enrich_scatter": _make_f1_enrich_scatter,
}
