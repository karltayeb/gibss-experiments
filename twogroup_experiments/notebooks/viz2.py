import marimo

__generated_with = "0.19.8"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from pathlib import Path
    import yaml


@app.function
def method_color_map() -> dict[str, str]:
    return {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }


@app.function
def method_line_style_map() -> dict[str, str]:
    return {
        "logistic_threshold": "-",
        "cox_light_threshold": "-",
        "twogroup": "--",
        "twogroup_oracle": "--",
        "logistic_oracle": "--",
        "cox_heavy": "--",
    }


@app.function
def method_display_order() -> list[str]:
    return [
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
    ]


@app.function
def base_chart_theme() -> dict[str, object]:
    return {
        "width": 4.2,
        "height": 3.0,
    }


@app.function
def make_placeholder_chart(title: str):
    theme = base_chart_theme()
    fig, ax = plt.subplots(figsize=(theme["width"], theme["height"]))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


@app.function
def method_label_map() -> dict[str, str]:
    return {
        "logistic_threshold": "Logistic Threshold",
        "cox_light_threshold": "Cox Light Threshold",
        "twogroup": "Twogroup",
        "twogroup_oracle": "Twogroup Oracle",
        "logistic_oracle": "Logistic Oracle",
        "cox_heavy": "Cox Heavy",
    }


@app.function
def filter_thresholded_methods(
    plot_data: pl.DataFrame, selected_threshold: float
) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    return plot_data.filter(
        (
            (pl.col("method") == "logistic_threshold")
            & (pl.col("threshold") == selected_threshold)
        )
        | (
            (pl.col("method") == "cox_light_threshold")
            & (pl.col("threshold") == selected_threshold)
        )
        | (~pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"]))
    )


@app.function
def add_method_display_labels(
    plot_data: pl.DataFrame, selected_threshold: float
) -> pl.DataFrame:
    label_map = method_label_map()
    display_df = plot_data.with_columns(
        pl.col("method")
        .replace(label_map, default=pl.col("method"))
        .alias("method_display")
    )
    return display_df.with_columns(
        pl.when(pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"]))
        .then(
            pl.format(
                "{} @ {}", pl.col("method_display"), pl.lit(f"{selected_threshold:g}")
            )
        )
        .otherwise(pl.col("method_display"))
        .alias("series_label")
    )


@app.function
def filter_selected_methods(plot_data: pl.DataFrame, selected_methods: set[str]) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    return plot_data.filter(pl.col("method").is_in(list(selected_methods)))


@app.function
def ordered_method_display_labels() -> list[str]:
    label_map = method_label_map()
    return [
        label_map.get(method_name, method_name)
        for method_name in method_display_order()
    ]


@app.function
def empty_pip_threshold_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
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


@app.function
def empty_causal_pip_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
            "causal_pip": pl.Float64,
            "max_pip": pl.Float64,
            "batch_hash": pl.String,
            "batch_name": pl.String,
            "simulation_name": pl.String,
        }
    )


@app.function
def empty_cs_component_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
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


@app.function
def empty_cs_truth_plot_data() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "replicate": pl.Int64,
            "method": pl.String,
            "threshold": pl.Float64,
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


@app.function
def make_pip_calibration_summary(plot_data: pl.DataFrame) -> pl.DataFrame:
    calibration_df = plot_data.with_columns(
        (pl.col("pip_threshold") * 20)
        .floor()
        .clip(0, 19)
        .cast(pl.Int64)
        .alias("pip_bin_index")
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
        .with_columns(
            pl.when(pl.col("n_total") > 0)
            .then(pl.col("n_causal") / pl.col("n_total"))
            .otherwise(None)
            .alias("empirical_rate")
        )
        .sort("simulation_name", "batch_hash", "replicate", "series_label", "pip_mid")
    )


@app.function
def prepare_power_fdp_plot_data_frame(
    plot_data: pl.DataFrame,
    *,
    selected_threshold: float,
    include_logistic_oracle: bool,
    include_twogroup_oracle: bool,
    include_logistic_threshold: bool,
    include_cox_light_threshold: bool,
    include_twogroup: bool,
    include_cox_heavy: bool,
    show_background_threshold_traces: bool,
) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    label_map = method_label_map()
    method_filtered = plot_data.filter(
        ((pl.col("method") != "logistic_oracle") | pl.lit(include_logistic_oracle))
        & ((pl.col("method") != "twogroup_oracle") | pl.lit(include_twogroup_oracle))
        & ((pl.col("method") != "logistic_threshold") | pl.lit(include_logistic_threshold))
        & ((pl.col("method") != "cox_light_threshold") | pl.lit(include_cox_light_threshold))
        & ((pl.col("method") != "twogroup") | pl.lit(include_twogroup))
        & ((pl.col("method") != "cox_heavy") | pl.lit(include_cox_heavy))
    ).with_columns(
        pl.col("method")
        .replace(label_map, default=pl.col("method"))
        .alias("method_display"),
        pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"]).alias("is_thresholded"),
    ).with_columns(
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
        .then(pl.format("{} @ {}", pl.col("method_display"), pl.col("threshold")))
        .otherwise(pl.col("method_display"))
        .alias("trace_label"),
        pl.when(pl.col("is_selected_threshold"))
        .then(
            pl.when(pl.col("is_thresholded"))
            .then(pl.format("{} @ {}", pl.col("method_display"), pl.lit(f"{selected_threshold:g}")))
            .otherwise(pl.col("method_display"))
        )
        .otherwise(None)
        .alias("legend_label"),
    )


@app.function
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


@app.function
def make_causal_pip_summary(plot_data: pl.DataFrame) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    label_map = method_label_map()
    return (
        plot_data.with_columns(
            pl.col("method")
            .replace(label_map, default=pl.col("method"))
            .alias("method_display")
        )
        .group_by("simulation_name", "method", "method_display", "threshold")
        .agg(pl.col("causal_pip").mean().alias("mean_causal_pip"))
        .sort("simulation_name", "method_display", "threshold")
    )


@app.function
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
                return_dtype=pl.List(
                    pl.Struct({"beta": pl.Float64, "cs_size": pl.Int64})
                ),
            ).alias("beta_cs_pairs")
        )
        .explode("beta_cs_pairs")
        .unnest("beta_cs_pairs")
        .drop("betas", "cs_sizes", "ordered_pips")
    )


@app.function
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
                return_dtype=pl.List(
                    pl.Struct({"beta": pl.Float64, "covered": pl.Boolean})
                ),
            ).alias("beta_covered_pairs")
        )
        .drop("betas", "covered")
        .explode("beta_covered_pairs")
        .unnest("beta_covered_pairs")
    )


@app.function
def make_conditional_cs_summary(
    cs_component_plot_data: pl.DataFrame,
    cs_truth_plot_data: pl.DataFrame,
    *,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    join_threshold_sentinel = -999999.0
    component_beta_rows = explode_cs_component_beta_rows(cs_component_plot_data).with_columns(
        pl.col("threshold").fill_null(join_threshold_sentinel).alias("threshold_key")
    )
    truth_beta_rows = explode_cs_truth_beta_rows(cs_truth_plot_data).with_columns(
        pl.col("threshold").fill_null(join_threshold_sentinel).alias("threshold_key")
    )
    selected_component_rows = component_beta_rows.filter(
        pl.col("beta") == nominal_coverage
    )
    qualifying_components = selected_component_rows.filter(
        (pl.col("cs_size") <= max_cs_size) & (pl.col("ser_log_bf") >= min_ser_log_bf)
    )
    qualifying_component_flags = qualifying_components.select(
        "simulation_name",
        "batch_hash",
        "replicate",
        "method",
        "threshold",
        "threshold_key",
        "component",
        "cs_size",
    )
    truth_at_nominal_coverage = truth_beta_rows.filter(pl.col("beta") == nominal_coverage)
    truth_with_qualifying_components = (
        truth_at_nominal_coverage
        .join(
            qualifying_component_flags,
            on=[
                "simulation_name",
                "batch_hash",
                "replicate",
                "method",
                "threshold_key",
                "component",
            ],
            how="inner",
        )
    )
    coverage_components = (
        truth_with_qualifying_components.group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "component",
        )
        .agg(pl.col("covered").any().cast(pl.Float64).alias("coverage"))
    )
    size_summary = qualifying_components.group_by(
        "simulation_name",
        "method",
        "threshold",
    ).agg(pl.col("cs_size").mean().alias("value")).with_columns(
        pl.lit("CS Size").alias("metric")
    )
    total_causal_counts = (
        truth_at_nominal_coverage.select(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "threshold_key",
            "causal_variable",
        )
        .unique()
        .group_by("simulation_name", "method", "threshold", "threshold_key")
        .agg(pl.len().alias("n_total_causal_variants"))
    )
    discovered_causal_variants = (
        truth_with_qualifying_components.group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "threshold_key",
            "causal_variable",
        )
        .agg(pl.col("covered").any().alias("discovered"))
        .filter(pl.col("discovered"))
        .group_by("simulation_name", "method", "threshold", "threshold_key")
        .agg(pl.len().alias("n_discovered_causal_variants"))
    )
    power_summary = (
        total_causal_counts.join(
            discovered_causal_variants,
            on=["simulation_name", "method", "threshold_key"],
            how="left",
        )
        .with_columns(pl.col("n_discovered_causal_variants").fill_null(0))
        .with_columns(
            (
                pl.col("n_discovered_causal_variants")
                / pl.col("n_total_causal_variants")
            ).alias("value")
        )
        .select(
            "simulation_name",
            "method",
            "threshold",
            "value",
        )
        .with_columns(pl.lit("Power").alias("metric"))
    )
    coverage_summary = coverage_components.group_by(
        "simulation_name",
        "method",
        "threshold",
    ).agg(pl.col("coverage").mean().alias("value")).with_columns(
        pl.lit("Coverage").alias("metric")
    )
    by_simulation_summary = pl.concat(
        [power_summary, coverage_summary, size_summary],
        how="diagonal_relaxed",
    )
    aggregate_summary = (
        by_simulation_summary.group_by("method", "threshold", "metric")
        .agg(pl.col("value").mean().alias("value"))
        .with_columns(pl.lit("Aggregate").alias("simulation_name"))
    )
    return (
        aggregate_summary.sort("metric", "method", "threshold"),
        by_simulation_summary.sort("simulation_name", "metric", "method", "threshold"),
    )


@app.function
def make_conditional_cs_replicate_summary(
    cs_component_plot_data: pl.DataFrame,
    cs_truth_plot_data: pl.DataFrame,
    *,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
) -> pl.DataFrame:
    join_threshold_sentinel = -999999.0
    component_beta_rows = explode_cs_component_beta_rows(cs_component_plot_data).with_columns(
        pl.col("threshold").fill_null(join_threshold_sentinel).alias("threshold_key")
    )
    truth_beta_rows = explode_cs_truth_beta_rows(cs_truth_plot_data).with_columns(
        pl.col("threshold").fill_null(join_threshold_sentinel).alias("threshold_key")
    )
    selected_component_rows = component_beta_rows.filter(pl.col("beta") == nominal_coverage)
    qualifying_components = selected_component_rows.filter(
        (pl.col("cs_size") <= max_cs_size) & (pl.col("ser_log_bf") >= min_ser_log_bf)
    )
    qualifying_component_flags = qualifying_components.select(
        "simulation_name",
        "batch_hash",
        "replicate",
        "method",
        "threshold",
        "threshold_key",
        "component",
        "cs_size",
    )
    truth_at_nominal_coverage = truth_beta_rows.filter(pl.col("beta") == nominal_coverage)
    truth_with_qualifying_components = truth_at_nominal_coverage.join(
        qualifying_component_flags,
        on=[
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold_key",
            "component",
        ],
        how="inner",
    )
    size_summary = qualifying_components.group_by(
        "simulation_name",
        "batch_hash",
        "replicate",
        "method",
        "threshold",
    ).agg(pl.col("cs_size").mean().alias("value")).with_columns(
        pl.lit("CS Size").alias("metric")
    )
    total_causal_counts = (
        truth_at_nominal_coverage.select(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "threshold_key",
            "causal_variable",
        )
        .unique()
        .group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "threshold_key",
        )
        .agg(pl.len().alias("n_total_causal_variants"))
    )
    discovered_causal_variants = (
        truth_with_qualifying_components.group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "threshold_key",
            "causal_variable",
        )
        .agg(pl.col("covered").any().alias("discovered"))
        .filter(pl.col("discovered"))
        .group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "threshold_key",
        )
        .agg(pl.len().alias("n_discovered_causal_variants"))
    )
    power_summary = (
        total_causal_counts.join(
            discovered_causal_variants,
            on=[
                "simulation_name",
                "batch_hash",
                "replicate",
                "method",
                "threshold_key",
            ],
            how="left",
        )
        .with_columns(pl.col("n_discovered_causal_variants").fill_null(0))
        .with_columns(
            (
                pl.col("n_discovered_causal_variants")
                / pl.col("n_total_causal_variants")
            ).alias("value")
        )
        .select(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "value",
        )
        .with_columns(pl.lit("Power").alias("metric"))
    )
    coverage_summary = (
        truth_with_qualifying_components.group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
            "component",
        )
        .agg(pl.col("covered").any().cast(pl.Float64).alias("component_covered"))
        .group_by(
            "simulation_name",
            "batch_hash",
            "replicate",
            "method",
            "threshold",
        )
        .agg(pl.col("component_covered").mean().alias("value"))
        .with_columns(pl.lit("Coverage").alias("metric"))
    )
    return pl.concat(
        [power_summary, size_summary, coverage_summary],
        how="diagonal_relaxed",
    ).sort("simulation_name", "replicate", "metric", "method", "threshold")


@app.function
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
    grouped = (
        replicate_metric_summary.group_by(*group_cols)
        .agg(pl.col("value").alias("value_samples"))
        .sort(*group_cols)
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for row in grouped.iter_rows(named=True):
        values = np.asarray(row["value_samples"], dtype=float)
        mean_value = float(np.mean(values)) if len(values) else np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        if len(values):
            bootstrap_idx = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
            bootstrap_means = values[bootstrap_idx].mean(axis=1)
            ci_lower = float(np.quantile(bootstrap_means, 0.025))
            ci_upper = float(np.quantile(bootstrap_means, 0.975))
        record = {column: row[column] for column in group_cols}
        record.update({"value": mean_value, "ci_lower": ci_lower, "ci_upper": ci_upper})
        records.append(record)
    return pl.DataFrame(records).sort(*group_cols)


@app.function
def select_current_threshold_cs_rows(
    plot_data: pl.DataFrame, *, selected_threshold: float
) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    return plot_data.filter(
        (
            pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"])
            & (pl.col("threshold") == selected_threshold)
        )
        | (~pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"]))
    )


@app.function
def render_conditional_cs_scenario_points_chart(
    scenario_summary: pl.DataFrame,
    *,
    selected_threshold: float,
):
    if scenario_summary.is_empty():
        return make_placeholder_chart("No scenario conditional CS summary data")
    theme = base_chart_theme()
    color_map = method_color_map()
    label_map = method_label_map()
    metrics = ["Power", "CS Size", "Coverage"]
    method_names = [
        method_name
        for method_name in method_display_order()
        if method_name in scenario_summary.get_column("method").unique().to_list()
    ]
    simulation_names = scenario_summary.get_column("simulation_name").unique().sort().to_list()
    x_positions = {simulation_name: idx for idx, simulation_name in enumerate(simulation_names)}
    if len(method_names) > 1:
        jitter_offsets = np.linspace(-0.28, 0.28, len(method_names))
    else:
        jitter_offsets = np.array([0.0])
    jitter_map = {method_name: float(offset) for method_name, offset in zip(method_names, jitter_offsets, strict=True)}
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(theme["width"] * len(metrics), theme["height"] * 1.35),
        squeeze=False,
        sharex=True,
    )
    for metric_index, metric_name in enumerate(metrics):
        ax = axes[0, metric_index]
        metric_df = scenario_summary.filter(pl.col("metric") == metric_name)
        y_min = None
        y_max = None
        for method_name in method_names:
            method_df = metric_df.filter(pl.col("method") == method_name).sort("simulation_name")
            if method_df.is_empty():
                continue
            x = np.array([x_positions[name] + jitter_map[method_name] for name in method_df["simulation_name"].to_list()])
            y = method_df["value"].to_numpy()
            lower = method_df["ci_lower"].to_numpy()
            upper = method_df["ci_upper"].to_numpy()
            yerr = np.vstack([
                np.clip(y - lower, a_min=0.0, a_max=None),
                np.clip(upper - y, a_min=0.0, a_max=None),
            ])
            method_label = label_map.get(method_name, method_name)
            if method_name in {"logistic_threshold", "cox_light_threshold"}:
                method_label = f"{method_label} @ {selected_threshold:g}"
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                color=color_map[method_name],
                ecolor=color_map[method_name],
                elinewidth=1.2,
                capsize=2,
                markersize=5,
                label=method_label,
            )
            y_min = np.nanmin(y) if y_min is None else min(y_min, float(np.nanmin(y)))
            y_max = np.nanmax(y) if y_max is None else max(y_max, float(np.nanmax(y)))
        ax.set_title(metric_name)
        ax.set_xticks(range(len(simulation_names)), simulation_names, rotation=25, ha="right")
        ax.set_xlabel("Simulation scenario")
        ax.set_box_aspect(0.95)
        if metric_name == "Coverage":
            ax.set_ylim(-0.02, 1.02)
        elif y_min is not None and y_max is not None:
            pad = max(0.05, 0.08 * max(y_max - y_min, 0.2))
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
        if metric_index == 0:
            ax.set_ylabel("Value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
        fig.tight_layout(rect=(0, 0, 0.88, 1), w_pad=0.4)
    else:
        fig.tight_layout()
    return fig


@app.function
def select_current_threshold_method_rows(
    plot_data: pl.DataFrame, *, selected_threshold: float
) -> pl.DataFrame:
    if plot_data.is_empty():
        return plot_data
    return plot_data.filter(
        (
            pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"])
            & (pl.col("threshold") == selected_threshold)
        )
        | (~pl.col("method").is_in(["logistic_threshold", "cox_light_threshold"]))
    )


@app.function
def prepare_cs_histogram_data(
    cs_component_plot_data: pl.DataFrame,
    *,
    nominal_coverage: float,
    selected_threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    component_beta_rows = explode_cs_component_beta_rows(cs_component_plot_data)
    selected_threshold_rows = select_current_threshold_method_rows(
        component_beta_rows,
        selected_threshold=selected_threshold,
    )
    cs_size_histogram_data = selected_threshold_rows.filter(
        pl.col("beta") == nominal_coverage
    ).select("method", "simulation_name", "threshold", "cs_size")
    ser_log_bf_histogram_data = selected_threshold_rows.select(
        "method", "simulation_name", "threshold", "ser_log_bf"
    )
    return cs_size_histogram_data, ser_log_bf_histogram_data


@app.function
def summarize_calibration_with_bootstrap(
    calibration_summary: pl.DataFrame,
    group_cols: list[str],
    *,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pl.DataFrame:
    if calibration_summary.is_empty():
        return pl.DataFrame(
            schema={
                **{column: calibration_summary.schema[column] for column in group_cols},
                "empirical_rate": pl.Float64,
                "ci_lower": pl.Float64,
                "ci_upper": pl.Float64,
            }
        )
    grouped = (
        calibration_summary.group_by(
            *group_cols,
            "pip_bin_index",
            "pip_left",
            "pip_right",
            "pip_mid",
        )
        .agg(
            pl.col("n_total").alias("n_total_values"),
            pl.col("n_causal").alias("n_causal_values"),
        )
        .sort(*group_cols, "pip_mid")
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for row in grouped.iter_rows(named=True):
        totals = np.asarray(row["n_total_values"], dtype=float)
        causals = np.asarray(row["n_causal_values"], dtype=float)
        total_sum = totals.sum()
        empirical_rate = float(causals.sum() / total_sum) if total_sum > 0 else np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        if len(totals) > 0:
            bootstrap_idx = rng.integers(0, len(totals), size=(n_bootstrap, len(totals)))
            bootstrap_totals = totals[bootstrap_idx].sum(axis=1)
            bootstrap_causals = causals[bootstrap_idx].sum(axis=1)
            valid = bootstrap_totals > 0
            if valid.any():
                bootstrap_rates = bootstrap_causals[valid] / bootstrap_totals[valid]
                ci_lower = float(np.quantile(bootstrap_rates, 0.025))
                ci_upper = float(np.quantile(bootstrap_rates, 0.975))
        record = {column: row[column] for column in group_cols}
        record.update(
            {
                "pip_bin_index": row["pip_bin_index"],
                "pip_left": row["pip_left"],
                "pip_right": row["pip_right"],
                "pip_mid": row["pip_mid"],
                "empirical_rate": empirical_rate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
        records.append(record)
    return pl.DataFrame(records).sort(*group_cols, "pip_mid")


@app.function
def render_calibration_chart_from_summary(
    calibration_summary: pl.DataFrame, *, facet_by_simulation: bool
):
    if calibration_summary.is_empty():
        return make_placeholder_chart("No PIP calibration data")
    theme = base_chart_theme()
    color_map = method_color_map()
    label_map = method_label_map()
    ordered_methods = [
        label_map.get(method_name, method_name) for method_name in method_display_order()
    ]

    if facet_by_simulation:
        simulation_names = calibration_summary.get_column("simulation_name").unique().sort().to_list()
        method_labels = [label for label in ordered_methods if label in calibration_summary.get_column("method_display").unique().to_list()]
        fig, axes = plt.subplots(
            len(simulation_names),
            len(method_labels),
            figsize=(theme["width"] * len(method_labels), theme["height"] * len(simulation_names)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row_index, simulation_name in enumerate(simulation_names):
            for col_index, method_label in enumerate(method_labels):
                ax = axes[row_index, col_index]
                panel_df = calibration_summary.filter(
                    (pl.col("simulation_name") == simulation_name)
                    & (pl.col("method_display") == method_label)
                ).sort("pip_mid")
                method_name = next(
                    name for name in method_display_order() if label_map.get(name, name) == method_label
                )
                ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle=":", linewidth=1.0, zorder=1)
                if not panel_df.is_empty():
                    x = panel_df["pip_mid"].to_numpy()
                    y = panel_df["empirical_rate"].to_numpy()
                    lower = panel_df["ci_lower"].to_numpy()
                    upper = panel_df["ci_upper"].to_numpy()
                    yerr = np.vstack([
                        np.clip(y - lower, a_min=0.0, a_max=None),
                        np.clip(upper - y, a_min=0.0, a_max=None),
                    ])
                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt="o",
                        color=color_map[method_name],
                        ecolor=color_map[method_name],
                        elinewidth=1.0,
                        capsize=2,
                        markersize=4,
                        zorder=2,
                    )
                if row_index == 0:
                    ax.set_title(method_label)
                if col_index == 0:
                    ax.set_ylabel(f"{simulation_name}\nEmpirical causal frequency")
                else:
                    ax.set_ylabel("")
                if row_index == len(simulation_names) - 1:
                    ax.set_xlabel("PIP bin midpoint")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                ax.set_box_aspect(1)
        fig.subplots_adjust(wspace=0.08, hspace=0.18)
        return fig

    method_labels = [label for label in ordered_methods if label in calibration_summary.get_column("method_display").unique().to_list()]
    fig, axes = plt.subplots(
        1,
        len(method_labels),
        figsize=(theme["width"] * len(method_labels), theme["height"]),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for col_index, method_label in enumerate(method_labels):
        ax = axes[0, col_index]
        panel_df = calibration_summary.filter(pl.col("method_display") == method_label).sort("pip_mid")
        method_name = next(
            name for name in method_display_order() if label_map.get(name, name) == method_label
        )
        ax.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle=":", linewidth=1.0, zorder=1)
        if not panel_df.is_empty():
            x = panel_df["pip_mid"].to_numpy()
            y = panel_df["empirical_rate"].to_numpy()
            lower = panel_df["ci_lower"].to_numpy()
            upper = panel_df["ci_upper"].to_numpy()
            yerr = np.vstack([
                np.clip(y - lower, a_min=0.0, a_max=None),
                np.clip(upper - y, a_min=0.0, a_max=None),
            ])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                color=color_map[method_name],
                ecolor=color_map[method_name],
                elinewidth=1.0,
                capsize=2,
                markersize=4,
                zorder=2,
            )
        ax.set_title(method_label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("PIP bin midpoint")
        if col_index == 0:
            ax.set_ylabel("Empirical causal frequency")
        ax.set_box_aspect(1)
    fig.subplots_adjust(wspace=0.08)
    return fig


@app.function
def render_power_fdp_chart_from_summary(
    power_fdp_summary: pl.DataFrame, *, facet: bool, max_fdp: float, fixed_y_scale: bool
):
    if power_fdp_summary.is_empty():
        return make_placeholder_chart("No power/FDP data")
    theme = base_chart_theme()
    color_map = method_color_map()
    line_style_map = method_line_style_map()
    label_map = method_label_map()
    x_max = min(1.0, max_fdp + 0.01)
    visible_power_fdp_summary = power_fdp_summary.filter(pl.col("fdp") <= x_max)
    def _compute_y_limits(panel_df: pl.DataFrame) -> tuple[float, float]:
        y_values = (
            panel_df.filter(pl.col("fdp") <= x_max).get_column("power").to_numpy()
            if not panel_df.filter(pl.col("fdp") <= x_max).is_empty()
            else panel_df.get_column("power").to_numpy()
        )
        if fixed_y_scale:
            return 0.0, 1.02
        y_min_data = float(np.nanmin(y_values)) if len(y_values) else 0.0
        y_max_data = float(np.nanmax(y_values)) if len(y_values) else 1.0
        y_padding = max(0.02, 0.05 * max(y_max_data - y_min_data, 0.1))
        return max(0.0, y_min_data - y_padding), min(1.02, y_max_data + y_padding)
    series_metadata = (
        power_fdp_summary.select(
            "method",
            "method_display",
            "trace_label",
            "legend_label",
            "is_selected_threshold",
        )
        .unique()
        .with_columns(
            pl.col("method")
            .replace(
                {method_name: index for index, method_name in enumerate(method_display_order())},
                default=len(method_display_order()),
            )
            .alias("method_order")
        )
        .sort("method_order", pl.col("is_selected_threshold").cast(pl.Int8), "trace_label", descending=[False, True, False])
    )
    if facet:
        simulation_names = (
            power_fdp_summary.get_column("simulation_name").unique().sort().to_list()
        )
        n_panels = len(simulation_names)
        ncols = min(2, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=((theme["width"] * 1.35) * ncols, theme["height"] * nrows),
            squeeze=False,
            sharex=True,
            sharey=fixed_y_scale,
        )
        axes_flat = axes.flatten()
        for ax, simulation_name in zip(axes_flat, simulation_names):
            simulation_df = power_fdp_summary.filter(
                pl.col("simulation_name") == simulation_name
            )
            y_min, y_max = _compute_y_limits(simulation_df)
            for series_row in series_metadata.iter_rows(named=True):
                method_name = series_row["method"]
                trace_label = series_row["trace_label"]
                legend_label = series_row["legend_label"]
                is_selected_threshold = series_row["is_selected_threshold"]
                method_df = simulation_df.filter(
                    (pl.col("method") == method_name)
                    & (pl.col("trace_label") == trace_label)
                ).sort("pip_threshold")
                if method_df.is_empty():
                    continue
                ax.plot(
                    method_df["fdp"].to_numpy(),
                    method_df["power"].to_numpy(),
                    label=legend_label if legend_label is not None else None,
                    color=color_map[method_name],
                    linestyle=line_style_map[method_name],
                    linewidth=2.0 if is_selected_threshold else 1.0,
                    alpha=1.0 if is_selected_threshold else 0.2,
                )
            ax.set_title(simulation_name)
            ax.set_xlim(0, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("FDP")
            ax.set_ylabel("Power")
            ax.set_box_aspect(0.72)
        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", frameon=False)
            fig.tight_layout(rect=(0, 0, 0.88, 1))
        else:
            fig.tight_layout()
        return fig
    fig, ax = plt.subplots(figsize=(theme["width"] * 1.45, theme["height"]))
    y_min, y_max = _compute_y_limits(power_fdp_summary)
    for series_row in series_metadata.iter_rows(named=True):
        method_name = series_row["method"]
        trace_label = series_row["trace_label"]
        legend_label = series_row["legend_label"]
        is_selected_threshold = series_row["is_selected_threshold"]
        method_df = power_fdp_summary.filter(
            (pl.col("method") == method_name)
            & (pl.col("trace_label") == trace_label)
        ).sort("pip_threshold")
        if method_df.is_empty():
            continue
        ax.plot(
            method_df["fdp"].to_numpy(),
            method_df["power"].to_numpy(),
            label=legend_label if legend_label is not None else None,
            color=color_map[method_name],
            linestyle=line_style_map[method_name],
            linewidth=2.0 if is_selected_threshold else 1.0,
            alpha=1.0 if is_selected_threshold else 0.2,
        )
    ax.set_title("Power vs FDP")
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("FDP")
    ax.set_ylabel("Power")
    ax.set_box_aspect(0.72)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig


@app.function
def render_causal_pip_chart_from_summary(
    causal_pip_summary: pl.DataFrame, *, facet: bool
):
    if causal_pip_summary.is_empty():
        return make_placeholder_chart("No causal PIP data")
    theme = base_chart_theme()
    color_map = method_color_map()
    line_style_map = method_line_style_map()
    label_map = method_label_map()
    thresholded_methods = ["logistic_threshold", "cox_light_threshold"]
    threshold_values = (
        causal_pip_summary.filter(pl.col("method").is_in(thresholded_methods))
        .get_column("threshold")
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    if threshold_values:
        x_min = float(min(threshold_values))
        x_max = float(max(threshold_values))
    else:
        x_min, x_max = 0.0, 4.0

    def _draw(ax, panel_df: pl.DataFrame) -> None:
        for method_name in method_display_order():
            method_label = label_map.get(method_name, method_name)
            method_df = panel_df.filter(pl.col("method") == method_name).sort("threshold")
            if method_df.is_empty():
                continue
            if method_name in thresholded_methods:
                ax.plot(
                    method_df["threshold"].to_numpy(),
                    method_df["mean_causal_pip"].to_numpy(),
                    color=color_map[method_name],
                    linestyle=line_style_map[method_name],
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    label=method_label,
                )
            else:
                y_value = float(method_df["mean_causal_pip"][0])
                ax.hlines(
                    y_value,
                    x_min,
                    x_max,
                    color=color_map[method_name],
                    linestyle=line_style_map[method_name],
                    linewidth=2.0,
                    label=method_label,
                )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Mean causal PIP")

    if facet:
        simulation_names = (
            causal_pip_summary.get_column("simulation_name").unique().sort().to_list()
        )
        n_panels = len(simulation_names)
        ncols = min(2, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=((theme["width"] * 1.25) * ncols, theme["height"] * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.flatten()
        for ax, simulation_name in zip(axes_flat, simulation_names):
            panel_df = causal_pip_summary.filter(pl.col("simulation_name") == simulation_name)
            _draw(ax, panel_df)
            ax.set_title(simulation_name)
            ax.set_box_aspect(0.8)
        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", frameon=False)
            fig.tight_layout(rect=(0, 0, 0.88, 1))
        else:
            fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(theme["width"] * 1.6, theme["height"]))
    _draw(ax, causal_pip_summary)
    ax.set_title("Causal PIP vs Threshold")
    ax.set_box_aspect(0.75)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    return fig


@app.function
def render_conditional_cs_summary_chart(
    cs_summary: pl.DataFrame,
    *,
    facet: bool,
    nominal_coverage: float,
    max_cs_size: int,
    min_ser_log_bf: float,
):
    if cs_summary.is_empty():
        return make_placeholder_chart("No conditional CS summary data")
    theme = base_chart_theme()
    color_map = method_color_map()
    line_style_map = method_line_style_map()
    label_map = method_label_map()
    metrics = ["Power", "CS Size", "Coverage"]
    thresholded_methods = ["logistic_threshold", "cox_light_threshold"]
    threshold_values = (
        cs_summary.filter(pl.col("method").is_in(thresholded_methods))
        .get_column("threshold")
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    if threshold_values:
        x_min = float(min(threshold_values))
        x_max = float(max(threshold_values))
    else:
        x_min, x_max = 0.0, 4.0
    threshold_text = (
        f"Nominal coverage: {nominal_coverage:.2f}\n"
        f"Max CS size: {max_cs_size}\n"
        f"Min SER log BF: {min_ser_log_bf:.1f}"
    )

    def _draw_row(axes_row, panel_df: pl.DataFrame) -> None:
        for metric_index, metric_name in enumerate(metrics):
            ax = axes_row[metric_index]
            metric_df = panel_df.filter(pl.col("metric") == metric_name)
            metric_values: list[float] = []
            for method_name in method_display_order():
                method_label = label_map.get(method_name, method_name)
                method_metric_df = metric_df.filter(pl.col("method") == method_name).sort("threshold")
                if method_metric_df.is_empty():
                    continue
                metric_values.extend(method_metric_df["value"].to_list())
                if method_name in thresholded_methods:
                    ax.plot(
                        method_metric_df["threshold"].to_numpy(),
                        method_metric_df["value"].to_numpy(),
                        color=color_map[method_name],
                        linestyle=line_style_map[method_name],
                        linewidth=2.0,
                        marker="o",
                        markersize=4,
                        label=method_label,
                    )
                else:
                    y_value = float(method_metric_df["value"][0])
                    ax.hlines(
                        y_value,
                        x_min,
                        x_max,
                        color=color_map[method_name],
                        linestyle=line_style_map[method_name],
                        linewidth=2.0,
                        label=method_label,
                    )
                ax.set_title(metric_name)
                ax.set_xlim(x_min, x_max)
                ax.set_xlabel("Threshold")
                ax.set_box_aspect(1)
            if metric_name == "Coverage":
                ax.set_ylim(-0.02, 1.02)
            elif metric_name == "Power" and metric_values:
                metric_min = float(np.nanmin(metric_values))
                metric_max = float(np.nanmax(metric_values))
                metric_pad = max(0.02, 0.08 * max(metric_max - metric_min, 0.1))
                ax.set_ylim(
                    max(0.0, metric_min - metric_pad),
                    min(1.02, metric_max + metric_pad),
                )
            if metric_index == 0:
                ax.set_ylabel(metric_name)

    if facet:
        simulation_names = cs_summary.get_column("simulation_name").unique().sort().to_list()
        fig, axes = plt.subplots(
            len(simulation_names),
            len(metrics),
            figsize=(theme["width"] * len(metrics), theme["height"] * len(simulation_names)),
            squeeze=False,
            sharex=True,
        )
        for row_index, simulation_name in enumerate(simulation_names):
            panel_df = cs_summary.filter(pl.col("simulation_name") == simulation_name)
            _draw_row(axes[row_index], panel_df)
            axes[row_index, 0].set_ylabel(f"{simulation_name}\n{metrics[0]}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center right", frameon=False)
            fig.text(
                0.915,
                0.18,
                threshold_text,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8"},
            )
            fig.tight_layout(rect=(0, 0, 0.88, 1), w_pad=0.2)
        else:
            fig.tight_layout()
        return fig

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(theme["width"] * len(metrics), theme["height"]),
        squeeze=False,
        sharex=True,
    )
    _draw_row(axes[0], cs_summary)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
        fig.text(
            0.915,
            0.18,
            threshold_text,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8"},
        )
        fig.tight_layout(rect=(0, 0, 0.88, 1), w_pad=0.2)
    else:
        fig.tight_layout()
    return fig


@app.function
def render_cs_histograms(
    cs_size_histogram_data: pl.DataFrame,
    ser_log_bf_histogram_data: pl.DataFrame,
    *,
    selected_threshold: float,
    max_cs_size: int,
    min_ser_log_bf: float,
):
    if cs_size_histogram_data.is_empty() and ser_log_bf_histogram_data.is_empty():
        return make_placeholder_chart("No CS histogram data")
    theme = base_chart_theme()
    color_map = method_color_map()
    label_map = method_label_map()
    size_methods = (
        cs_size_histogram_data.get_column("method").unique().to_list()
        if not cs_size_histogram_data.is_empty()
        else []
    )
    bf_methods = (
        ser_log_bf_histogram_data.get_column("method").unique().to_list()
        if not ser_log_bf_histogram_data.is_empty()
        else []
    )
    method_names = [
        method_name
        for method_name in method_display_order()
        if method_name in size_methods or method_name in bf_methods
    ]
    ncols = max(1, len(method_names))
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(2.35 * ncols, 4.8),
        squeeze=False,
    )
    log_bf_left_edge = -2.5
    log_bf_left_clip = -1.75
    log_bf_right_edge = 10.5
    log_bf_right_clip = 10.25
    log_bf_bin_edges = np.concatenate(
        [
            np.array([log_bf_left_edge, -2.0]),
            np.linspace(-2.0, 10.0, 19)[1:-1],
            np.array([10.0, 10.5]),
        ]
    )

    def _draw_partitioned_hist(
        ax,
        values: np.ndarray,
        bins: np.ndarray,
        *,
        passing_mask: np.ndarray,
        color: str,
    ) -> None:
        failing_values = values[~passing_mask]
        passing_values = values[passing_mask]
        if failing_values.size:
            ax.hist(
                failing_values,
                bins=bins,
                histtype="bar",
                facecolor="none",
                edgecolor=color,
                linewidth=1.1,
            )
        if passing_values.size:
            ax.hist(
                passing_values,
                bins=bins,
                histtype="bar",
                color=color,
                alpha=0.85,
                edgecolor="white",
            )

    for col_index, method_name in enumerate(method_names):
        method_label = label_map.get(method_name, method_name)
        title = (
            f"{method_label} @ {selected_threshold:g}"
            if method_name in {"logistic_threshold", "cox_light_threshold"}
            else method_label
        )

        size_ax = axes[0, col_index]
        size_df = cs_size_histogram_data.filter(pl.col("method") == method_name)
        if not size_df.is_empty():
            size_values = size_df["cs_size"].to_numpy()
            max_size_value = int(np.max(size_values))
            size_upper = max(max_size_value, max_cs_size)
            size_bins = np.linspace(0.5, size_upper + 0.5, 21)
            _draw_partitioned_hist(
                size_ax,
                size_values,
                size_bins,
                passing_mask=size_values <= max_cs_size,
                color=color_map[method_name],
            )
            size_ax.set_xlim(
                0.5,
                size_upper + 0.5,
            )
            size_fraction_passing = float(np.mean(size_values <= max_cs_size))
            size_ax.text(
                0.98,
                0.95,
                f"pass: {size_fraction_passing:.2f}",
                transform=size_ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "pad": 0.2},
            )
        size_ax.axvline(
            max_cs_size,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        size_ax.set_title(title, fontsize=9)
        size_ax.set_xlabel("CS Size")
        if col_index == 0:
            size_ax.set_ylabel("Count")
        size_ax.tick_params(axis="both", labelsize=8)
        size_ax.set_box_aspect(1)

        bf_ax = axes[1, col_index]
        bf_df = ser_log_bf_histogram_data.filter(pl.col("method") == method_name)
        if not bf_df.is_empty():
            bf_raw_values = bf_df["ser_log_bf"].to_numpy()
            bf_values = np.clip(bf_raw_values, log_bf_left_clip, log_bf_right_clip)
            _draw_partitioned_hist(
                ax=bf_ax,
                values=bf_values,
                bins=log_bf_bin_edges,
                passing_mask=bf_raw_values >= min_ser_log_bf,
                color=color_map[method_name],
            )
            bf_fraction_passing = float(np.mean(bf_df["ser_log_bf"].to_numpy() >= min_ser_log_bf))
            bf_ax.text(
                0.98,
                0.95,
                f"pass: {bf_fraction_passing:.2f}",
                transform=bf_ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.8", "pad": 0.2},
            )
        bf_ax.axvline(
            min_ser_log_bf,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        bf_ax.set_xlabel("SER log BF")
        if col_index == 0:
            bf_ax.set_ylabel("Count")
        bf_ax.set_xlim(log_bf_left_edge, log_bf_right_edge)
        bf_ax.set_xticks(
            [log_bf_left_clip, 0, 2, 4, 6, 8, log_bf_right_clip],
            ["< -2", "0", "2", "4", "6", "8", "> 10"],
        )
        bf_ax.set_title(title, fontsize=9)
        bf_ax.tick_params(axis="both", labelsize=8)
        bf_ax.set_box_aspect(1)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.17, wspace=0.28, hspace=0.38)
    return fig


@app.cell
def select_collection():
    collection_alias_root = Path("results") / "twogroup_experiments" / "by_alias"
    collection_options = sorted(
        path.name for path in collection_alias_root.iterdir() if path.is_dir()
    )
    collections_with_pip_plot_data = [
        collection_name
        for collection_name in collection_options
        if list(
            (
                collection_alias_root / collection_name
            ).glob("batches/*/fits/*/pip_threshold_plot_data.parquet")
        )
    ]
    if "c4_ser_nonlocal" in collections_with_pip_plot_data:
        default_collection = "c4_ser_nonlocal"
    elif collections_with_pip_plot_data:
        default_collection = collections_with_pip_plot_data[0]
    else:
        default_collection = collection_options[0]
    collection_dropdown = mo.ui.dropdown(
        options=collection_options,
        value=default_collection,
        label="collection",
    )
    collection_dropdown
    return (collection_dropdown,)


@app.cell
def instantiate_threshold_control():
    threshold_control = mo.ui.slider(
        start=0.0,
        stop=4.0,
        step=0.25,
        value=2.0,
        label="threshold",
        show_value=True,
    )
    return (threshold_control,)


@app.cell
def instantiate_max_cs_size_control(cs_component_plot_data):
    if cs_component_plot_data.is_empty():
        max_cs_size_stop = 50
    else:
        max_cs_size_stop = int(
            cs_component_plot_data.select(pl.col("ordered_pips").list.len().max()).item()
        )
    max_cs_size_control = mo.ui.slider(
        start=1,
        stop=max_cs_size_stop,
        step=1,
        value=min(20, max_cs_size_stop),
        label="max cs size",
        show_value=True,
    )
    return (max_cs_size_control,)


@app.cell
def instantiate_min_ser_log_bf_control():
    min_ser_log_bf_control = mo.ui.slider(
        start=-2.0,
        stop=5.0,
        step=0.1,
        value=2.0,
        label="min ser log bf",
        show_value=True,
    )
    return (min_ser_log_bf_control,)


@app.cell
def instantiate_max_fdp_control():
    max_fdp_control = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.2,
        label="max fdp",
        show_value=True,
    )
    return (max_fdp_control,)


@app.cell(hide_code=True)
def instantiate_nominal_coverage_control():
    nominal_coverage_control = mo.ui.slider(
        start=0.50,
        stop=0.99,
        step=0.01,
        value=0.95,
        label="nominal coverage",
        show_value=True,
    )
    return (nominal_coverage_control,)


@app.cell(hide_code=True)
def instantiate_method_multiselect():
    method_options = {
        "logistic oracle": "logistic_oracle",
        "twogroup oracle": "twogroup_oracle",
        "logistic threshold": "logistic_threshold",
        "cox-light threshold": "cox_light_threshold",
        "twogroup": "twogroup",
        "cox heavy": "cox_heavy",
    }
    method_multiselect = mo.ui.multiselect(
        options=method_options,
        value=list(method_options.keys()),
        label="methods",
    )
    return method_multiselect, method_options


@app.cell(hide_code=True)
def instantiate_show_background_threshold_traces_control():
    show_background_threshold_traces_control = mo.ui.checkbox(
        value=False,
        label="show faint non-selected thresholds",
    )
    return (show_background_threshold_traces_control,)


@app.cell(hide_code=True)
def instantiate_fixed_y_scale_control():
    fixed_y_scale_control = mo.ui.checkbox(
        value=False,
        label="fixed [0, 1] y scale",
    )
    return (fixed_y_scale_control,)


@app.cell(hide_code=True)
def notebook_title():
    mo.md("# Twogroup Experiment Visualization v2")
    return


@app.cell(hide_code=True)
def notebook_overview():
    mo.md(
        "This notebook will read prepared plot data from the twogroup alias tree "
        "and render aggregate and by-simulation views for PIP and credible set summaries."
    )
    return


@app.cell
def load_collection_manifest(collection_dropdown):
    collection_root = (
        Path("results")
        / "twogroup_experiments"
        / "by_alias"
        / collection_dropdown.value
    )
    manifest_path = collection_root / "collection_spec.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    return collection_root, manifest


@app.cell
def load_pip_threshold_plot_data(collection_root, empty_pip_threshold_plot_data):
    pip_threshold_paths = sorted(
        collection_root.glob("batches/*/fits/*/pip_threshold_plot_data.parquet")
    )
    pip_threshold_plot_data = (
        pl.concat(
            [pl.read_parquet(path) for path in pip_threshold_paths],
            how="diagonal_relaxed",
        )
        if pip_threshold_paths
        else empty_pip_threshold_plot_data()
    )
    return pip_threshold_plot_data, pip_threshold_paths


@app.cell
def load_causal_pip_plot_data(collection_root, empty_causal_pip_plot_data):
    causal_pip_paths = sorted(
        collection_root.glob("batches/*/fits/*/causal_pip_plot_data.parquet")
    )
    causal_pip_plot_data = (
        pl.concat(
            [pl.read_parquet(path) for path in causal_pip_paths],
            how="diagonal_relaxed",
        )
        if causal_pip_paths
        else empty_causal_pip_plot_data()
    )
    return causal_pip_plot_data, causal_pip_paths


@app.cell
def load_cs_component_plot_data(collection_root, empty_cs_component_plot_data):
    cs_component_paths = sorted(
        collection_root.glob("batches/*/fits/*/cs_component_plot_data.parquet")
    )
    cs_component_plot_data = (
        pl.concat(
            [pl.read_parquet(path) for path in cs_component_paths],
            how="diagonal_relaxed",
        )
        if cs_component_paths
        else empty_cs_component_plot_data()
    )
    return cs_component_plot_data, cs_component_paths


@app.cell
def load_cs_truth_plot_data(collection_root, empty_cs_truth_plot_data):
    cs_truth_paths = sorted(
        collection_root.glob("batches/*/fits/*/cs_truth_plot_data.parquet")
    )
    cs_truth_plot_data = (
        pl.concat(
            [pl.read_parquet(path) for path in cs_truth_paths],
            how="diagonal_relaxed",
        )
        if cs_truth_paths
        else empty_cs_truth_plot_data()
    )
    return cs_truth_plot_data, cs_truth_paths


@app.cell(hide_code=True)
def pip_section_heading():
    mo.md("## PIP")
    return


@app.cell(hide_code=True)
def pip_calibration_heading():
    mo.md("### Calibration")
    return


@app.cell(hide_code=True)
def pip_calibration_controls_local(threshold_control, method_multiselect):
    mo.hstack([threshold_control, method_multiselect], justify="start")
    return


@app.cell
def prepare_pip_calibration_plot_data(
    pip_threshold_plot_data, threshold_control, method_multiselect
):
    _selected_methods = set(method_multiselect.value)
    threshold_filtered_pip_calibration_data = filter_thresholded_methods(
        pip_threshold_plot_data,
        threshold_control.value,
    )
    method_filtered_pip_calibration_data = filter_selected_methods(
        threshold_filtered_pip_calibration_data,
        _selected_methods,
    )
    labeled_pip_calibration_data = add_method_display_labels(
        method_filtered_pip_calibration_data,
        threshold_control.value,
    )
    pip_calibration_summary = make_pip_calibration_summary(labeled_pip_calibration_data)
    return (pip_calibration_summary,)


@app.cell
def render_pip_calibration_aggregate(pip_calibration_summary):
    aggregate_pip_calibration_summary = summarize_calibration_with_bootstrap(
        pip_calibration_summary,
        group_cols=["method_display", "series_label"],
    )
    pip_calibration_aggregate_chart = render_calibration_chart_from_summary(
        aggregate_pip_calibration_summary,
        facet_by_simulation=False,
    )
    return (pip_calibration_aggregate_chart,)


@app.cell
def render_pip_calibration_by_scenario(pip_calibration_summary):
    pip_calibration_by_scenario_summary = summarize_calibration_with_bootstrap(
        pip_calibration_summary,
        group_cols=["simulation_name", "method_display", "series_label"],
    )
    pip_calibration_by_scenario_chart = render_calibration_chart_from_summary(
        pip_calibration_by_scenario_summary,
        facet_by_simulation=True,
    )
    return (pip_calibration_by_scenario_chart,)


@app.cell
def display_pip_calibration_tabs(
    pip_calibration_aggregate_chart,
    pip_calibration_by_scenario_chart,
):
    mo.ui.tabs(
        {
            "Aggregate": pip_calibration_aggregate_chart,
            "By Simulation Scenario": pip_calibration_by_scenario_chart,
        }
    )
    return


@app.cell(hide_code=True)
def pip_fdr_heading():
    mo.md("### Power vs FDP")
    return


@app.cell(hide_code=True)
def pip_fdr_controls_local(
    max_fdp_control,
    threshold_control,
    method_multiselect,
    show_background_threshold_traces_control,
    fixed_y_scale_control,
):
    mo.vstack(
        [
            mo.hstack(
                [threshold_control, max_fdp_control],
                justify="start",
                gap=2,
            ),
            mo.hstack(
                [method_multiselect],
                justify="start",
                gap=2,
            ),
            mo.hstack(
                [show_background_threshold_traces_control, fixed_y_scale_control],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def prepare_power_fdp_plot_data(
    pip_threshold_plot_data,
    threshold_control,
    method_multiselect,
    show_background_threshold_traces_control,
):
    _selected_methods = set(method_multiselect.value)
    prepared_power_fdp_data = prepare_power_fdp_plot_data_frame(
        pip_threshold_plot_data,
        selected_threshold=threshold_control.value,
        include_logistic_oracle="logistic_oracle" in _selected_methods,
        include_twogroup_oracle="twogroup_oracle" in _selected_methods,
        include_logistic_threshold="logistic_threshold" in _selected_methods,
        include_cox_light_threshold="cox_light_threshold" in _selected_methods,
        include_twogroup="twogroup" in _selected_methods,
        include_cox_heavy="cox_heavy" in _selected_methods,
        show_background_threshold_traces=show_background_threshold_traces_control.value,
    )
    power_fdp_summary = make_power_fdp_summary(prepared_power_fdp_data)
    return (power_fdp_summary,)


@app.cell
def render_power_vs_fdp_aggregate(power_fdp_summary, max_fdp_control, fixed_y_scale_control):
    aggregate_power_fdp_summary = (
        power_fdp_summary.group_by(
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
        .with_columns(pl.lit("Aggregate").alias("simulation_name"))
        .sort("method_display", "trace_label", "pip_threshold")
    )
    power_vs_fdp_aggregate_chart = render_power_fdp_chart_from_summary(
        aggregate_power_fdp_summary,
        facet=False,
        max_fdp=max_fdp_control.value,
        fixed_y_scale=fixed_y_scale_control.value,
    )
    return (power_vs_fdp_aggregate_chart,)


@app.cell
def render_power_vs_fdp_by_scenario(power_fdp_summary, max_fdp_control, fixed_y_scale_control):
    power_vs_fdp_by_scenario_chart = render_power_fdp_chart_from_summary(
        power_fdp_summary,
        facet=True,
        max_fdp=max_fdp_control.value,
        fixed_y_scale=fixed_y_scale_control.value,
    )
    return (power_vs_fdp_by_scenario_chart,)


@app.cell
def display_power_vs_fdp_tabs(
    power_vs_fdp_aggregate_chart,
    power_vs_fdp_by_scenario_chart,
):
    mo.ui.tabs(
        {
            "Aggregate": power_vs_fdp_aggregate_chart,
            "By Simulation Scenario": power_vs_fdp_by_scenario_chart,
        }
    )
    return


@app.cell(hide_code=True)
def max_pip_heading():
    mo.md("### Causal PIP vs Threshold")
    return


@app.cell(hide_code=True)
def max_pip_controls_local(method_multiselect):
    mo.hstack([method_multiselect], justify="start")
    return


@app.cell
def render_threshold_vs_max_pip_aggregate(causal_pip_plot_data, method_multiselect):
    _selected_methods = set(method_multiselect.value)
    aggregate_causal_pip_summary = make_causal_pip_summary(
        filter_selected_methods(
            causal_pip_plot_data, _selected_methods
        ).with_columns(pl.lit("Aggregate").alias("simulation_name"))
    )
    threshold_vs_max_pip_aggregate_chart = render_causal_pip_chart_from_summary(
        aggregate_causal_pip_summary,
        facet=False,
    )
    return (threshold_vs_max_pip_aggregate_chart,)


@app.cell
def render_threshold_vs_max_pip_by_scenario(causal_pip_plot_data, method_multiselect):
    _selected_methods = set(method_multiselect.value)
    causal_pip_by_scenario_summary = make_causal_pip_summary(
        filter_selected_methods(causal_pip_plot_data, _selected_methods)
    )
    threshold_vs_max_pip_by_scenario_chart = render_causal_pip_chart_from_summary(
        causal_pip_by_scenario_summary,
        facet=True,
    )
    return (threshold_vs_max_pip_by_scenario_chart,)


@app.cell
def display_threshold_vs_max_pip_tabs(
    threshold_vs_max_pip_aggregate_chart,
    threshold_vs_max_pip_by_scenario_chart,
):
    mo.ui.tabs(
        {
            "Aggregate": threshold_vs_max_pip_aggregate_chart,
            "By Simulation Scenario": threshold_vs_max_pip_by_scenario_chart,
        }
    )
    return


@app.cell(hide_code=True)
def cs_section_heading():
    mo.md("## Credible Sets")
    return


@app.cell(hide_code=True)
def unconditional_cs_heading():
    mo.md("### Unconditional Coverage")
    return


@app.cell(hide_code=True)
def unconditional_cs_controls_local(threshold_control, method_multiselect):
    mo.vstack(
        [
            mo.hstack([threshold_control], justify="start", gap=2),
            mo.hstack([method_multiselect], justify="start", gap=2),
        ],
        gap=1,
    )
    return


@app.cell
def render_unconditional_cs_aggregate():
    unconditional_cs_aggregate_chart = make_placeholder_chart(
        "Aggregate unconditional CS coverage"
    )
    return (unconditional_cs_aggregate_chart,)


@app.cell
def render_unconditional_cs_by_scenario():
    unconditional_cs_by_scenario_chart = make_placeholder_chart(
        "Unconditional CS coverage by simulation scenario"
    )
    return (unconditional_cs_by_scenario_chart,)


@app.cell
def display_unconditional_cs_tabs(
    unconditional_cs_aggregate_chart,
    unconditional_cs_by_scenario_chart,
):
    mo.ui.tabs(
        {
            "Aggregate": unconditional_cs_aggregate_chart,
            "By Simulation Scenario": unconditional_cs_by_scenario_chart,
        }
    )
    return


@app.cell(hide_code=True)
def cs_summary_heading():
    mo.md("### Power, Size, Coverage")
    return


@app.cell(hide_code=True)
def cs_summary_controls_local(
    nominal_coverage_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    mo.vstack(
        [
            mo.hstack(
                [nominal_coverage_control, max_cs_size_control, min_ser_log_bf_control],
                justify="start",
                gap=2,
            ),
            mo.hstack([method_multiselect], justify="start", gap=2),
        ],
        gap=1,
    )
    return


@app.cell
def render_cs_summary_aggregate(
    cs_component_plot_data,
    cs_truth_plot_data,
    nominal_coverage_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    _selected_methods = set(method_multiselect.value)
    aggregate_cs_summary, _ = make_conditional_cs_summary(
        filter_selected_methods(cs_component_plot_data, _selected_methods),
        filter_selected_methods(cs_truth_plot_data, _selected_methods),
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    cs_summary_aggregate_chart = render_conditional_cs_summary_chart(
        aggregate_cs_summary,
        facet=False,
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    return (cs_summary_aggregate_chart,)


@app.cell
def render_cs_summary_by_scenario(
    cs_component_plot_data,
    cs_truth_plot_data,
    nominal_coverage_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    _selected_methods = set(method_multiselect.value)
    _, by_simulation_cs_summary = make_conditional_cs_summary(
        filter_selected_methods(cs_component_plot_data, _selected_methods),
        filter_selected_methods(cs_truth_plot_data, _selected_methods),
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    cs_summary_by_scenario_chart = render_conditional_cs_summary_chart(
        by_simulation_cs_summary,
        facet=True,
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    return (cs_summary_by_scenario_chart,)


@app.cell
def display_cs_summary_tabs(
    cs_summary_aggregate_chart,
    cs_summary_by_scenario_chart,
):
    mo.ui.tabs(
        {
            "Aggregate": cs_summary_aggregate_chart,
            "By Simulation Scenario": cs_summary_by_scenario_chart,
        }
    )
    return


@app.cell(hide_code=True)
def cs_summary_scenario_points_heading():
    mo.md("### Power, Size, Coverage by Scenario at Selected Threshold")
    return


@app.cell(hide_code=True)
def cs_summary_scenario_points_controls_local(
    nominal_coverage_control,
    threshold_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    mo.vstack(
        [
            mo.hstack(
                [
                    nominal_coverage_control,
                    threshold_control,
                    max_cs_size_control,
                    min_ser_log_bf_control,
                ],
                justify="start",
                gap=2,
            ),
            mo.hstack([method_multiselect], justify="start", gap=2),
        ],
        gap=1,
    )
    return


@app.cell
def render_cs_summary_scenario_points(
    cs_component_plot_data,
    cs_truth_plot_data,
    nominal_coverage_control,
    threshold_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    _selected_methods = set(method_multiselect.value)
    replicate_cs_summary = make_conditional_cs_replicate_summary(
        filter_selected_methods(cs_component_plot_data, _selected_methods),
        filter_selected_methods(cs_truth_plot_data, _selected_methods),
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    selected_threshold_cs_summary = select_current_threshold_cs_rows(
        replicate_cs_summary,
        selected_threshold=threshold_control.value,
    )
    scenario_points_summary = summarize_replicate_metric_with_bootstrap(
        selected_threshold_cs_summary,
        group_cols=["simulation_name", "method", "metric"],
    )
    cs_summary_scenario_points_chart = render_conditional_cs_scenario_points_chart(
        scenario_points_summary,
        selected_threshold=threshold_control.value,
    )
    return (cs_summary_scenario_points_chart,)


@app.cell
def display_cs_summary_scenario_points(cs_summary_scenario_points_chart):
    cs_summary_scenario_points_chart
    return


@app.cell(hide_code=True)
def cs_histogram_heading():
    mo.md("### CS Size and SER log BF Histograms")
    return


@app.cell(hide_code=True)
def cs_histogram_controls_local(
    nominal_coverage_control,
    threshold_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    mo.vstack(
        [
            mo.hstack(
                [
                    nominal_coverage_control,
                    threshold_control,
                    max_cs_size_control,
                    min_ser_log_bf_control,
                ],
                justify="start",
                gap=2,
            ),
            mo.hstack([method_multiselect], justify="start", gap=2),
        ],
        gap=1,
    )
    return


@app.cell
def render_cs_histograms_panel(
    cs_component_plot_data,
    nominal_coverage_control,
    threshold_control,
    max_cs_size_control,
    min_ser_log_bf_control,
    method_multiselect,
):
    _selected_methods = set(method_multiselect.value)
    cs_size_histogram_data, ser_log_bf_histogram_data = prepare_cs_histogram_data(
        filter_selected_methods(cs_component_plot_data, _selected_methods),
        nominal_coverage=nominal_coverage_control.value,
        selected_threshold=threshold_control.value,
    )
    cs_histograms_figure = render_cs_histograms(
        cs_size_histogram_data,
        ser_log_bf_histogram_data,
        selected_threshold=threshold_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    return (cs_histograms_figure,)


@app.cell
def display_cs_histograms(cs_histograms_figure):
    cs_histograms_figure
    return


if __name__ == "__main__":
    app.run()
