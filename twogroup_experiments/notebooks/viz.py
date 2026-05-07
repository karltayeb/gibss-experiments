import marimo

__generated_with = "0.19.8"
app = marimo.App(width="columns")

with app.setup:
    # all imports here
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    from pathlib import Path
    import yaml


@app.cell
def select_collection():
    collection_alias_root = Path("results") / "twogroup_experiments" / "by_alias"
    collection_options = sorted(
        path.name for path in collection_alias_root.iterdir() if path.is_dir()
    )
    default_collection = (
        "hallmark_ser_local_c"
        if "hallmark_ser_local_c" in collection_options
        else collection_options[0]
    )
    collection_dropdown = mo.ui.dropdown(
        options=collection_options,
        value=default_collection,
        label="collection",
    )
    collection_dropdown
    return (collection_dropdown,)


@app.cell(hide_code=True)
def load_data(collection_dropdown):
    collection_id = collection_dropdown.value
    collection_root = (
        Path("results") / "twogroup_experiments" / "by_alias" / collection_id
    )
    manifest_path = collection_root / "collection_spec.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())

    sim_frames = []
    fit_frames = []
    for batch in manifest["batches"]:
        batch_hash = batch["__spec_hash__"]
        batch_root = collection_root / "batches" / batch_hash

        sim_frames.append(
            pl.read_parquet(batch_root / "simulations.parquet").with_columns(
                pl.lit(batch_hash).alias("batch_hash"),
                pl.lit(batch["name"]).alias("batch_name"),
                pl.lit(batch["simulation_spec"]["fields"]["name"]).alias(
                    "simulation_name"
                ),
            )
        )

        for fit_path in sorted(batch_root.glob("fits/*/fits.parquet")):
            fit_frames.append(
                pl.read_parquet(fit_path).with_columns(
                    pl.lit(batch_hash).alias("batch_hash"),
                    pl.lit(fit_path.parent.name).alias("method_hash"),
                    pl.lit(batch["name"]).alias("batch_name"),
                )
            )

    sims = pl.concat(sim_frames, how="diagonal_relaxed")
    fits = pl.concat(fit_frames, how="diagonal_relaxed")
    df = fits.join(
        sims,
        on=["batch_hash", "batch_name", "replicate"],
        how="left",
    )
    print(
        {
            "collection_id": collection_id,
            "n_batches": len(manifest["batches"]),
            "n_sim_rows": sims.height,
            "n_fit_rows": fits.height,
            "n_joined_rows": df.height,
        }
    )
    return (df,)


@app.cell
def precompute_cs_calibration_grid(df):
    beta_grid = np.round(np.arange(0.50, 1.00, 0.01), 2)
    cs_grid_rows = []

    for cs_grid_row in df.iter_rows(named=True):
        alpha = np.asarray(cs_grid_row["ser_posterior"]["alpha"], dtype=float)
        causal_indices = np.asarray(
            cs_grid_row["credible_set"]["causal_indices"], dtype=int
        )
        if causal_indices.size == 0:
            continue

        order = np.argsort(-alpha)
        sorted_alpha = alpha[order]
        cumulative_alpha = np.cumsum(sorted_alpha)
        causal_index = int(causal_indices[0])
        causal_rank = int(np.where(order == causal_index)[0][0]) + 1
        ser_log_bf = float(cs_grid_row["ser_posterior"]["ser_log_bf"])

        for beta in beta_grid:
            cs_size = int(np.searchsorted(cumulative_alpha, beta, side="left")) + 1
            cs_grid_rows.append(
                {
                    "simulation_name": cs_grid_row["simulation_name"],
                    "batch_hash": cs_grid_row["batch_hash"],
                    "batch_name": cs_grid_row["batch_name"],
                    "replicate": cs_grid_row["replicate"],
                    "method": cs_grid_row["method"],
                    "threshold": cs_grid_row["threshold"],
                    "beta": float(beta),
                    "cs_size": cs_size,
                    "causal_rank": causal_rank,
                    "ser_log_bf": ser_log_bf,
                    "covered": causal_rank <= cs_size,
                }
            )

    cs_calibration_grid = pd.DataFrame(cs_grid_rows)
    return (cs_calibration_grid,)


@app.cell
def fdp_controls(pip_bin_grid):
    method_options = sorted(pip_bin_grid["method"].dropna().unique())
    default_methods = method_options
    method_multiselect = mo.ui.multiselect(
        options=method_options,
        value=default_methods,
        label="methods",
    )
    max_fdp_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.2,
        label="max fdp",
        show_value=True,
    )
    return max_fdp_slider, method_multiselect


@app.function
def aggregate_fdp_power_curves(pip_bin_grid, facet_column=None):
    group_columns = ["method", "threshold", "pip_threshold"]
    if facet_column is not None:
        group_columns = [facet_column, *group_columns]
    return (
        pip_bin_grid.groupby(group_columns, dropna=False)[["power", "fdp"]]
        .mean()
        .reset_index()
    )


@app.function
def select_highlight_threshold(method_df, max_fdp):
    eligible_df = method_df[method_df["fdp"] <= max_fdp]
    if eligible_df.empty:
        return None
    threshold_scores = []
    for threshold_value in sorted(eligible_df["threshold"].dropna().unique()):
        threshold_df = eligible_df[
            eligible_df["threshold"] == threshold_value
        ].sort_values("fdp")
        if threshold_df.empty:
            continue
        auc_score = np.trapezoid(
            threshold_df["power"].to_numpy(),
            threshold_df["fdp"].to_numpy(),
        )
        normalized_auc = auc_score / max(max_fdp, 1e-12)
        threshold_scores.append(
            {
                "threshold": threshold_value,
                "normalized_auc": normalized_auc,
                "max_power": threshold_df["power"].max(),
            }
        )
    if not threshold_scores:
        return None
    threshold_score_df = pd.DataFrame(threshold_scores).sort_values(
        ["normalized_auc", "max_power", "threshold"],
        ascending=[False, False, True],
    )
    return threshold_score_df.iloc[0]["threshold"]


@app.cell(column=1, hide_code=True)
def pip_section():
    pip_header = mo.md(
        """
        <a id="pip"></a>
        ## PIP
        """
    )
    pip_header
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Calibration
    """)
    return


@app.cell(hide_code=True)
def render_pip_calibration_plot_facet(pip_calibration_grid, threshold_slider):
    pip_facet_selected_threshold = threshold_slider.value
    pip_facet_filtered_df = pip_calibration_grid[
        (
            (pip_calibration_grid["method"] == "logistic_threshold")
            & (pip_calibration_grid["threshold"] == pip_facet_selected_threshold)
        )
        | (
            (pip_calibration_grid["method"] == "cox_light_threshold")
            & (pip_calibration_grid["threshold"] == pip_facet_selected_threshold)
        )
        | (
            ~pip_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    pip_facet_summary_df = (
        pip_facet_filtered_df.groupby(
            ["simulation_name", "method", "pip_left", "pip_right", "pip_mid"],
            dropna=False,
        )[["n_total", "n_causal"]]
        .sum()
        .reset_index()
    )
    pip_facet_summary_df["empirical_rate"] = (
        pip_facet_summary_df["n_causal"] / pip_facet_summary_df["n_total"]
    )
    pip_facet_summary_df.loc[pip_facet_summary_df["n_total"] == 0, "empirical_rate"] = (
        np.nan
    )
    pip_facet_simulation_names = sorted(
        pip_facet_summary_df["simulation_name"].dropna().unique()
    )
    pip_facet_n_cols = 2
    pip_facet_n_rows = int(np.ceil(len(pip_facet_simulation_names) / pip_facet_n_cols))
    pip_facet_fig, pip_facet_axes = plt.subplots(
        pip_facet_n_rows,
        pip_facet_n_cols,
        figsize=(12, 4 * pip_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    pip_facet_axes = np.atleast_1d(pip_facet_axes).ravel()

    pip_facet_method_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for pip_facet_ax, pip_facet_simulation_name in zip(
        pip_facet_axes, pip_facet_simulation_names
    ):
        pip_facet_sim_df = pip_facet_summary_df[
            pip_facet_summary_df["simulation_name"] == pip_facet_simulation_name
        ]
        for pip_facet_method in sorted(pip_facet_sim_df["method"].dropna().unique()):
            pip_facet_method_df = pip_facet_sim_df[
                pip_facet_sim_df["method"] == pip_facet_method
            ].sort_values("pip_mid")
            pip_facet_label = pip_facet_method
            if pip_facet_method in {"logistic_threshold", "cox_light_threshold"}:
                pip_facet_label = (
                    f"{pip_facet_method} @ {pip_facet_selected_threshold:g}"
                )
            pip_facet_ax.plot(
                pip_facet_method_df["pip_mid"],
                pip_facet_method_df["empirical_rate"],
                linewidth=2,
                marker="o",
                color=pip_facet_method_colors.get(pip_facet_method),
                label=pip_facet_label,
            )
        pip_facet_ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="x = y",
        )
        pip_facet_ax.set_title(pip_facet_simulation_name)
        pip_facet_ax.set_xlim(0.0, 1.0)
        pip_facet_ax.set_ylim(0.0, 1.02)
        pip_facet_ax.grid(alpha=0.3)

    for pip_facet_ax in pip_facet_axes[len(pip_facet_simulation_names) :]:
        pip_facet_ax.set_visible(False)

    for pip_facet_ax in pip_facet_axes[: len(pip_facet_simulation_names)]:
        pip_facet_ax.set_xlabel("pip bin midpoint")
        pip_facet_ax.set_ylabel("empirical causal frequency")

    pip_facet_handles, pip_facet_labels = pip_facet_axes[0].get_legend_handles_labels()
    pip_facet_fig.legend(
        pip_facet_handles,
        pip_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    pip_facet_fig.suptitle("PIP calibration by simulation", y=1.02)
    pip_facet_fig.tight_layout()
    return (pip_facet_fig,)


@app.cell(hide_code=True)
def render_pip_calibration_plot(pip_calibration_grid, threshold_slider):
    pip_overall_selected_threshold = threshold_slider.value
    pip_filtered_df = pip_calibration_grid[
        (
            (pip_calibration_grid["method"] == "logistic_threshold")
            & (pip_calibration_grid["threshold"] == pip_overall_selected_threshold)
        )
        | (
            (pip_calibration_grid["method"] == "cox_light_threshold")
            & (pip_calibration_grid["threshold"] == pip_overall_selected_threshold)
        )
        | (
            ~pip_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    pip_summary_df = (
        pip_filtered_df.groupby(
            ["method", "pip_left", "pip_right", "pip_mid"], dropna=False
        )[["n_total", "n_causal"]]
        .sum()
        .reset_index()
    )
    pip_summary_df["empirical_rate"] = (
        pip_summary_df["n_causal"] / pip_summary_df["n_total"]
    )
    pip_summary_df.loc[pip_summary_df["n_total"] == 0, "empirical_rate"] = np.nan

    pip_fig, pip_ax = plt.subplots(figsize=(8, 5))
    pip_method_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for pip_method in sorted(pip_summary_df["method"].dropna().unique()):
        pip_method_df = pip_summary_df[
            pip_summary_df["method"] == pip_method
        ].sort_values("pip_mid")
        pip_label = pip_method
        if pip_method in {"logistic_threshold", "cox_light_threshold"}:
            pip_label = f"{pip_method} @ {pip_overall_selected_threshold:g}"
        pip_ax.plot(
            pip_method_df["pip_mid"],
            pip_method_df["empirical_rate"],
            linewidth=2,
            marker="o",
            color=pip_method_colors.get(pip_method),
            label=pip_label,
        )

    pip_ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="x = y",
    )
    pip_ax.set_xlabel("pip bin midpoint")
    pip_ax.set_ylabel("empirical causal frequency")
    pip_ax.set_title("PIP calibration")
    pip_ax.set_xlim(0.0, 1.0)
    pip_ax.set_ylim(0.0, 1.02)
    pip_ax.grid(alpha=0.3)
    pip_ax.legend()
    pip_fig.tight_layout()
    return (pip_fig,)


@app.cell(hide_code=True)
def pip_calibration_controls_local(threshold_slider):
    mo.hstack([threshold_slider], justify="start")
    return


@app.cell
def _(pip_facet_fig, pip_fig):
    mo.tabs({"Aggregate": pip_fig, "By Simulation Scenario": pip_facet_fig})
    return


@app.cell(hide_code=True)
def pip_power_fdp_section():
    pip_power_header = mo.md(
        """
        <a id="pip-power-fdp"></a>
        ### Power vs FDP

        Controls: `method` selection and `max fdp`. The `max fdp` cutoff also determines which threshold curve is highlighted.
        """
    )
    pip_power_header
    return


@app.cell(hide_code=True)
def precompute_pip_calibration_grid(df):
    pip_bin_edges = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    pip_calibration_rows = []

    for pip_calibration_row in df.iter_rows(named=True):
        pip_alpha = np.asarray(
            pip_calibration_row["ser_posterior"]["alpha"], dtype=float
        )
        pip_causal_indices = np.asarray(
            pip_calibration_row["credible_set"]["causal_indices"], dtype=int
        )
        pip_causal_mask = np.zeros(pip_alpha.shape[0], dtype=bool)
        pip_causal_mask[pip_causal_indices] = True

        for left_edge, right_edge in zip(pip_bin_edges[:-1], pip_bin_edges[1:]):
            if np.isclose(right_edge, 1.0):
                pip_in_bin = (pip_alpha >= left_edge) & (pip_alpha <= right_edge)
            else:
                pip_in_bin = (pip_alpha >= left_edge) & (pip_alpha < right_edge)
            n_total = int(np.sum(pip_in_bin))
            n_causal = int(np.sum(pip_causal_mask & pip_in_bin))
            pip_calibration_rows.append(
                {
                    "simulation_name": pip_calibration_row["simulation_name"],
                    "batch_hash": pip_calibration_row["batch_hash"],
                    "batch_name": pip_calibration_row["batch_name"],
                    "replicate": pip_calibration_row["replicate"],
                    "method": pip_calibration_row["method"],
                    "threshold": pip_calibration_row["threshold"],
                    "pip_left": float(left_edge),
                    "pip_right": float(right_edge),
                    "pip_mid": float((left_edge + right_edge) / 2.0),
                    "n_total": n_total,
                    "n_causal": n_causal,
                }
            )

    pip_calibration_grid = pd.DataFrame(pip_calibration_rows)
    return (pip_calibration_grid,)


@app.cell(hide_code=True)
def render_fdp_plot_facet(max_fdp_slider, method_multiselect, pip_bin_grid):
    facet_selected_methods = method_multiselect.value
    facet_max_fdp = max_fdp_slider.value
    facet_filtered_pip_bin_grid = pip_bin_grid[
        pip_bin_grid["method"].isin(facet_selected_methods)
    ]
    fdp_facet_summary_df = aggregate_fdp_power_curves(
        facet_filtered_pip_bin_grid, facet_column="simulation_name"
    )
    fdp_facet_summary_df = fdp_facet_summary_df[
        fdp_facet_summary_df["fdp"] <= facet_max_fdp
    ]

    fdp_facet_simulation_names = sorted(
        fdp_facet_summary_df["simulation_name"].dropna().unique()
    )
    fdp_facet_n_cols = 2
    fdp_facet_n_rows = int(np.ceil(len(fdp_facet_simulation_names) / fdp_facet_n_cols))
    fdp_facet_fig, fdp_facet_axes = plt.subplots(
        fdp_facet_n_rows,
        fdp_facet_n_cols,
        figsize=(12, 4 * fdp_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    fdp_facet_axes = np.atleast_1d(fdp_facet_axes).ravel()

    fdp_facet_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    fdp_facet_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for fdp_facet_ax, fdp_facet_simulation_name in zip(
        fdp_facet_axes, fdp_facet_simulation_names
    ):
        fdp_facet_sim_df = fdp_facet_summary_df[
            fdp_facet_summary_df["simulation_name"] == fdp_facet_simulation_name
        ]

        for facet_selected_method in facet_selected_methods:
            if facet_selected_method in fdp_facet_curve_colors:
                fdp_facet_method_df = fdp_facet_sim_df[
                    fdp_facet_sim_df["method"] == facet_selected_method
                ]
                facet_highlight_threshold = select_highlight_threshold(
                    fdp_facet_method_df, facet_max_fdp
                )
                for fdp_facet_threshold_value in sorted(
                    fdp_facet_method_df["threshold"].dropna().unique()
                ):
                    fdp_facet_curve = fdp_facet_method_df[
                        fdp_facet_method_df["threshold"] == fdp_facet_threshold_value
                    ].sort_values("pip_threshold")
                    fdp_facet_ax.plot(
                        fdp_facet_curve["fdp"],
                        fdp_facet_curve["power"],
                        color=fdp_facet_curve_colors[facet_selected_method],
                        alpha=0.15,
                    )
                    if fdp_facet_threshold_value == facet_highlight_threshold:
                        fdp_facet_ax.plot(
                            fdp_facet_curve["fdp"],
                            fdp_facet_curve["power"],
                            color=fdp_facet_curve_colors[facet_selected_method],
                            linewidth=2.5,
                            alpha=1.0,
                            label=f"{facet_selected_method} @ {fdp_facet_threshold_value:g}",
                        )

        for facet_selected_method in facet_selected_methods:
            if facet_selected_method in fdp_facet_line_colors:
                fdp_facet_curve = fdp_facet_sim_df[
                    fdp_facet_sim_df["method"] == facet_selected_method
                ].sort_values("pip_threshold")
                if not fdp_facet_curve.empty:
                    fdp_facet_ax.plot(
                        fdp_facet_curve["fdp"],
                        fdp_facet_curve["power"],
                        color=fdp_facet_line_colors[facet_selected_method],
                        linewidth=2,
                        label=facet_selected_method,
                    )

        fdp_facet_ax.set_title(fdp_facet_simulation_name)
        fdp_facet_ax.set_xlim(-0.02, facet_max_fdp + 0.02)
        fdp_facet_ax.set_ylim(-0.02, 1.02)
        fdp_facet_ax.grid(alpha=0.3)

    for fdp_facet_ax in fdp_facet_axes[len(fdp_facet_simulation_names) :]:
        fdp_facet_ax.set_visible(False)

    for fdp_facet_ax in fdp_facet_axes[: len(fdp_facet_simulation_names)]:
        fdp_facet_ax.set_xlabel("fdp")
        fdp_facet_ax.set_ylabel("power")

    fdp_facet_handles, fdp_facet_labels = fdp_facet_axes[0].get_legend_handles_labels()
    fdp_facet_fig.legend(
        fdp_facet_handles,
        fdp_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fdp_facet_fig.suptitle("FDP vs power by simulation", y=1.02)
    fdp_facet_fig.tight_layout()
    return (fdp_facet_fig,)


@app.cell(hide_code=True)
def render_fdp_plot(max_fdp_slider, method_multiselect, pip_bin_grid):
    overall_selected_methods = method_multiselect.value
    overall_max_fdp = max_fdp_slider.value
    overall_filtered_pip_bin_grid = pip_bin_grid[
        pip_bin_grid["method"].isin(overall_selected_methods)
    ]
    fdp_overall_summary_df = aggregate_fdp_power_curves(overall_filtered_pip_bin_grid)
    fdp_overall_summary_df = fdp_overall_summary_df[
        fdp_overall_summary_df["fdp"] <= overall_max_fdp
    ]

    fdp_overall_fig, fdp_overall_ax = plt.subplots(figsize=(8, 5))
    fdp_overall_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    for overall_selected_method in overall_selected_methods:
        if overall_selected_method in fdp_overall_curve_colors:
            fdp_overall_method_df = fdp_overall_summary_df[
                fdp_overall_summary_df["method"] == overall_selected_method
            ].sort_values("pip_threshold")
            overall_highlight_threshold = select_highlight_threshold(
                fdp_overall_method_df, overall_max_fdp
            )
            for fdp_overall_threshold_value in sorted(
                fdp_overall_method_df["threshold"].dropna().unique()
            ):
                fdp_overall_curve = fdp_overall_method_df[
                    fdp_overall_method_df["threshold"] == fdp_overall_threshold_value
                ].sort_values("pip_threshold")
                fdp_overall_ax.plot(
                    fdp_overall_curve["fdp"],
                    fdp_overall_curve["power"],
                    color=fdp_overall_curve_colors[overall_selected_method],
                    alpha=0.15,
                )
                if fdp_overall_threshold_value == overall_highlight_threshold:
                    fdp_overall_ax.plot(
                        fdp_overall_curve["fdp"],
                        fdp_overall_curve["power"],
                        color=fdp_overall_curve_colors[overall_selected_method],
                        linewidth=2.5,
                        alpha=1.0,
                        label=f"{overall_selected_method} @ {fdp_overall_threshold_value:g}",
                    )

    fdp_overall_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    for overall_selected_method in overall_selected_methods:
        if overall_selected_method in fdp_overall_line_colors:
            fdp_overall_curve = fdp_overall_summary_df[
                fdp_overall_summary_df["method"] == overall_selected_method
            ].sort_values("pip_threshold")
            if not fdp_overall_curve.empty:
                fdp_overall_ax.plot(
                    fdp_overall_curve["fdp"],
                    fdp_overall_curve["power"],
                    color=fdp_overall_line_colors[overall_selected_method],
                    linewidth=2,
                    label=overall_selected_method,
                )

    fdp_overall_ax.set_xlabel("fdp")
    fdp_overall_ax.set_ylabel("power")
    fdp_overall_ax.set_title("FDP vs power")
    fdp_overall_ax.set_xlim(-0.02, overall_max_fdp + 0.02)
    fdp_overall_ax.set_ylim(-0.02, 1.02)
    fdp_overall_ax.grid(alpha=0.3)
    fdp_overall_ax.legend()
    fdp_overall_fig.tight_layout()
    return (fdp_overall_fig,)


@app.cell
def pip_power_fdp_controls_local(
    max_fdp_slider,
    method_multiselect,
    threshold_slider,
):
    mo.hstack(
        [method_multiselect, max_fdp_slider, threshold_slider],
        justify="start",
        gap=2,
    )
    return


@app.cell
def _(fdp_facet_fig, fdp_overall_fig):
    mo.tabs({"Aggregate": fdp_overall_fig, "By Simulation Scenario": fdp_facet_fig})
    return


@app.cell(hide_code=True)
def precompute_pip_bin_grid(df):
    pip_grid = np.arange(0.001, 1.001, 0.001)
    pip_grid_rows = []

    for pip_grid_row in df.iter_rows(named=True):
        pip_grid_alpha = np.asarray(pip_grid_row["ser_posterior"]["alpha"], dtype=float)
        pip_grid_causal_indices = np.asarray(
            pip_grid_row["credible_set"]["causal_indices"], dtype=int
        )

        pip_grid_total_counts = np.zeros(pip_grid.size, dtype=int)
        pip_grid_causal_counts = np.zeros(pip_grid.size, dtype=int)

        pip_grid_bin_index = np.floor(pip_grid_alpha * 1000).astype(int) - 1
        pip_grid_valid_mask = pip_grid_bin_index >= 0
        np.add.at(
            pip_grid_total_counts,
            pip_grid_bin_index[pip_grid_valid_mask],
            1,
        )

        pip_grid_causal_alpha = pip_grid_alpha[pip_grid_causal_indices]
        pip_grid_causal_bin_index = (
            np.floor(pip_grid_causal_alpha * 1000).astype(int) - 1
        )
        pip_grid_causal_valid_mask = pip_grid_causal_bin_index >= 0
        np.add.at(
            pip_grid_causal_counts,
            pip_grid_causal_bin_index[pip_grid_causal_valid_mask],
            1,
        )

        pip_grid_selected = np.cumsum(pip_grid_total_counts[::-1])[::-1]
        pip_grid_true_selected = np.cumsum(pip_grid_causal_counts[::-1])[::-1]
        pip_grid_false_selected = pip_grid_selected - pip_grid_true_selected
        pip_grid_fdp = np.divide(
            pip_grid_false_selected,
            pip_grid_selected,
            out=np.zeros_like(pip_grid_selected, dtype=float),
            where=pip_grid_selected > 0,
        )
        pip_grid_power = np.divide(
            pip_grid_true_selected,
            max(len(pip_grid_causal_indices), 1),
            out=np.zeros_like(pip_grid_true_selected, dtype=float),
            where=True,
        )

        for pip_threshold, power_value, fdp_value in zip(
            pip_grid,
            pip_grid_power,
            pip_grid_fdp,
        ):
            pip_grid_rows.append(
                {
                    "simulation_name": pip_grid_row["simulation_name"],
                    "batch_hash": pip_grid_row["batch_hash"],
                    "batch_name": pip_grid_row["batch_name"],
                    "replicate": pip_grid_row["replicate"],
                    "method": pip_grid_row["method"],
                    "threshold": pip_grid_row["threshold"],
                    "pip_threshold": float(pip_threshold),
                    "power": float(power_value),
                    "fdp": float(fdp_value),
                }
            )

    pip_bin_grid = pd.DataFrame(pip_grid_rows)
    return (pip_bin_grid,)


@app.cell(hide_code=True)
def pip_causal_pips_section():
    mo.md("""
    ### Causal PIPs
    """)
    return


@app.cell(hide_code=True)
def render_max_pip_plot(df):
    overall_plot_df = (
        df.with_columns(
            pl.col("fit_summary").struct.field("causal_pip").alias("causal_pip")
        )
        .group_by(["method", "threshold"])
        .agg(pl.mean("causal_pip").alias("mean_causal_pip"))
        .sort(["method", "threshold"])
    )
    overall_curve_df = overall_plot_df.filter(pl.col("threshold").is_not_null())
    overall_line_df = overall_plot_df.filter(pl.col("threshold").is_null())

    overall_fig, overall_ax = plt.subplots(figsize=(8, 5))
    overall_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    for overall_method, overall_color in overall_curve_colors.items():
        overall_method_df = overall_curve_df.filter(pl.col("method") == overall_method)
        overall_ax.plot(
            overall_method_df["threshold"].to_list(),
            overall_method_df["mean_causal_pip"].to_list(),
            marker="o",
            linewidth=2,
            color=overall_color,
            label=overall_method,
        )

    overall_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    for overall_method, overall_color in overall_line_colors.items():
        overall_method_line = overall_line_df.filter(pl.col("method") == overall_method)
        if overall_method_line.height == 0:
            continue
        overall_ax.axhline(
            y=overall_method_line["mean_causal_pip"].item(),
            color=overall_color,
            linestyle="--",
            linewidth=2,
            label=overall_method,
        )

    overall_ax.set_xlabel("threshold")
    overall_ax.set_ylabel("mean causal_pip")
    overall_ax.set_title("Mean causal_pip vs threshold")
    overall_ax.legend()
    overall_ax.grid(alpha=0.3)
    overall_fig.tight_layout()
    return (overall_fig,)


@app.cell(hide_code=True)
def render_max_pip_plot_facet(df):
    facet_plot_df = (
        df.with_columns(
            pl.col("fit_summary").struct.field("causal_pip").alias("causal_pip")
        )
        .group_by(["simulation_name", "method", "threshold"])
        .agg(pl.mean("causal_pip").alias("mean_causal_pip"))
        .sort(["simulation_name", "method", "threshold"])
    )
    facet_simulation_names = facet_plot_df["simulation_name"].unique().sort().to_list()
    facet_n_cols = 2
    facet_n_rows = int(np.ceil(len(facet_simulation_names) / facet_n_cols))
    facet_fig, facet_axes = plt.subplots(
        facet_n_rows,
        facet_n_cols,
        figsize=(12, 4 * facet_n_rows),
        sharex=True,
        sharey=True,
    )
    facet_axes = np.atleast_1d(facet_axes).ravel()

    facet_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    facet_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for facet_ax, facet_simulation_name in zip(facet_axes, facet_simulation_names):
        facet_sim_df = facet_plot_df.filter(
            pl.col("simulation_name") == facet_simulation_name
        )
        facet_curve_df = facet_sim_df.filter(pl.col("threshold").is_not_null())
        facet_line_df = facet_sim_df.filter(pl.col("threshold").is_null())

        for facet_method, facet_color in facet_curve_colors.items():
            facet_method_df = facet_curve_df.filter(pl.col("method") == facet_method)
            facet_ax.plot(
                facet_method_df["threshold"].to_list(),
                facet_method_df["mean_causal_pip"].to_list(),
                marker="o",
                linewidth=2,
                color=facet_color,
                label=facet_method,
            )

        for facet_method, facet_color in facet_line_colors.items():
            facet_method_line = facet_line_df.filter(pl.col("method") == facet_method)
            if facet_method_line.height == 0:
                continue
            facet_ax.axhline(
                y=facet_method_line["mean_causal_pip"].item(),
                color=facet_color,
                linestyle="--",
                linewidth=2,
                label=facet_method,
            )

        facet_ax.set_title(facet_simulation_name)
        facet_ax.grid(alpha=0.3)

    for facet_ax in facet_axes[len(facet_simulation_names) :]:
        facet_ax.set_visible(False)

    for facet_ax in facet_axes[: len(facet_simulation_names)]:
        facet_ax.set_xlabel("threshold")
        facet_ax.set_ylabel("mean causal_pip")

    facet_handles, facet_labels = facet_axes[0].get_legend_handles_labels()
    facet_fig.legend(
        facet_handles, facet_labels, loc="upper center", ncol=3, frameon=False
    )
    facet_fig.suptitle("Mean causal_pip vs threshold by simulation", y=1.02)
    facet_fig.tight_layout()
    return (facet_fig,)


@app.cell(hide_code=True)
def _(facet_fig, overall_fig):
    mo.tabs({"Aggregate": overall_fig, "By Simulation Scenario": facet_fig})
    return


@app.cell(column=2, hide_code=True)
def credible_sets_unconditional_section():
    mo.md(r"""
    ## Unconditional Credible Sets

    We compute the $\alpha$ credible sets for each SER without filtering to those that we would consider discoveries.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Coverage
    """)
    return


@app.cell(hide_code=True)
def render_cs_calibration_plot(cs_calibration_grid, threshold_slider):
    overall_selected_threshold = threshold_slider.value
    cs_filtered_df = cs_calibration_grid[
        (
            (cs_calibration_grid["method"] == "logistic_threshold")
            & (cs_calibration_grid["threshold"] == overall_selected_threshold)
        )
        | (
            (cs_calibration_grid["method"] == "cox_light_threshold")
            & (cs_calibration_grid["threshold"] == overall_selected_threshold)
        )
        | (
            ~cs_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    cs_summary_df = (
        cs_filtered_df.groupby(["method", "beta"], dropna=False)[["covered"]]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )

    cs_fig, cs_ax = plt.subplots(figsize=(8, 5))
    cs_method_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_method in sorted(cs_summary_df["method"].dropna().unique()):
        cs_method_df = cs_summary_df[cs_summary_df["method"] == cs_method].sort_values(
            "beta"
        )
        cs_label = cs_method
        if cs_method == "logistic_threshold":
            cs_label = f"{cs_method} @ {overall_selected_threshold:g}"
        elif cs_method == "cox_light_threshold":
            cs_label = f"{cs_method} @ {overall_selected_threshold:g}"
        cs_ax.plot(
            cs_method_df["beta"],
            cs_method_df["empirical_coverage"],
            linewidth=2,
            color=cs_method_colors.get(cs_method),
            label=cs_label,
        )

    cs_ax.plot(
        [0.5, 0.99],
        [0.5, 0.99],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="x = y",
    )
    cs_ax.set_xlabel("nominal coverage")
    cs_ax.set_ylabel("empirical coverage")
    cs_ax.set_title("Credible set calibration")
    cs_ax.set_xlim(0.5, 0.99)
    cs_ax.set_ylim(0.0, 1.02)
    cs_ax.grid(alpha=0.3)
    cs_ax.legend()
    cs_fig.tight_layout()
    cs_fig
    return (cs_fig,)


@app.cell
def credible_sets_calibration_controls_local(threshold_slider):
    mo.hstack([threshold_slider], justify="start")
    return


@app.cell
def _(cs_facet_fig, cs_fig):
    mo.tabs({"Aggregated": cs_fig, "By Simulation Scenario": cs_facet_fig})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Credible set size
    """)
    return


@app.cell
def _(cs_size_facet_fig, cs_size_fig):
    mo.tabs({"Aggregate": cs_size_fig, "By Simulation Scenario": cs_size_facet_fig})
    return


@app.cell(hide_code=True)
def credible_sets_conditional_section():
    mo.md(r"""
    ### Conditional

    Everything below depends on `max cs size` or `min ser log bf`.


    We compute the $\alpha$ credible sets for each SER. We filter credible sets by CS size and the SER log Bayes factor.
    """)
    return


@app.cell
def cs_threshold_controls(cs_calibration_grid):
    logistic_thresholds = sorted(
        cs_calibration_grid.loc[
            cs_calibration_grid["method"] == "logistic_threshold", "threshold"
        ]
        .dropna()
        .unique()
        .tolist()
    )
    cox_thresholds = sorted(
        cs_calibration_grid.loc[
            cs_calibration_grid["method"] == "cox_light_threshold", "threshold"
        ]
        .dropna()
        .unique()
        .tolist()
    )
    shared_thresholds = sorted(set(logistic_thresholds) & set(cox_thresholds))
    threshold_slider = mo.ui.slider(
        start=float(min(shared_thresholds)),
        stop=float(max(shared_thresholds)),
        step=float(shared_thresholds[1] - shared_thresholds[0]),
        value=2.0,
        label="threshold",
        show_value=True,
    )
    return (threshold_slider,)


@app.cell(hide_code=True)
def render_cs_size_histogram(
    cs_calibration_grid,
    max_cs_size_slider,
    nominal_coverage_slider,
    threshold_slider,
):
    selected_beta = nominal_coverage_slider.value
    selected_threshold = threshold_slider.value
    cs_hist_df = cs_calibration_grid[
        (
            (cs_calibration_grid["method"] == "logistic_threshold")
            & (cs_calibration_grid["threshold"] == selected_threshold)
        )
        | (
            (cs_calibration_grid["method"] == "cox_light_threshold")
            & (cs_calibration_grid["threshold"] == selected_threshold)
        )
        | (
            ~cs_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    cs_hist_df = cs_hist_df[cs_hist_df["beta"] == selected_beta]

    cs_hist_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    cs_hist_method_order = [
        "logistic_threshold",
        "cox_light_threshold",
        "twogroup",
        "twogroup_oracle",
        "logistic_oracle",
        "cox_heavy",
    ]

    max_size = int(cs_hist_df["cs_size"].max()) if not cs_hist_df.empty else 1
    bins = np.arange(0.5, max_size + 1.5, 1.0)
    cs_hist_fig, cs_hist_axes = plt.subplots(
        len(cs_hist_method_order),
        1,
        figsize=(5.5, 1.35 * len(cs_hist_method_order)),
        sharex=True,
    )
    cs_hist_axes = np.atleast_1d(cs_hist_axes).ravel()

    for cs_hist_ax, hist_method in zip(cs_hist_axes, cs_hist_method_order):
        hist_method_df = cs_hist_df[cs_hist_df["method"] == hist_method]
        hist_label = hist_method
        if hist_method in {"logistic_threshold", "cox_light_threshold"}:
            hist_label = f"{hist_method} @ {selected_threshold:g}"
        if hist_method_df.empty:
            cs_hist_ax.set_visible(False)
            continue
        cs_hist_ax.hist(
            hist_method_df["cs_size"],
            bins=bins,
            histtype="step",
            linewidth=2,
            color=cs_hist_colors.get(hist_method),
        )
        cs_hist_ax.axvline(
            max_cs_size_slider.value,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        cs_hist_ax.grid(alpha=0.3)
        cs_hist_ax.set_ylabel("n")
        cs_hist_ax.set_title(hist_label, fontsize=10, loc="left")

    cs_hist_axes[-1].set_xlabel("credible set size")
    cs_hist_fig.suptitle(f"CS sizes at {selected_beta:.0%} nominal coverage", y=1.0)
    cs_hist_fig.tight_layout()
    return (cs_hist_fig,)


@app.cell(hide_code=True)
def render_ser_log_bf_histogram(
    cs_calibration_grid,
    nominal_coverage_slider,
    ser_log_bf_slider,
    threshold_slider,
):
    ser_hist_selected_beta = nominal_coverage_slider.value
    ser_hist_selected_threshold = threshold_slider.value
    ser_hist_df = cs_calibration_grid[
        (
            (cs_calibration_grid["method"] == "logistic_threshold")
            & (cs_calibration_grid["threshold"] == ser_hist_selected_threshold)
        )
        | (
            (cs_calibration_grid["method"] == "cox_light_threshold")
            & (cs_calibration_grid["threshold"] == ser_hist_selected_threshold)
        )
        | (
            ~cs_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    ser_hist_df = ser_hist_df[ser_hist_df["beta"] == ser_hist_selected_beta]

    ser_hist_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    ser_hist_method_order = [
        "logistic_threshold",
        "cox_light_threshold",
        "twogroup",
        "twogroup_oracle",
        "logistic_oracle",
        "cox_heavy",
    ]
    ser_hist_fig, ser_hist_axes = plt.subplots(
        len(ser_hist_method_order),
        1,
        figsize=(5.5, 1.35 * len(ser_hist_method_order)),
        sharex=True,
    )
    ser_hist_axes = np.atleast_1d(ser_hist_axes).ravel()

    for ser_hist_ax, ser_hist_method in zip(ser_hist_axes, ser_hist_method_order):
        ser_hist_method_df = ser_hist_df[ser_hist_df["method"] == ser_hist_method]
        ser_hist_label = ser_hist_method
        if ser_hist_method in {"logistic_threshold", "cox_light_threshold"}:
            ser_hist_label = f"{ser_hist_method} @ {ser_hist_selected_threshold:g}"
        if ser_hist_method_df.empty:
            ser_hist_ax.set_visible(False)
            continue
        ser_hist_ax.hist(
            ser_hist_method_df["ser_log_bf"],
            bins=25,
            histtype="step",
            linewidth=2,
            color=ser_hist_colors.get(ser_hist_method),
        )
        ser_hist_ax.axvline(
            ser_log_bf_slider.value,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        ser_hist_ax.grid(alpha=0.3)
        ser_hist_ax.set_ylabel("n")
        ser_hist_ax.set_title(ser_hist_label, fontsize=10, loc="left")

    ser_hist_axes[-1].set_xlabel("ser log bf")
    ser_hist_fig.suptitle(
        f"SER log BF at {ser_hist_selected_beta:.0%} nominal coverage",
        y=1.0,
    )
    ser_hist_fig.tight_layout()
    return (ser_hist_fig,)


@app.cell(hide_code=True)
def credible_sets_conditional_controls_local(
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
    threshold_slider,
):
    mo.vstack(
        [
            mo.hstack([threshold_slider], justify="start"),
            mo.hstack(
                [nominal_coverage_slider, max_cs_size_slider, ser_log_bf_slider],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def display_cs_histograms(cs_hist_fig, ser_hist_fig):
    mo.hstack([cs_hist_fig, ser_hist_fig], widths="equal", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conditional credible set size
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Conditional Calibration
    """)
    return


@app.cell(hide_code=True)
def _(cs_conditional_facet_fig, cs_conditional_fig):
    mo.tabs(
        {
            "Aggregated": cs_conditional_fig,
            "By Simulation Scenario": cs_conditional_facet_fig,
        }
    )
    return


@app.cell(hide_code=True)
def render_cs_calibration_plot_facet(cs_calibration_grid, threshold_slider):
    facet_selected_threshold = threshold_slider.value
    cs_facet_filtered_df = cs_calibration_grid[
        (
            (cs_calibration_grid["method"] == "logistic_threshold")
            & (cs_calibration_grid["threshold"] == facet_selected_threshold)
        )
        | (
            (cs_calibration_grid["method"] == "cox_light_threshold")
            & (cs_calibration_grid["threshold"] == facet_selected_threshold)
        )
        | (
            ~cs_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    cs_facet_summary_df = (
        cs_facet_filtered_df.groupby(
            ["simulation_name", "method", "beta"], dropna=False
        )[["covered"]]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )
    cs_facet_simulation_names = sorted(
        cs_facet_summary_df["simulation_name"].dropna().unique()
    )
    cs_facet_n_cols = 2
    cs_facet_n_rows = int(np.ceil(len(cs_facet_simulation_names) / cs_facet_n_cols))
    cs_facet_fig, cs_facet_axes = plt.subplots(
        cs_facet_n_rows,
        cs_facet_n_cols,
        figsize=(12, 4 * cs_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    cs_facet_axes = np.atleast_1d(cs_facet_axes).ravel()

    cs_facet_method_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_facet_ax, cs_facet_simulation_name in zip(
        cs_facet_axes, cs_facet_simulation_names
    ):
        cs_facet_sim_df = cs_facet_summary_df[
            cs_facet_summary_df["simulation_name"] == cs_facet_simulation_name
        ]
        for cs_facet_method in sorted(cs_facet_sim_df["method"].dropna().unique()):
            cs_facet_method_df = cs_facet_sim_df[
                cs_facet_sim_df["method"] == cs_facet_method
            ].sort_values("beta")
            cs_facet_label = cs_facet_method
            if cs_facet_method == "logistic_threshold":
                cs_facet_label = f"{cs_facet_method} @ {facet_selected_threshold:g}"
            elif cs_facet_method == "cox_light_threshold":
                cs_facet_label = f"{cs_facet_method} @ {facet_selected_threshold:g}"
            cs_facet_ax.plot(
                cs_facet_method_df["beta"],
                cs_facet_method_df["empirical_coverage"],
                linewidth=2,
                color=cs_facet_method_colors.get(cs_facet_method),
                label=cs_facet_label,
            )
        cs_facet_ax.plot(
            [0.5, 0.99],
            [0.5, 0.99],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="x = y",
        )
        cs_facet_ax.set_title(cs_facet_simulation_name)
        cs_facet_ax.set_xlim(0.5, 0.99)
        cs_facet_ax.set_ylim(0.0, 1.02)
        cs_facet_ax.grid(alpha=0.3)

    for cs_facet_ax in cs_facet_axes[len(cs_facet_simulation_names) :]:
        cs_facet_ax.set_visible(False)

    for cs_facet_ax in cs_facet_axes[: len(cs_facet_simulation_names)]:
        cs_facet_ax.set_xlabel("nominal coverage")
        cs_facet_ax.set_ylabel("empirical coverage")

    cs_facet_handles, cs_facet_labels = cs_facet_axes[0].get_legend_handles_labels()
    cs_facet_fig.legend(
        cs_facet_handles,
        cs_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    cs_facet_fig.suptitle("Credible set calibration by simulation", y=1.02)
    cs_facet_fig.tight_layout()
    return (cs_facet_fig,)


@app.cell(hide_code=True)
def render_cs_size_plot_facet(cs_calibration_grid, nominal_coverage_slider):
    cs_size_facet_df = (
        cs_calibration_grid[
            cs_calibration_grid["beta"] == nominal_coverage_slider.value
        ]
        .groupby(["simulation_name", "method", "threshold"], dropna=False)[["cs_size"]]
        .mean()
        .reset_index()
        .rename(columns={"cs_size": "mean_cs_size"})
    )
    cs_size_facet_simulation_names = sorted(
        cs_size_facet_df["simulation_name"].dropna().unique()
    )
    cs_size_facet_n_cols = 2
    cs_size_facet_n_rows = int(
        np.ceil(len(cs_size_facet_simulation_names) / cs_size_facet_n_cols)
    )
    cs_size_facet_fig, cs_size_facet_axes = plt.subplots(
        cs_size_facet_n_rows,
        cs_size_facet_n_cols,
        figsize=(12, 4 * cs_size_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    cs_size_facet_axes = np.atleast_1d(cs_size_facet_axes).ravel()

    cs_size_facet_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    cs_size_facet_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_size_facet_ax, cs_size_facet_simulation_name in zip(
        cs_size_facet_axes, cs_size_facet_simulation_names
    ):
        cs_size_facet_sim_df = cs_size_facet_df[
            cs_size_facet_df["simulation_name"] == cs_size_facet_simulation_name
        ]
        for (
            cs_size_facet_method,
            cs_size_facet_color,
        ) in cs_size_facet_curve_colors.items():
            cs_size_facet_method_df = cs_size_facet_sim_df[
                cs_size_facet_sim_df["method"] == cs_size_facet_method
            ].sort_values("threshold")
            if cs_size_facet_method_df.empty:
                continue
            cs_size_facet_ax.plot(
                cs_size_facet_method_df["threshold"],
                cs_size_facet_method_df["mean_cs_size"],
                marker="o",
                linewidth=2,
                color=cs_size_facet_color,
                label=cs_size_facet_method,
            )
        for (
            cs_size_facet_method,
            cs_size_facet_color,
        ) in cs_size_facet_line_colors.items():
            cs_size_facet_method_df = cs_size_facet_sim_df[
                cs_size_facet_sim_df["method"] == cs_size_facet_method
            ]
            if cs_size_facet_method_df.empty:
                continue
            cs_size_facet_ax.axhline(
                y=cs_size_facet_method_df["mean_cs_size"].iloc[0],
                color=cs_size_facet_color,
                linestyle="--",
                linewidth=2,
                label=cs_size_facet_method,
            )
        cs_size_facet_ax.set_title(cs_size_facet_simulation_name)
        cs_size_facet_ax.grid(alpha=0.3)

    for cs_size_facet_ax in cs_size_facet_axes[len(cs_size_facet_simulation_names) :]:
        cs_size_facet_ax.set_visible(False)
    for cs_size_facet_ax in cs_size_facet_axes[: len(cs_size_facet_simulation_names)]:
        cs_size_facet_ax.set_xlabel("threshold")
        cs_size_facet_ax.set_ylabel("mean cs size")

    cs_size_facet_handles, cs_size_facet_labels = cs_size_facet_axes[
        0
    ].get_legend_handles_labels()
    cs_size_facet_fig.legend(
        cs_size_facet_handles,
        cs_size_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    cs_size_facet_fig.suptitle(
        f"Mean {nominal_coverage_slider.value:.0%} CS size vs threshold by simulation",
        y=1.02,
    )
    cs_size_facet_fig.tight_layout()
    return (cs_size_facet_fig,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Credible set size
    """)
    return


@app.cell(hide_code=True)
def render_cs_discovery_plot_facet(
    cs_calibration_grid,
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
):
    discovery_facet_df = cs_calibration_grid[
        cs_calibration_grid["beta"] == nominal_coverage_slider.value
    ].copy()
    discovery_facet_df["discovered"] = (
        (discovery_facet_df["covered"]).astype(int)
        * (discovery_facet_df["cs_size"] <= max_cs_size_slider.value).astype(int)
        * (discovery_facet_df["ser_log_bf"] > ser_log_bf_slider.value).astype(int)
    )
    cs_discovery_facet_df = (
        discovery_facet_df.groupby(
            ["simulation_name", "method", "threshold"], dropna=False
        )[["discovered"]]
        .mean()
        .reset_index()
        .rename(columns={"discovered": "discovery_rate"})
    )
    cs_discovery_facet_simulation_names = sorted(
        cs_discovery_facet_df["simulation_name"].dropna().unique()
    )
    cs_discovery_facet_n_cols = 2
    cs_discovery_facet_n_rows = int(
        np.ceil(len(cs_discovery_facet_simulation_names) / cs_discovery_facet_n_cols)
    )
    cs_discovery_facet_fig, cs_discovery_facet_axes = plt.subplots(
        cs_discovery_facet_n_rows,
        cs_discovery_facet_n_cols,
        figsize=(12, 4 * cs_discovery_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    cs_discovery_facet_axes = np.atleast_1d(cs_discovery_facet_axes).ravel()

    cs_discovery_facet_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    cs_discovery_facet_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_discovery_facet_ax, cs_discovery_facet_simulation_name in zip(
        cs_discovery_facet_axes, cs_discovery_facet_simulation_names
    ):
        cs_discovery_facet_sim_df = cs_discovery_facet_df[
            cs_discovery_facet_df["simulation_name"]
            == cs_discovery_facet_simulation_name
        ]
        for (
            cs_discovery_facet_method,
            cs_discovery_facet_color,
        ) in cs_discovery_facet_curve_colors.items():
            cs_discovery_facet_method_df = cs_discovery_facet_sim_df[
                cs_discovery_facet_sim_df["method"] == cs_discovery_facet_method
            ].sort_values("threshold")
            if cs_discovery_facet_method_df.empty:
                continue
            cs_discovery_facet_ax.plot(
                cs_discovery_facet_method_df["threshold"],
                cs_discovery_facet_method_df["discovery_rate"],
                marker="o",
                linewidth=2,
                color=cs_discovery_facet_color,
                label=cs_discovery_facet_method,
            )
        for (
            cs_discovery_facet_method,
            cs_discovery_facet_color,
        ) in cs_discovery_facet_line_colors.items():
            cs_discovery_facet_method_df = cs_discovery_facet_sim_df[
                (cs_discovery_facet_sim_df["method"] == cs_discovery_facet_method)
                & (cs_discovery_facet_sim_df["threshold"].isna())
            ]
            if cs_discovery_facet_method_df.empty:
                continue
            cs_discovery_facet_ax.axhline(
                y=cs_discovery_facet_method_df["discovery_rate"].iloc[0],
                color=cs_discovery_facet_color,
                linestyle="--",
                linewidth=2,
                label=cs_discovery_facet_method,
            )
        cs_discovery_facet_ax.set_title(cs_discovery_facet_simulation_name)
        cs_discovery_facet_ax.set_ylim(0.0, 1.02)
        cs_discovery_facet_ax.grid(alpha=0.3)

    for cs_discovery_facet_ax in cs_discovery_facet_axes[
        len(cs_discovery_facet_simulation_names) :
    ]:
        cs_discovery_facet_ax.set_visible(False)
    for cs_discovery_facet_ax in cs_discovery_facet_axes[
        : len(cs_discovery_facet_simulation_names)
    ]:
        cs_discovery_facet_ax.set_xlabel("threshold")
        cs_discovery_facet_ax.set_ylabel("discovery rate")

    cs_discovery_facet_handles, cs_discovery_facet_labels = cs_discovery_facet_axes[
        0
    ].get_legend_handles_labels()
    cs_discovery_facet_fig.legend(
        cs_discovery_facet_handles,
        cs_discovery_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    cs_discovery_facet_fig.suptitle(
        f"{nominal_coverage_slider.value:.0%} CS discovery rate vs threshold by simulation",
        y=1.10,
    )
    cs_discovery_facet_fig.tight_layout()
    return (cs_discovery_facet_fig,)


@app.cell(hide_code=True)
def render_cs_size_plot(cs_calibration_grid, nominal_coverage_slider):
    cs_size_df = (
        cs_calibration_grid[
            cs_calibration_grid["beta"] == nominal_coverage_slider.value
        ]
        .groupby(["method", "threshold"], dropna=False)[["cs_size"]]
        .mean()
        .reset_index()
        .rename(columns={"cs_size": "mean_cs_size"})
    )
    cs_size_curve_df = cs_size_df[cs_size_df["threshold"].notna()]
    cs_size_line_df = cs_size_df[cs_size_df["threshold"].isna()]

    cs_size_fig, cs_size_ax = plt.subplots(figsize=(8, 5))
    cs_size_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    for cs_size_method, cs_size_color in cs_size_curve_colors.items():
        cs_size_method_df = cs_size_curve_df[
            cs_size_curve_df["method"] == cs_size_method
        ].sort_values("threshold")
        if cs_size_method_df.empty:
            continue
        cs_size_ax.plot(
            cs_size_method_df["threshold"],
            cs_size_method_df["mean_cs_size"],
            marker="o",
            linewidth=2,
            color=cs_size_color,
            label=cs_size_method,
        )

    cs_size_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    for cs_size_method, cs_size_color in cs_size_line_colors.items():
        cs_size_method_df = cs_size_line_df[cs_size_line_df["method"] == cs_size_method]
        if cs_size_method_df.empty:
            continue
        cs_size_ax.axhline(
            y=cs_size_method_df["mean_cs_size"].iloc[0],
            color=cs_size_color,
            linestyle="--",
            linewidth=2,
            label=cs_size_method,
        )

    cs_size_ax.set_xlabel("threshold")
    cs_size_ax.set_ylabel("mean cs size")
    cs_size_ax.set_title(
        f"Mean {nominal_coverage_slider.value:.0%} CS size vs threshold"
    )
    cs_size_ax.grid(alpha=0.3)
    cs_size_ax.legend()
    cs_size_fig.tight_layout()
    return (cs_size_fig,)


@app.cell
def credible_sets_unconditional_controls_local(
    nominal_coverage_slider,
    threshold_slider,
):
    mo.hstack([nominal_coverage_slider, threshold_slider], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def render_cs_conditional_coverage_plot(
    cs_calibration_grid,
    max_cs_size_slider,
    ser_log_bf_slider,
    threshold_slider,
):
    conditional_threshold = threshold_slider.value
    cs_conditional_df = cs_calibration_grid[
        (
            (cs_calibration_grid["method"] == "logistic_threshold")
            & (cs_calibration_grid["threshold"] == conditional_threshold)
        )
        | (
            (cs_calibration_grid["method"] == "cox_light_threshold")
            & (cs_calibration_grid["threshold"] == conditional_threshold)
        )
        | (
            ~cs_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    cs_conditional_df = cs_conditional_df[
        (cs_conditional_df["cs_size"] <= max_cs_size_slider.value)
        & (cs_conditional_df["ser_log_bf"] > ser_log_bf_slider.value)
    ]
    cs_conditional_summary_df = (
        cs_conditional_df.groupby(["method", "beta"], dropna=False)[["covered"]]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )

    cs_conditional_fig, cs_conditional_ax = plt.subplots(figsize=(8, 5))
    cs_conditional_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_conditional_method in sorted(
        cs_conditional_summary_df["method"].dropna().unique()
    ):
        cs_conditional_method_df = cs_conditional_summary_df[
            cs_conditional_summary_df["method"] == cs_conditional_method
        ].sort_values("beta")
        cs_conditional_label = cs_conditional_method
        if cs_conditional_method in {"logistic_threshold", "cox_light_threshold"}:
            cs_conditional_label = (
                f"{cs_conditional_method} @ {conditional_threshold:g}"
            )
        cs_conditional_ax.plot(
            cs_conditional_method_df["beta"],
            cs_conditional_method_df["empirical_coverage"],
            linewidth=2,
            color=cs_conditional_colors.get(cs_conditional_method),
            label=cs_conditional_label,
        )

    cs_conditional_ax.plot(
        [0.5, 0.99],
        [0.5, 0.99],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="x = y",
    )
    cs_conditional_ax.set_xlabel("nominal coverage")
    cs_conditional_ax.set_ylabel("empirical coverage")
    cs_conditional_ax.set_title("Conditional credible set coverage")
    cs_conditional_ax.set_xlim(0.5, 0.99)
    cs_conditional_ax.set_ylim(0.0, 1.02)
    cs_conditional_ax.grid(alpha=0.3)
    cs_conditional_ax.legend()
    cs_conditional_fig.tight_layout()
    return (cs_conditional_fig,)


@app.cell(hide_code=True)
def render_cs_conditional_size_plot(
    cs_calibration_grid,
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
):
    cs_conditional_size_df = cs_calibration_grid[
        cs_calibration_grid["beta"] == nominal_coverage_slider.value
    ].copy()
    cs_conditional_size_df = cs_conditional_size_df[
        (cs_conditional_size_df["cs_size"] <= max_cs_size_slider.value)
        & (cs_conditional_size_df["ser_log_bf"] > ser_log_bf_slider.value)
    ]
    cs_conditional_size_df = (
        cs_conditional_size_df.groupby(["method", "threshold"], dropna=False)[
            ["cs_size"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"cs_size": "mean_cs_size"})
    )
    cs_conditional_size_curve_df = cs_conditional_size_df[
        cs_conditional_size_df["threshold"].notna()
    ]
    cs_conditional_size_line_df = cs_conditional_size_df[
        cs_conditional_size_df["threshold"].isna()
    ]

    cs_conditional_size_fig, cs_conditional_size_ax = plt.subplots(figsize=(8, 5))
    cs_conditional_size_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    for (
        cs_conditional_size_method,
        cs_conditional_size_color,
    ) in cs_conditional_size_curve_colors.items():
        cs_conditional_size_method_df = cs_conditional_size_curve_df[
            cs_conditional_size_curve_df["method"] == cs_conditional_size_method
        ].sort_values("threshold")
        if cs_conditional_size_method_df.empty:
            continue
        cs_conditional_size_ax.plot(
            cs_conditional_size_method_df["threshold"],
            cs_conditional_size_method_df["mean_cs_size"],
            marker="o",
            linewidth=2,
            color=cs_conditional_size_color,
            label=cs_conditional_size_method,
        )

    cs_conditional_size_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    for (
        cs_conditional_size_method,
        cs_conditional_size_color,
    ) in cs_conditional_size_line_colors.items():
        cs_conditional_size_method_df = cs_conditional_size_line_df[
            cs_conditional_size_line_df["method"] == cs_conditional_size_method
        ]
        if cs_conditional_size_method_df.empty:
            continue
        cs_conditional_size_ax.axhline(
            y=cs_conditional_size_method_df["mean_cs_size"].iloc[0],
            color=cs_conditional_size_color,
            linestyle="--",
            linewidth=2,
            label=cs_conditional_size_method,
        )

    cs_conditional_size_ax.set_xlabel("threshold")
    cs_conditional_size_ax.set_ylabel("mean cs size")
    cs_conditional_size_ax.set_title(
        f"Mean conditional {nominal_coverage_slider.value:.0%} CS size vs threshold"
    )
    cs_conditional_size_ax.grid(alpha=0.3)
    cs_conditional_size_ax.legend()
    cs_conditional_size_fig.tight_layout()
    return (cs_conditional_size_fig,)


@app.cell(hide_code=True)
def render_cs_conditional_size_plot_facet(
    cs_calibration_grid,
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
):
    cs_conditional_size_facet_df = cs_calibration_grid[
        cs_calibration_grid["beta"] == nominal_coverage_slider.value
    ].copy()
    cs_conditional_size_facet_df = cs_conditional_size_facet_df[
        (cs_conditional_size_facet_df["cs_size"] <= max_cs_size_slider.value)
        & (cs_conditional_size_facet_df["ser_log_bf"] > ser_log_bf_slider.value)
    ]
    cs_conditional_size_facet_df = (
        cs_conditional_size_facet_df.groupby(
            ["simulation_name", "method", "threshold"], dropna=False
        )[["cs_size"]]
        .mean()
        .reset_index()
        .rename(columns={"cs_size": "mean_cs_size"})
    )
    cs_conditional_size_facet_names = sorted(
        cs_conditional_size_facet_df["simulation_name"].dropna().unique()
    )
    cs_conditional_size_facet_n_cols = 2
    cs_conditional_size_facet_n_rows = int(
        np.ceil(len(cs_conditional_size_facet_names) / cs_conditional_size_facet_n_cols)
    )
    cs_conditional_size_facet_fig, cs_conditional_size_facet_axes = plt.subplots(
        cs_conditional_size_facet_n_rows,
        cs_conditional_size_facet_n_cols,
        figsize=(12, 4 * cs_conditional_size_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    cs_conditional_size_facet_axes = np.atleast_1d(
        cs_conditional_size_facet_axes
    ).ravel()

    cs_conditional_size_facet_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    cs_conditional_size_facet_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_conditional_size_facet_ax, cs_conditional_size_facet_name in zip(
        cs_conditional_size_facet_axes, cs_conditional_size_facet_names
    ):
        cs_conditional_size_facet_sim_df = cs_conditional_size_facet_df[
            cs_conditional_size_facet_df["simulation_name"]
            == cs_conditional_size_facet_name
        ]
        for (
            cs_conditional_size_facet_method,
            cs_conditional_size_facet_color,
        ) in cs_conditional_size_facet_curve_colors.items():
            cs_conditional_size_facet_method_df = cs_conditional_size_facet_sim_df[
                cs_conditional_size_facet_sim_df["method"]
                == cs_conditional_size_facet_method
            ].sort_values("threshold")
            if cs_conditional_size_facet_method_df.empty:
                continue
            cs_conditional_size_facet_ax.plot(
                cs_conditional_size_facet_method_df["threshold"],
                cs_conditional_size_facet_method_df["mean_cs_size"],
                marker="o",
                linewidth=2,
                color=cs_conditional_size_facet_color,
                label=cs_conditional_size_facet_method,
            )
        for (
            cs_conditional_size_facet_method,
            cs_conditional_size_facet_color,
        ) in cs_conditional_size_facet_line_colors.items():
            cs_conditional_size_facet_method_df = cs_conditional_size_facet_sim_df[
                cs_conditional_size_facet_sim_df["method"]
                == cs_conditional_size_facet_method
            ]
            if cs_conditional_size_facet_method_df.empty:
                continue
            cs_conditional_size_facet_ax.axhline(
                y=cs_conditional_size_facet_method_df["mean_cs_size"].iloc[0],
                color=cs_conditional_size_facet_color,
                linestyle="--",
                linewidth=2,
                label=cs_conditional_size_facet_method,
            )
        cs_conditional_size_facet_ax.set_title(cs_conditional_size_facet_name)
        cs_conditional_size_facet_ax.grid(alpha=0.3)

    for cs_conditional_size_facet_ax in cs_conditional_size_facet_axes[
        len(cs_conditional_size_facet_names) :
    ]:
        cs_conditional_size_facet_ax.set_visible(False)
    for cs_conditional_size_facet_ax in cs_conditional_size_facet_axes[
        : len(cs_conditional_size_facet_names)
    ]:
        cs_conditional_size_facet_ax.set_xlabel("threshold")
        cs_conditional_size_facet_ax.set_ylabel("mean cs size")

    (
        cs_conditional_size_facet_handles,
        cs_conditional_size_facet_labels,
    ) = cs_conditional_size_facet_axes[0].get_legend_handles_labels()
    cs_conditional_size_facet_fig.legend(
        cs_conditional_size_facet_handles,
        cs_conditional_size_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    cs_conditional_size_facet_fig.suptitle(
        f"Mean conditional {nominal_coverage_slider.value:.0%} CS size vs threshold by simulation",
        y=1.02,
    )
    cs_conditional_size_facet_fig.tight_layout()
    return (cs_conditional_size_facet_fig,)


@app.cell(hide_code=True)
def render_cs_conditional_coverage_plot_facet(
    cs_calibration_grid,
    max_cs_size_slider,
    ser_log_bf_slider,
    threshold_slider,
):
    conditional_facet_threshold = threshold_slider.value
    cs_conditional_facet_df = cs_calibration_grid[
        (
            (cs_calibration_grid["method"] == "logistic_threshold")
            & (cs_calibration_grid["threshold"] == conditional_facet_threshold)
        )
        | (
            (cs_calibration_grid["method"] == "cox_light_threshold")
            & (cs_calibration_grid["threshold"] == conditional_facet_threshold)
        )
        | (
            ~cs_calibration_grid["method"].isin(
                ["logistic_threshold", "cox_light_threshold"]
            )
        )
    ]
    cs_conditional_facet_df = cs_conditional_facet_df[
        (cs_conditional_facet_df["cs_size"] <= max_cs_size_slider.value)
        & (cs_conditional_facet_df["ser_log_bf"] > ser_log_bf_slider.value)
    ]
    cs_conditional_facet_summary_df = (
        cs_conditional_facet_df.groupby(
            ["simulation_name", "method", "beta"], dropna=False
        )[["covered"]]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )
    cs_conditional_facet_names = sorted(
        cs_conditional_facet_summary_df["simulation_name"].dropna().unique()
    )
    cs_conditional_facet_n_cols = 2
    cs_conditional_facet_n_rows = int(
        np.ceil(len(cs_conditional_facet_names) / cs_conditional_facet_n_cols)
    )
    cs_conditional_facet_fig, cs_conditional_facet_axes = plt.subplots(
        cs_conditional_facet_n_rows,
        cs_conditional_facet_n_cols,
        figsize=(12, 4 * cs_conditional_facet_n_rows),
        sharex=True,
        sharey=True,
    )
    cs_conditional_facet_axes = np.atleast_1d(cs_conditional_facet_axes).ravel()

    cs_conditional_facet_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    for cs_conditional_facet_ax, cs_conditional_facet_name in zip(
        cs_conditional_facet_axes, cs_conditional_facet_names
    ):
        cs_conditional_facet_sim_df = cs_conditional_facet_summary_df[
            cs_conditional_facet_summary_df["simulation_name"]
            == cs_conditional_facet_name
        ]
        for cs_conditional_facet_method in sorted(
            cs_conditional_facet_sim_df["method"].dropna().unique()
        ):
            cs_conditional_facet_method_df = cs_conditional_facet_sim_df[
                cs_conditional_facet_sim_df["method"] == cs_conditional_facet_method
            ].sort_values("beta")
            cs_conditional_facet_label = cs_conditional_facet_method
            if cs_conditional_facet_method in {
                "logistic_threshold",
                "cox_light_threshold",
            }:
                cs_conditional_facet_label = (
                    f"{cs_conditional_facet_method} @ {conditional_facet_threshold:g}"
                )
            cs_conditional_facet_ax.plot(
                cs_conditional_facet_method_df["beta"],
                cs_conditional_facet_method_df["empirical_coverage"],
                linewidth=2,
                color=cs_conditional_facet_colors.get(cs_conditional_facet_method),
                label=cs_conditional_facet_label,
            )
        cs_conditional_facet_ax.plot(
            [0.5, 0.99],
            [0.5, 0.99],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="x = y",
        )
        cs_conditional_facet_ax.set_title(cs_conditional_facet_name)
        cs_conditional_facet_ax.set_xlim(0.5, 0.99)
        cs_conditional_facet_ax.set_ylim(0.0, 1.02)
        cs_conditional_facet_ax.grid(alpha=0.3)

    for cs_conditional_facet_ax in cs_conditional_facet_axes[
        len(cs_conditional_facet_names) :
    ]:
        cs_conditional_facet_ax.set_visible(False)

    for cs_conditional_facet_ax in cs_conditional_facet_axes[
        : len(cs_conditional_facet_names)
    ]:
        cs_conditional_facet_ax.set_xlabel("nominal coverage")
        cs_conditional_facet_ax.set_ylabel("empirical coverage")

    cs_conditional_facet_handles, cs_conditional_facet_labels = (
        cs_conditional_facet_axes[0].get_legend_handles_labels()
    )
    cs_conditional_facet_fig.legend(
        cs_conditional_facet_handles,
        cs_conditional_facet_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    cs_conditional_facet_fig.suptitle(
        "Conditional credible set coverage by simulation", y=1.02
    )
    cs_conditional_facet_fig.tight_layout()
    return (cs_conditional_facet_fig,)


@app.cell
def cs_controls():
    nominal_coverage_slider = mo.ui.slider(
        start=0.50,
        stop=0.95,
        step=0.05,
        value=0.95,
        label="nominal coverage",
        show_value=True,
    )
    max_cs_size_slider = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=50,
        label="max cs size",
        show_value=True,
    )
    ser_log_bf_slider = mo.ui.slider(
        start=-1.0,
        stop=10.0,
        step=0.5,
        value=2.0,
        label="min ser log bf",
        show_value=True,
    )
    return max_cs_size_slider, nominal_coverage_slider, ser_log_bf_slider


@app.cell(hide_code=True)
def credible_sets_section():
    mo.md("""
    ## Credible Set Summary
    """)
    return


@app.cell(hide_code=True)
def credible_sets_summary_controls_local(
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
):
    mo.hstack(
        [nominal_coverage_slider, max_cs_size_slider, ser_log_bf_slider],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def render_cs_summary_panel(
    cs_calibration_grid,
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
):
    summary_df = cs_calibration_grid[
        cs_calibration_grid["beta"] == nominal_coverage_slider.value
    ].copy()
    summary_df["passes_filter"] = (
        summary_df["cs_size"] <= max_cs_size_slider.value
    ) & (summary_df["ser_log_bf"] > ser_log_bf_slider.value)
    summary_df["discovered"] = summary_df["covered"].astype(int) * summary_df[
        "passes_filter"
    ].astype(int)
    power_summary_df = (
        summary_df.groupby(["method", "threshold"], dropna=False)[["discovered"]]
        .mean()
        .reset_index()
        .rename(columns={"discovered": "discovery_rate"})
    )
    filtered_df = summary_df[summary_df["passes_filter"]]
    size_summary_df = (
        filtered_df.groupby(["method", "threshold"], dropna=False)[["cs_size"]]
        .mean()
        .reset_index()
        .rename(columns={"cs_size": "mean_cs_size"})
    )
    coverage_summary_df = (
        filtered_df.groupby(["method", "threshold"], dropna=False)[["covered"]]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )

    curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }

    panel_specs = [
        ("Power", power_summary_df, "discovery_rate", (0.0, 1.02)),
        ("CS Size", size_summary_df, "mean_cs_size", None),
        ("Coverage", coverage_summary_df, "empirical_coverage", (0.0, 1.02)),
    ]
    cs_summary_fig, cs_summary_axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True)

    for cs_summary_ax, (panel_title, panel_df, value_col, ylim) in zip(
        cs_summary_axes, panel_specs
    ):
        curve_df = panel_df[panel_df["threshold"].notna()]
        line_df = panel_df[panel_df["threshold"].isna()]

        for curve_method, curve_color in curve_colors.items():
            curve_method_df = curve_df[curve_df["method"] == curve_method].sort_values(
                "threshold"
            )
            if curve_method_df.empty:
                continue
            cs_summary_ax.plot(
                curve_method_df["threshold"],
                curve_method_df[value_col],
                marker="o",
                linewidth=2,
                color=curve_color,
                label=curve_method,
            )

        for line_method, line_color in line_colors.items():
            line_method_df = line_df[line_df["method"] == line_method]
            if line_method_df.empty:
                continue
            cs_summary_ax.axhline(
                y=line_method_df[value_col].iloc[0],
                color=line_color,
                linestyle="--",
                linewidth=2,
                label=line_method,
            )

        cs_summary_ax.set_title(panel_title)
        cs_summary_ax.set_xlabel("threshold")
        cs_summary_ax.set_box_aspect(1)
        if ylim is not None:
            cs_summary_ax.set_ylim(*ylim)
        cs_summary_ax.grid(alpha=0.3)

    cs_summary_axes[0].set_ylabel("proportion discovered")
    cs_summary_axes[1].set_ylabel("mean cs size")
    cs_summary_axes[2].set_ylabel("empirical coverage")
    handles, labels = cs_summary_axes[0].get_legend_handles_labels()
    cs_summary_fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.62),
        frameon=False,
    )
    cs_summary_fig.suptitle(
        f"{nominal_coverage_slider.value:.0%} Credible Set",
        y=0.98,
    )
    legend_note = "\n".join(
        [
            f"Nominal coverage: {nominal_coverage_slider.value:.0%}",
            f"Max CS size: {max_cs_size_slider.value}",
            f"Min SER log BF: {ser_log_bf_slider.value:.1f}",
        ]
    )
    cs_summary_fig.text(
        1.01,
        0.36,
        legend_note,
        transform=cs_summary_fig.transFigure,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    cs_summary_fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.95))
    return (cs_summary_fig,)


@app.cell
def render_cs_summary_panel_facet(
    cs_calibration_grid,
    max_cs_size_slider,
    nominal_coverage_slider,
    ser_log_bf_slider,
):
    csf_df = cs_calibration_grid[
        cs_calibration_grid["beta"] == nominal_coverage_slider.value
    ].copy()
    csf_df["csf_passes_filter"] = (
        (csf_df["cs_size"] <= max_cs_size_slider.value)
        & (csf_df["ser_log_bf"] > ser_log_bf_slider.value)
    )
    csf_df["csf_discovered"] = (
        csf_df["covered"].astype(int) * csf_df["csf_passes_filter"].astype(int)
    )
    csf_power_df = (
        csf_df.groupby(["simulation_name", "method", "threshold"], dropna=False)[
            ["csf_discovered"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"csf_discovered": "discovery_rate"})
    )
    csf_filtered_df = csf_df[csf_df["csf_passes_filter"]]
    csf_size_df = (
        csf_filtered_df.groupby(
            ["simulation_name", "method", "threshold"], dropna=False
        )[["cs_size"]]
        .mean()
        .reset_index()
        .rename(columns={"cs_size": "mean_cs_size"})
    )
    csf_coverage_df = (
        csf_filtered_df.groupby(
            ["simulation_name", "method", "threshold"], dropna=False
        )[["covered"]]
        .mean()
        .reset_index()
        .rename(columns={"covered": "empirical_coverage"})
    )

    csf_curve_colors = {
        "logistic_threshold": "#1f77b4",
        "cox_light_threshold": "#ff7f0e",
    }
    csf_line_colors = {
        "twogroup": "#2ca02c",
        "twogroup_oracle": "#d62728",
        "logistic_oracle": "#9467bd",
        "cox_heavy": "#8c564b",
    }
    csf_panel_specs = [
        ("Power", csf_power_df, "discovery_rate", (0.0, 1.02)),
        ("CS Size", csf_size_df, "mean_cs_size", None),
        ("Coverage", csf_coverage_df, "empirical_coverage", (0.0, 1.02)),
    ]
    csf_simulation_names = sorted(csf_df["simulation_name"].dropna().unique())
    csf_n_rows = len(csf_simulation_names)
    cs_summary_facet_fig, cs_summary_facet_axes = plt.subplots(
        csf_n_rows,
        3,
        figsize=(12, 4.2 * csf_n_rows),
        sharex=True,
    )
    cs_summary_facet_axes = np.atleast_2d(cs_summary_facet_axes)

    for csf_row_idx, csf_simulation_name in enumerate(csf_simulation_names):
        for csf_col_idx, (
            csf_panel_title,
            csf_panel_df,
            csf_value_col,
            csf_ylim,
        ) in enumerate(csf_panel_specs):
            csf_ax = cs_summary_facet_axes[csf_row_idx, csf_col_idx]
            csf_simulation_df = csf_panel_df[
                csf_panel_df["simulation_name"] == csf_simulation_name
            ]
            csf_curve_df = csf_simulation_df[csf_simulation_df["threshold"].notna()]
            csf_line_df = csf_simulation_df[csf_simulation_df["threshold"].isna()]

            for csf_curve_method, csf_curve_color in csf_curve_colors.items():
                csf_curve_method_df = csf_curve_df[
                    csf_curve_df["method"] == csf_curve_method
                ].sort_values("threshold")
                if csf_curve_method_df.empty:
                    continue
                csf_ax.plot(
                    csf_curve_method_df["threshold"],
                    csf_curve_method_df[csf_value_col],
                    marker="o",
                    linewidth=2,
                    color=csf_curve_color,
                    label=csf_curve_method,
                )

            for csf_line_method, csf_line_color in csf_line_colors.items():
                csf_line_method_df = csf_line_df[
                    csf_line_df["method"] == csf_line_method
                ]
                if csf_line_method_df.empty:
                    continue
                csf_ax.axhline(
                    y=csf_line_method_df[csf_value_col].iloc[0],
                    color=csf_line_color,
                    linestyle="--",
                    linewidth=2,
                    label=csf_line_method,
                )

            if csf_row_idx == 0:
                csf_ax.set_title(csf_panel_title)
            if csf_col_idx == 0:
                csf_ax.set_ylabel(
                    f"{csf_simulation_name}\n\nproportion discovered"
                )
            elif csf_col_idx == 1:
                csf_ax.set_ylabel("mean cs size")
            else:
                csf_ax.set_ylabel("empirical coverage")
            csf_ax.set_xlabel("threshold")
            csf_ax.set_box_aspect(1)
            if csf_ylim is not None:
                csf_ax.set_ylim(*csf_ylim)
            csf_ax.grid(alpha=0.3)

    csf_handles, csf_labels = cs_summary_facet_axes[0, 0].get_legend_handles_labels()
    cs_summary_facet_fig.legend(
        csf_handles,
        csf_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.62),
        frameon=False,
    )
    csf_legend_note = "\n".join(
        [
            f"Nominal coverage: {nominal_coverage_slider.value:.0%}",
            f"Max CS size: {max_cs_size_slider.value}",
            f"Min SER log BF: {ser_log_bf_slider.value:.1f}",
        ]
    )
    cs_summary_facet_fig.text(
        1.01,
        0.36,
        csf_legend_note,
        transform=cs_summary_facet_fig.transFigure,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    cs_summary_facet_fig.suptitle(
        f"{nominal_coverage_slider.value:.0%} Credible Set",
        y=0.995,
    )
    cs_summary_facet_fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.97))
    return (cs_summary_facet_fig,)


@app.cell
def _(cs_summary_facet_fig, cs_summary_fig):
    mo.tabs(
        {
            "Aggregate": cs_summary_fig,
            "By Simulation Scenario": cs_summary_facet_fig,
        }
    )
    return


if __name__ == "__main__":
    app.run()
