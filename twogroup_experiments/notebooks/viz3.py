import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import polars as pl

    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    import viz3_utils


@app.cell(hide_code=True)
def title_cell():
    # Render notebook title.
    title_md = mo.md("# Viz3")
    title_md
    return


@app.cell
def collection_selector_cell():
    # Build collection selector from materialized alias specs.
    collection_alias_root = Path(__file__).parent.parent / "results" / "by_alias"
    collection_specs = viz3_utils.load_collection_specs(collection_alias_root)
    collection_names = sorted(collection_specs)
    collections_with_pip_plot_data = [
        collection_name
        for collection_name in collection_names
        if any(
            (collection_alias_root / collection_name).glob(
                "batches/*/fits/*/pip_threshold_plot_data.parquet"
            )
        )
    ]
    default_collection = (
        "hallmark__ser_enrich__loc"
        if "hallmark__ser_enrich__loc" in collections_with_pip_plot_data
        else collections_with_pip_plot_data[0]
    )
    collection_dropdown = mo.ui.dropdown(
        options=collections_with_pip_plot_data,
        value=default_collection,
        label="collection",
    )
    return collection_alias_root, collection_dropdown


@app.cell
def collection_bundle_cell(collection_alias_root, collection_dropdown):
    # Load selected collection bundle from alias root.
    selected_root = collection_alias_root / collection_dropdown.value
    collection_bundle = viz3_utils.load_collection_bundle(selected_root)
    return (collection_bundle,)


@app.cell
def shared_method_controls_cell(collection_bundle):
    # Build shared method-family and L controls from loaded data.
    method_source = collection_bundle.pip_threshold_plot_data
    method_families = viz3_utils.available_method_families(method_source)
    L_values = viz3_utils.available_L_values(method_source)
    method_family_multiselect = mo.ui.multiselect(
        options=method_families,
        value=method_families,
        label="method family",
    )
    L_dropdown = mo.ui.dropdown(
        options=L_values,
        value=L_values[0],
        label="L",
    )
    thresholds = viz3_utils.available_thresholds(method_source)
    threshold_control = mo.ui.dropdown(
        options=thresholds,
        value=thresholds[0] if thresholds else 1.0,
        label="threshold",
    )
    return L_dropdown, method_family_multiselect, threshold_control


@app.cell
def selected_methods_cell(
    L_dropdown,
    collection_bundle,
    method_family_multiselect,
):
    # Derive selected method names from shared controls.
    selected_methods = viz3_utils.selected_method_names(
        collection_bundle.pip_threshold_plot_data,
        selected_method_families=method_family_multiselect.value,
        selected_L=L_dropdown.value,
    )
    return (selected_methods,)


@app.cell(hide_code=True)
def notebook_overview_cell(
    L_dropdown,
    collection_dropdown,
    method_family_multiselect,
    threshold_control,
):
    # Render visible shell controls and planned plot-family list.
    overview_controls = mo.vstack(
        [
            mo.hstack(
                [collection_dropdown, method_family_multiselect, L_dropdown],
                justify="start",
                gap=2,
            ),
            mo.hstack([threshold_control], justify="start", gap=2),
        ],
        gap=1,
    )
    planned_sections_md = mo.md(
        "## Planned Plot Families\n"
        "1. PIP Calibration\n"
        "2. Power vs FDP\n"
        "3. Causal PIP vs Threshold\n"
        "4. Credible Set Summary\n"
        "5. Credible Set Scenario Points\n"
        "6. Credible Set Histograms\n"
    )
    overview_stack = mo.vstack([overview_controls, planned_sections_md], gap=2)
    overview_stack
    return


@app.cell(hide_code=True)
def pip_calibration_heading_cell():
    # Render calibration section heading.
    pip_calibration_heading_md = mo.md("## PIP Calibration")
    pip_calibration_heading_md
    return


@app.cell
def pip_calibration_summary_cell(
    collection_bundle,
    selected_methods,
    threshold_control,
):
    # Prepare calibration summary from selected methods and threshold.
    calibration_input = viz3_utils.filter_selected_methods(
        viz3_utils.filter_thresholded_methods(
            collection_bundle.pip_threshold_plot_data,
            selected_threshold=threshold_control.value,
        ),
        selected_methods,
    )
    calibration_input = viz3_utils.add_method_display_labels(
        calibration_input,
        selected_threshold=threshold_control.value,
    )
    pip_calibration_summary = viz3_utils.summarize_pip_calibration(calibration_input)
    return (pip_calibration_summary,)


@app.cell
def pip_calibration_charts_cell(pip_calibration_summary):
    # Render aggregate and by-simulation calibration charts.
    pip_calibration_aggregate_input = pip_calibration_summary.group_by(
        "method_family",
        "method_display",
        "series_label",
        "pip_bin_index",
        "pip_left",
        "pip_right",
        "pip_mid",
    ).agg(
        pl.col("n_total").sum().alias("n_total"),
        pl.col("n_causal").sum().alias("n_causal"),
    )
    pip_calibration_aggregate_summary = viz3_utils.summarize_calibration_with_bootstrap(
        pip_calibration_aggregate_input,
        group_cols=["method_family", "method_display", "series_label"],
    ).with_columns(pl.lit("Aggregate").alias("simulation_name"))
    pip_calibration_by_sim_summary = viz3_utils.summarize_calibration_with_bootstrap(
        pip_calibration_summary,
        group_cols=["simulation_name", "method_family", "method_display", "series_label"],
    )
    pip_calibration_aggregate_chart = viz3_utils.render_pip_calibration(
        pip_calibration_aggregate_summary,
        facet_by_simulation=False,
    )
    pip_calibration_by_sim_chart = viz3_utils.render_pip_calibration(
        pip_calibration_by_sim_summary,
        facet_by_simulation=True,
    )
    return pip_calibration_aggregate_chart, pip_calibration_by_sim_chart


@app.cell(hide_code=True)
def pip_calibration_tabs_cell(
    pip_calibration_aggregate_chart,
    pip_calibration_by_sim_chart,
):
    # Display calibration charts in aggregate and by-simulation tabs.
    pip_calibration_tabs = mo.ui.tabs(
        {
            "Aggregate": pip_calibration_aggregate_chart,
            "By Simulation Scenario": pip_calibration_by_sim_chart,
        }
    )
    pip_calibration_tabs
    return


@app.cell(hide_code=True)
def power_fdp_heading_cell():
    # Render power-vs-FDP section heading.
    power_fdp_heading_md = mo.md("## Power vs FDP")
    power_fdp_heading_md
    return


@app.cell
def power_fdp_controls_cell():
    # Create plot-local controls for power-vs-FDP views.
    max_fdp_control = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
        value=0.5,
        label="max FDP",
    )
    fixed_y_scale_control = mo.ui.checkbox(value=True, label="fixed y scale")
    show_background_threshold_traces_control = mo.ui.checkbox(
        value=True,
        label="show faint non-selected thresholds",
    )
    return (
        fixed_y_scale_control,
        max_fdp_control,
        show_background_threshold_traces_control,
    )


@app.cell(hide_code=True)
def power_fdp_controls_view_cell(
    fixed_y_scale_control,
    max_fdp_control,
    show_background_threshold_traces_control,
):
    # Render plot-local controls for power-vs-FDP.
    power_fdp_controls = mo.hstack(
        [
            max_fdp_control,
            show_background_threshold_traces_control,
            fixed_y_scale_control,
        ],
        justify="start",
        gap=2,
    )
    power_fdp_controls
    return


@app.cell
def power_fdp_summary_cell(
    collection_bundle,
    selected_methods,
    show_background_threshold_traces_control,
    threshold_control,
):
    # Prepare power-vs-FDP summary from selected methods and controls.
    prepared_power_fdp_data = viz3_utils.prepare_power_fdp_plot_data_frame(
        collection_bundle.pip_threshold_plot_data,
        selected_threshold=threshold_control.value,
        selected_methods=selected_methods,
        show_background_threshold_traces=show_background_threshold_traces_control.value,
    )
    power_fdp_summary = viz3_utils.make_power_fdp_summary(prepared_power_fdp_data)
    return (power_fdp_summary,)


@app.cell
def power_fdp_charts_cell(
    fixed_y_scale_control,
    max_fdp_control,
    power_fdp_summary,
):
    # Render aggregate and by-simulation power-vs-FDP charts.
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
    power_fdp_aggregate_chart = viz3_utils.render_power_fdp_chart(
        aggregate_power_fdp_summary,
        facet=False,
        max_fdp=max_fdp_control.value,
        fixed_y_scale=fixed_y_scale_control.value,
    )
    power_fdp_by_sim_chart = viz3_utils.render_power_fdp_chart(
        power_fdp_summary,
        facet=True,
        max_fdp=max_fdp_control.value,
        fixed_y_scale=fixed_y_scale_control.value,
    )
    return power_fdp_aggregate_chart, power_fdp_by_sim_chart


@app.cell(hide_code=True)
def power_fdp_tabs_cell(power_fdp_aggregate_chart, power_fdp_by_sim_chart):
    # Display power-vs-FDP charts in aggregate and by-simulation tabs.
    power_fdp_tabs = mo.ui.tabs(
        {
            "Aggregate": power_fdp_aggregate_chart,
            "By Simulation Scenario": power_fdp_by_sim_chart,
        }
    )
    power_fdp_tabs
    return


@app.cell(hide_code=True)
def causal_pip_heading_cell():
    # Render causal-PIP section heading.
    causal_pip_heading_md = mo.md("## Causal PIP vs Threshold")
    causal_pip_heading_md
    return


@app.cell
def causal_pip_charts_cell(collection_bundle, selected_methods):
    # Render aggregate and by-simulation causal-PIP charts.
    selected_causal_pip_data = viz3_utils.filter_selected_methods(
        collection_bundle.causal_pip_plot_data,
        selected_methods,
    )
    causal_pip_aggregate_summary = viz3_utils.make_causal_pip_summary(
        selected_causal_pip_data.with_columns(
            pl.lit("Aggregate").alias("simulation_name")
        )
    )
    causal_pip_by_sim_summary = viz3_utils.make_causal_pip_summary(
        selected_causal_pip_data
    )
    causal_pip_aggregate_chart = viz3_utils.render_causal_pip_chart(
        causal_pip_aggregate_summary,
        facet=False,
    )
    causal_pip_by_sim_chart = viz3_utils.render_causal_pip_chart(
        causal_pip_by_sim_summary,
        facet=True,
    )
    return causal_pip_aggregate_chart, causal_pip_by_sim_chart


@app.cell(hide_code=True)
def causal_pip_tabs_cell(causal_pip_aggregate_chart, causal_pip_by_sim_chart):
    # Display causal-PIP charts in aggregate and by-simulation tabs.
    causal_pip_tabs = mo.ui.tabs(
        {
            "Aggregate": causal_pip_aggregate_chart,
            "By Simulation Scenario": causal_pip_by_sim_chart,
        }
    )
    causal_pip_tabs
    return


@app.cell(hide_code=True)
def cs_summary_heading_cell():
    # Render credible-set summary section heading.
    cs_summary_heading_md = mo.md("## Credible Set Summary")
    cs_summary_heading_md
    return


@app.cell
def cs_summary_controls_cell():
    # Create plot-local controls for credible-set summary views.
    nominal_coverage_control = mo.ui.slider(
        start=0.5,
        stop=0.99,
        step=0.01,
        value=0.95,
        label="nominal coverage",
    )
    max_cs_size_control = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=20,
        label="max CS size",
    )
    min_ser_log_bf_control = mo.ui.slider(
        start=0.0,
        stop=20.0,
        step=0.5,
        value=2.0,
        label="min SER log BF",
    )
    return (
        max_cs_size_control,
        min_ser_log_bf_control,
        nominal_coverage_control,
    )


@app.cell(hide_code=True)
def cs_summary_controls_view_cell(
    max_cs_size_control,
    min_ser_log_bf_control,
    nominal_coverage_control,
):
    # Render plot-local controls for credible-set summary.
    cs_summary_controls = mo.hstack(
        [nominal_coverage_control, max_cs_size_control, min_ser_log_bf_control],
        justify="start",
        gap=2,
    )
    cs_summary_controls
    return


@app.cell
def cs_summary_data_cell(
    collection_bundle,
    max_cs_size_control,
    min_ser_log_bf_control,
    nominal_coverage_control,
    selected_methods,
):
    # Prepare aggregate and by-simulation credible-set summaries.
    aggregate_cs_summary, by_sim_cs_summary = viz3_utils.make_conditional_cs_summary(
        viz3_utils.filter_selected_methods(
            collection_bundle.cs_component_plot_data, selected_methods
        ),
        viz3_utils.filter_selected_methods(
            collection_bundle.cs_truth_plot_data, selected_methods
        ),
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    return aggregate_cs_summary, by_sim_cs_summary


@app.cell
def cs_summary_charts_cell(
    aggregate_cs_summary,
    by_sim_cs_summary,
    max_cs_size_control,
    min_ser_log_bf_control,
    nominal_coverage_control,
):
    # Render aggregate and by-simulation credible-set summary charts.
    cs_summary_aggregate_chart = viz3_utils.render_conditional_cs_summary_chart(
        aggregate_cs_summary,
        facet=False,
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    cs_summary_by_sim_chart = viz3_utils.render_conditional_cs_summary_chart(
        by_sim_cs_summary,
        facet=True,
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    return cs_summary_aggregate_chart, cs_summary_by_sim_chart


@app.cell(hide_code=True)
def cs_summary_tabs_cell(cs_summary_aggregate_chart, cs_summary_by_sim_chart):
    # Display credible-set summary charts in aggregate and by-simulation tabs.
    cs_summary_tabs = mo.ui.tabs(
        {
            "Aggregate": cs_summary_aggregate_chart,
            "By Simulation Scenario": cs_summary_by_sim_chart,
        }
    )
    cs_summary_tabs
    return


@app.cell(hide_code=True)
def cs_scenario_points_heading_cell():
    # Render credible-set scenario-points section heading.
    cs_scenario_points_heading_md = mo.md("## Credible Set Scenario Points")
    cs_scenario_points_heading_md
    return


@app.cell
def cs_scenario_points_data_cell(
    collection_bundle,
    max_cs_size_control,
    min_ser_log_bf_control,
    nominal_coverage_control,
    selected_methods,
    threshold_control,
):
    # Prepare replicate-level CS summary and bootstrap scenario points.
    replicate_cs_summary = viz3_utils.make_conditional_cs_replicate_summary(
        viz3_utils.filter_selected_methods(
            collection_bundle.cs_component_plot_data, selected_methods
        ),
        viz3_utils.filter_selected_methods(
            collection_bundle.cs_truth_plot_data, selected_methods
        ),
        nominal_coverage=nominal_coverage_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    selected_threshold_cs_summary = viz3_utils.select_current_threshold_cs_rows(
        replicate_cs_summary,
        selected_threshold=threshold_control.value,
    )
    scenario_points_summary = viz3_utils.summarize_replicate_metric_with_bootstrap(
        selected_threshold_cs_summary,
        group_cols=["simulation_name", "method", "method_display", "metric"],
    )
    return (scenario_points_summary,)


@app.cell
def cs_scenario_points_chart_cell(scenario_points_summary, threshold_control):
    # Render credible-set scenario points chart.
    cs_scenario_points_chart = viz3_utils.render_conditional_cs_scenario_points_chart(
        scenario_points_summary,
        selected_threshold=threshold_control.value,
    )
    return (cs_scenario_points_chart,)


@app.cell(hide_code=True)
def cs_scenario_points_view_cell(cs_scenario_points_chart):
    # Display credible-set scenario points chart.
    cs_scenario_points_chart
    return


@app.cell(hide_code=True)
def cs_histograms_heading_cell():
    # Render credible-set histogram section heading.
    cs_histograms_heading_md = mo.md("## Credible Set Histograms")
    cs_histograms_heading_md
    return


@app.cell
def cs_histogram_data_cell(
    collection_bundle,
    nominal_coverage_control,
    selected_methods,
    threshold_control,
):
    # Prepare CS size and SER log BF histogram inputs.
    histogram_component_data = viz3_utils.filter_selected_methods(
        collection_bundle.cs_component_plot_data,
        selected_methods,
    )
    cs_size_histogram_data, ser_log_bf_histogram_data = (
        viz3_utils.prepare_cs_histogram_data(
            histogram_component_data,
            nominal_coverage=nominal_coverage_control.value,
            selected_threshold=threshold_control.value,
        )
    )
    return cs_size_histogram_data, ser_log_bf_histogram_data


@app.cell
def cs_histogram_chart_cell(
    cs_size_histogram_data,
    max_cs_size_control,
    min_ser_log_bf_control,
    ser_log_bf_histogram_data,
    threshold_control,
):
    # Render CS histogram panel.
    cs_histograms_figure = viz3_utils.render_cs_histograms(
        cs_size_histogram_data,
        ser_log_bf_histogram_data,
        selected_threshold=threshold_control.value,
        max_cs_size=max_cs_size_control.value,
        min_ser_log_bf=min_ser_log_bf_control.value,
    )
    return (cs_histograms_figure,)


@app.cell(hide_code=True)
def cs_histogram_view_cell(cs_histograms_figure):
    # Display CS histogram panel.
    cs_histograms_figure
    return


if __name__ == "__main__":
    app.run()
