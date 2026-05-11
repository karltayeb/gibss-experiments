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

    import plot_ready
    import viz3_utils


@app.cell(hide_code=True)
def title_cell():
    _title_md = mo.md("# Viz4: Plot-Ready")
    _title_md
    return


@app.cell
def collection_selector_cell():
    collection_alias_root = Path(__file__).parent.parent / "results" / "by_alias"
    collections = plot_ready.available_plot_ready_collections(collection_alias_root)
    collection_dropdown = mo.ui.dropdown(
        options=collections,
        value=None,
        allow_select_none=True,
        label="collection",
    )
    collection_dropdown
    return collection_alias_root, collection_dropdown


@app.cell
def bundle_cell(collection_alias_root, collection_dropdown):
    mo.stop(
        collection_dropdown.value is None,
        mo.md("Select a collection to load plot-ready data."),
    )
    bundle = plot_ready.load_plot_ready_collection(
        collection_alias_root / collection_dropdown.value
    )
    return (bundle,)


@app.cell
def controls_cell(bundle, collection_dropdown):
    _method_metadata = bundle["method_metadata"]

    # Threshold: all distinct non-null thresholds from method_metadata
    _all_thresholds = sorted(
        _method_metadata.filter(pl.col("threshold").is_not_null())["threshold"].unique().to_list()
    )
    threshold_dropdown = mo.ui.dropdown(
        options=_all_thresholds,
        value=_all_thresholds[0] if _all_thresholds else None,
        label="threshold",
    )

    # Method family multiselect
    _all_families = sorted(_method_metadata["method_family"].unique().to_list())
    method_family_multiselect = mo.ui.multiselect(
        options=_all_families,
        value=_all_families,
        label="method family",
    )

    # L dropdown
    _all_L = sorted(_method_metadata["L"].unique().to_list())
    L_dropdown = mo.ui.dropdown(
        options=_all_L,
        value=_all_L[0] if _all_L else 1,
        label="L",
    )

    mo.hstack([collection_dropdown, method_family_multiselect, L_dropdown, threshold_dropdown])
    return L_dropdown, method_family_multiselect, threshold_dropdown


@app.cell
def selected_methods_cell(bundle, L_dropdown, method_family_multiselect, threshold_dropdown):
    _method_metadata = bundle["method_metadata"]
    selected_threshold = threshold_dropdown.value

    # Foreground: matches selected family + L, and (non-thresholded OR selected threshold OR null threshold)
    _fg_mask = (
        pl.col("method_family").is_in(method_family_multiselect.value)
        & (pl.col("L") == L_dropdown.value)
        & (
            ~pl.col("is_thresholded")
            | (pl.col("threshold") == selected_threshold)
            | pl.col("threshold").is_null()
        )
    )
    foreground_methods = set(
        _method_metadata.filter(_fg_mask)["method"].to_list()
    )

    # Background: thresholded methods in selected family/L but other thresholds
    _bg_mask = (
        pl.col("method_family").is_in(method_family_multiselect.value)
        & (pl.col("L") == L_dropdown.value)
        & pl.col("is_thresholded")
        & (pl.col("threshold") != selected_threshold)
    )
    _bg_filtered = _method_metadata.filter(_bg_mask)
    background_methods_thresholds = set(
        zip(
            _bg_filtered["method"].to_list(),
            _bg_filtered["threshold"].to_list(),
        )
    )
    return background_methods_thresholds, foreground_methods, selected_threshold


@app.cell(hide_code=True)
def pip_calibration_heading_cell():
    _pip_calibration_heading_md = mo.md("## PIP Calibration")
    _pip_calibration_heading_md
    return


@app.cell
def pip_calibration_cell(bundle, foreground_methods):
    _pip_cal = bundle["pip_calibration"]
    _method_meta = bundle["method_metadata"]

    # Join with method_metadata to get display columns, then add series_label
    _enriched = (
        _pip_cal
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select("method", "threshold", "method_display", "method_family"),
            on=["method", "threshold"],
            how="left",
        )
        .with_columns(
            pl.col("method_display").alias("series_label"),
            pl.lit("Aggregate").alias("simulation_name"),
        )
    )

    if _enriched.is_empty():
        pip_cal_chart = viz3_utils.make_placeholder_chart("No PIP calibration data")
    else:
        pip_cal_chart = viz3_utils.render_pip_calibration(_enriched, facet_by_simulation=False)

    pip_cal_chart
    return


@app.cell(hide_code=True)
def power_fdp_heading_cell():
    _power_fdp_heading_md = mo.md("## Power vs FDP")
    _power_fdp_heading_md
    return


@app.cell
def power_fdp_cell(bundle, background_methods_thresholds, foreground_methods):
    _power_fdp = bundle["power_fdp"]
    _method_meta = bundle["method_metadata"]

    # Build foreground dataframe (is_selected_threshold=True)
    _fg_df = (
        _power_fdp
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select(
                "method", "threshold", "method_display", "method_family",
                "method_label_base", "is_thresholded",
            ),
            on=["method", "threshold"],
            how="left",
        )
        .with_columns(pl.lit(True).alias("is_selected_threshold"))
    )

    # Build background dataframe (is_selected_threshold=False)
    _bg_rows = list(background_methods_thresholds)
    if _bg_rows:
        _bg_methods = [r[0] for r in _bg_rows]
        _bg_thresholds = [r[1] for r in _bg_rows]
        _bg_df = (
            _power_fdp
            .filter(pl.col("method").is_in(_bg_methods))
            .join(
                _method_meta.select(
                    "method", "threshold", "method_display", "method_family",
                    "method_label_base", "is_thresholded",
                ),
                on=["method", "threshold"],
                how="left",
            )
            .filter(pl.col("threshold").is_in(_bg_thresholds))
            .with_columns(pl.lit(False).alias("is_selected_threshold"))
        )
        _combined = pl.concat([_fg_df, _bg_df])
    else:
        _combined = _fg_df

    # Add trace_label and legend_label
    _combined = _combined.with_columns(
        pl.when(pl.col("is_thresholded"))
        .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
        .otherwise(pl.col("method_display"))
        .alias("trace_label"),
        pl.when(pl.col("is_selected_threshold"))
        .then(pl.col("method_display"))
        .otherwise(None)
        .alias("legend_label"),
    )

    if _combined.is_empty():
        power_fdp_chart = viz3_utils.make_placeholder_chart("No power/FDP data")
    else:
        power_fdp_chart = viz3_utils.render_power_fdp_chart(
            _combined, facet=False, max_fdp=0.5, fixed_y_scale=True
        )

    power_fdp_chart
    return


@app.cell(hide_code=True)
def causal_pip_heading_cell():
    _causal_pip_heading_md = mo.md("## Causal PIP vs Threshold")
    _causal_pip_heading_md
    return


@app.cell
def causal_pip_cell(bundle, foreground_methods):
    _causal_pip = bundle["causal_pip"]
    _method_meta = bundle["method_metadata"]

    _enriched = (
        _causal_pip
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select("method", "threshold", "method_display", "method_family"),
            on=["method", "threshold"],
            how="left",
        )
        .with_columns(pl.lit("Aggregate").alias("simulation_name"))
    )

    if _enriched.is_empty():
        causal_pip_chart = viz3_utils.make_placeholder_chart("No causal PIP data")
    else:
        causal_pip_chart = viz3_utils.render_causal_pip_chart(_enriched, facet=False)

    causal_pip_chart
    return


@app.cell(hide_code=True)
def cs_summary_heading_cell():
    _cs_summary_heading_md = mo.md("## Credible Set Summary")
    _cs_summary_heading_md
    return


@app.cell
def cs_summary_cell(bundle, foreground_methods):
    import matplotlib.pyplot as _plt

    _cs_summary = bundle["cs_summary"]
    _method_meta = bundle["method_metadata"]

    _enriched = (
        _cs_summary
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select("method", "threshold", "method_display", "method_family"),
            on=["method", "threshold"],
            how="left",
        )
    )

    if _enriched.is_empty():
        cs_summary_chart = viz3_utils.make_placeholder_chart("No CS summary data")
    else:
        _theme = viz3_utils.base_chart_theme()
        _metrics = ["Power", "Coverage", "CS Size"]
        _methods = sorted(_enriched["method_display"].unique().to_list())
        _fig, _axes = _plt.subplots(
            1, len(_metrics),
            figsize=(_theme["width"] * len(_metrics), _theme["height"]),
            squeeze=False,
        )
        for _col_idx, _metric in enumerate(_metrics):
            _ax = _axes[0, _col_idx]
            _metric_df = _enriched.filter(pl.col("metric") == _metric)
            for _md in _methods:
                _row_df = _metric_df.filter(pl.col("method_display") == _md)
                if _row_df.height > 0:
                    _color = viz3_utils.method_color(_row_df["method"][0])
                    _ax.bar(_md, _row_df["value"][0], color=_color)
            _ax.set_title(_metric)
            _ax.set_xticks(range(len(_methods)))
            _ax.set_xticklabels(_methods, rotation=30, ha="right", fontsize=7)
        _fig.tight_layout()
        cs_summary_chart = _fig

    cs_summary_chart
    return


@app.cell(hide_code=True)
def histograms_heading_cell():
    _histograms_heading_md = mo.md("## Credible Set Histograms")
    _histograms_heading_md
    return


@app.cell
def histograms_cell(bundle, foreground_methods):
    import matplotlib.pyplot as _plt

    _cs_size_hist = bundle["cs_size_histogram"]
    _ser_log_bf_hist = bundle["ser_log_bf_histogram"]
    _method_meta = bundle["method_metadata"]

    _cs_hist_filtered = (
        _cs_size_hist
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select("method", "threshold", "method_display"),
            on=["method", "threshold"],
            how="left",
        )
    )
    _log_bf_filtered = (
        _ser_log_bf_hist
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select("method", "threshold", "method_display"),
            on=["method", "threshold"],
            how="left",
        )
    )

    if _cs_hist_filtered.is_empty() and _log_bf_filtered.is_empty():
        hist_chart = viz3_utils.make_placeholder_chart("No histogram data")
    else:
        _theme = viz3_utils.base_chart_theme()
        _fig, _axes = _plt.subplots(
            1, 2,
            figsize=(_theme["width"] * 2, _theme["height"]),
            squeeze=False,
        )

        # CS size histogram
        _ax0 = _axes[0, 0]
        if not _cs_hist_filtered.is_empty():
            for _md in sorted(_cs_hist_filtered["method_display"].unique().to_list()):
                _rows = _cs_hist_filtered.filter(pl.col("method_display") == _md)
                _method_name = _rows["method"][0]
                _data = _rows["cs_size"].to_numpy()
                _ax0.hist(
                    _data, bins=20, alpha=0.5,
                    label=_md,
                    color=viz3_utils.method_color(_method_name),
                )
            _ax0.set_xlabel("CS Size")
            _ax0.set_ylabel("Count")
            _ax0.legend(fontsize=6)
        _ax0.set_title("CS Size")

        # SER log BF histogram
        _ax1 = _axes[0, 1]
        if not _log_bf_filtered.is_empty():
            for _md in sorted(_log_bf_filtered["method_display"].unique().to_list()):
                _rows = _log_bf_filtered.filter(pl.col("method_display") == _md)
                _method_name = _rows["method"][0]
                _data = _rows["ser_log_bf"].to_numpy()
                _ax1.hist(
                    _data, alpha=0.5,
                    label=_md,
                    color=viz3_utils.method_color(_method_name),
                )
            _ax1.set_xlabel("SER log BF")
            _ax1.legend(fontsize=6)
        _ax1.set_title("SER log BF")

        _fig.tight_layout()
        hist_chart = _fig

    hist_chart
    return


if __name__ == "__main__":
    app.run()
