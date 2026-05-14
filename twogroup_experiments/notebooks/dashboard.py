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
    import viz_utils


@app.cell(hide_code=True)
def title_cell():
    _title_md = mo.md("# Viz4: Plot-Ready")
    _title_md
    return


@app.cell(hide_code=True)
def prepare_heading_cell():
    mo.md("""
    ## Prepare Collections
    """)
    return


@app.cell
def unprepared_cell():
    _collections_root = Path(__file__).parent.parent / "results" / "collections"
    _not_ready = sorted(
        p.name
        for p in _collections_root.iterdir()
        if p.is_dir()
        and (p / "collection_spec.yaml").exists()
        and not (p / "plot_ready" / "out.txt").exists()
    )
    unprepared_table = mo.ui.table(
        pl.DataFrame({"name": _not_ready}),
        selection="multi",
    )
    _display = (
        mo.vstack([
            mo.md(f"**{len(_not_ready)} collections not yet plot-ready:**"),
            unprepared_table,
        ])
        if _not_ready
        else mo.md("All collections are plot-ready.")
    )
    _display
    return (unprepared_table,)


@app.cell(hide_code=True)
def snakemake_cores_cell():
    cores_input = mo.ui.number(value=1, start=1, stop=64, label="Cores (-c)")
    return (cores_input,)


@app.cell
def snakemake_prepare_cell(unprepared_table, cores_input):
    import re as _re
    import subprocess as _subprocess

    _selected = unprepared_table.value["name"].to_list() if len(unprepared_table.value) else []

    if not _selected:
        mo.stop(True, mo.md("Select collections above to prepare."))

    _cwd = str(Path(__file__).parent.parent)
    _targets = [
        f"results/collections/{n}/plot_ready/out.txt" for n in _selected
    ]
    _base_cmd = ["uv", "run", "snakemake", "--snakefile", "twogroup_experiments.snk"] + _targets
    _cores = int(cores_input.value)

    def _parse_dry_run(text):
        if "Nothing to be done" in text:
            return "Nothing to be done (all up to date)."
        lines = text.splitlines()
        in_stats = False
        rows = []
        for line in lines:
            if _re.search(r"[Jj]ob\s+(stats|counts)", line):
                in_stats = True
                continue
            if in_stats:
                stripped = line.strip()
                if not stripped or _re.match(r"^-+", stripped):
                    continue
                if _re.match(r"^job\s+count", stripped, _re.I):
                    continue
                parts = stripped.split()
                if len(parts) == 2:
                    try:
                        count = int(parts[1])
                        rule = parts[0]
                    except ValueError:
                        try:
                            count = int(parts[0])
                            rule = parts[1]
                        except ValueError:
                            continue
                    if rule.lower() != "total":
                        rows.append(f"- {rule}: {count}")
                elif len(parts) == 1 and parts[0].isdigit():
                    break
        if rows:
            return "\n".join(rows)
        return text[:400].strip() or "No output."

    def _do_dry_run(_):
        result = _subprocess.run(
            _base_cmd + ["--dry-run"],
            cwd=_cwd, capture_output=True, text=True,
        )
        return _parse_dry_run(result.stdout + result.stderr)

    def _do_run(_):
        _subprocess.Popen(
            _base_cmd + [f"-c{_cores}"],
            cwd=_cwd, start_new_session=True,
        )

    dry_run_btn = mo.ui.button(label="Dry run", on_click=_do_dry_run, value="")
    run_btn = mo.ui.button(
        label=f"Run ({len(_selected)} collections, -c{_cores})",
        on_click=_do_run,
    )
    _cmd_str = " \\\n    ".join(_base_cmd + [f"-c{_cores}"])
    mo.vstack([
        mo.md(f"```bash\n{_cmd_str}\n```"),
        mo.hstack([cores_input, dry_run_btn, run_btn]),
    ])
    return (dry_run_btn,)


@app.cell(hide_code=True)
def dry_run_output_cell(dry_run_btn):
    mo.md(f"**Job summary:**\n{dry_run_btn.value}") if dry_run_btn.value else mo.md("")


@app.cell(hide_code=True)
def view_heading_cell():
    mo.md("""
    ## View Collections
    """)
    return


@app.cell
def collection_selector_cell():
    collection_alias_root = Path(__file__).parent.parent / "results" / "collections"
    _names = plot_ready.available_plot_ready_collections(collection_alias_root)
    collection_table = mo.ui.table(
        pl.DataFrame({"name": _names}),
        selection="multi",
    )
    collection_table
    return collection_alias_root, collection_table


@app.cell
def bundles_cell(collection_alias_root, collection_table):
    mo.stop(
        len(collection_table.value) == 0,
        mo.md("Select one or more collections above."),
    )
    _selected = collection_table.value["name"].to_list()
    _bundles = {
        name: plot_ready.load_plot_ready_collection(collection_alias_root / name)
        for name in _selected
    }

    combined_method_metadata = (
        pl.concat([b["method_metadata"] for b in _bundles.values()])
        .unique(subset=["method", "threshold"])
    )

    def _tag(key):
        return pl.concat([
            b[key].with_columns(pl.lit(name).alias("collection_name"))
            for name, b in _bundles.items()
        ])

    combined_data = {
        "method_metadata": combined_method_metadata,
        "collection_names": _selected,
        "pip_calibration": _tag("pip_calibration"),
        "power_fdp": _tag("power_fdp"),
        "causal_pip": _tag("causal_pip"),
        "cs_raw": _tag("cs_raw"),
        "cs_size_histogram": _tag("cs_size_histogram"),
        "ser_log_bf_histogram": _tag("ser_log_bf_histogram"),
    }
    return (combined_data,)


@app.cell
def controls_cell(combined_data):
    _method_metadata = combined_data["method_metadata"]

    _all_thresholds = sorted(
        _method_metadata.filter(pl.col("threshold").is_not_null())["threshold"]
        .unique()
        .to_list()
    )
    threshold_dropdown = mo.ui.dropdown(
        options=_all_thresholds,
        value=_all_thresholds[0] if _all_thresholds else None,
        label="threshold",
    )

    _all_families = sorted(_method_metadata["method_family"].unique().to_list())
    method_family_multiselect = mo.ui.multiselect(
        options=_all_families,
        value=_all_families,
        label="method family",
    )

    _all_L = sorted(_method_metadata["L"].unique().to_list())
    L_dropdown = mo.ui.dropdown(
        options=_all_L,
        value=_all_L[0] if _all_L else 1,
        label="L",
    )

    max_fdp_slider = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=0.5,
        label="max FDP",
    )

    mo.hstack([method_family_multiselect, L_dropdown, threshold_dropdown, max_fdp_slider])
    return (
        L_dropdown,
        max_fdp_slider,
        method_family_multiselect,
        threshold_dropdown,
    )


@app.cell
def selected_methods_cell(
    L_dropdown,
    combined_data,
    method_family_multiselect,
    threshold_dropdown,
):
    _method_metadata = combined_data["method_metadata"]
    selected_threshold = threshold_dropdown.value

    _fg_mask = (
        pl.col("method_family").is_in(method_family_multiselect.value)
        & (pl.col("L") == L_dropdown.value)
        & (
            ~pl.col("is_thresholded")
            | (pl.col("threshold") == selected_threshold)
            | pl.col("threshold").is_null()
        )
    )
    foreground_methods = set(_method_metadata.filter(_fg_mask)["method"].to_list())

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
    return (
        background_methods_thresholds,
        foreground_methods,
        selected_threshold,
    )


@app.cell(hide_code=True)
def pip_calibration_heading_cell():
    mo.md("""
    ## PIP Calibration
    """)
    return


@app.cell
def pip_calibration_cell(
    combined_data,
    foreground_methods,
    threshold_dropdown,
):
    _pip_cal = combined_data["pip_calibration"]
    _method_meta = combined_data["method_metadata"]
    _selected_threshold = threshold_dropdown.value

    _enriched = (
        _pip_cal.filter(
            pl.col("threshold").is_null()
            | (pl.col("threshold") == _selected_threshold)
        )
        .filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select("method", "threshold", "method_display", "method_family"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(
            pl.col("method_display").alias("series_label"),
            pl.col("collection_name").alias("simulation_name"),
        )
    )

    if _enriched.is_empty():
        pip_cal_chart = viz_utils.make_placeholder_chart("No PIP calibration data")
    else:
        pip_cal_chart = viz_utils.render_pip_calibration(
            _enriched, facet_by_simulation=True
        )

    pip_cal_chart
    return


@app.cell(hide_code=True)
def power_fdp_heading_cell():
    mo.md("""
    ## Power vs FDP
    """)
    return


@app.cell
def power_fdp_cell(
    background_methods_thresholds,
    combined_data,
    foreground_methods,
    max_fdp_slider,
    selected_threshold,
):
    _power_fdp = combined_data["power_fdp"]
    _method_meta = combined_data["method_metadata"]
    _meta_cols = ["method", "threshold", "method_display", "method_family", "method_label_base", "is_thresholded"]

    _fg_df = (
        _power_fdp.filter(
            pl.col("method").is_in(foreground_methods)
            & (pl.col("threshold").is_null() | (pl.col("threshold") == selected_threshold))
        )
        .join(
            _method_meta.select(_meta_cols),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(pl.lit(True).alias("is_selected_threshold"))
    )

    _bg_pairs_df = (
        pl.DataFrame(
            list(background_methods_thresholds),
            schema={"method": pl.String, "threshold": pl.Float64},
            orient="row",
        )
        if background_methods_thresholds
        else pl.DataFrame(schema={"method": pl.String, "threshold": pl.Float64})
    )
    if not _bg_pairs_df.is_empty():
        _bg_df = (
            _power_fdp.join(
                _bg_pairs_df, on=["method", "threshold"], how="inner", nulls_equal=True
            )
            .join(
                _method_meta.select(_meta_cols),
                on=["method", "threshold"],
                how="left",
                nulls_equal=True,
            )
            .with_columns(pl.lit(False).alias("is_selected_threshold"))
        )
        _combined = pl.concat([_fg_df, _bg_df])
    else:
        _combined = _fg_df

    _combined = _combined.with_columns(
        pl.when(pl.col("is_thresholded"))
        .then(pl.format("{} (@{})", pl.col("method_label_base"), pl.col("threshold")))
        .otherwise(pl.col("method_display"))
        .alias("trace_label"),
        pl.when(pl.col("is_selected_threshold"))
        .then(pl.col("method_display"))
        .otherwise(None)
        .alias("legend_label"),
        pl.col("collection_name").alias("simulation_name"),
    )

    if _combined.is_empty():
        power_fdp_chart = viz_utils.make_placeholder_chart("No power/FDP data")
    else:
        power_fdp_chart = viz_utils.render_power_fdp_chart(
            _combined,
            facet=True,
            max_fdp=max_fdp_slider.value,
            fixed_y_scale=True,
            legend_outside=True,
            square_axes=True,
        )

    power_fdp_chart
    return


@app.cell(hide_code=True)
def causal_pip_heading_cell():
    mo.md("""
    ## Causal PIP vs Threshold
    """)
    return


@app.cell
def causal_pip_cell(combined_data, foreground_methods):
    _causal_pip = combined_data["causal_pip"]
    _method_meta = combined_data["method_metadata"]

    _enriched = (
        _causal_pip.filter(pl.col("method").is_in(foreground_methods))
        .join(
            _method_meta.select(
                "method",
                "threshold",
                "method_display",
                "method_display_base",
                "method_family",
                "is_thresholded",
            ),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(pl.col("collection_name").alias("simulation_name"))
    )

    _method_order = (
        _method_meta.filter(pl.col("method").is_in(foreground_methods))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )

    if _enriched.is_empty():
        causal_pip_chart = viz_utils.make_placeholder_chart("No causal PIP data")
    else:
        causal_pip_chart = viz_utils.render_causal_pip_chart(
            _enriched,
            facet=True,
            legend_outside=True,
            square_axes=True,
            method_order=_method_order,
        )

    causal_pip_chart
    return


@app.cell(hide_code=True)
def cs_summary_heading_cell():
    mo.md("""
    ## Credible Set Summary
    """)
    return


@app.cell
def histogram_controls_cell(combined_data):
    _cs_size_hist = combined_data["cs_size_histogram"]
    _ser_log_bf_hist = combined_data["ser_log_bf_histogram"]

    _max_cs = (
        int(_cs_size_hist["cs_size"].max())
        if not _cs_size_hist.is_empty()
        else 100
    )
    _lbf_min = (
        float(_ser_log_bf_hist["ser_log_bf"].min())
        if not _ser_log_bf_hist.is_empty()
        else -10.0
    )
    _lbf_max = (
        float(_ser_log_bf_hist["ser_log_bf"].max())
        if not _ser_log_bf_hist.is_empty()
        else 10.0
    )
    _lbf_step = round((_lbf_max - _lbf_min) / 100, 2) or 0.1

    max_cs_size_slider = mo.ui.slider(
        start=1,
        stop=_max_cs,
        value=_max_cs,
        step=1,
        label="max CS size",
    )
    min_log_bf_slider = mo.ui.slider(
        start=round(_lbf_min, 2),
        stop=round(_lbf_max, 2),
        value=round(_lbf_min, 2),
        step=_lbf_step,
        label="min log BF",
    )

    mo.hstack([max_cs_size_slider, min_log_bf_slider])
    return max_cs_size_slider, min_log_bf_slider


@app.cell
def cs_summary_cell(
    combined_data,
    foreground_methods,
    max_cs_size_slider,
    min_log_bf_slider,
    selected_threshold,
):
    import matplotlib.pyplot as _plt

    _cs_raw = combined_data["cs_raw"]
    _method_meta = combined_data["method_metadata"]
    _collection_names = combined_data["collection_names"]
    _max_cs = max_cs_size_slider.value
    _min_lbf = min_log_bf_slider.value

    _raw = (
        _cs_raw.filter(
            pl.col("method").is_in(foreground_methods)
            & (pl.col("threshold").is_null() | (pl.col("threshold") == selected_threshold))
        )
        .join(
            _method_meta.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(
            ((pl.col("cs_size") <= _max_cs) & (pl.col("ser_log_bf") >= _min_lbf)).alias("valid_cs")
        )
    )

    _agg = (
        _raw.group_by("collection_name", "method", "threshold", "method_display", "is_thresholded")
        .agg(
            (pl.col("causal_in_cs") & pl.col("valid_cs")).cast(pl.Float64).mean().alias("Power"),
            pl.when(pl.col("valid_cs")).then(pl.col("causal_in_cs").cast(pl.Float64)).mean().alias("Coverage"),
            pl.when(pl.col("valid_cs")).then(pl.col("cs_size").cast(pl.Float64)).mean().alias("CS Size"),
        )
    )

    if _agg.is_empty():
        cs_summary_chart = viz_utils.make_placeholder_chart("No CS summary data")
    else:
        _theme = viz_utils.base_chart_theme()
        _metrics = ["Power", "Coverage", "CS Size"]
        _n_coll = len(_collection_names)
        _h = _theme["height"]
        _fig, _axes = _plt.subplots(
            _n_coll,
            len(_metrics),
            figsize=(_h * len(_metrics), _h * _n_coll),
            squeeze=False,
        )
        for _row_idx, _coll in enumerate(_collection_names):
            _coll_agg = _agg.filter(pl.col("collection_name") == _coll)
            _method_order = (
                _coll_agg.select("method_display", "is_thresholded")
                .unique()
                .drop_nulls(subset=["method_display"])
                .sort(["is_thresholded", "method_display"])["method_display"]
                .to_list()
            )
            for _col_idx, _metric in enumerate(_metrics):
                _ax = _axes[_row_idx, _col_idx]
                for _i, _md in enumerate(_method_order):
                    _row_df = _coll_agg.filter(pl.col("method_display") == _md)
                    if _row_df.height > 0:
                        _color = viz_utils.method_color(_row_df["method"][0])
                        _val = _row_df[_metric][0]
                        if _val is not None:
                            _ax.scatter([_i], [_val], color=_color, zorder=3, s=60, label=_md)
                if _row_idx == 0:
                    _ax.set_title(_metric)
                _ax.set_xticks(range(len(_method_order)))
                _ax.set_xticklabels(_method_order, rotation=45, ha="right", fontsize=7)
                _ax.set_xlim(-0.5, len(_method_order) - 0.5)
                _ax.set_box_aspect(1)
                if _col_idx == 0:
                    _ax.set_ylabel(_coll, fontsize=10, fontweight='bold')
        _fig.tight_layout()
        cs_summary_chart = _fig

    cs_summary_chart
    return


@app.cell(hide_code=True)
def cs_power_fdp_heading_cell():
    mo.md("""
    ## CS Power vs FDP (log BF threshold)
    """)
    return


@app.cell
def cs_power_fdp_cell(
    combined_data,
    foreground_methods,
    max_fdp_slider,
    selected_threshold,
):
    import matplotlib.pyplot as _plt2
    import numpy as _np

    _cs_raw2 = combined_data["cs_raw"]
    _method_meta2 = combined_data["method_metadata"]
    _collection_names2 = combined_data["collection_names"]

    _raw2 = (
        _cs_raw2.filter(
            pl.col("method").is_in(foreground_methods)
            & (pl.col("threshold").is_null() | (pl.col("threshold") == selected_threshold))
        )
        .join(
            _method_meta2.select("method", "threshold", "method_display", "is_thresholded"),
            on=["method", "threshold"],
            how="left",
            nulls_equal=True,
        )
    )

    if _raw2.is_empty():
        cs_power_fdp_chart = viz_utils.make_placeholder_chart("No CS data")
    else:
        _lbf_lo2 = float(_raw2["ser_log_bf"].min())
        _lbf_hi2 = float(_raw2["ser_log_bf"].max())
        _lbf_grid2 = _np.linspace(_lbf_lo2, _lbf_hi2, 60)[::-1]
        _method_groups2 = (
            _raw2.select("method", "threshold", "method_display", "is_thresholded")
            .unique()
            .sort(["is_thresholded", "method_display"])
        )

        _rows2 = []
        for _coll_name in _collection_names2:
            _coll_raw = _raw2.filter(pl.col("collection_name") == _coll_name)
            for _mg in _method_groups2.iter_rows(named=True):
                _thresh_filter = (
                    pl.col("threshold").is_null()
                    if _mg["threshold"] is None
                    else (pl.col("threshold") == _mg["threshold"])
                )
                _m_data = _coll_raw.filter(
                    (pl.col("method") == _mg["method"]) & _thresh_filter
                )
                if _m_data.is_empty():
                    continue
                _n_total2 = _m_data.height
                _causal2 = _m_data["causal_in_cs"].to_numpy()
                _lbf2 = _m_data["ser_log_bf"].to_numpy()
                for _t in _lbf_grid2:
                    _disc = _lbf2 >= _t
                    _hit = _disc & _causal2
                    _n_disc = int(_disc.sum())
                    _n_hit = int(_hit.sum())
                    _rows2.append({
                        "collection_name": _coll_name,
                        "method": _mg["method"],
                        "threshold": _mg["threshold"],
                        "method_display": _mg["method_display"],
                        "is_thresholded": _mg["is_thresholded"],
                        "pip_threshold": float(_t),
                        "power": float(_n_hit / max(_n_total2, 1)),
                        "fdp": float((_n_disc - _n_hit) / max(_n_disc, 1)),
                    })

        _cs_pf = pl.from_dicts(
            _rows2,
            schema={
                "collection_name": pl.String,
                "method": pl.String,
                "threshold": pl.Float64,
                "method_display": pl.String,
                "is_thresholded": pl.Boolean,
                "pip_threshold": pl.Float64,
                "power": pl.Float64,
                "fdp": pl.Float64,
            },
        ).with_columns(
            pl.col("method_display").alias("trace_label"),
            pl.col("method_display").alias("legend_label"),
            pl.lit(True).alias("is_selected_threshold"),
            pl.col("collection_name").alias("simulation_name"),
        )

        cs_power_fdp_chart = viz_utils.render_power_fdp_chart(
            _cs_pf,
            facet=True,
            max_fdp=max_fdp_slider.value,
            fixed_y_scale=True,
            legend_outside=True,
            square_axes=True,
        )

    cs_power_fdp_chart
    return


@app.cell(hide_code=True)
def histograms_heading_cell():
    mo.md("""
    ## Credible Set Histograms
    """)
    return


@app.cell
def histograms_cell(
    combined_data,
    foreground_methods,
    max_cs_size_slider,
    min_log_bf_slider,
    threshold_dropdown,
):
    import matplotlib.pyplot as _plt3
    import numpy as _np3

    _cs_size_hist = combined_data["cs_size_histogram"]
    _ser_log_bf_hist = combined_data["ser_log_bf_histogram"]
    _method_meta = combined_data["method_metadata"]
    _collection_names = combined_data["collection_names"]
    _selected_threshold = threshold_dropdown.value

    _cs_filtered = _cs_size_hist.filter(
        pl.col("method").is_in(foreground_methods)
        & (pl.col("threshold").is_null() | (pl.col("threshold") == _selected_threshold))
    ).join(
        _method_meta.select("method", "threshold", "method_display", "is_thresholded"),
        on=["method", "threshold"],
        how="left",
        nulls_equal=True,
    )
    _lbf_filtered = _ser_log_bf_hist.filter(
        pl.col("method").is_in(foreground_methods)
        & (pl.col("threshold").is_null() | (pl.col("threshold") == _selected_threshold))
    ).join(
        _method_meta.select("method", "threshold", "method_display", "is_thresholded"),
        on=["method", "threshold"],
        how="left",
        nulls_equal=True,
    )

    if _cs_filtered.is_empty() and _lbf_filtered.is_empty():
        hist_chart = viz_utils.make_placeholder_chart("No histogram data")
    else:
        _all_displays = sorted(
            m
            for m in set(
                _cs_filtered["method_display"].drop_nulls().to_list()
                + _lbf_filtered["method_display"].drop_nulls().to_list()
            )
        )
        _n = len(_all_displays)
        _n_coll = len(_collection_names)
        _theme = viz_utils.base_chart_theme()
        _per_w = max(2.5, _theme["width"] / 3)

        _cs_max_val = (
            int(_cs_filtered["cs_size"].max()) + 1
            if not _cs_filtered.is_empty()
            else 100
        )
        _lbf_lo = float(_lbf_filtered["ser_log_bf"].min()) if not _lbf_filtered.is_empty() else -10.0
        _lbf_hi = float(_lbf_filtered["ser_log_bf"].max()) if not _lbf_filtered.is_empty() else 10.0
        _cs_bins = _np3.linspace(0, _cs_max_val, 21)
        _lbf_bins = _np3.linspace(_lbf_lo, _lbf_hi, 21)
        _max_cs = max_cs_size_slider.value
        _min_lbf = min_log_bf_slider.value

        # 2 rows per collection (CS size, log BF), N method columns
        _fig, _axes = _plt3.subplots(
            2 * _n_coll,
            _n,
            figsize=(_per_w * _n, _theme["height"] * 2 * _n_coll),
            squeeze=False,
        )

        # CS size block: rows 0 to N_coll-1
        for _ci, _coll in enumerate(_collection_names):
            _cs_coll = _cs_filtered.filter(pl.col("collection_name") == _coll)
            for _j, _md in enumerate(_all_displays):
                _ax = _axes[_ci, _j]
                _cs_rows = _cs_coll.filter(pl.col("method_display") == _md)
                if not _cs_rows.is_empty():
                    _color = viz_utils.method_color(_cs_rows["method"][0])
                    _cs_data = _cs_rows["cs_size"].to_numpy()
                    _pass = _cs_data[_cs_data <= _max_cs]
                    _fail = _cs_data[_cs_data > _max_cs]
                    if len(_pass) > 0:
                        _ax.hist(_pass, bins=_cs_bins, color=_color, alpha=0.8)
                    if len(_fail) > 0:
                        _ax.hist(_fail, bins=_cs_bins, facecolor="none", edgecolor=_color, linewidth=0.8)
                _ax.set_xlim(0, _cs_max_val)
                if _ci == 0:
                    _ax.set_title(_md, fontsize=7)
                if _j == 0:
                    _ax.set_ylabel(_coll, fontsize=10, fontweight='bold')
                _ax.set_box_aspect(1)

        # Log BF block: rows N_coll to 2*N_coll-1
        for _ci, _coll in enumerate(_collection_names):
            _lbf_coll = _lbf_filtered.filter(pl.col("collection_name") == _coll)
            for _j, _md in enumerate(_all_displays):
                _ax = _axes[_n_coll + _ci, _j]
                _lbf_rows = _lbf_coll.filter(pl.col("method_display") == _md)
                if not _lbf_rows.is_empty():
                    _color = viz_utils.method_color(_lbf_rows["method"][0])
                    _lbf_data = _lbf_rows["ser_log_bf"].to_numpy()
                    _pass_l = _lbf_data[_lbf_data >= _min_lbf]
                    _fail_l = _lbf_data[_lbf_data < _min_lbf]
                    if len(_pass_l) > 0:
                        _ax.hist(_pass_l, bins=_lbf_bins, color=_color, alpha=0.8)
                    if len(_fail_l) > 0:
                        _ax.hist(_fail_l, bins=_lbf_bins, facecolor="none", edgecolor=_color, linewidth=0.8)
                _ax.set_xlim(_lbf_lo, _lbf_hi)
                if _ci == 0:
                    _ax.set_title(_md, fontsize=7)
                if _j == 0:
                    _ax.set_ylabel(_coll, fontsize=10, fontweight='bold')
                _ax.set_box_aspect(1)

        _fig.tight_layout()
        hist_chart = _fig

    hist_chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
