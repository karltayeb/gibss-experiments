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

    import config
    import plot_ready
    import viz_utils


@app.cell(hide_code=True)
def title_cell():
    _title_md = mo.md("# Viz4: Plot-Ready")
    _title_md
    return


@app.cell(hide_code=True)
def view_heading_cell():
    mo.md("""
    ## View Collections
    """)
    return


@app.cell
def config_select_cell():
    import yaml as _yaml

    _config_path = Path(__file__).parent / "plot_config.yaml"
    _all_configs: dict = _yaml.safe_load(_config_path.read_text()) or {} if _config_path.exists() else {}
    _supercollections = _all_configs.get("supercollections", {})
    _settings_presets = _all_configs.get("settings", {})

    _sc_names = list(_supercollections.keys())
    _ps_names = ["(default)"] + list(_settings_presets.keys())

    supercollection_dropdown = mo.ui.dropdown(
        options=_sc_names,
        value=_sc_names[0] if _sc_names else None,
        label="supercollection",
    )
    plot_settings_dropdown = mo.ui.dropdown(
        options=_ps_names,
        value=_ps_names[0] if _ps_names else None,
        label="plot settings",
    )

    def _compute_state(sc, ps):
        if not sc:
            return {"selected": [], "aliases": {}, "settings": {}}
        sc_cfg = _supercollections.get(sc, {})
        coll_list = sc_cfg.get("collections", [])
        selected = [item["name"] for item in coll_list]
        aliases = {item["name"]: item.get("alias", item["name"]) for item in coll_list}
        defaults = sc_cfg.get("default_settings", {})
        overrides = _settings_presets.get(ps, {}) if ps != "(default)" else {}
        settings = {**defaults, **overrides}
        return {"selected": selected, "aliases": aliases, "settings": settings}

    def _apply(_):
        return _compute_state(supercollection_dropdown.value, plot_settings_dropdown.value)

    _initial_val = _compute_state(
        _sc_names[0] if _sc_names else None,
        _ps_names[0] if _ps_names else None,
    )
    apply_btn = mo.ui.button(label="Apply", on_click=_apply, value=_initial_val)

    mo.hstack([supercollection_dropdown, plot_settings_dropdown, apply_btn])
    return (apply_btn,)


@app.cell
def bundles_cell(apply_btn):
    collection_alias_root = Path(__file__).parent.parent / "results" / "collections"
    _settings = apply_btn.value
    _selected = _settings["selected"]
    _aliases: dict[str, str] = _settings["aliases"]

    mo.stop(not _selected, mo.md("Select a supercollection above."))

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
            b[key].with_columns(
                pl.lit(_aliases.get(name, name)).alias("collection_name")
            )
            for name, b in _bundles.items()
        ])

    combined_data = {
        "method_metadata": combined_method_metadata,
        "collection_names": [_aliases.get(n, n) for n in _selected],
        "pip_plot_data": _tag("pip_plot_data"),
        "cs_plot_data": _tag("cs_plot_data"),
    }
    return (combined_data,)


@app.cell
def controls_cell(combined_data, apply_btn):
    _method_metadata = combined_data["method_metadata"]
    _settings_cfg: dict = apply_btn.value.get("settings", {})

    _all_thresholds = sorted(
        _method_metadata.filter(pl.col("threshold").is_not_null())["threshold"]
        .unique()
        .to_list()
    )
    _saved_thresholds = _settings_cfg.get("thresholds", _all_thresholds)
    _valid_thresholds = [t for t in _saved_thresholds if t in _all_thresholds]
    threshold_multiselect = mo.ui.multiselect(
        options=_all_thresholds,
        value=_valid_thresholds if _valid_thresholds else _all_thresholds,
        label="thresholds",
    )

    _all_families = sorted(_method_metadata["method_family"].unique().to_list())
    _saved_families = _settings_cfg.get("method_families", _all_families)
    _valid_families = [f for f in _saved_families if f in _all_families]
    method_family_multiselect = mo.ui.multiselect(
        options=_all_families,
        value=_valid_families if _valid_families else _all_families,
        label="method family",
    )

    _all_L = sorted(_method_metadata["L"].unique().to_list())
    L_dropdown = mo.ui.dropdown(
        options=_all_L,
        value=_settings_cfg.get("L", _all_L[0] if _all_L else 1),
        label="L",
    )

    max_fdp_slider = mo.ui.slider(
        start=0.05,
        stop=1.0,
        step=0.05,
        value=_settings_cfg.get("max_fdp", 0.5),
        label="max FDP",
    )

    mo.hstack([method_family_multiselect, L_dropdown, threshold_multiselect, max_fdp_slider])
    return (
        L_dropdown,
        max_fdp_slider,
        method_family_multiselect,
        threshold_multiselect,
    )


@app.cell
def selected_methods_cell(
    L_dropdown,
    combined_data,
    method_family_multiselect,
    threshold_multiselect,
):
    _method_metadata = combined_data["method_metadata"]
    selected_thresholds = threshold_multiselect.value or None

    _thresh_mask = (
        ~pl.col("is_thresholded")
        | pl.col("threshold").is_null()
        | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
    )
    _fg_mask = (
        pl.col("method_family").is_in(method_family_multiselect.value)
        & (pl.col("L") == L_dropdown.value)
        & _thresh_mask
    )
    foreground_methods = set(_method_metadata.filter(_fg_mask)["method"].to_list())
    return (
        foreground_methods,
        selected_thresholds,
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
    selected_thresholds,
):
    _pip_plot = combined_data["pip_plot_data"]
    _method_meta = combined_data["method_metadata"]
    _pip_filtered = _pip_plot.filter(pl.col("method").is_in(foreground_methods))
    _pip_cal_summary = viz_utils.expand_pip_calibration_from_compact(
        _pip_filtered, _method_meta, selected_thresholds=selected_thresholds,
    )
    if _pip_cal_summary.is_empty():
        pip_cal_chart = viz_utils.make_placeholder_chart("No PIP calibration data")
    else:
        pip_cal_chart = viz_utils.render_pip_calibration(
            _pip_cal_summary, facet_by_simulation=True,
            collection_names=combined_data["collection_names"],
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
    combined_data,
    foreground_methods,
    max_fdp_slider,
    selected_thresholds,
):
    _pip_plot = combined_data["pip_plot_data"]
    _method_meta = combined_data["method_metadata"]
    _power_fdp = viz_utils.expand_power_fdp_from_compact(
        _pip_plot, _method_meta,
        selected_methods=foreground_methods,
        selected_thresholds=selected_thresholds,
    )
    if _power_fdp.is_empty():
        power_fdp_chart = viz_utils.make_placeholder_chart("No power/FDP data")
    else:
        _summary = viz_utils.make_power_fdp_summary(_power_fdp)
        power_fdp_chart = viz_utils.render_power_fdp_chart(
            _summary,
            facet=True,
            max_fdp=max_fdp_slider.value,
            fixed_y_scale=True,
            legend_outside=True,
            square_axes=True,
            collection_names=combined_data["collection_names"],
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
    _pip_plot = combined_data["pip_plot_data"]
    _method_meta = combined_data["method_metadata"]
    _causal_pip = viz_utils.expand_causal_pip_from_compact(_pip_plot, _method_meta)
    _filtered = _causal_pip.filter(pl.col("method").is_in(foreground_methods))
    _method_order = (
        _method_meta.filter(pl.col("method").is_in(foreground_methods))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )
    if _filtered.is_empty():
        causal_pip_chart = viz_utils.make_placeholder_chart("No causal PIP data")
    else:
        _summary = viz_utils.make_causal_pip_summary(_filtered)
        causal_pip_chart = viz_utils.render_causal_pip_chart(
            _summary,
            facet=True,
            legend_outside=True,
            square_axes=True,
            method_order=_method_order,
            collection_names=combined_data["collection_names"],
        )
    causal_pip_chart
    return


@app.cell(hide_code=True)
def causal_rank_heading_cell():
    mo.md("""
    ## Mean Causal Rank
    """)
    return


@app.cell
def causal_rank_cell(combined_data, foreground_methods):
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _method_meta = combined_data["method_metadata"]

    _method_order = (
        _method_meta.filter(pl.col("method").is_in(foreground_methods))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )

    if _cs_data.is_empty():
        causal_rank_chart = viz_utils.make_placeholder_chart("No CS beta trace data")
    else:
        _rank_summary = viz_utils.make_causal_rank_summary(
            _cs_data,
            _method_meta,
            selected_methods=set(foreground_methods),
        )
        if _rank_summary.is_empty():
            causal_rank_chart = viz_utils.make_placeholder_chart("No causal rank data")
        else:
            causal_rank_chart = viz_utils.render_causal_rank_chart(
                _rank_summary,
                facet=True,
                legend_outside=True,
                square_axes=True,
                method_order=_method_order,
                collection_names=combined_data["collection_names"],
            )

    causal_rank_chart
    return


@app.cell(hide_code=True)
def mass_above_causal_heading_cell():
    mo.md("""
    ## Mean Mass Above Causal
    """)
    return


@app.cell
def mass_above_causal_cell(combined_data, foreground_methods):
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _method_meta = combined_data["method_metadata"]
    _method_order = (
        _method_meta.filter(pl.col("method").is_in(foreground_methods))
        .select("method", "is_thresholded")
        .unique()
        .sort(["is_thresholded", "method"])["method"]
        .to_list()
    )
    if _cs_data.is_empty():
        mass_above_causal_chart = viz_utils.make_placeholder_chart("No CS data")
    else:
        _expanded = viz_utils.expand_mass_above_causal_from_compact(
            _cs_data.filter(pl.col("method").is_in(foreground_methods)),
            _method_meta,
        )
        if _expanded.is_empty():
            mass_above_causal_chart = viz_utils.make_placeholder_chart("No mass above causal data")
        else:
            _summary = viz_utils.make_mass_above_causal_summary(_expanded)
            mass_above_causal_chart = viz_utils.render_mass_above_causal_chart(
                _summary,
                facet=True,
                legend_outside=True,
                square_axes=True,
                method_order=_method_order,
                collection_names=combined_data["collection_names"],
            )
    mass_above_causal_chart
    return


@app.cell(hide_code=True)
def cs_summary_heading_cell():
    mo.md("""
    ## Credible Set Summary
    """)
    return


@app.cell
def histogram_controls_cell(combined_data, apply_btn):
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _settings_cfg: dict = apply_btn.value.get("settings", {})

    _BETA_095_IDX = 45  # CS_BETA_GRID[45] == 0.95
    _max_cs = (
        int(_cs_data.with_columns(
            pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("_cs95")
        )["_cs95"].max())
        if not _cs_data.is_empty()
        else 100
    )
    max_cs_size_slider = mo.ui.slider(
        start=1,
        stop=_max_cs,
        value=min(_settings_cfg.get("max_cs_size", _max_cs), _max_cs),
        step=1,
        label="max CS size",
    )
    min_log_bf_slider = mo.ui.slider(
        start=-1,
        stop=50,
        value=_settings_cfg.get("min_log_bf", 2),
        step=0.1,
        label="min log BF",
    )

    cs_beta_slider = mo.ui.slider(
        start=0.50,
        stop=0.99,
        step=0.01,
        value=_settings_cfg.get("cs_beta", 0.95),
        label="nominal coverage (β)",
    )

    mo.hstack([max_cs_size_slider, min_log_bf_slider, cs_beta_slider])
    return max_cs_size_slider, min_log_bf_slider, cs_beta_slider


@app.cell
def cs_summary_cell(
    combined_data,
    foreground_methods,
    max_cs_size_slider,
    min_log_bf_slider,
    cs_beta_slider,
    selected_thresholds,
):
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _method_meta = combined_data["method_metadata"]
    _collection_names = combined_data["collection_names"]
    _max_cs = max_cs_size_slider.value
    _min_lbf = min_log_bf_slider.value
    _selected_beta = round(cs_beta_slider.value, 2)

    if _cs_data.is_empty():
        cs_summary_chart = viz_utils.make_placeholder_chart("No CS beta trace data")
    else:
        _summary = viz_utils.make_cs_beta_trace_summary(
            _cs_data,
            _method_meta,
            selected_methods=set(foreground_methods),
            selected_thresholds=selected_thresholds,
            max_cs_size=_max_cs,
            min_ser_log_bf=_min_lbf,
        )
        cs_summary_chart = viz_utils.render_cs_dot_summary_chart(
            _summary,
            collection_names=_collection_names,
            selected_beta=_selected_beta,
            max_cs_size=_max_cs,
            min_ser_log_bf=_min_lbf,
        )

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
    selected_thresholds,
):
    import matplotlib.pyplot as _plt2
    import numpy as _np

    _BETA_095_IDX = 45  # CS_BETA_GRID[45] == 0.95
    _cs_data2 = combined_data.get("cs_plot_data", pl.DataFrame())
    if not _cs_data2.is_empty():
        _cs_raw2 = _cs_data2.with_columns(
            pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size"),
            pl.when(pl.col("rank_of_causal").list.len() > 0)
            .then(pl.col("rank_of_causal").list.min() < pl.col("cs_sizes").list.get(_BETA_095_IDX))
            .otherwise(False)
            .alias("causal_in_cs"),
        ).select("collection_name", "sample_id", "method", "threshold", "l", "cs_size", "causal_in_cs", "ser_log_bf")
    else:
        _cs_raw2 = pl.DataFrame(schema={
            "collection_name": pl.String, "sample_id": pl.String,
            "method": pl.String, "threshold": pl.Float64,
            "l": pl.Int64, "cs_size": pl.Int64, "causal_in_cs": pl.Boolean, "ser_log_bf": pl.Float64,
        })
    _method_meta2 = combined_data["method_metadata"]
    _collection_names2 = combined_data["collection_names"]

    _thresh_mask2 = (
        pl.col("threshold").is_null()
        | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
    )
    _raw2 = (
        _cs_raw2.filter(pl.col("method").is_in(foreground_methods) & _thresh_mask2)
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
            collection_names=_collection_names2,
        )

    cs_power_fdp_chart
    return


@app.cell(hide_code=True)
def cs_beta_trace_heading_cell():
    mo.md("""
    ## Power / Coverage / CS Size vs Nominal Coverage (β sweep)
    """)
    return


@app.cell
def cs_beta_trace_cell(
    combined_data,
    foreground_methods,
    max_cs_size_slider,
    min_log_bf_slider,
    selected_thresholds,
):
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    _method_meta = combined_data["method_metadata"]
    _collection_names = combined_data["collection_names"]

    if _cs_data.is_empty():
        cs_beta_trace_chart = viz_utils.make_placeholder_chart("No CS beta trace data")
    else:
        _beta_summary = viz_utils.make_cs_beta_trace_summary(
            _cs_data,
            _method_meta,
            selected_methods=set(foreground_methods),
            selected_thresholds=selected_thresholds,
            max_cs_size=max_cs_size_slider.value,
            min_ser_log_bf=min_log_bf_slider.value,
        )
        cs_beta_trace_chart = viz_utils.render_cs_beta_trace_chart(
            _beta_summary,
            collection_names=_collection_names,
            selected_thresholds=selected_thresholds,
            max_cs_size=max_cs_size_slider.value,
            min_ser_log_bf=min_log_bf_slider.value,
        )

    cs_beta_trace_chart
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
    selected_thresholds,
):
    import matplotlib.pyplot as _plt3
    import numpy as _np3

    _BETA_095_IDX = 45  # CS_BETA_GRID[45] == 0.95
    _cs_data = combined_data.get("cs_plot_data", pl.DataFrame())
    if not _cs_data.is_empty():
        _cs_size_hist = _cs_data.with_columns(
            pl.col("cs_sizes").list.get(_BETA_095_IDX).alias("cs_size")
        ).select("collection_name", "method", "threshold", "l", "cs_size", "ser_log_bf")
        _ser_log_bf_hist = _cs_data.select("collection_name", "method", "threshold", "l", "ser_log_bf")
    else:
        _cs_size_hist = pl.DataFrame(schema={
            "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
            "l": pl.Int64, "cs_size": pl.Int64, "ser_log_bf": pl.Float64,
        })
        _ser_log_bf_hist = pl.DataFrame(schema={
            "collection_name": pl.String, "method": pl.String, "threshold": pl.Float64,
            "l": pl.Int64, "ser_log_bf": pl.Float64,
        })
    _method_meta = combined_data["method_metadata"]
    _collection_names = combined_data["collection_names"]
    _thresh_mask_hist = (
        pl.col("threshold").is_null()
        | (pl.lit(True) if selected_thresholds is None else pl.col("threshold").is_in(selected_thresholds))
    )

    _cs_filtered = _cs_size_hist.filter(
        pl.col("method").is_in(foreground_methods) & _thresh_mask_hist
    ).join(
        _method_meta.select("method", "threshold", "method_display", "is_thresholded"),
        on=["method", "threshold"],
        how="left",
        nulls_equal=True,
    )
    _lbf_filtered = _ser_log_bf_hist.filter(
        pl.col("method").is_in(foreground_methods) & _thresh_mask_hist
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

        # 2 rows (CS size, log BF) × N method columns — all collections pooled
        _fig, _axes = _plt3.subplots(
            2, _n,
            figsize=(_per_w * _n, _theme["height"] * 2),
            squeeze=False,
        )

        for _j, _md in enumerate(_all_displays):
            # CS size row (row 0)
            _ax_cs = _axes[0, _j]
            _cs_rows = _cs_filtered.filter(pl.col("method_display") == _md)
            if not _cs_rows.is_empty():
                _color = viz_utils.method_color(_cs_rows["method"][0])
                _cs_data = _cs_rows["cs_size"].to_numpy()
                _pass = _cs_data[_cs_data <= _max_cs]
                _fail = _cs_data[_cs_data > _max_cs]
                if len(_pass) > 0:
                    _ax_cs.hist(_pass, bins=_cs_bins, color=_color, alpha=0.8)
                if len(_fail) > 0:
                    _ax_cs.hist(_fail, bins=_cs_bins, facecolor="none", edgecolor=_color, linewidth=0.8)
            _ax_cs.set_xlim(0, _cs_max_val)
            _ax_cs.set_title(_md, fontsize=7)
            _ax_cs.set_box_aspect(1)
            if _j == 0:
                _ax_cs.set_ylabel("CS Size", fontsize=8)

            # Log BF row (row 1)
            _ax_lbf = _axes[1, _j]
            _lbf_rows = _lbf_filtered.filter(pl.col("method_display") == _md)
            if not _lbf_rows.is_empty():
                _color = viz_utils.method_color(_lbf_rows["method"][0])
                _lbf_data = _lbf_rows["ser_log_bf"].to_numpy()
                _pass_l = _lbf_data[_lbf_data >= _min_lbf]
                _fail_l = _lbf_data[_lbf_data < _min_lbf]
                if len(_pass_l) > 0:
                    _ax_lbf.hist(_pass_l, bins=_lbf_bins, color=_color, alpha=0.8)
                if len(_fail_l) > 0:
                    _ax_lbf.hist(_fail_l, bins=_lbf_bins, facecolor="none", edgecolor=_color, linewidth=0.8)
            _ax_lbf.set_xlim(_lbf_lo, _lbf_hi)
            _ax_lbf.set_box_aspect(1)
            if _j == 0:
                _ax_lbf.set_ylabel("SER log BF", fontsize=8)

        _fig.tight_layout()
        hist_chart = _fig

    hist_chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
