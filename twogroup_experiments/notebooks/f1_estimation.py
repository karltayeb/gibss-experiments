import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import yaml

    RESULTS_ROOT = Path(__file__).parent.parent / "results"
    PLOT_CONFIG_PATH = Path(__file__).parent / "plot_config.yaml"

    _parent = str(Path(__file__).parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

    from viz_utils import method_family_color_map, method_family_label_map


@app.cell(hide_code=True)
def title_cell():
    mo.md(
        """
        # f1 Estimation Diagnostics

        Pick supercollection → select scenarios → inspect estimated f1 loc/scale per replicate.
        Black line/star = truth.
        """
    )
    return


@app.cell
def config_cell():
    manifest = json.loads((RESULTS_ROOT / "manifest.json").read_text())
    plot_config = yaml.safe_load(PLOT_CONFIG_PATH.read_text()) or {}

    # batch hash → {true_loc, true_scale, sim_name}
    batch_true_f1 = {}
    for _bh, _bn in manifest["batches"].items():
        _f1f = _bn["simulation_spec"]["fields"]["f1"]["fields"]
        batch_true_f1[_bh] = {
            "true_loc": _f1f["loc"],
            "true_scale": _f1f["scale"],
            "sim_name": _bn["simulation_spec"]["fields"]["name"],
            "batch_name": _bn.get("name", ""),
        }

    # sim_name → list of batch hashes
    sim_to_batches: dict[str, list[str]] = {}
    for _bh, _ti in batch_true_f1.items():
        sim_to_batches.setdefault(_ti["sim_name"], []).append(_bh)

    # method hash → meta
    method_meta = {}
    for _mh, _mn in manifest["method_specs"].items():
        _name = _mn.get("name") or _mn.get("fields", {}).get("name", "")
        _kwargs = _mn.get("fields", _mn).get("kwargs", {})
        _f1k = _kwargs.get("f1") or {}
        _f1f = _f1k.get("fields", {}) if isinstance(_f1k, dict) else {}
        method_meta[_mh] = {
            "name": _name,
            "family": _name.rsplit("_L", 1)[0] if "_L" in _name else _name,
            "L": _kwargs.get("L", 1),
            "estimate_loc": _f1f.get("estimate_loc", True),
            "estimate_scale": _f1f.get("estimate_scale", True),
        }

    return batch_true_f1, manifest, method_meta, plot_config, sim_to_batches


@app.cell
def load_f1_data_cell(batch_true_f1, method_meta):
    """Load all existing twogroup fits into a flat DataFrame (cached)."""
    _rows = []
    for _fp in sorted(RESULTS_ROOT.glob("by_batch/*/fits/*/fits.parquet")):
        _bh = _fp.parts[-4]
        _mh = _fp.parts[-2]
        if _bh not in batch_true_f1 or _mh not in method_meta:
            continue
        _mi = method_meta[_mh]
        if not _mi["family"].startswith("twogroup"):
            continue
        _df = pl.read_parquet(_fp)
        if _df["two_group_state"][0] is None:
            continue
        _tgs = _df.select(pl.col("two_group_state").struct.unnest())
        _f1 = _tgs.select(pl.col("f1").struct.unnest())
        _ti = batch_true_f1[_bh]
        for _rep, _eloc, _escale in zip(
            _df["replicate"].to_list(),
            _f1["loc"].to_list(),
            _f1["scale"].to_list(),
        ):
            _rows.append({
                "method": _mi["name"],
                "family": _mi["family"],
                "L": _mi["L"],
                "estimate_loc": _mi["estimate_loc"],
                "estimate_scale": _mi["estimate_scale"],
                "replicate": _rep,
                "est_loc": _eloc,
                "est_scale": _escale,
                "true_loc": _ti["true_loc"],
                "true_scale": _ti["true_scale"],
                "sim_name": _ti["sim_name"],
                "batch_hash": _bh,
            })

    f1_data = pl.from_dicts(_rows) if _rows else pl.DataFrame()
    return (f1_data,)


@app.cell
def supercollection_cell(plot_config):
    _sc_names = sorted(plot_config.get("supercollections", {}).keys())
    supercollection_dd = mo.ui.dropdown(
        options=_sc_names,
        value=_sc_names[0] if _sc_names else None,
        label="Supercollection",
    )
    supercollection_dd
    return (supercollection_dd,)


@app.cell
def scenario_table_cell(supercollection_dd, plot_config, sim_to_batches, batch_true_f1):
    mo.stop(not supercollection_dd.value, mo.md("Select supercollection."))

    _sc_cfg = plot_config["supercollections"][supercollection_dd.value]
    _table_rows = []
    _coll_to_batch_hashes: dict[str, list[str]] = {}

    for _coll in _sc_cfg["collections"]:
        _cname = _coll["name"]
        _alias = _coll.get("alias", _cname)
        _bhs = sim_to_batches.get(_cname, [])
        _coll_to_batch_hashes[_cname] = _bhs
        if _bhs:
            _ti = batch_true_f1[_bhs[0]]
            _table_rows.append({
                "alias": _alias,
                "true_loc": _ti["true_loc"],
                "true_scale": _ti["true_scale"],
                "n_batches": len(_bhs),
                "collection": _cname,
            })
        else:
            _table_rows.append({
                "alias": _alias,
                "true_loc": None,
                "true_scale": None,
                "n_batches": 0,
                "collection": _cname,
            })

    scenario_table = mo.ui.table(
        _table_rows,
        selection="multi",
        show_download=False,
        label="Select scenarios",
        initial_selection=list(range(len(_table_rows))),
    )
    coll_to_batch_hashes = _coll_to_batch_hashes
    scenario_table
    return scenario_table, coll_to_batch_hashes


@app.cell
def method_controls_cell(f1_data):
    mo.stop(f1_data.is_empty(), mo.md("No twogroup fits found."))

    _l_opts = sorted(str(v) for v in f1_data["L"].unique().to_list())
    l_select = mo.ui.multiselect(options=_l_opts, value=_l_opts, label="L")

    _fam_opts = sorted(f1_data["family"].unique().to_list())
    family_select = mo.ui.multiselect(options=_fam_opts, value=_fam_opts, label="Method families")

    mo.hstack([l_select, family_select])
    return family_select, l_select


@app.cell
def filtered_data_cell(f1_data, scenario_table, coll_to_batch_hashes, l_select, family_select):
    mo.stop(not scenario_table.value, mo.md("Select at least one scenario."))

    _selected_bhs = set()
    for _row in scenario_table.value:
        for _bh in coll_to_batch_hashes.get(_row["collection"], []):
            _selected_bhs.add(_bh)

    _sel_l = [int(x) for x in l_select.value]
    _sel_fam = family_select.value

    filtered = f1_data.filter(
        pl.col("batch_hash").is_in(list(_selected_bhs))
        & pl.col("L").is_in(_sel_l)
        & pl.col("family").is_in(_sel_fam)
    )
    return (filtered,)


@app.cell
def plot_cell(filtered, scenario_table):
    import matplotlib.gridspec as gridspec

    mo.stop(filtered.is_empty(), mo.md("No data for selected filters."))

    _color_map = method_family_color_map()
    _label_map = method_family_label_map()

    _ordered_sims = [r["collection"] for r in scenario_table.value]
    _scenario_info = {
        r["collection"]: (r["true_loc"], r["true_scale"], r["alias"])
        for r in scenario_table.value
        if r["true_loc"] is not None
    }
    _scenarios = [
        (cn, *_scenario_info[cn])
        for cn in _ordered_sims
        if cn in _scenario_info
    ]

    # columns = individual methods (respects L filter)
    _methods = sorted(filtered["method"].unique().to_list())
    _n_sims = len(_scenarios)
    _n_methods = len(_methods)

    _cell = 2.8  # inches per cell
    _fig = plt.figure(figsize=(_n_methods * _cell, _n_sims * _cell))
    _outer = _fig.add_gridspec(
        _n_sims, _n_methods, hspace=0.55, wspace=0.45
    )

    _row_ref = {}  # ri -> first _ax_s in that row (for cross-method axis sharing)

    for _ri, (_cname, _tl, _ts, _alias) in enumerate(_scenarios):
        for _ci, _mname in enumerate(_methods):
            _inner = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=_outer[_ri, _ci],
                width_ratios=[3, 1],
                height_ratios=[1, 3],
                hspace=0.05, wspace=0.05,
            )
            _share_kw = {"sharex": _row_ref[_ri], "sharey": _row_ref[_ri]} if _ri in _row_ref else {}
            _ax_s = _fig.add_subplot(_inner[1, 0], **_share_kw)
            if _ri not in _row_ref:
                _row_ref[_ri] = _ax_s
            _ax_t = _fig.add_subplot(_inner[0, 0], sharex=_ax_s)
            _ax_r = _fig.add_subplot(_inner[1, 1], sharey=_ax_s)
            _ax_corner = _fig.add_subplot(_inner[0, 1])
            _ax_corner.axis("off")

            _fd = filtered.filter(
                (pl.col("sim_name") == _cname) & (pl.col("method") == _mname)
            )
            _fam = _fd["family"][0] if not _fd.is_empty() else None
            _color = _color_map.get(_fam, "steelblue") if _fam else "steelblue"

            if not _fd.is_empty():
                _elocs = _fd["est_loc"].to_numpy()
                _escales = _fd["est_scale"].to_numpy()
                _ax_s.scatter(_elocs, _escales, alpha=0.4, color=_color, s=12, linewidths=0)
                _ax_t.hist(_elocs, bins=20, color=_color, alpha=0.75)
                _ax_r.hist(_escales, bins=20, color=_color, alpha=0.75, orientation="horizontal")

            # truth
            _ax_s.scatter([_tl], [_ts], color="black", marker="*", s=80, zorder=6)
            _ax_s.axvline(_tl, color="black", linewidth=0.7, alpha=0.4)
            _ax_s.axhline(_ts, color="black", linewidth=0.7, alpha=0.4)
            _ax_t.axvline(_tl, color="black", linewidth=1.0, alpha=0.7)
            _ax_r.axhline(_ts, color="black", linewidth=1.0, alpha=0.7)

            # hide shared-axis tick labels
            plt.setp(_ax_t.get_xticklabels(), visible=False)
            plt.setp(_ax_r.get_yticklabels(), visible=False)
            _ax_t.tick_params(axis="x", which="both", bottom=False)
            _ax_r.tick_params(axis="y", which="both", left=False)

            # tick label size
            for _ax in (_ax_s, _ax_t, _ax_r):
                _ax.tick_params(labelsize=6)

            # axis labels: left edge and bottom edge only
            if _ci == 0:
                _ax_s.set_ylabel("scale", fontsize=7)
            else:
                _ax_s.set_ylabel("")
                plt.setp(_ax_s.get_yticklabels(), visible=False)
            if _ri == _n_sims - 1:
                _ax_s.set_xlabel("loc", fontsize=7)

            # col header (method name, top row only)
            if _ri == 0:
                _lbl = _label_map.get(_fam, _mname) if _fam else _mname
                _ax_t.set_title(_lbl, fontsize=8, pad=3)

            # row label (sim alias, leftmost col only)
            if _ci == 0:
                _ax_s.set_ylabel(f"{_alias}\nscale", fontsize=7)

    _fig
    return


@app.cell(hide_code=True)
def enrichment_title_cell():
    mo.md("## Enrichment Model: Intercept & Effect at Causal Variant")
    return


@app.cell
def load_enrichment_data_cell(batch_true_f1, method_meta):
    """Load intercept and mu[causal_idx] for all twogroup fits."""
    _rows = []
    for _fp in sorted(RESULTS_ROOT.glob("by_batch/*/fits/*/fits.parquet")):
        _bh = _fp.parts[-4]
        _mh = _fp.parts[-2]
        if _bh not in batch_true_f1 or _mh not in method_meta:
            continue
        _mi = method_meta[_mh]
        if not _mi["family"].startswith("twogroup"):
            continue

        _df = pl.read_parquet(_fp)
        if _df["two_group_state"][0] is None:
            continue

        _sims_path = RESULTS_ROOT / "by_batch" / _bh / "simulations.parquet"
        _sims = pl.read_parquet(_sims_path).select(
            "replicate",
            pl.col("simulation").struct.unnest(),
        )

        _fs = _df.select(
            "replicate",
            pl.col("family_state").struct.field("intercept").alias("est_intercept"),
            pl.col("single_effects").list.get(0).struct.field("mu").alias("mu_list"),
        )
        _joined = _fs.join(_sims.select(
            "replicate",
            pl.col("causal_indices").list.get(0).alias("causal_idx"),
            pl.col("causal_effects").list.get(0).alias("true_effect"),
            pl.col("intercept").alias("true_intercept"),
        ), on="replicate")

        _joined = _joined.with_columns(
            pl.col("mu_list").list.get(pl.col("causal_idx")).alias("mu_at_causal")
        )

        _ti = batch_true_f1[_bh]
        for _row in _joined.iter_rows(named=True):
            _rows.append({
                "method": _mi["name"],
                "family": _mi["family"],
                "L": _mi["L"],
                "replicate": _row["replicate"],
                "est_intercept": _row["est_intercept"],
                "mu_at_causal": _row["mu_at_causal"],
                "true_intercept": _row["true_intercept"],
                "true_effect": _row["true_effect"],
                "true_loc": _ti["true_loc"],
                "true_scale": _ti["true_scale"],
                "sim_name": _ti["sim_name"],
                "batch_hash": _bh,
            })

    enrich_data = pl.from_dicts(_rows) if _rows else pl.DataFrame()
    return (enrich_data,)


@app.cell
def enrichment_filtered_cell(enrich_data, scenario_table, coll_to_batch_hashes, l_select, family_select):
    mo.stop(not scenario_table.value, mo.md("Select scenarios above."))

    _sel_bhs = set()
    for _row in scenario_table.value:
        for _bh in coll_to_batch_hashes.get(_row["collection"], []):
            _sel_bhs.add(_bh)

    enrich_filtered = enrich_data.filter(
        pl.col("batch_hash").is_in(list(_sel_bhs))
        & pl.col("L").is_in([int(x) for x in l_select.value])
        & pl.col("family").is_in(family_select.value)
    )
    return (enrich_filtered,)


@app.cell
def enrichment_plot_cell(enrich_filtered, scenario_table):
    import matplotlib.gridspec as _gs2

    mo.stop(enrich_filtered.is_empty(), mo.md("No enrichment data for selected filters."))

    _color_map = method_family_color_map()
    _label_map = method_family_label_map()

    _ordered_sims = [r["collection"] for r in scenario_table.value]
    _scenario_info = {
        r["collection"]: (r["true_loc"], r["true_scale"], r["alias"])
        for r in scenario_table.value
        if r["true_loc"] is not None
    }
    _scenarios = [
        (cn, *_scenario_info[cn])
        for cn in _ordered_sims
        if cn in _scenario_info
    ]

    _methods = sorted(enrich_filtered["method"].unique().to_list())
    _n_sims = len(_scenarios)
    _n_methods = len(_methods)

    _cell = 2.8
    _fig2 = plt.figure(figsize=(_n_methods * _cell, _n_sims * _cell))
    _outer2 = _fig2.add_gridspec(_n_sims, _n_methods, hspace=0.55, wspace=0.45)

    for _ri, (_cname, _tl, _ts, _alias) in enumerate(_scenarios):
        for _ci, _mname in enumerate(_methods):
            _inner2 = _gs2.GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=_outer2[_ri, _ci],
                width_ratios=[3, 1],
                height_ratios=[1, 3],
                hspace=0.05, wspace=0.05,
            )
            _ax_s = _fig2.add_subplot(_inner2[1, 0])
            _ax_t = _fig2.add_subplot(_inner2[0, 0], sharex=_ax_s)
            _ax_r = _fig2.add_subplot(_inner2[1, 1], sharey=_ax_s)
            _ax_c = _fig2.add_subplot(_inner2[0, 1])
            _ax_c.axis("off")

            _fd = enrich_filtered.filter(
                (pl.col("sim_name") == _cname) & (pl.col("method") == _mname)
            )
            _fam = _fd["family"][0] if not _fd.is_empty() else None
            _color = _color_map.get(_fam, "steelblue") if _fam else "steelblue"

            if not _fd.is_empty():
                _x = _fd["est_intercept"].to_numpy()
                _y = _fd["mu_at_causal"].to_numpy()
                _true_intercept = float(_fd["true_intercept"][0])
                _true_effect = float(_fd["true_effect"][0])

                _ax_s.scatter(_x, _y, alpha=0.4, color=_color, s=12, linewidths=0)
                _ax_t.hist(_x, bins=20, color=_color, alpha=0.75)
                _ax_r.hist(_y, bins=20, color=_color, alpha=0.75, orientation="horizontal")

                _ax_s.scatter([_true_intercept], [_true_effect],
                              color="black", marker="*", s=80, zorder=6)
                _ax_s.axvline(_true_intercept, color="black", linewidth=0.7, alpha=0.4)
                _ax_s.axhline(_true_effect, color="black", linewidth=0.7, alpha=0.4)
                _ax_t.axvline(_true_intercept, color="black", linewidth=1.0, alpha=0.7)
                _ax_r.axhline(_true_effect, color="black", linewidth=1.0, alpha=0.7)

            plt.setp(_ax_t.get_xticklabels(), visible=False)
            plt.setp(_ax_r.get_yticklabels(), visible=False)
            _ax_t.tick_params(axis="x", which="both", bottom=False)
            _ax_r.tick_params(axis="y", which="both", left=False)
            for _ax in (_ax_s, _ax_t, _ax_r):
                _ax.tick_params(labelsize=6)

            if _ci == 0:
                _ax_s.set_ylabel(f"{_alias}\nmu_causal", fontsize=7)
            else:
                _ax_s.set_ylabel("")
                plt.setp(_ax_s.get_yticklabels(), visible=False)
            if _ri == _n_sims - 1:
                _ax_s.set_xlabel("intercept", fontsize=7)

            if _ri == 0:
                _lbl = _label_map.get(_fam, _mname) if _fam else _mname
                _ax_t.set_title(_lbl, fontsize=8, pad=3)

    _fig2
    return


@app.cell(hide_code=True)
def ez_title_cell():
    mo.md("## Ez Conditional Histograms\n\nDistribution of posterior E[z] for true z=0 (top) and z=1 (bottom) observations.")
    return


@app.cell
def load_ez_data_cell(batch_true_f1, method_meta):
    """Load Ez arrays and true z per (batch, method), grouped for plotting."""
    _items = []
    for _fp in sorted(RESULTS_ROOT.glob("by_batch/*/fits/*/fits.parquet")):
        _bh = _fp.parts[-4]
        _mh = _fp.parts[-2]
        if _bh not in batch_true_f1 or _mh not in method_meta:
            continue
        _mi = method_meta[_mh]
        if not _mi["family"].startswith("twogroup"):
            continue
        _df = pl.read_parquet(_fp)
        if _df["two_group_state"][0] is None:
            continue
        _tgs = _df.select(pl.col("two_group_state").struct.unnest())
        if "Ez" not in _tgs.columns:
            continue

        _sims_path = RESULTS_ROOT / "by_batch" / _bh / "simulations.parquet"
        _sims = pl.read_parquet(_sims_path)
        _rep_to_z = {
            row["replicate"]: np.array(row["z"], dtype=int)
            for row in _sims.select(
                "replicate",
                pl.col("simulation").struct.field("z"),
            ).iter_rows(named=True)
        }

        _ez_z0, _ez_z1 = [], []
        for _rep, _ez_raw in zip(_df["replicate"].to_list(), _tgs["Ez"].to_list()):
            _z = _rep_to_z.get(_rep)
            if _z is None:
                continue
            _ez_arr = np.asarray(_ez_raw, dtype=float)
            _ez_z0.extend(_ez_arr[_z == 0].tolist())
            _ez_z1.extend(_ez_arr[_z == 1].tolist())

        _ti = batch_true_f1[_bh]
        _items.append({
            "method": _mi["name"],
            "family": _mi["family"],
            "L": _mi["L"],
            "sim_name": _ti["sim_name"],
            "batch_hash": _bh,
            "ez_z0": np.array(_ez_z0),
            "ez_z1": np.array(_ez_z1),
        })

    ez_data = _items
    return (ez_data,)


@app.cell
def ez_plot_cell(ez_data, scenario_table, coll_to_batch_hashes, l_select, family_select):
    import matplotlib.gridspec as _gs3

    mo.stop(not scenario_table.value, mo.md("Select scenarios above."))

    _sel_bhs_ez = set()
    for _row in scenario_table.value:
        for _bh in coll_to_batch_hashes.get(_row["collection"], []):
            _sel_bhs_ez.add(_bh)
    _sel_l_ez = [int(x) for x in l_select.value]
    _sel_fam_ez = family_select.value

    # Aggregate Ez arrays by (sim_name, method)
    _ez_agg = {}
    for _item in ez_data:
        if (
            _item["batch_hash"] not in _sel_bhs_ez
            or _item["L"] not in _sel_l_ez
            or _item["family"] not in _sel_fam_ez
        ):
            continue
        _key = (_item["sim_name"], _item["method"])
        if _key not in _ez_agg:
            _ez_agg[_key] = {"ez_z0": [], "ez_z1": [], "family": _item["family"]}
        _ez_agg[_key]["ez_z0"].append(_item["ez_z0"])
        _ez_agg[_key]["ez_z1"].append(_item["ez_z1"])

    mo.stop(not _ez_agg, mo.md("No Ez data for selected filters."))

    _color_map_ez = method_family_color_map()
    _label_map_ez = method_family_label_map()

    _ordered_sims_ez = [r["collection"] for r in scenario_table.value]
    _scenario_info_ez = {r["collection"]: r["alias"] for r in scenario_table.value if r["true_loc"] is not None}
    _scenarios_ez = [(cn, _scenario_info_ez[cn]) for cn in _ordered_sims_ez if cn in _scenario_info_ez]

    _all_methods_ez = sorted(set(m for _, m in _ez_agg.keys()))
    _n_sims_ez = len(_scenarios_ez)
    _n_methods_ez = len(_all_methods_ez)

    _cell_ez = 2.2
    _fig_ez = plt.figure(figsize=(_n_methods_ez * _cell_ez, _n_sims_ez * _cell_ez * 1.5))
    _outer_ez = _fig_ez.add_gridspec(_n_sims_ez, _n_methods_ez, hspace=0.65, wspace=0.4)

    for _ri, (_cname, _alias) in enumerate(_scenarios_ez):
        for _ci, _mname in enumerate(_all_methods_ez):
            _key = (_cname, _mname)
            _inner_ez = _gs3.GridSpecFromSubplotSpec(2, 1, subplot_spec=_outer_ez[_ri, _ci], hspace=0.05)
            _ax_z0 = _fig_ez.add_subplot(_inner_ez[0])
            _ax_z1 = _fig_ez.add_subplot(_inner_ez[1], sharex=_ax_z0)

            if _key in _ez_agg:
                _d = _ez_agg[_key]
                _fam = _d["family"]
                _color = _color_map_ez.get(_fam, "steelblue")
                _arr_z0 = np.concatenate(_d["ez_z0"]) if _d["ez_z0"] else np.array([])
                _arr_z1 = np.concatenate(_d["ez_z1"]) if _d["ez_z1"] else np.array([])
                if len(_arr_z0):
                    _ax_z0.hist(_arr_z0, bins=30, color=_color, alpha=0.75, density=True)
                if len(_arr_z1):
                    _ax_z1.hist(_arr_z1, bins=30, color=_color, alpha=0.75, density=True)
            else:
                _fam = None

            plt.setp(_ax_z0.get_xticklabels(), visible=False)
            _ax_z0.tick_params(axis="x", which="both", bottom=False)
            for _ax in (_ax_z0, _ax_z1):
                _ax.tick_params(labelsize=6)
                _ax.set_xlim(0, 1)

            if _ci == 0:
                _ax_z0.set_ylabel(f"{_alias}\nz=0", fontsize=7)
                _ax_z1.set_ylabel("z=1", fontsize=7)
            else:
                plt.setp(_ax_z0.get_yticklabels(), visible=False)
                plt.setp(_ax_z1.get_yticklabels(), visible=False)

            if _ri == _n_sims_ez - 1:
                _ax_z1.set_xlabel("Ez", fontsize=7)

            if _ri == 0:
                _lbl = _label_map_ez.get(_mname.rsplit("_L", 1)[0], _mname) if _fam else _mname
                _ax_z0.set_title(_lbl, fontsize=8, pad=3)

    _fig_ez
    return


if __name__ == "__main__":
    app.run()
