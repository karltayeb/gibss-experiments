import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import json
    import sys
    from pathlib import Path

    import marimo as mo

    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    import plot_ready
    from utils import write_yaml


@app.cell(hide_code=True)
def title_cell():
    _md = mo.md("# Collection Creator")
    _md
    return


@app.cell
def manifest_cell():
    _manifest_path = Path(__file__).parent.parent / "results" / "manifest.json"
    manifest = json.loads(_manifest_path.read_text())
    mo.md(f"Loaded manifest: **{len(manifest['batches'])} batches**, **{len(manifest['method_specs'])} methods**, **{len(manifest['simulation_specs'])} simulations**")
    return (manifest,)


@app.cell
def sim_index_cell(manifest):
    """Parse simulation dimensions from batch nodes for filter UI."""
    def _parse_sim_name(name: str) -> dict:
        parts = name.split("__")
        d = {"name": name, "design": parts[0] if parts else ""}
        if len(parts) > 1:
            d["regime"] = parts[1]
        if len(parts) > 2:
            param_part = parts[2]
            idx = param_part.rfind("_")
            if idx > 0:
                d["parameter"] = param_part[:idx]
                try:
                    d["value"] = float(param_part[idx + 1:])
                except ValueError:
                    d["parameter"] = param_part
            else:
                d["parameter"] = param_part
        return d

    def _parse_method_node(node: dict) -> dict:
        fields = node.get("fields", {})
        name = fields.get("name", "")
        kwargs = fields.get("kwargs", {})
        return {
            "hash": node["__spec_hash__"],
            "name": name,
            "L": kwargs.get("L"),
            "threshold": kwargs.get("threshold"),
            "family": name.rsplit("_L", 1)[0] if "_L" in name else name,
        }

    # batch hash → {batch_node, sim_dims}
    batch_index = {}
    for _bh, _bn in manifest["batches"].items():
        _sim_name = _bn["simulation_spec"]["fields"]["name"]
        batch_index[_bh] = {**_bn, "sim_dims": _parse_sim_name(_sim_name)}

    method_index = {
        mh: _parse_method_node(mn)
        for mh, mn in manifest["method_specs"].items()
    }

    return batch_index, method_index


@app.cell
def mode_cell():
    mode_tabs = mo.ui.tabs({
        "Build": mo.md(""),
        "Batch-generate": mo.md(""),
        "Compose": mo.md(""),
    })
    mode_tabs
    return (mode_tabs,)


@app.cell
def build_filters_cell(batch_index, method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Build")

    _all_designs = sorted({v["sim_dims"].get("design", "") for v in batch_index.values() if v["sim_dims"].get("design")})
    _all_regimes = sorted({v["sim_dims"].get("regime", "") for v in batch_index.values() if v["sim_dims"].get("regime")})
    _all_params = sorted({v["sim_dims"].get("parameter", "") for v in batch_index.values() if v["sim_dims"].get("parameter")})
    _all_families = sorted({v["family"] for v in method_index.values()})

    design_filter = mo.ui.multiselect(options=_all_designs, value=_all_designs, label="Design")
    regime_filter = mo.ui.multiselect(options=_all_regimes, value=_all_regimes, label="Regime")
    param_filter = mo.ui.multiselect(options=_all_params, value=_all_params, label="Parameter")
    family_filter = mo.ui.multiselect(options=_all_families, value=_all_families, label="Method family")

    mo.hstack([design_filter, regime_filter, param_filter, family_filter])
    return design_filter, family_filter, param_filter, regime_filter


@app.cell
def build_selectors_cell(batch_index, design_filter, family_filter, method_index, mode_tabs, param_filter, regime_filter):
    mo.stop(mode_tabs.value != "Build")

    _filtered_batches = {
        bh: bv for bh, bv in batch_index.items()
        if bv["sim_dims"].get("design") in design_filter.value
        and bv["sim_dims"].get("regime", "") in (regime_filter.value or [""])
        and bv["sim_dims"].get("parameter", "") in (param_filter.value or [""])
    }
    _filtered_methods = {
        mh: mv for mh, mv in method_index.items()
        if mv["family"] in family_filter.value
    }

    _batch_options = {bh: bv["name"] for bh, bv in _filtered_batches.items()}
    _method_options = {mh: f"{mv['name']} (L={mv['L']})" for mh, mv in _filtered_methods.items()}

    batch_multiselect = mo.ui.multiselect(
        options=_batch_options, value=list(_batch_options.keys()), label="Batches"
    )
    method_multiselect = mo.ui.multiselect(
        options=_method_options, value=list(_method_options.keys()), label="Methods"
    )
    collection_name_input = mo.ui.text(placeholder="my_collection_name", label="Collection name")

    mo.vstack([
        mo.hstack([batch_multiselect, method_multiselect]),
        collection_name_input,
        mo.md(f"**{len(batch_multiselect.value)} batches × {len(method_multiselect.value)} methods selected**"),
    ])
    return batch_multiselect, collection_name_input, method_multiselect


@app.cell
def build_write_cell(batch_multiselect, collection_name_input, manifest, method_multiselect, mode_tabs):
    mo.stop(mode_tabs.value != "Build")
    mo.stop(not collection_name_input.value.strip())

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _collections_dir.mkdir(parents=True, exist_ok=True)
    _out_path = _collections_dir / f"{collection_name_input.value.strip()}.yaml"

    def _do_write(_btn):
        _batch_nodes = [manifest["batches"][h] for h in batch_multiselect.value]
        _method_nodes = [manifest["method_specs"][h] for h in method_multiselect.value]
        _node = plot_ready.build_collection_yaml_node(
            name=collection_name_input.value.strip(),
            batch_nodes=_batch_nodes,
            method_nodes=_method_nodes,
        )
        write_yaml(_node, str(_out_path))

    write_btn = mo.ui.button(label=f"Write {_out_path.name}", on_click=_do_write)
    write_btn
    return (write_btn,)


@app.cell
def batchgen_cell(batch_index, manifest, method_index, mode_tabs):
    mo.stop(mode_tabs.value != "Batch-generate")

    _all_method_options = {
        mh: f"{mv['name']} (L={mv['L']})"
        for mh, mv in method_index.items()
    }

    batchgen_method_select = mo.ui.multiselect(
        options=_all_method_options,
        value=list(_all_method_options.keys()),
        label="Methods to include in every generated collection",
    )

    # one collection per unique simulation name (grouping all its batches)
    sim_to_batches: dict[str, list[str]] = {}
    for _bh, _bv in batch_index.items():
        _sname = _bv["simulation_spec"]["fields"]["name"]
        sim_to_batches.setdefault(_sname, []).append(_bh)

    _preview_rows = [
        {"collection": sim_name, "n_batches": len(batch_hashes)}
        for sim_name, batch_hashes in sorted(sim_to_batches.items())
    ]

    import polars as _pl
    _preview_df = _pl.from_dicts(_preview_rows)

    mo.vstack([
        batchgen_method_select,
        mo.md(f"Will generate **{len(sim_to_batches)} collection yamls**, one per simulation."),
        mo.ui.table(_preview_df, selection=None),
    ])
    return batchgen_method_select, sim_to_batches


@app.cell
def batchgen_write_cell(batchgen_method_select, manifest, mode_tabs, sim_to_batches):
    mo.stop(mode_tabs.value != "Batch-generate")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_batchgen(_btn):
        _collections_dir.mkdir(parents=True, exist_ok=True)
        _method_nodes = [manifest["method_specs"][h] for h in batchgen_method_select.value]
        for _sim_name, _batch_hashes in sim_to_batches.items():
            _batch_nodes = [manifest["batches"][h] for h in _batch_hashes]
            _node = plot_ready.build_collection_yaml_node(
                name=_sim_name,
                batch_nodes=_batch_nodes,
                method_nodes=_method_nodes,
            )
            write_yaml(_node, str(_collections_dir / f"{_sim_name}.yaml"))

    batchgen_write_btn = mo.ui.button(
        label=f"Generate {len(sim_to_batches)} collection yamls",
        on_click=_do_batchgen,
    )
    batchgen_write_btn
    return (batchgen_write_btn,)


@app.cell
def compose_cell(mode_tabs):
    mo.stop(mode_tabs.value != "Compose")

    _collections_dir = Path(__file__).parent.parent / "results" / "collections"
    _existing = {
        p.stem: str(p)
        for p in sorted(_collections_dir.glob("*.yaml"))
    }

    compose_select = mo.ui.multiselect(
        options=_existing,
        label="Collections to union",
    )
    compose_name_input = mo.ui.text(placeholder="my_union_collection", label="Union collection name")

    if compose_select.value:
        import yaml as _yaml
        _nodes = [_yaml.safe_load(open(p)) for p in compose_select.value]
        _all_batches = {b["__spec_hash__"] for n in _nodes for b in n["batches"]}
        _all_methods = {m["__spec_hash__"] for n in _nodes for m in n["method_specs"]}
        _preview_md = mo.md(
            f"Union: **{len(_all_batches)} batches**, **{len(_all_methods)} methods** (after dedup)"
        )
    else:
        _preview_md = mo.md("Select collections above to preview the union.")

    mo.vstack([compose_select, compose_name_input, _preview_md])
    return compose_name_input, compose_select


@app.cell
def compose_write_cell(compose_name_input, compose_select, mode_tabs):
    mo.stop(mode_tabs.value != "Compose")
    mo.stop(not compose_name_input.value.strip())
    mo.stop(not compose_select.value)

    import yaml as _yaml
    _collections_dir = Path(__file__).parent.parent / "results" / "collections"

    def _do_compose(_btn):
        _nodes = [_yaml.safe_load(open(p)) for p in compose_select.value]
        _union_node = plot_ready.union_collection_yaml_nodes(
            name=compose_name_input.value.strip(),
            collection_nodes=_nodes,
        )
        _out = _collections_dir / f"{compose_name_input.value.strip()}.yaml"
        write_yaml(_union_node, str(_out))

    compose_write_btn = mo.ui.button(
        label=f"Write union to {compose_name_input.value.strip()}.yaml",
        on_click=_do_compose,
    )
    compose_write_btn
    return (compose_write_btn,)
